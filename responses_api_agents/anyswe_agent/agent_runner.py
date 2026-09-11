# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import base64
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from fastapi import Request


sys.path.insert(0, "/nemo_gym_mount")
agent_deps_dir = os.environ.get("NGSWE_AGENT_DEPS_DIR", "/agent_deps_mount")

# Capture the task image's own python3 BEFORE prepending the agent-deps runtime to PATH, so
# we can point the agent's code tools at the repo's interpreter even on images that don't
# ship the classic conda env (e.g. SWE-bench Pro's /app images use /usr/local/bin/python3).
_image_python = shutil.which("python3") or shutil.which("python")

os.environ["PATH"] = f"{agent_deps_dir}/bin:" + os.environ.get("PATH", "")

# Activate the repo's Python environment so the agent's code-execution tools (execute_code,
# `python`) import the repo's packages and can run its test suite. hermes' execute_code picks
# its interpreter from VIRTUAL_ENV/CONDA_PREFIX (tools/code_execution_tool.py
# _resolve_child_python) and otherwise falls back to the isolated agent-deps interpreter
# above -- which lacks the repo's packages, so `import <repo pkg>` / pytest raise
# ModuleNotFoundError. Detect the repo env across image families instead of hardcoding one:
#   - classic / SWE-bench Verified: /opt/miniconda3/envs/testbed (conda)
#   - SWE-bench Pro (/app images):  the image's system python prefix (e.g. /usr/local, py3.11)
_testbed_venv = "/opt/miniconda3/envs/testbed"
if os.path.isdir(_testbed_venv):  # SWE Verified
    _repo_prefix = _testbed_venv
elif _image_python:  # SWE Pro
    _cand = os.path.dirname(os.path.dirname(_image_python))
    _repo_prefix = _cand if os.path.isfile(os.path.join(_cand, "bin", os.path.basename(_image_python))) else None
else:
    _repo_prefix = None
if _repo_prefix:
    os.environ["VIRTUAL_ENV"] = _repo_prefix
    os.environ["PATH"] = os.path.join(_repo_prefix, "bin") + ":" + os.environ.get("PATH", "")

_REPO_CANDIDATES = ("/testbed", "/workspace/repo", "/app", "/root/repo")


def _json_env(name: str) -> dict:
    encoded = os.environ.get(f"{name}_B64")
    if encoded:
        return json.loads(base64.b64decode(encoded).decode())
    return json.loads(os.environ.get(name, "{}"))


def _find_repo() -> Path | None:
    return next(
        (repo for candidate in _REPO_CANDIDATES if (repo := Path(candidate)).exists() and (repo / ".git").exists()),
        None,
    )


def _alternate_index_env(index_path: Path) -> dict[str, str]:
    return {**os.environ, "GIT_INDEX_FILE": str(index_path)}


def _snapshot_repo(repo: Path, index_path: Path) -> str:
    """Write the task image's initial working tree to an alternate Git index."""
    index_path.unlink(missing_ok=True)
    env = _alternate_index_env(index_path)
    subprocess.run(["git", "read-tree", "HEAD"], check=True, cwd=repo, env=env)
    subprocess.run(["git", "add", "-A"], check=True, cwd=repo, env=env)
    return subprocess.run(
        ["git", "write-tree"],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo,
        env=env,
    ).stdout.strip()


def _extract_patch(repo: Path, index_path: Path, baseline_tree: str) -> str:
    """Return only changes made after the agent started, including untracked files."""
    env = _alternate_index_env(index_path)
    subprocess.run(["git", "add", "-A"], check=True, cwd=repo, env=env)
    return subprocess.run(
        ["git", "diff", "--no-color", "--cached", baseline_tree],
        capture_output=True,
        text=True,
        errors="replace",
        check=True,
        cwd=repo,
        env=env,
    ).stdout


def main() -> None:
    model_url = os.environ.get("NGSWE_MODEL_URL", "")
    model_name = os.environ["NGSWE_MODEL_NAME"]
    instruction = Path("/trajectories_mount/instruction.txt").read_text()
    agent_kwargs = _json_env("NGSWE_AGENT_KWARGS")
    sampling = _json_env("NGSWE_SAMPLING")

    # nemo_gym adds the current working directory to sys.path on import (for its own plugin
    # discovery). Here cwd is the task repo (e.g. /testbed), so a repo that ships a top-level
    # folder matching one of the agent's own module names (e.g. sphinx's `utils`, requests'
    # `requests`) would shadow it and crash the agent at startup. Drop cwd back out afterward.
    _cwd = os.getcwd()
    from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
    from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.server_utils import ServerClient

    sys.path[:] = [p for p in sys.path if p not in (_cwd, "", ".")]

    module = importlib.import_module(os.environ["NGSWE_AGENT_MODULE"])
    agent_class = getattr(module, os.environ["NGSWE_AGENT_CLASS"])
    config_class = getattr(module, os.environ["NGSWE_AGENT_CONFIG_CLASS"])

    server_name = "policy_model"
    global_config = (
        {server_name: {"responses_api_models": {"model": {"host": "0.0.0.0", "port": 0}}}} if model_url else {}
    )
    client = ServerClient.model_construct(global_config_dict=global_config)
    client._build_server_base_url = lambda config: model_url
    config_sampling = {key: value for key, value in sampling.items() if key in config_class.model_fields}
    model_server = ModelServerRef(name=server_name, type="responses_api_models") if model_url else None
    config = config_class(
        host="0.0.0.0",
        port=0,
        name=agent_class.__name__.lower(),
        entrypoint="app.py",
        model_server=model_server,
        resources_server=ResourcesServerRef(name="anyswe", type="resources_servers"),
        **{**agent_kwargs, **config_sampling},
    )
    agent = agent_class(config=config, server_client=client)

    # Some published task images contain pre-existing working-tree changes. Snapshot that exact
    # state through an alternate index so the submitted patch contains only the agent's edits and
    # does not mutate the repository index the agent sees.
    repo = _find_repo()
    index_path = Path("/trajectories_mount/anyswe-baseline.index")
    baseline_tree = _snapshot_repo(repo, index_path) if repo else None

    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[NeMoGymEasyInputMessage(role="user", content=instruction)],
        model=model_name,
        **sampling,
    )
    request = Request({"type": "http", "path_params": {}})
    response = asyncio.run(agent.responses(request=request, body=body))
    Path("/trajectories_mount/response.json").write_text(response.model_dump_json())
    print(f"agent finished: {len(response.output)} output items", flush=True)

    patch = _extract_patch(repo, index_path, baseline_tree) if repo and baseline_tree else ""
    index_path.unlink(missing_ok=True)
    if repo:
        print(f"patch: {len(patch)} chars from {repo}", flush=True)
    Path("/trajectories_mount/patch.diff").write_text(patch)


if __name__ == "__main__":
    main()
