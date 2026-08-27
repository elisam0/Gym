# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""swe-bench-pro harness — host-side (flat) graded.

Runs ScaleAI's SWE-bench Pro (https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro)
tasks. Unlike ``swe-bench``/``swe-bench-ext``, Pro ships a per-instance,
pre-pinned evaluation recipe (``run_script`` + ``parser_script`` + Dockerfile
ENV lines) that resets, patches, tests, and JSON-parses the result as a single
generated ``entryscript.sh`` — mirroring the upstream evaluator's own
orchestration rather than a generic pytest/junit framework runner. That's why
this is its own harness instead of a ``swe-bench-ext`` row: the grading
contract (one opaque script + one JSON parser, both supplied by the dataset
row) doesn't fit the framework-based path.

Provenance: the grading recipe below (``_parse_string_list``,
``_strip_binary_hunks``, ``_create_entryscript``, ``_assemble_workspace_files``,
``_grade_output``) is ported from this repo's own
``resources_servers/swebench_pro/verification.py``, introduced in
https://github.com/NVIDIA-NeMo/Gym/pull/2498 (public), which in turn ports
functions from the upstream evaluator (MIT licensed):
https://github.com/scaleapi/SWE-bench_Pro-os/blob/ca10a60a5fcae51e6948ffe1485d4153d421e6c5/swe_bench_pro_eval.py
Each ported function below keeps a comment noting it is copied/adapted from
that source, same as ``verification.py`` does.
"""

from __future__ import annotations

import ast
import json
import re
from typing import TYPE_CHECKING, Any

from nemo_gym.sandbox import SandboxResources, SandboxSpec
from responses_api_agents.swe_env.harness import EvalArtifacts, SweEvalReport, SweTask, SweTaskHarness


if TYPE_CHECKING:
    from responses_api_agents.swe_env.sandbox import AsyncSweEnvironment


_WORKSPACE_DIR = "/workspace"
_PATCH_PATH = f"{_WORKSPACE_DIR}/patch.diff"
_RUN_SCRIPT_PATH = f"{_WORKSPACE_DIR}/run_script.sh"
_PARSER_PATH = f"{_WORKSPACE_DIR}/parser.py"
_ENTRY_SCRIPT_PATH = f"{_WORKSPACE_DIR}/entryscript.sh"
_STDOUT_PATH = f"{_WORKSPACE_DIR}/stdout.log"
_STDERR_PATH = f"{_WORKSPACE_DIR}/stderr.log"
_OUTPUT_PATH = f"{_WORKSPACE_DIR}/output.json"
_PATCH_STATUS_PATH = f"{_WORKSPACE_DIR}/patch_apply_status"


def _parse_string_list(value: Any) -> list[str]:
    """Parse a JSON/Python list-of-strings value without executing dataset content.

    Ported from ``resources_servers/swebench_pro/verification.py::parse_string_list``
    (itself the NeMo Gym safe-parsing replacement for upstream's ``eval()`` call).
    """
    if isinstance(value, list):
        parsed = value
    elif isinstance(value, str):
        if not value.strip():
            parsed = []
        else:
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = ast.literal_eval(value)
    else:
        parsed = list(value or [])

    if not isinstance(parsed, (list, tuple)) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"Expected a list of strings, got {value!r}")
    return list(parsed)


def _strip_binary_hunks(patch: str) -> str:
    """Remove binary diff sections from a git patch.

    Copied verbatim (structure-wise) from upstream's ``strip_binary_hunks``, via
    ``resources_servers/swebench_pro/verification.py``.
    """
    if not patch:
        return patch
    sections = re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)
    kept: list[str] = []
    for section in sections:
        if not section.strip():
            continue
        if re.search(r"^Binary files .* differ$", section, re.MULTILINE):
            continue
        if re.search(r"^GIT binary patch$", section, re.MULTILINE):
            continue
        kept.append(section)
    return "".join(kept)


def _create_entryscript(sample: dict[str, Any]) -> str:
    """Build the in-container script: reset, checkout, patch, test, parse.

    Adapted from upstream's ``create_entryscript`` via
    ``resources_servers/swebench_pro/verification.py``; see that file for the
    per-line "NeMo Gym change" annotations this reproduces (safe list parsing
    instead of ``eval()``, dataset-embedded Dockerfiles instead of a live
    upstream checkout, optional Go module prefetch).
    """
    before_repo_set_cmd = (sample.get("before_repo_set_cmd") or "").strip().split("\n")[-1]
    selected_test_files_to_run = ",".join(_parse_string_list(sample.get("selected_test_files_to_run", [])))
    base_commit = sample["base_commit"]

    should_prefetch_go_modules = sample.get("prefetch_go_modules", False) and (
        str(sample.get("repo_language", "")).lower() == "go" or "go test " in sample.get("run_script", "")
    )
    go_module_prefetch_cmd = ""
    if should_prefetch_go_modules:
        go_module_prefetch_cmd = (
            "# NeMo Gym change: prefetch modules without modifying the upstream test script.\n"
            "if [ -f go.mod ]; then\n"
            "  go mod download\n"
            "fi"
        )

    env_cmds: list[str] = []
    for dockerfile_content in (sample.get("base_dockerfile", ""), sample.get("instance_dockerfile", "")):
        for line in dockerfile_content.split("\n"):
            line = line.strip()
            if line.startswith("ENV"):
                env_cmds.append(line.replace("ENV", "export", 1))
    env_cmds_block = "\n".join(env_cmds)

    return f"""\
{env_cmds_block}
# apply patch
cd /app
git reset --hard {base_commit}
git checkout {base_commit}
git apply -v {_PATCH_PATH}
PATCH_APPLY_STATUS=$?
{before_repo_set_cmd}
{go_module_prefetch_cmd}
bash {_RUN_SCRIPT_PATH} {selected_test_files_to_run} > {_STDOUT_PATH} 2> {_STDERR_PATH}
python {_PARSER_PATH} {_STDOUT_PATH} {_STDERR_PATH} {_OUTPUT_PATH}
printf '%s\\n' "$PATCH_APPLY_STATUS" > {_PATCH_STATUS_PATH}
"""


def _assemble_workspace_files(patch: str, sample: dict[str, Any]) -> dict[str, str]:
    """Build the {path: contents} map the entryscript expects on disk.

    Adapted from upstream's ``assemble_workspace_files`` via
    ``resources_servers/swebench_pro/verification.py``.
    """
    return {
        _PATCH_PATH: _strip_binary_hunks(patch),
        _RUN_SCRIPT_PATH: sample.get("run_script", ""),
        _PARSER_PATH: sample.get("parser_script", ""),
        _ENTRY_SCRIPT_PATH: _create_entryscript(sample),
    }


def _grade_output(output: dict[str, Any], sample: dict[str, Any]) -> bool:
    """Apply the upstream resolution rule: every required test PASSED.

    Ported from ``resources_servers/swebench_pro/verification.py::grade_output``
    (itself extracted from upstream's ``main()``).
    """
    passed_tests = {t["name"] for t in output.get("tests", []) if t.get("status") == "PASSED"}
    f2p = set(_parse_string_list(sample.get("fail_to_pass", [])))
    p2p = set(_parse_string_list(sample.get("pass_to_pass", [])))
    return (f2p | p2p) <= passed_tests


class SweBenchProHarness(SweTaskHarness):
    """Harness for the swe-bench-pro family of SWE tasks (host-side / flat graded).

    Grading runs a single dataset-supplied ``entryscript.sh`` (reset -> checkout
    -> patch -> run_script -> parser_script -> JSON) rather than the standard
    swebench harness or a generic framework parser, so ``reset_repo`` is a no-op
    here: the entryscript performs its own ``git reset --hard`` + checkout.
    """

    name = "swe-bench-pro"
    grade_strategy = "flat-host-grade"

    def build_spec(self, task: SweTask) -> SandboxSpec:
        """Build the sandbox spec for a swe-bench-pro task.

        Args:
            task: The SWE task; ``task.image`` must already resolve to the
                row's ``dockerhub_tag`` and ``task.repo_workdir`` should be
                ``"/app"`` (Pro's repo checkout root, not ``/testbed``).

        Returns:
            SandboxSpec: The populated sandbox spec.
        """
        return SandboxSpec(
            image=task.image,
            workdir=task.repo_workdir,
            ttl_s=task.metadata.get("ttl_s", 1800),
            ready_timeout_s=task.metadata.get("ready_timeout_s", 1200),
            env={"GIT_PAGER": "cat"},
            metadata={
                "instance_id": task.instance_id[:63],
                "benchmark": task.benchmark,
                "harness": self.name,
            },
            resources=SandboxResources.from_mapping(task.metadata.get("resources", {})),
            provider_options=dict(task.metadata.get("provider_options", {})),
        )

    def supports_provider(self, provider_name: str) -> bool:
        """Flat/host-graded: works on any exec-capable provider.

        Args:
            provider_name: The sandbox provider name.

        Returns:
            bool: Always ``True``.
        """
        return True

    async def materialize(self, env: "AsyncSweEnvironment", task: SweTask) -> None:
        """Write the model patch and the dataset's pinned eval scripts into the sandbox.

        Args:
            env: The started environment to write into.
            task: The task carrying the model patch and, in
                ``task.metadata["instance_dict"]``, the pinned ``run_script``/
                ``parser_script``/Dockerfile/``before_repo_set_cmd`` assets.
        """
        sample = dict(task.metadata.get("instance_dict") or {})
        sample.setdefault("base_commit", task.base_commit)
        sample.setdefault("fail_to_pass", task.fail_to_pass)
        sample.setdefault("pass_to_pass", task.pass_to_pass)
        files = _assemble_workspace_files(task.model_patch, sample)
        for remote_path, contents in files.items():
            await env.write_text(remote_path, contents)
        await env.execute(f"chmod +x {_RUN_SCRIPT_PATH} {_ENTRY_SCRIPT_PATH}", cwd="/app")

    async def reset_repo(self, env: "AsyncSweEnvironment", task: SweTask) -> None:
        """No-op: the generated entryscript performs its own reset + checkout.

        Args:
            env: The active environment (unused).
            task: The task (unused).
        """
        return None

    async def run_eval(self, env: "AsyncSweEnvironment", task: SweTask) -> EvalArtifacts:
        """Run the generated entryscript and capture its raw output for host-side grading.

        Args:
            env: The active environment to execute commands in.
            task: The SWE task being evaluated.

        Returns:
            EvalArtifacts: ``test_output`` holds the combined stdout/stderr log;
            ``raw`` additionally carries ``output_json`` (the parser's raw text,
            parsed in ``grade``) and ``patch_apply_status``.
        """
        result = await env.execute(
            f"bash {_ENTRY_SCRIPT_PATH}",
            cwd="/app",
            is_eval=True,
            timeout_s=task.metadata.get("tests_timeout", 1800),
        )
        error_type = result.get("error_type")
        if error_type in ("sandbox", "timeout"):
            # The entryscript redirects run_script.sh's own stdout/stderr into files
            # (see _create_entryscript), so `result["output"]` -- the *outer* bash
            # command's own stdout -- is empty even on a genuine hang partway through
            # run_script.sh. Best-effort read those files back: the sandbox may still
            # be reachable after only the timed-out command was killed, and whatever
            # was flushed before the kill is the only way to tell a hang in
            # prepare_test_environment (e.g. a service that never came up) apart from
            # a plain sandbox failure.
            partial_stdout = partial_stderr = ""
            try:
                partial_stdout = (await env.execute(f"cat {_STDOUT_PATH}", cwd="/app")).get("stdout", "")
                partial_stderr = (await env.execute(f"cat {_STDERR_PATH}", cwd="/app")).get("stdout", "")
            except Exception:
                pass
            test_output = result.get("output", "")
            if partial_stdout or partial_stderr:
                test_output = f"{test_output}\n\nPARTIAL STDOUT (before {error_type}):\n{partial_stdout}\n\nPARTIAL STDERR:\n{partial_stderr}"
            # anyswe_agent's own metrics schema (shared across every family) has no field for
            # test output, so it would otherwise be silently dropped before reaching the
            # published rollout/metrics JSON. Print it instead so it lands in the run's own
            # (already-published) console log -- the only way to see *where* a grading timeout
            # actually stalled (e.g. a service wait-loop that never got satisfied) versus
            # guessing from the outside.
            print(
                f"[swe-bench-pro] {task.instance_id}: run_eval hit {error_type!r}. "
                f"Last ~2000 chars of captured output:\n{test_output[-2000:]}",
                flush=True,
            )
            return EvalArtifacts(test_output=test_output, return_code=-1, raw={"error_type": error_type})

        # Read via "stdout" specifically, not the combined "output": some sandbox providers emit
        # their own informational messages on stderr for every exec call, which would otherwise
        # get spliced into file contents here and corrupt the patch-apply-status comparison and
        # the JSON parse below.
        stdout_log = (await env.execute(f"cat {_STDOUT_PATH}", cwd="/app")).get("stdout", "")
        stderr_log = (await env.execute(f"cat {_STDERR_PATH}", cwd="/app")).get("stdout", "")
        output_json = (await env.execute(f"cat {_OUTPUT_PATH}", cwd="/app")).get("stdout", "")
        patch_status = (await env.execute(f"cat {_PATCH_STATUS_PATH}", cwd="/app")).get("stdout", "").strip()

        return EvalArtifacts(
            test_output=f"STDOUT:\n{stdout_log}\n\nSTDERR:\n{stderr_log}",
            return_code=result.get("returncode", -1),
            patch_applied=(patch_status == "0"),
            raw={"output_json": output_json, "error_type": error_type},
        )

    def grade(self, task: SweTask, artifacts: EvalArtifacts) -> SweEvalReport:
        """Parse the entryscript's JSON output and apply Pro's resolution rule.

        Mirrors ``resources_servers/swebench_pro/verification.py::run_verification``'s
        post-processing: an execution/timeout/sandbox failure or invalid JSON masks
        the sample via ``error_kind`` rather than scoring it unresolved; otherwise
        resolution requires both a clean patch apply and every FAIL_TO_PASS/
        PASS_TO_PASS test reporting PASSED (``_grade_output``).

        Args:
            task: The SWE task being graded.
            artifacts: The raw output captured by ``run_eval``.

        Returns:
            SweEvalReport: The graded result.
        """
        if artifacts.raw.get("error_type") in {"sandbox", "timeout"}:
            return SweEvalReport(
                instance_id=task.instance_id,
                patch_exists=bool(task.model_patch),
                patch_applied=artifacts.patch_applied,
                error_kind=artifacts.raw["error_type"],
            )

        output_json = artifacts.raw.get("output_json", "")
        try:
            test_results = json.loads(output_json) if output_json else None
        except json.JSONDecodeError:
            test_results = None

        if not isinstance(test_results, dict):
            # Same visibility gap as the run_eval timeout branch: anyswe_agent's own metrics
            # schema (shared across every family) has no field to carry tests_status through
            # to the published rollout/metrics JSON, so print it -- this is currently the
            # dominant error_kind across a live SWE-bench Pro batch, and without this there is
            # no way to tell "parser crashed" from "test command never ran" from the outside.
            print(
                f"[swe-bench-pro] {task.instance_id}: grade() got non-JSON/missing output.json "
                f"(patch_applied={artifacts.patch_applied}). Raw output.json (first 500 chars): "
                f"{output_json[:500]!r}\nLast ~2000 chars of captured stdout/stderr:\n{artifacts.test_output[-2000:]}",
                flush=True,
            )
            return SweEvalReport(
                instance_id=task.instance_id,
                patch_exists=bool(task.model_patch),
                patch_applied=artifacts.patch_applied,
                error_kind="eval_error",
                tests_status={"raw_output_json": output_json},
            )

        sample = dict(task.metadata.get("instance_dict") or {})
        sample.setdefault("fail_to_pass", task.fail_to_pass)
        sample.setdefault("pass_to_pass", task.pass_to_pass)
        resolved = artifacts.patch_applied and _grade_output(test_results, sample)

        return SweEvalReport(
            instance_id=task.instance_id,
            resolved=resolved,
            patch_applied=artifacts.patch_applied,
            patch_exists=bool(task.model_patch),
            tests_status=test_results,
        )
