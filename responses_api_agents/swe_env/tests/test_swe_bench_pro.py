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

"""Unit tests for the swe-bench-pro flat (host-graded) harness.

Mirrors ``test_r2egym.py``'s scripted-provider pattern: a ``_FakeProvider``
returns canned ``exec``/``upload`` results, so these tests exercise the
harness's own logic (entryscript generation, file placement, JSON-output
grading) without touching a real sandbox, Docker image, or the SWE-bench Pro
dataset's actual run/parser scripts.
"""

from __future__ import annotations

import asyncio
import json

from nemo_gym.sandbox import (
    SandboxExecResult,
    SandboxHandle,
    SandboxStatus,
    register_provider,
)
from responses_api_agents.swe_env.harness import SweTask, reward_from_report
from responses_api_agents.swe_env.harnesses.swe_bench_pro import SweBenchProHarness


_PASSING_OUTPUT = json.dumps({"tests": [{"name": "t::a", "status": "PASSED"}, {"name": "t::b", "status": "PASSED"}]})
_FAILING_OUTPUT = json.dumps({"tests": [{"name": "t::a", "status": "FAILED"}, {"name": "t::b", "status": "PASSED"}]})


class _FakeProvider:
    """Scripted provider: serves canned file reads (cat) and an eval exit code; records uploads."""

    name = "fake-swebench-pro"

    def __init__(self, *, files: dict[str, str] | None = None, exec_rc: int = 0, **_):
        self._files = dict(files or {})
        self._exec_rc = exec_rc
        self.uploaded: dict[str, str] = {}

    async def create(self, spec):
        return SandboxHandle(sandbox_id="fake", provider_name=self.name, raw={"workdir": spec.workdir})

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        if command.startswith("cat "):
            path = command[len("cat ") :].strip()
            return SandboxExecResult(stdout=self._files.get(path, ""), stderr="", return_code=0)
        return SandboxExecResult(stdout="", stderr="", return_code=self._exec_rc)

    async def upload_file(self, handle, local_path, remote_path):
        try:
            with open(local_path, encoding="utf-8") as fh:
                self.uploaded[remote_path] = fh.read()
        except OSError:
            self.uploaded[remote_path] = ""
        return None

    async def download_file(self, *a, **k):
        return None

    async def status(self, handle):
        return SandboxStatus.RUNNING

    async def close(self, handle):
        return None

    async def aclose(self):
        return None


register_provider("fake-swebench-pro", _FakeProvider, override=True)


_INSTANCE_DICT = {
    "base_commit": "abc123",
    "run_script": "#!/bin/bash\necho run\n",
    "parser_script": "print('parse')\n",
    "before_repo_set_cmd": "pip install -e .\n",
    "selected_test_files_to_run": ["t::a", "t::b"],
    "base_dockerfile": "FROM python:3.12\nENV FOO=bar\n",
    "instance_dockerfile": "ENV BAZ=qux\n",
    "prefetch_go_modules": False,
    "repo_language": "python",
}


def _task(**overrides) -> SweTask:
    """Build a swe-bench-pro ``SweTask`` with sensible defaults."""
    base = dict(
        instance_id="repo__inst-1",
        image="docker.io/jefzda/sweap-images:repo__inst-1",
        base_commit="abc123",
        repo_workdir="/app",
        model_patch="diff --git a/x b/x\n",
        fail_to_pass=["t::a"],
        pass_to_pass=["t::b"],
        benchmark="swe-bench-pro",
        split="test",
        metadata={"instance_dict": dict(_INSTANCE_DICT)},
    )
    base.update(overrides)
    return SweTask(**base)


def test_harness_identity():
    harness = SweBenchProHarness()
    assert harness.name == "swe-bench-pro"
    assert harness.grade_strategy == "flat-host-grade"


def test_build_spec_image_workdir_metadata():
    spec = SweBenchProHarness().build_spec(_task())
    assert spec.image == "docker.io/jefzda/sweap-images:repo__inst-1"
    assert spec.workdir == "/app"
    assert spec.metadata["harness"] == "swe-bench-pro"


def test_supports_provider_any_exec_capable():
    harness = SweBenchProHarness()
    assert harness.supports_provider("docker") is True
    assert harness.supports_provider("apptainer") is True


def test_reset_repo_is_noop():
    from responses_api_agents.swe_env.sandbox import AsyncSweEnvironment

    async def run():
        harness = SweBenchProHarness()
        task = _task()
        env = await AsyncSweEnvironment.start({"fake-swebench-pro": {}}, harness.build_spec(task))
        await harness.reset_repo(env, task)  # must not raise

    asyncio.run(run())


def test_materialize_writes_workspace_files():
    from responses_api_agents.swe_env.sandbox import AsyncSweEnvironment

    async def run():
        harness = SweBenchProHarness()
        task = _task()
        env = await AsyncSweEnvironment.start({"fake-swebench-pro": {}}, harness.build_spec(task))
        await harness.materialize(env, task)
        return env.sandbox._provider

    provider = asyncio.run(run())
    assert provider.uploaded["/workspace/patch.diff"] == "diff --git a/x b/x\n"
    assert provider.uploaded["/workspace/run_script.sh"] == "#!/bin/bash\necho run\n"
    assert provider.uploaded["/workspace/parser.py"] == "print('parse')\n"
    entryscript = provider.uploaded["/workspace/entryscript.sh"]
    assert "git reset --hard abc123" in entryscript
    assert "git checkout abc123" in entryscript
    assert "export FOO=bar" in entryscript
    assert "export BAZ=qux" in entryscript
    assert "pip install -e ." in entryscript
    assert "bash /workspace/run_script.sh t::a,t::b" in entryscript


def test_run_eval_then_grade_resolved():
    from responses_api_agents.swe_env.sandbox import AsyncSweEnvironment

    async def run():
        harness = SweBenchProHarness()
        task = _task()
        files = {
            "/workspace/stdout.log": "ok\n",
            "/workspace/stderr.log": "",
            "/workspace/output.json": _PASSING_OUTPUT,
            "/workspace/patch_apply_status": "0\n",
        }
        env = await AsyncSweEnvironment.start({"fake-swebench-pro": {"files": files}}, harness.build_spec(task))
        artifacts = await harness.run_eval(env, task)
        return harness.grade(task, artifacts)

    report = asyncio.run(run())
    assert report.error_kind is None
    assert report.patch_applied is True
    assert report.resolved is True
    assert reward_from_report(report) == 1.0


def test_run_eval_then_grade_unresolved_on_failed_test():
    from responses_api_agents.swe_env.sandbox import AsyncSweEnvironment

    async def run():
        harness = SweBenchProHarness()
        task = _task()
        files = {
            "/workspace/stdout.log": "ok\n",
            "/workspace/stderr.log": "",
            "/workspace/output.json": _FAILING_OUTPUT,
            "/workspace/patch_apply_status": "0\n",
        }
        env = await AsyncSweEnvironment.start({"fake-swebench-pro": {"files": files}}, harness.build_spec(task))
        artifacts = await harness.run_eval(env, task)
        return harness.grade(task, artifacts)

    report = asyncio.run(run())
    assert report.error_kind is None
    assert report.resolved is False
    assert reward_from_report(report) == 0.0


def test_grade_masks_on_infra_error():
    from responses_api_agents.swe_env.sandbox import AsyncSweEnvironment

    async def run():
        harness = SweBenchProHarness()
        task = _task()
        env = await AsyncSweEnvironment.start({"fake-swebench-pro": {"exec_rc": 0}}, harness.build_spec(task))
        artifacts = await harness.run_eval(env, task)
        artifacts.raw["error_type"] = "timeout"
        return harness.grade(task, artifacts)

    report = asyncio.run(run())
    assert report.error_kind == "timeout"
    assert report.resolved is False
    assert reward_from_report(report) == 0.0


def test_grade_masks_on_invalid_json_output():
    from responses_api_agents.swe_env.sandbox import AsyncSweEnvironment

    async def run():
        harness = SweBenchProHarness()
        task = _task()
        files = {
            "/workspace/stdout.log": "ok\n",
            "/workspace/stderr.log": "",
            "/workspace/output.json": "not json",
            "/workspace/patch_apply_status": "0\n",
        }
        env = await AsyncSweEnvironment.start({"fake-swebench-pro": {"files": files}}, harness.build_spec(task))
        artifacts = await harness.run_eval(env, task)
        return harness.grade(task, artifacts)

    report = asyncio.run(run())
    assert report.error_kind == "eval_error"
    assert report.resolved is False


def test_patch_not_applied_gates_resolved_false_even_if_tests_pass():
    from responses_api_agents.swe_env.sandbox import AsyncSweEnvironment

    async def run():
        harness = SweBenchProHarness()
        task = _task()
        files = {
            "/workspace/stdout.log": "ok\n",
            "/workspace/stderr.log": "",
            "/workspace/output.json": _PASSING_OUTPUT,
            "/workspace/patch_apply_status": "1\n",
        }
        env = await AsyncSweEnvironment.start({"fake-swebench-pro": {"files": files}}, harness.build_spec(task))
        artifacts = await harness.run_eval(env, task)
        return harness.grade(task, artifacts)

    report = asyncio.run(run())
    assert report.patch_applied is False
    assert report.resolved is False
