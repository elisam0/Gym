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

"""Unit tests for ``provision_and_collect``'s ``pre_launch_cmd`` step.

Mirrors ``test_swe_bench_pro.py``'s scripted-provider pattern: a ``_FakeProvider`` records
every exec/upload call in order, so these tests exercise ordering (stage_files -> pre_launch_cmd
-> agent_launch_command) without touching a real sandbox or Docker image.
"""

from __future__ import annotations

import asyncio

from nemo_gym.sandbox import SandboxExecResult, SandboxHandle, SandboxStatus, register_provider
from responses_api_agents.swe_env.harness import SweTask
from responses_api_agents.swe_env.self_drive import provision_and_collect


class _FakeProvider:
    """Scripted provider: records every exec/upload call, in order, for ordering assertions."""

    name = "fake-self-drive"

    def __init__(self, **_):
        self.calls: list[tuple[str, str]] = []

    async def create(self, spec):
        return SandboxHandle(sandbox_id="fake", provider_name=self.name, raw={"workdir": spec.workdir})

    async def exec(self, handle, command, *, cwd=None, env=None, timeout_s=None, user=None):
        self.calls.append(("exec", command))
        if command.startswith("cd ") and "git diff --cached" in command:
            return SandboxExecResult(stdout="diff --git a/x b/x\n", stderr="", return_code=0)
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    async def upload_file(self, handle, local_path, remote_path):
        with open(local_path, encoding="utf-8") as fh:
            content = fh.read()
        self.calls.append(("upload", f"{remote_path}:{content}"))

    async def download_file(self, *a, **k):
        return None

    async def status(self, handle):
        return SandboxStatus.RUNNING

    async def close(self, handle):
        return None

    async def aclose(self):
        return None


register_provider("fake-self-drive", _FakeProvider, override=True)


def _task(**overrides) -> SweTask:
    base = dict(
        instance_id="repo__inst-1",
        image="docker.io/jefzda/sweap-images:repo__inst-1",
        base_commit="abc123",
        repo_workdir="/app",
        model_patch="",
        fail_to_pass=[],
        pass_to_pass=[],
        benchmark="swe-bench-pro",
        split="test",
        metadata={"instance_dict": {"run_script": "", "parser_script": "", "repo_language": "python"}},
    )
    base.update(overrides)
    return SweTask(**base)


def test_pre_launch_cmd_runs_after_stage_files_and_before_agent_launch():
    async def run():
        result = await provision_and_collect(
            _task(),
            provider={"fake-self-drive": {}},
            agent_launch_command="run-the-agent",
            stage_files={"/app/anti_cheat_setup.sh": "echo hi"},
            pre_launch_cmd="bash anti_cheat_setup.sh",
        )
        return result

    result = asyncio.run(run())
    assert result["patch"] == "diff --git a/x b/x\n"


def test_pre_launch_cmd_ordering_and_content():
    provider = _FakeProvider()

    async def run():
        return await provision_and_collect(
            _task(),
            provider=provider,
            agent_launch_command="run-the-agent",
            stage_files={"/app/anti_cheat_setup.sh": "echo hi"},
            pre_launch_cmd="bash anti_cheat_setup.sh",
        )

    asyncio.run(run())
    kinds = [kind for kind, _ in provider.calls]
    upload_idx = kinds.index("upload")
    pre_launch_idx = next(i for i, (k, v) in enumerate(provider.calls) if k == "exec" and v == "bash anti_cheat_setup.sh")
    launch_idx = next(i for i, (k, v) in enumerate(provider.calls) if k == "exec" and v == "run-the-agent")
    assert upload_idx < pre_launch_idx < launch_idx


def test_no_pre_launch_cmd_is_a_no_op():
    provider = _FakeProvider()

    async def run():
        return await provision_and_collect(
            _task(),
            provider=provider,
            agent_launch_command="run-the-agent",
        )

    asyncio.run(run())
    assert all(cmd != "bash anti_cheat_setup.sh" for kind, cmd in provider.calls if kind == "exec")
