# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_gym.server_utils import ServerClient
from responses_api_agents.gdpval_agent.app import (
    GDPValAgent,
    GDPValAgentConfig,
    GDPValAgentRunRequest,
    agent_key,
    build_user_prompt,
    is_deliverable,
)


def _config(tmp_path: Path, **overrides) -> GDPValAgentConfig:
    base = dict(
        host="0.0.0.0",
        port=0,
        name="gdpval_agent",
        entrypoint="app.py",
        resources_server={"type": "resources_servers", "name": "gdpval"},
        model_server={"type": "responses_api_models", "name": "policy_model"},
        image="/abs/gdpval.sif",
        persist_deliverables_dir=str(tmp_path / "deliverables"),
    )
    base.update(overrides)
    return GDPValAgentConfig(**base)


def _agent(config: GDPValAgentConfig) -> GDPValAgent:
    return GDPValAgent(config=config, server_client=ServerClient.model_construct(global_config_dict={}))


def test_agent_key_maps_module_to_deps_script_name():
    assert agent_key("responses_api_agents.hermes_agent.app") == "hermes_agent"


def test_prompt_names_the_output_directory_and_reference_paths():
    prompt = build_user_prompt("Draft a memo.", ["/workspace/input/a.pdf"], "/workspace/output")
    assert "Draft a memo." in prompt
    assert "/workspace/input/a.pdf" in prompt
    assert "/workspace/output" in prompt
    assert "finish tool" not in prompt


def test_prompt_reports_no_reference_files():
    assert "None" in build_user_prompt("Task.", [], "/workspace/output")


@pytest.mark.parametrize("name", ["report.docx", "model.xlsx", "deck.pptx", "out.pdf"])
def test_deliverable_extensions_are_kept(name):
    assert is_deliverable(name)


@pytest.mark.parametrize("name", ["build.py", "run.sh", "agent.log", "notes.ipynb"])
def test_scratch_extensions_are_dropped(name):
    assert not is_deliverable(name)


def test_persist_dir_must_be_absolute(tmp_path):
    with pytest.raises(ValueError, match="absolute"):
        _config(tmp_path, persist_deliverables_dir="relative/path")


def test_deliverables_dir_uses_task_id_and_rollout_index(tmp_path):
    agent = _agent(_config(tmp_path))
    body = GDPValAgentRunRequest(responses_create_params={"input": []}, task_id="abc123", _ng_rollout_index=2)
    assert agent._deliverables_dir(body).parts[-2:] == ("task_abc123", "repeat_2")


def test_deliverables_dir_defaults_to_first_repeat(tmp_path):
    agent = _agent(_config(tmp_path))
    body = GDPValAgentRunRequest(responses_create_params={"input": []}, task_id="abc123")
    assert agent._deliverables_dir(body).parts[-1] == "repeat_0"


def test_deps_prefix_is_bound_read_only(tmp_path):
    agent = _agent(_config(tmp_path, model_server=None))
    body = GDPValAgentRunRequest(responses_create_params={"input": []})
    spec = agent._build_spec(body, "instruction", "/abs/gdpval.sif", tmp_path / "deps")
    assert f"{tmp_path / 'deps'}:/agent_deps_mount:ro" in spec.provider_options["binds"]


class _StubBox:
    """Sandbox stub whose exec returns canned find output and whose download writes a file."""

    def __init__(self, listings):
        self.listings = list(listings)
        self.commands = []

    async def exec(self, command, **kwargs):
        self.commands.append(command)
        return SimpleNamespace(stdout=self.listings.pop(0), stderr="", return_code=0)

    async def download(self, remote, local):
        Path(local).write_bytes(b"content")


@pytest.mark.asyncio
async def test_collect_flattens_and_drops_scratch_files(tmp_path):
    agent = _agent(_config(tmp_path))
    box = _StubBox(["/workspace/output/report.docx\n/workspace/output/sub/data.xlsx\n/workspace/output/gen.py\n"])
    target = tmp_path / "out"

    collected = await agent._collect(box, target)

    assert collected == 2
    assert sorted(p.name for p in target.iterdir()) == ["data.xlsx", "report.docx"]
    assert not any(p.is_dir() for p in target.iterdir())


@pytest.mark.asyncio
async def test_collect_falls_back_to_workspace_sweep_when_output_is_empty(tmp_path):
    agent = _agent(_config(tmp_path))
    box = _StubBox(["", "/workspace/memo.docx\n"])
    target = tmp_path / "out"

    collected = await agent._collect(box, target)

    assert collected == 1
    assert (target / "memo.docx").exists()
    assert "-newer" in box.commands[1]


@pytest.mark.asyncio
async def test_collect_clears_a_previous_attempt(tmp_path):
    agent = _agent(_config(tmp_path))
    target = tmp_path / "out"
    target.mkdir()
    (target / "stale.docx").write_text("old")

    await agent._collect(_StubBox(["/workspace/output/fresh.docx\n"]), target)

    assert not (target / "stale.docx").exists()
    assert (target / "fresh.docx").exists()
