# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atom with an MCP grant: the adapter writes the settings and the approval, and the agent sees the grant."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime import MCPToolAccess
from responses_api_agents.nooa_agent.atom_mcp_adapter import configure_mcp, invoke_atom_with_mcp, server_key


pytest.importorskip("nooa_atom.gym", reason="nooa-atom not installed in this venv")


def grant(name: str = "tools", token: str = "secret-token", url: str = "http://resources:8000/mcp") -> MCPToolAccess:
    return MCPToolAccess.model_validate(
        {
            "name": name,
            "required": True,
            "connection": {"url": url, "headers": {"x-session-token": token}},
        }
    )


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    for name in ("GYM_MCP_TOOLS_URL", "GYM_MCP_TOOLS_HEADER_0"):
        monkeypatch.delenv(name, raising=False)


def test_server_names_become_identifiers():
    assert server_key("tools") == "tools"
    assert server_key("my-server.v2") == "my_server_v2"
    assert server_key("2fast").isidentifier()


def test_the_settings_hold_placeholders_never_the_token(tmp_path: Path):
    configure_mcp(tmp_path, (grant(),))
    text = (tmp_path / ".nooa" / "settings.yaml").read_text()
    assert "secret-token" not in text and "resources:8000" not in text
    atom = yaml.safe_load(text)["atom"]
    assert atom["mcp_auto_connect"] == ["tools"]
    assert atom["mcp_servers"]["tools"]["url"] == "${GYM_MCP_TOOLS_URL}"
    assert atom["mcp_servers"]["tools"]["headers"] == {"x-session-token": "${GYM_MCP_TOOLS_HEADER_0}"}


def test_the_values_go_to_the_environment(tmp_path: Path):
    import os

    configure_mcp(tmp_path, (grant(),))
    assert os.environ["GYM_MCP_TOOLS_URL"] == "http://resources:8000/mcp"
    assert os.environ["GYM_MCP_TOOLS_HEADER_0"] == "secret-token"


def test_the_definition_is_approved_for_this_workspace_only(tmp_path: Path):
    from nooa_atom.skills.mcp_servers import MCPServers
    from nooa_atom.workspace.options import AtomOptions

    workspace = tmp_path / "work"
    workspace.mkdir()
    configure_mcp(workspace, (grant(),))
    options = AtomOptions.load(workspace)
    approved = MCPServers(servers=options.mcp_servers, project_dir=workspace / ".nooa")
    assert approved.is_approved("tools")
    other = tmp_path / "other"
    (other / ".nooa").mkdir(parents=True)
    elsewhere = MCPServers(servers=options.mcp_servers, project_dir=other / ".nooa")
    assert not elsewhere.is_approved("tools")


def test_a_changed_definition_is_not_approved(tmp_path: Path):
    from nooa_atom.skills.mcp_servers import MCPServers

    configure_mcp(tmp_path, (grant(),))
    changed = {"tools": {"type": "http", "url": "http://evil/mcp", "headers": {}}}
    assert not MCPServers(servers=changed, project_dir=tmp_path / ".nooa").is_approved("tools")


def test_existing_settings_are_kept(tmp_path: Path):
    (tmp_path / ".nooa").mkdir()
    (tmp_path / ".nooa" / "settings.yaml").write_text(
        yaml.safe_dump(
            {"atom": {"mcp_auto_connect": ["other"], "mcp_servers": {"other": {"type": "http", "url": "x"}}}, "k": 1}
        )
    )
    configure_mcp(tmp_path, (grant(),))
    settings = yaml.safe_load((tmp_path / ".nooa" / "settings.yaml").read_text())
    assert settings["k"] == 1
    assert settings["atom"]["mcp_auto_connect"] == ["other", "tools"]
    assert set(settings["atom"]["mcp_servers"]) == {"tools", "other"}


def test_two_grants_with_the_same_server_name_are_refused(tmp_path: Path):
    with pytest.raises(ValueError, match="share the server name"):
        configure_mcp(tmp_path, (grant("a-b"), grant("a.b")))


@pytest.mark.asyncio
async def test_the_adapter_configures_then_hands_the_task_to_atom(tmp_path: Path):
    agent = SimpleNamespace(workspace=tmp_path, gym_mcp_accesses=(grant(),), solve=AsyncMock(return_value="result"))
    request = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "do the task"}])
    assert await invoke_atom_with_mcp(agent, request) == "result"
    agent.solve.assert_awaited_once_with("do the task")
    assert json.loads(json.dumps(yaml.safe_load((tmp_path / ".nooa" / "settings.yaml").read_text())))["atom"]


@pytest.mark.asyncio
async def test_without_a_grant_the_adapter_fails_loudly(tmp_path: Path):
    agent = SimpleNamespace(workspace=tmp_path, gym_mcp_accesses=(), solve=AsyncMock())
    request = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "x"}])
    with pytest.raises(RuntimeError, match="needs an MCP grant"):
        await invoke_atom_with_mcp(agent, request)
    agent.solve.assert_not_awaited()


def test_the_adapter_declares_that_it_uses_mcp_grants():
    assert invoke_atom_with_mcp.consumes_mcp_access is True
