# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Atom with the MCP servers the environment granted.

Atom reaches external tools through MCP servers that its workspace settings name and that an operator has approved. A
Gym rollout has nobody to approve, and the environment grants a server per rollout, with its address and a token that
change every time. Before Atom starts, this adapter writes the settings for each granted server, with the address and
token as environment placeholders, and records the approval of exactly that definition. The approval is the operator's
act, done here by harness code before the model runs: the model cannot approve a server.

The harness never connects a remembered server by itself, so the task prompt has to tell the model to activate it
(`await self.skills.activate(["<server name>"])`) before its first use.

Run it only in a sandbox, where one rollout owns the process: it sets environment variables and writes Atom's user
approval store, which belong to a single rollout.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime import MCPToolAccess
from responses_api_agents.nooa_agent.atom_adapter import invoke_atom


if TYPE_CHECKING:
    from nooa.interactive import TaskResult
    from nooa_atom.gym import AtomHarness


def server_key(name: str) -> str:
    """The server's name as a Python identifier, which is how Atom exposes it (``self.<name>``)."""
    key = re.sub(r"\W", "_", name)
    return key if key.isidentifier() else f"mcp_{key}"


def configure_mcp(workspace: Path, accesses: tuple[MCPToolAccess, ...]) -> list[str]:
    """Write the workspace settings for the granted servers and approve their definitions; return their names."""
    from nooa_atom.skills.mcp_servers import MCPServers

    servers: dict[str, dict[str, Any]] = {}
    for access in accesses:
        key = server_key(access.name)
        if key in servers:
            raise ValueError(f"MCP grants {access.name!r} and another share the server name {key!r}")
        url_variable = f"GYM_MCP_{key.upper()}_URL"
        os.environ[url_variable] = str(access.connection.url)
        headers = {}
        for index, (header, value) in enumerate(sorted(access.connection.headers.items())):
            variable = f"GYM_MCP_{key.upper()}_HEADER_{index}"
            os.environ[variable] = value
            headers[header] = f"${{{variable}}}"
        servers[key] = {"type": "http", "url": f"${{{url_variable}}}", "headers": headers}

    # Atom's approval store is the user's; point it at a directory of this rollout's own unless the caller chose one.
    os.environ.setdefault("NEMO_OO_USER_DIR", tempfile.mkdtemp(prefix="gym-atom-user-"))
    project = workspace / ".nooa"
    project.mkdir(parents=True, exist_ok=True)
    settings_path = project / "settings.yaml"
    settings = yaml.safe_load(settings_path.read_text()) if settings_path.exists() else None
    settings = settings if isinstance(settings, dict) else {}
    atom = settings.setdefault("atom", {})
    atom["mcp_servers"] = {**(atom.get("mcp_servers") or {}), **servers}
    atom["mcp_auto_connect"] = sorted({*(atom.get("mcp_auto_connect") or []), *servers})
    settings_path.write_text(yaml.safe_dump(settings))

    manager = MCPServers(servers=servers, project_dir=project)
    for key in servers:
        manager.approve(key, manager.approval_request(key).confirmation)
    return sorted(servers)


async def invoke_atom_with_mcp(agent: AtomHarness, request: NeMoGymResponseCreateParamsNonStreaming) -> TaskResult:
    """Configure and approve the granted MCP servers, then run Atom on the task."""
    accesses = tuple(getattr(agent, "gym_mcp_accesses", ()))
    if not accesses:
        raise RuntimeError(
            "invoke_atom_with_mcp needs an MCP grant from the environment (resources_tool_transports: [mcp])"
        )
    configure_mcp(Path(agent.workspace), accesses)
    return await invoke_atom(agent, request)


# The server checks this before it accepts an MCP grant, so a grant is never silently ignored.
invoke_atom_with_mcp.consumes_mcp_access = True  # type: ignore[attr-defined]
