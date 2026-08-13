# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai.types.responses.response_output_item import (
    McpApprovalRequest,
    McpCall,
    McpListTools,
)
from pydantic import ValidationError

from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseMcpApprovalRequest,
    NeMoGymResponseMcpCall,
    NeMoGymResponseMcpListTools,
)


def _response_with_output(output: list) -> dict:
    return {
        "id": "resp_1",
        "created_at": 0.0,
        "model": "gpt-oss-120b",
        "object": "response",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": output,
    }


class TestOpenAIUtils:
    async def test_NeMoGymAsyncOpenAI(self) -> None:
        NeMoGymAsyncOpenAI(api_key="abc", base_url="https://api.openai.com/v1")

    @staticmethod
    def _response(status: int, retry_after: str | None = None) -> MagicMock:
        response = MagicMock()
        response.status = status
        response.headers = {} if retry_after is None else {"Retry-After": retry_after}
        response.content.read = AsyncMock(return_value=b"error")
        return response

    async def test_rate_limit_retries_stop_before_retry_deadline(self) -> None:
        response = self._response(429)
        now = 0.0
        sleep_delays = []

        async def fake_sleep(delay: float) -> None:
            nonlocal now
            sleep_delays.append(delay)
            now += delay

        with (
            patch("nemo_gym.openai_utils.request", new=AsyncMock(return_value=response)) as request_mock,
            patch("nemo_gym.openai_utils.raise_for_status", new=AsyncMock()) as raise_mock,
            patch("nemo_gym.openai_utils.time.monotonic", side_effect=lambda: now),
            patch("nemo_gym.openai_utils.sleep", side_effect=fake_sleep),
        ):
            client = NeMoGymAsyncOpenAI(api_key="abc", base_url="https://api.openai.com/v1")
            await client._request_with_retry(method="POST", url="https://api.openai.com/v1/chat/completions")

        assert sleep_delays == [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        assert request_mock.await_count == 8
        raise_mock.assert_awaited_once_with(response)

    async def test_retry_after_seconds_overrides_exponential_backoff(self) -> None:
        rate_limited = self._response(429, retry_after="30")
        success = self._response(200)
        now = 0.0
        sleep_delays = []

        async def fake_sleep(delay: float) -> None:
            nonlocal now
            sleep_delays.append(delay)
            now += delay

        with (
            patch(
                "nemo_gym.openai_utils.request",
                new=AsyncMock(side_effect=[rate_limited, success]),
            ),
            patch("nemo_gym.openai_utils.time.monotonic", side_effect=lambda: now),
            patch("nemo_gym.openai_utils.sleep", side_effect=fake_sleep),
        ):
            client = NeMoGymAsyncOpenAI(api_key="abc", base_url="https://api.openai.com/v1")
            response = await client._request_with_retry(
                method="POST",
                url="https://api.openai.com/v1/chat/completions",
            )

        assert response is success
        assert sleep_delays == [30.0]

    async def test_retry_after_http_date_is_converted_to_seconds(self) -> None:
        wall_clock_now = datetime(2026, 8, 13, tzinfo=timezone.utc)
        retry_after = format_datetime(wall_clock_now + timedelta(seconds=45), usegmt=True)
        rate_limited = self._response(429, retry_after=retry_after)
        success = self._response(200)
        now = 0.0
        sleep_delays = []

        async def fake_sleep(delay: float) -> None:
            nonlocal now
            sleep_delays.append(delay)
            now += delay

        with (
            patch(
                "nemo_gym.openai_utils.request",
                new=AsyncMock(side_effect=[rate_limited, success]),
            ),
            patch("nemo_gym.openai_utils.datetime", wraps=datetime) as datetime_mock,
            patch("nemo_gym.openai_utils.time.monotonic", side_effect=lambda: now),
            patch("nemo_gym.openai_utils.sleep", side_effect=fake_sleep),
        ):
            datetime_mock.now.return_value = wall_clock_now
            client = NeMoGymAsyncOpenAI(api_key="abc", base_url="https://api.openai.com/v1")
            response = await client._request_with_retry(
                method="POST",
                url="https://api.openai.com/v1/chat/completions",
            )

        assert response is success
        assert sleep_delays == [45.0]

    async def test_retry_after_past_deadline_is_not_slept(self) -> None:
        response = self._response(429, retry_after="120")

        with (
            patch("nemo_gym.openai_utils.request", new=AsyncMock(return_value=response)) as request_mock,
            patch("nemo_gym.openai_utils.raise_for_status", new=AsyncMock()) as raise_mock,
            patch("nemo_gym.openai_utils.time.monotonic", return_value=0.0),
            patch("nemo_gym.openai_utils.sleep", new=AsyncMock()) as sleep_mock,
        ):
            client = NeMoGymAsyncOpenAI(api_key="abc", base_url="https://api.openai.com/v1")
            await client._request_with_retry(method="POST", url="https://api.openai.com/v1/chat/completions")

        request_mock.assert_awaited_once()
        sleep_mock.assert_not_awaited()
        raise_mock.assert_awaited_once_with(response)


class TestNeMoGymResponseCreateParamsNonStreaming:
    def test_seed_rejected_at_top_level(self) -> None:
        """seed is not part of the OpenAI Responses schema; it must be passed via metadata.extra_body."""
        with pytest.raises(ValidationError):
            NeMoGymResponseCreateParamsNonStreaming(input="hello", seed=42)

    def test_seed_via_metadata_extra_body(self) -> None:
        """seed passed through metadata.extra_body round-trips through the strict schema."""
        params = NeMoGymResponseCreateParamsNonStreaming(input="hello", metadata={"extra_body": '{"seed": 42}'})
        assert params.metadata["extra_body"] == '{"seed": 42}'

    def test_unknown_field_still_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            NeMoGymResponseCreateParamsNonStreaming(input="hello", not_a_real_field=1)


class TestNeMoGymResponseHostedMcpItems:
    """Hosted-MCP output items (``mcp_call`` etc.) must validate rather than 500.

    Endpoints that run tools server-side (e.g. NVIDIA-hosted gpt-oss surfacing
    its built-in python tool as MCP) emit these in ``response.output``; before
    they were in the union, ``NeMoGymResponse.model_validate`` raised and the
    model server returned a 500 that aborted the whole rollout collection.
    """

    def test_mcp_call_in_response_output_validates(self) -> None:
        mcp_call = {
            "type": "mcp_call",
            "id": "mcp_1",
            "name": "python",
            "server_label": "exec",
            "arguments": '{"code": "print(42)"}',
            "output": "42\n",
            "status": "completed",
        }
        response = NeMoGymResponse.model_validate(
            _response_with_output(
                [
                    {"type": "reasoning", "id": "r1", "summary": []},
                    mcp_call,
                    {
                        "type": "message",
                        "id": "m1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "(Answer: 42)", "annotations": []}],
                    },
                ]
            )
        )
        call = response.output[1]
        assert isinstance(call, NeMoGymResponseMcpCall)
        assert call.type == "mcp_call"
        assert call.output == "42\n"

    def test_mcp_call_tolerates_missing_optional_fields(self) -> None:
        call = NeMoGymResponseMcpCall.model_validate({"type": "mcp_call", "name": "python", "arguments": "{}"})
        assert call.id is None and call.server_label is None and call.output is None

    def test_mcp_list_tools_and_approval_request_validate(self) -> None:
        listing = NeMoGymResponseMcpListTools.model_validate(
            {"type": "mcp_list_tools", "id": "l1", "server_label": "s", "tools": [{"name": "python"}]}
        )
        approval = NeMoGymResponseMcpApprovalRequest.model_validate(
            {"type": "mcp_approval_request", "id": "a1", "name": "python", "arguments": "{}", "server_label": "s"}
        )
        assert listing.tools == [{"name": "python"}]
        assert approval.name == "python"

    def test_hosted_mcp_items_inherit_upstream_types(self) -> None:
        # These must inherit the upstream openai typing (only relaxing the fields
        # NVIDIA-hosted endpoints omit/widen) rather than redefine it from scratch.
        assert issubclass(NeMoGymResponseMcpCall, McpCall)
        assert issubclass(NeMoGymResponseMcpListTools, McpListTools)
        assert issubclass(NeMoGymResponseMcpApprovalRequest, McpApprovalRequest)
