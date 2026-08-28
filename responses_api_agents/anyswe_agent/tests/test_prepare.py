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

from pathlib import Path

from nemo_gym.sandbox.providers import SandboxImagePrepareRequest, SandboxImagePrepareResult
from responses_api_agents.anyswe_agent import prepare


class FakeImagePrepareProvider:
    def __init__(self) -> None:
        self.requests: list[SandboxImagePrepareRequest] = []

    def prepare_image(self, request: SandboxImagePrepareRequest) -> SandboxImagePrepareResult:
        self.requests.append(request)
        return SandboxImagePrepareResult(
            image=str(request.target_dir / request.target_name),
            ok=True,
            prepared=True,
            detail="built",
        )


def test_mangled_instance_id() -> None:
    assert prepare._mangled_instance_id("Astropy__astropy-12907") == "astropy_1776_astropy-12907"


def test_source_image() -> None:
    image = prepare._source_image("astropy__astropy-12907")
    assert image == "docker://swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"


def test_build_one_image_delegates_to_provider_prepare_image(tmp_path: Path) -> None:
    provider = FakeImagePrepareProvider()

    instance_id, ok, detail, image = prepare._build_one_image(
        provider,
        "astropy__astropy-12907",
        "docker://swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest",
        tmp_path,
    )

    assert (instance_id, ok, detail) == ("astropy__astropy-12907", True, "built")
    assert image == str(tmp_path / "astropy__astropy-12907")
    assert provider.requests == [
        SandboxImagePrepareRequest(
            image="docker://swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest",
            target_dir=tmp_path,
            target_name="astropy__astropy-12907",
            attempts=prepare.IMAGE_BUILD_ATTEMPTS,
            retry_delay_s=prepare.IMAGE_BUILD_RETRY_DELAY_SECONDS,
        )
    ]


def test_build_images_returns_image_per_instance(tmp_path: Path) -> None:
    provider = FakeImagePrepareProvider()
    rows = [
        {"responses_create_params": {"metadata": {"instance_id": "astropy__astropy-12907"}}},
        {"responses_create_params": {"metadata": {"instance_id": "django__django-11099"}}},
    ]

    images = prepare.build_images(rows, tmp_path, jobs=2, provider=provider)

    assert images == {
        "astropy__astropy-12907": str(tmp_path / "astropy__astropy-12907"),
        "django__django-11099": str(tmp_path / "django__django-11099"),
    }
