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

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from responses_api_agents.anyswe_agent.prepare import (
    DOCKER_IMAGE_TMPL,
    HF_DATASET_DEFAULT,
    IMAGE_BUILD_ATTEMPTS,
    SWE_BENCH_PRO_IMAGE_REPOSITORY,
    _build_one_sif,
    _docker_tag,
    _source_image,
    _swebench_pro_enrich,
    _swebench_pro_load_digest_cache,
    _swebench_pro_write_digest_cache,
    _to_gym_row,
    build_dataset,
    make_swebench_pro_row_hook,
)


def _staged_path(command: list[str]) -> Path:
    return Path(command[3])


def test_build_retries_then_atomically_installs_image(tmp_path: Path) -> None:
    attempts = 0

    def run(command: list[str], **_kwargs) -> SimpleNamespace:
        nonlocal attempts
        attempts += 1
        output = _staged_path(command)
        output.write_text("partial" if attempts == 1 else "complete")
        return SimpleNamespace(returncode=1 if attempts == 1 else 0, stderr="truncated manifest", stdout="")

    with (
        patch("responses_api_agents.anyswe_agent.prepare.subprocess.run", side_effect=run),
        patch("responses_api_agents.anyswe_agent.prepare.time.sleep") as sleep,
    ):
        instance_id, ok, detail = _build_one_sif("django__django-14011", tmp_path, force=False)

    assert (instance_id, ok, detail) == ("django__django-14011", True, "built after 2 attempts")
    assert attempts == 2
    assert (tmp_path / "django__django-14011.sif").read_text() == "complete"
    assert list(tmp_path.iterdir()) == [tmp_path / "django__django-14011.sif"]
    sleep.assert_called_once_with(2)


def test_build_removes_intermediate_output_after_three_failures(tmp_path: Path) -> None:
    def run(command: list[str], **_kwargs) -> SimpleNamespace:
        _staged_path(command).write_text("partial")
        return SimpleNamespace(returncode=1, stderr="unexpected end of JSON input", stdout="")

    with (
        patch("responses_api_agents.anyswe_agent.prepare.subprocess.run", side_effect=run) as run_mock,
        patch("responses_api_agents.anyswe_agent.prepare.time.sleep"),
    ):
        instance_id, ok, detail = _build_one_sif("django__django-14011", tmp_path, force=False)

    assert (instance_id, ok) == ("django__django-14011", False)
    assert run_mock.call_count == IMAGE_BUILD_ATTEMPTS
    assert f"attempt {IMAGE_BUILD_ATTEMPTS}/{IMAGE_BUILD_ATTEMPTS}" in detail
    assert "unexpected end of JSON input" in detail
    assert list(tmp_path.iterdir()) == []


def test_failed_forced_rebuild_preserves_existing_image(tmp_path: Path) -> None:
    image = tmp_path / "django__django-14011.sif"
    image.write_text("known-good")

    def run(command: list[str], **_kwargs) -> SimpleNamespace:
        _staged_path(command).write_text("partial")
        return SimpleNamespace(returncode=1, stderr="registry unavailable", stdout="")

    with (
        patch("responses_api_agents.anyswe_agent.prepare.subprocess.run", side_effect=run),
        patch("responses_api_agents.anyswe_agent.prepare.time.sleep"),
    ):
        _instance_id, ok, _detail = _build_one_sif("django__django-14011", tmp_path, force=True)

    assert not ok
    assert image.read_text() == "known-good"
    assert list(tmp_path.iterdir()) == [image]


def test_build_one_sif_uses_explicit_image_over_default_template(tmp_path: Path) -> None:
    seen_images: list[str] = []

    def run(command: list[str], **_kwargs) -> SimpleNamespace:
        seen_images.append(command[-1])
        _staged_path(command).write_text("complete")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    with patch("responses_api_agents.anyswe_agent.prepare.subprocess.run", side_effect=run):
        _build_one_sif(
            "repo__inst-1",
            tmp_path,
            force=False,
            image="docker://docker.io/jefzda/sweap-images:repo__inst-1",
        )

    assert seen_images == ["docker://docker.io/jefzda/sweap-images:repo__inst-1"]


def test_build_one_sif_falls_back_to_default_template_without_image(tmp_path: Path) -> None:
    seen_images: list[str] = []

    def run(command: list[str], **_kwargs) -> SimpleNamespace:
        seen_images.append(command[-1])
        _staged_path(command).write_text("complete")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    with patch("responses_api_agents.anyswe_agent.prepare.subprocess.run", side_effect=run):
        _build_one_sif("django__django-14011", tmp_path, force=False)

    assert seen_images == [DOCKER_IMAGE_TMPL.format(tag=_docker_tag("django__django-14011"))]


class TestSourceImage:
    def test_prefers_row_dockerhub_tag(self) -> None:
        # dockerhub_tag is only the tag portion; it must be combined with the fixed
        # repository SWE-bench Pro images are published under.
        img = _source_image({"dockerhub_tag": "nodebb.nodebb-repo__inst-1"}, "repo__inst-1")
        assert img == "docker://docker.io/jefzda/sweap-images:nodebb.nodebb-repo__inst-1"

    def test_falls_back_to_swebench_template_without_dockerhub_tag(self) -> None:
        img = _source_image({}, "django__django-14011")
        assert img == DOCKER_IMAGE_TMPL.format(tag=_docker_tag("django__django-14011"))


class TestToGymRow:
    def test_dataset_name_is_threaded_through(self) -> None:
        row = _to_gym_row(
            {"instance_id": "repo__inst-1", "problem_statement": "fix it"},
            "ScaleAI/SWE-bench_Pro",
            "test",
            {"model": "m"},
        )
        assert row["responses_create_params"]["metadata"]["dataset_name"] == "ScaleAI/SWE-bench_Pro"
        assert row["responses_create_params"]["metadata"]["instance_id"] == "repo__inst-1"

    def test_default_dataset_unchanged(self) -> None:
        row = _to_gym_row(
            {"instance_id": "django__django-14011", "problem_statement": "fix it"},
            HF_DATASET_DEFAULT,
            "test",
            {"model": "m"},
        )
        assert row["responses_create_params"]["metadata"]["dataset_name"] == HF_DATASET_DEFAULT


def _write_upstream_asset_tree(root: Path, instance_id: str) -> None:
    (root / "run_scripts" / instance_id).mkdir(parents=True)
    (root / "run_scripts" / instance_id / "run_script.sh").write_text("#!/bin/bash\necho run\n")
    (root / "run_scripts" / instance_id / "parser.py").write_text("print('parse')\n")
    (root / "dockerfiles" / "base_dockerfile" / instance_id).mkdir(parents=True)
    (root / "dockerfiles" / "base_dockerfile" / instance_id / "Dockerfile").write_text("FROM python:3.12\n")
    (root / "dockerfiles" / "instance_dockerfile" / instance_id).mkdir(parents=True)
    (root / "dockerfiles" / "instance_dockerfile" / instance_id / "Dockerfile").write_text("ENV FOO=bar\n")


class TestSwebenchProEnrich:
    def test_attaches_assets_and_digest(self, tmp_path: Path) -> None:
        _write_upstream_asset_tree(tmp_path, "repo__inst-1")
        inst = {"instance_id": "repo__inst-1", "dockerhub_tag": "some-tag"}

        enriched = _swebench_pro_enrich(inst, tmp_path, "sha256:deadbeef")

        assert enriched["run_script"] == "#!/bin/bash\necho run\n"
        assert enriched["parser_script"] == "print('parse')\n"
        assert enriched["base_dockerfile"] == "FROM python:3.12\n"
        assert enriched["instance_dockerfile"] == "ENV FOO=bar\n"
        assert enriched["image_digest"] == "sha256:deadbeef"
        # Original row fields are preserved, and the input dict isn't mutated in place.
        assert enriched["dockerhub_tag"] == "some-tag"
        assert "run_script" not in inst

    def test_missing_asset_raises(self, tmp_path: Path) -> None:
        inst = {"instance_id": "repo__inst-missing"}
        try:
            _swebench_pro_enrich(inst, tmp_path, "sha256:deadbeef")
        except FileNotFoundError as exc:
            assert "repo__inst-missing" in str(exc)
        else:
            raise AssertionError("expected FileNotFoundError for a missing upstream asset")


class TestDigestCache:
    def test_write_then_load_roundtrip(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "digests.json"
        _swebench_pro_write_digest_cache(cache_path, {"tag-a": "sha256:aaa", "tag-b": "sha256:bbb"})

        loaded = _swebench_pro_load_digest_cache(cache_path)

        assert loaded == {"tag-a": "sha256:aaa", "tag-b": "sha256:bbb"}

    def test_missing_or_invalid_files_are_skipped(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist.json"
        invalid = tmp_path / "invalid.json"
        invalid.write_text("not json")

        assert _swebench_pro_load_digest_cache(missing, invalid) == {}


class TestMakeSwebenchProRowHook:
    def test_enriches_and_caches_digest_across_rows(self, tmp_path: Path) -> None:
        _write_upstream_asset_tree(tmp_path, "repo__inst-1")
        _write_upstream_asset_tree(tmp_path, "repo__inst-2")
        fetch_calls: list[str] = []

        def fake_fetch_digest(tag: str, max_attempts: int = 8) -> str:
            fetch_calls.append(tag)
            return f"sha256:{tag}"

        with (
            patch(
                "responses_api_agents.anyswe_agent.prepare._swebench_pro_fetch_upstream_assets",
                return_value=tmp_path,
            ),
            patch(
                "responses_api_agents.anyswe_agent.prepare._swebench_pro_fetch_image_digest",
                side_effect=fake_fetch_digest,
            ),
        ):
            hook = make_swebench_pro_row_hook(cache_dir=tmp_path)
            row1 = hook({"instance_id": "repo__inst-1", "dockerhub_tag": "same-tag"})
            row2 = hook({"instance_id": "repo__inst-2", "dockerhub_tag": "same-tag"})

        assert row1["image_digest"] == "sha256:same-tag"
        assert row2["image_digest"] == "sha256:same-tag"
        # Same dockerhub_tag across both rows: the digest is fetched (and cached) once.
        assert fetch_calls == ["same-tag"]
        assert (tmp_path / "swebench_pro_image_digests.json").is_file()


class TestBuildDatasetRowHookAndExpectedCount:
    def test_row_hook_and_id_to_image_applied(self, tmp_path: Path) -> None:
        rows = [
            {"instance_id": "repo__inst-1", "problem_statement": "fix it", "dockerhub_tag": "tag-1"},
        ]

        def fake_load_dataset(dataset, split, revision, token):
            return rows

        def row_hook(inst: dict) -> dict:
            return dict(inst, enriched=True)

        with patch("datasets.load_dataset", side_effect=fake_load_dataset):
            ids, id_to_image = build_dataset(
                tmp_path / "out.jsonl",
                "ScaleAI/SWE-bench_Pro",
                "test",
                None,
                None,
                {"model": "m"},
                row_hook=row_hook,
            )

        assert ids == ["repo__inst-1"]
        assert id_to_image["repo__inst-1"] == f"docker://{SWE_BENCH_PRO_IMAGE_REPOSITORY}:tag-1"
        written = json.loads((tmp_path / "out.jsonl").read_text().splitlines()[0])
        written_inst = json.loads(written["responses_create_params"]["metadata"]["instance_dict"])
        assert written_inst["enriched"] is True

    def test_expected_count_mismatch_exits(self, tmp_path: Path) -> None:
        rows = [{"instance_id": "a", "problem_statement": "x"}, {"instance_id": "b", "problem_statement": "y"}]

        def fake_load_dataset(dataset, split, revision, token):
            return rows

        with patch("datasets.load_dataset", side_effect=fake_load_dataset):
            try:
                build_dataset(
                    tmp_path / "out.jsonl",
                    "ScaleAI/SWE-bench_Pro",
                    "test",
                    None,
                    None,
                    {"model": "m"},
                    expected_count=3,
                )
            except SystemExit as exc:
                assert "Expected 3" in str(exc)
            else:
                raise AssertionError("expected SystemExit on row-count mismatch")
