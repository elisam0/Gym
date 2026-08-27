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
"""
    python prepare.py                          # full SWE-bench Verified + all SIFs
    python prepare.py --limit 5                # 5 instances + their 5 SIFs (smoke test)
    python prepare.py --instance-id django__django-13741
    python prepare.py --no-images              # dataset only, skip image builds
    python prepare.py --no-dataset --sif-dir PATH # build images only

    # Other dataset families (output defaults to data/<dataset-basename>.jsonl):
    python prepare.py --dataset ScaleAI/SWE-bench_Pro

schema anyswe_agent expects: each line has
`responses_create_params.metadata` with `instance_id`, `dataset_name`, `split`,
`problem_statement`, and `instance_dict` (the full SWE-bench instance the eval
harness needs). Images are Apptainer SIFs named `{instance_id}.sif` so the
agent's container_formatter is simply `<sif-dir>/{instance_id}.sif`.

Image source: SWE-bench (Verified/Multilingual)-style datasets publish images at a
predictable `swebench/sweb.eval.x86_64.<tag>` registry path derived purely from
`instance_id`, so the SIF is built from that derived tag. SWE-bench Pro instead
publishes a per-row Docker Hub tag (`dockerhub_tag`) under a single fixed
repository; see the "SWE-bench Pro asset enrichment" section below for why that
tag alone isn't enough to prepare a runnable row.

Prerequisites for image builds: `apptainer` on PATH and network access to the
image registry. Each SIF is multiple GB, building all of SWE-bench Verified
(500 tasks) needs hundreds of GB of disk. Can use --limit and iterate.

SWE-bench Pro asset enrichment: unlike SWE-bench Verified, the raw
`ScaleAI/SWE-bench_Pro` HF rows do NOT include the per-instance `run_script`,
`parser_script`, or Dockerfiles the swe-bench-pro harness
(`responses_api_agents/swe_env/harnesses/swe_bench_pro.py`) needs — those live
in a separate upstream evaluator repository. `--dataset ScaleAI/SWE-bench_Pro`
therefore downloads a pinned commit of that repo once (cached under
`data/swebench_pro_upstream/`), reads each instance's assets out of it, and
resolves + caches an immutable Docker Hub image digest per `dockerhub_tag`
(cached under `data/swebench_pro_image_digests.json`) before writing each row.
This logic (constants, `_swebench_pro_*` helpers) is ported from this repo's
own `benchmarks/swebench/pro/prepare.py`, introduced in
https://github.com/NVIDIA-NeMo/Gym/pull/2498 (public), which in turn pins a
commit of the upstream evaluator (MIT licensed):
https://github.com/scaleapi/SWE-bench_Pro-os
Only the *sourcing* of these assets is ported here — the row shape they're
folded into is anyswe_agent's own (empty `responses_create_params.input`,
everything under `metadata.instance_dict`), not PR #2498's prompt-templated
row, since anyswe_agent's runner reads `problem_statement` from metadata
directly rather than from a rendered prompt.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import sleep
from urllib.error import HTTPError
from urllib.parse import quote


HF_DATASET_DEFAULT = "princeton-nlp/SWE-bench_Verified"
HF_DATASET = HF_DATASET_DEFAULT  # backward-compat alias
DEFAULT_SPLIT = "test"
# SWE-bench publishes eval images with `__` -> `_1776_` and lowercased.
DOCKER_IMAGE_TMPL = "docker://swebench/sweb.eval.x86_64.{tag}:latest"
DEFAULT_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
IMAGE_BUILD_ATTEMPTS = 3
IMAGE_BUILD_RETRY_DELAY_SECONDS = 2

# SWE-bench Pro: dataset id, pinned versions, and asset locations. Values match
# https://github.com/NVIDIA-NeMo/Gym/pull/2498's benchmarks/swebench/pro/prepare.py exactly,
# so preparing this dataset here is reproducible with that PR's own output.
SWE_BENCH_PRO_HF_DATASET = "ScaleAI/SWE-bench_Pro"
SWE_BENCH_PRO_DATASET_REVISION = "7ab5114912baf22bb098818e604c02fe7ad2c11f"
SWE_BENCH_PRO_UPSTREAM_COMMIT = "ca10a60a5fcae51e6948ffe1485d4153d421e6c5"
SWE_BENCH_PRO_UPSTREAM_ARCHIVE_URL = (
    f"https://github.com/scaleapi/SWE-bench_Pro-os/archive/{SWE_BENCH_PRO_UPSTREAM_COMMIT}.tar.gz"
)
SWE_BENCH_PRO_IMAGE_REPOSITORY = "docker.io/jefzda/sweap-images"
SWE_BENCH_PRO_EXPECTED_COUNT = 731

_THIS_DIR = Path(__file__).parent


def _docker_tag(instance_id: str) -> str:
    return instance_id.replace("__", "_1776_").lower()


def _source_image(inst: dict, instance_id: str) -> str:
    """Resolve the docker image to build a SIF from.

    A row-provided registry tag (SWE-bench Pro's ``dockerhub_tag``) is combined with the
    fixed repository those tags are published under when present, since Pro doesn't follow
    the SWE-bench-Verified ``swebench/sweb.eval.x86_64.<tag>`` naming ``DOCKER_IMAGE_TMPL``
    assumes. Otherwise the tag is derived from ``instance_id`` the way SWE-bench
    (Verified/Multilingual) does.

    Args:
        inst: The dataset row, which may carry ``dockerhub_tag``.
        instance_id: The benchmark instance id.

    Returns:
        str: An ``apptainer build``-able ``docker://...`` image reference.
    """
    tag = inst.get("dockerhub_tag")
    if tag:
        return f"docker://{SWE_BENCH_PRO_IMAGE_REPOSITORY}:{tag}"
    return DOCKER_IMAGE_TMPL.format(tag=_docker_tag(instance_id))


# --- SWE-bench Pro asset enrichment ------------------------------------------------------
# Ported from https://github.com/NVIDIA-NeMo/Gym/pull/2498's
# benchmarks/swebench/pro/prepare.py (public), which in turn pins a commit of the
# upstream, MIT-licensed evaluator (https://github.com/scaleapi/SWE-bench_Pro-os). See the
# module docstring above for what's ported vs. adapted.


def _swebench_pro_fetch_upstream_assets(cache_dir: Path) -> Path:
    """Download and extract the pinned upstream evaluator source once, idempotently.

    Args:
        cache_dir: Directory the archive is downloaded and extracted into.

    Returns:
        Path: The extracted upstream source root.
    """
    root = cache_dir / f"SWE-bench_Pro-os-{SWE_BENCH_PRO_UPSTREAM_COMMIT}"
    if root.is_dir():
        return root

    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / f"{SWE_BENCH_PRO_UPSTREAM_COMMIT}.tar.gz"
    urllib.request.urlretrieve(SWE_BENCH_PRO_UPSTREAM_ARCHIVE_URL, archive_path)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(cache_dir, filter="data")
    if not root.is_dir():
        raise FileNotFoundError(f"Expected extracted upstream directory at {root}")
    return root


def _swebench_pro_read_asset(upstream_root: Path, relative_path: str) -> str:
    path = upstream_root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Missing SWE-bench Pro evaluator asset: {path}")
    return path.read_text(encoding="utf-8")


def _swebench_pro_fetch_image_digest(dockerhub_tag: str, max_attempts: int = 8) -> str:
    """Resolve a case-sensitive Docker Hub tag to an immutable digest, retrying on rate limits.

    Args:
        dockerhub_tag: The tag to resolve (under ``SWE_BENCH_PRO_IMAGE_REPOSITORY``).
        max_attempts: Maximum attempts before giving up on repeated HTTP 429s.

    Returns:
        str: The resolved ``sha256:...`` digest.
    """
    url = f"https://hub.docker.com/v2/repositories/jefzda/sweap-images/tags/{quote(dockerhub_tag, safe='')}"
    metadata: dict = {}
    for attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                metadata = json.load(response)
            break
        except HTTPError as exc:
            if exc.code != 429 or attempt == max_attempts - 1:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else min(2**attempt, 60)
            print(f"Docker Hub rate-limited {dockerhub_tag}; retrying in {delay:g}s", flush=True)
            sleep(delay)
    digest = metadata.get("digest")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise ValueError(f"Docker Hub did not return a digest for {dockerhub_tag}")
    return digest


def _swebench_pro_load_digest_cache(*paths: Path) -> dict[str, str]:
    cache: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            continue
        try:
            cache.update(
                {str(tag): str(digest) for tag, digest in json.loads(path.read_text(encoding="utf-8")).items()}
            )
        except (json.JSONDecodeError, OSError):
            continue
    return cache


def _swebench_pro_write_digest_cache(path: Path, cache: Mapping[str, str]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(dict(cache), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _swebench_pro_enrich(inst: dict, upstream_root: Path, image_digest: str) -> dict:
    """Attach the verifier-only assets a raw SWE-bench Pro row is missing.

    Args:
        inst: The raw dataset row.
        upstream_root: The extracted upstream evaluator source root.
        image_digest: The resolved digest for this row's ``dockerhub_tag``.

    Returns:
        dict: A copy of ``inst`` with ``run_script``, ``parser_script``,
        ``base_dockerfile``, ``instance_dockerfile``, and ``image_digest`` added.
    """
    instance_id = str(inst["instance_id"])
    enriched = dict(inst)
    enriched.update(
        {
            "run_script": _swebench_pro_read_asset(upstream_root, f"run_scripts/{instance_id}/run_script.sh"),
            "parser_script": _swebench_pro_read_asset(upstream_root, f"run_scripts/{instance_id}/parser.py"),
            "base_dockerfile": _swebench_pro_read_asset(
                upstream_root, f"dockerfiles/base_dockerfile/{instance_id}/Dockerfile"
            ),
            "instance_dockerfile": _swebench_pro_read_asset(
                upstream_root, f"dockerfiles/instance_dockerfile/{instance_id}/Dockerfile"
            ),
            "image_digest": image_digest,
        }
    )
    return enriched


def make_swebench_pro_row_hook(cache_dir: Path | None = None):
    """Build a ``build_dataset`` row hook that enriches raw SWE-bench Pro rows in place.

    Fetches the pinned upstream evaluator source once (cached), then returns a callable
    that, per row, reads that instance's assets and resolves + caches its image digest
    (a real Docker Hub API call per distinct ``dockerhub_tag``, cached to disk so a
    resumed/re-run prepare doesn't re-query already-resolved tags).

    Args:
        cache_dir: Base directory for the upstream-source and digest caches; defaults to
            ``data/`` next to this script.

    Returns:
        Callable[[dict], dict]: A row hook suitable for ``build_dataset(..., row_hook=...)``.
    """
    base = cache_dir or (_THIS_DIR / "data")
    upstream_root = _swebench_pro_fetch_upstream_assets(base / "swebench_pro_upstream")
    digest_cache_fpath = base / "swebench_pro_image_digests.json"
    digest_cache = _swebench_pro_load_digest_cache(digest_cache_fpath)

    def row_hook(inst: dict) -> dict:
        dockerhub_tag = str(inst["dockerhub_tag"])
        image_digest = digest_cache.get(dockerhub_tag)
        if image_digest is None:
            image_digest = _swebench_pro_fetch_image_digest(dockerhub_tag)
            digest_cache[dockerhub_tag] = image_digest
            _swebench_pro_write_digest_cache(digest_cache_fpath, digest_cache)
        return _swebench_pro_enrich(inst, upstream_root, image_digest)

    return row_hook


def _to_gym_row(inst: dict, dataset_name: str, split: str, sampling: dict) -> dict:
    # Keep rows runnable without collect-time overrides.
    return {
        "responses_create_params": {
            "input": [],
            **sampling,
            "metadata": {
                "instance_id": inst["instance_id"],
                "dataset_name": dataset_name,
                "split": split,
                "problem_statement": inst["problem_statement"],
                "instance_dict": json.dumps(inst),
            },
        },
    }


def build_dataset(
    output: Path,
    dataset: str,
    split: str,
    limit: int | None,
    instance_id: str | None,
    sampling: dict,
    *,
    revision: str | None = None,
    row_hook=None,
    expected_count: int | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Download a dataset split and write it to the anyswe_agent row format.

    Args:
        output: Path to write the prepared JSONL to.
        dataset: HF dataset id to load (also written into each row's ``dataset_name``,
            which ``anyswe_agent`` uses to pick the matching ``swe_env`` harness).
        split: The HF split to load.
        limit: Only take the first N instances (mutually exclusive with ``instance_id``).
        instance_id: Only take this one instance.
        sampling: Model/sampling fields baked into each row's ``responses_create_params``.
        revision: Optional pinned HF dataset revision, for reproducible preparation.
        row_hook: Optional ``dict -> dict`` transform applied to each raw row before it's
            written (e.g. ``make_swebench_pro_row_hook()``, to attach assets the raw row
            is missing). Ignored (an identity no-op) for families that don't need one.
        expected_count: If set, and neither ``limit`` nor ``instance_id`` narrows the
            selection, fail loudly when the loaded split's row count doesn't match — a
            cheap guard against a silently changed/incomplete upstream dataset.

    Returns:
        tuple[list[str], dict[str, str]]: The written instance ids, and a mapping from
        each id to the image ``build_images`` should build its SIF from.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("`datasets` is required for dataset prep: pip install datasets")

    # Read HF_TOKEN directly from the environment rather than through Gym's Hydra-backed
    # get_hf_token(): that call pulls in Hydra's own CLI-argument parser, which then chokes
    # on this script's plain argparse flags (--dataset, --limit, ...) and exits before
    # anything runs. Anonymous HF downloads also rate-limit more aggressively than
    # authenticated ones, so this still helps on repeated/CI runs even without Gym's config.
    print(f"Loading {dataset} [{split}]...", flush=True)
    rows = load_dataset(dataset, split=split, revision=revision, token=os.environ.get("HF_TOKEN"))

    if instance_id:
        rows = [r for r in rows if r["instance_id"] == instance_id]
        if not rows:
            sys.exit(f"instance_id {instance_id!r} not found in {dataset}")
    elif limit:
        rows = rows.select(range(min(limit, len(rows))))
    elif expected_count is not None and len(rows) != expected_count:
        sys.exit(f"Expected {expected_count} rows from {dataset}, got {len(rows)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    ids: list[str] = []
    id_to_image: dict[str, str] = {}
    with output.open("w") as f:
        for inst in rows:
            inst = dict(inst)
            if row_hook is not None:
                inst = row_hook(inst)
            f.write(json.dumps(_to_gym_row(inst, dataset, split, sampling)) + "\n")
            iid = inst["instance_id"]
            ids.append(iid)
            id_to_image[iid] = _source_image(inst, iid)
    print(f"Wrote {len(ids)} rows -> {output}", flush=True)
    return ids, id_to_image


def _build_one_sif(instance_id: str, sif_dir: Path, force: bool, *, image: str | None = None) -> tuple[str, bool, str]:
    sif_path = sif_dir / f"{instance_id}.sif"
    if sif_path.exists() and not force:
        return instance_id, True, "exists"

    image = image or DOCKER_IMAGE_TMPL.format(tag=_docker_tag(instance_id))
    failures: list[str] = []
    for attempt in range(1, IMAGE_BUILD_ATTEMPTS + 1):
        # Build beside the final image so the completed SIF can be atomically renamed into place.
        # Keeping each attempt in its own directory also makes all failed output easy to remove.
        build_dir = Path(tempfile.mkdtemp(prefix=f".{instance_id}-", dir=sif_dir))
        staged_path = build_dir / sif_path.name
        built = False
        try:
            proc = subprocess.run(
                ["apptainer", "build", "--force", str(staged_path), image],
                capture_output=True,
                text=True,
                errors="replace",
            )
            if proc.returncode != 0:
                error = proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}"
                failures.append(f"attempt {attempt}/{IMAGE_BUILD_ATTEMPTS}: {error[-500:]}")
            elif not staged_path.is_file():
                failures.append(
                    f"attempt {attempt}/{IMAGE_BUILD_ATTEMPTS}: apptainer succeeded without producing {staged_path.name}"
                )
            else:
                os.replace(staged_path, sif_path)
                built = True
        except OSError as exc:
            failures.append(f"attempt {attempt}/{IMAGE_BUILD_ATTEMPTS}: {exc}")

        try:
            shutil.rmtree(build_dir)
        except OSError as exc:
            failures.append(f"attempt {attempt}/{IMAGE_BUILD_ATTEMPTS}: failed to clean {build_dir}: {exc}")
            return instance_id, False, "\n".join(failures)

        if built:
            detail = "built" if attempt == 1 else f"built after {attempt} attempts"
            return instance_id, True, detail
        if attempt < IMAGE_BUILD_ATTEMPTS:
            time.sleep(IMAGE_BUILD_RETRY_DELAY_SECONDS * attempt)

    return instance_id, False, "\n".join(failures)


def build_images(
    instance_ids: list[str],
    sif_dir: Path,
    jobs: int,
    force: bool,
    *,
    id_to_image: dict[str, str] | None = None,
) -> None:
    """Build one SIF per instance id, in parallel.

    Args:
        instance_ids: The ids to build SIFs for.
        sif_dir: Directory the SIFs are written into, named ``{instance_id}.sif``.
        jobs: Number of parallel ``apptainer build`` workers.
        force: Rebuild SIFs that already exist.
        id_to_image: Optional per-id source image (e.g. from ``build_dataset``). An id
            missing from this map, or an omitted map entirely, falls back to the
            SWE-bench-Verified-style templated tag (``_build_one_sif``'s own default).
    """
    if not _which("apptainer"):
        sys.exit("`apptainer` not found on PATH. Install it or pass --no-images")
    sif_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building {len(instance_ids)} SIF(s) into {sif_dir} with {jobs} worker(s)...", flush=True)
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(_build_one_sif, iid, sif_dir, force, image=(id_to_image or {}).get(iid)): iid
            for iid in instance_ids
        }
        for done in as_completed(futures):
            iid, ok, detail = done.result()
            print(f"  [{'ok' if ok else 'FAIL'}] {iid}: {detail}", flush=True)
            if not ok:
                failures.append(iid)
    if failures:
        print(f"\n{len(failures)} image build(s) failed:", flush=True)
        for iid in failures:
            print(f"  - {iid}", flush=True)
        sys.exit(1)
    print(f"All images ready. Use: container_formatter='{sif_dir}/{{instance_id}}.sif'", flush=True)


def _which(name: str) -> bool:
    from shutil import which

    return which(name) is not None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=HF_DATASET_DEFAULT, help="HF dataset id to load")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="default: data/swebench_verified.jsonl for the default dataset, else data/<dataset-basename>.jsonl",
    )
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--limit", type=int, default=None, help="Only the first N instances (default: all)")
    p.add_argument("--instance-id", default=None, help="Only this instance")
    p.add_argument("--sif-dir", type=Path, default=_THIS_DIR / "data" / "sifs")
    p.add_argument("--no-dataset", action="store_true", help="Skip dataset build")
    p.add_argument("--no-images", action="store_true", help="Skip image build")
    p.add_argument("--jobs", type=int, default=4, help="Parallel image builds")
    p.add_argument("--force", action="store_true", help="Rebuild SIFs that already exist")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Default model baked into each row")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--max-output-tokens", type=int, default=12288)
    args = p.parse_args()

    if args.output is None:
        if args.dataset == HF_DATASET_DEFAULT:
            args.output = _THIS_DIR / "data" / "swebench_verified.jsonl"
        else:
            basename = args.dataset.rsplit("/", 1)[-1].lower().replace("-", "_")
            args.output = _THIS_DIR / "data" / f"{basename}.jsonl"

    sampling = {
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_output_tokens": args.max_output_tokens,
    }

    instance_ids: list[str]
    id_to_image: dict[str, str] = {}
    if args.no_dataset:
        if not args.output.exists():
            sys.exit(f"--no-dataset but {args.output} does not exist")
        instance_ids = []
        for line in args.output.read_text().splitlines():
            if not line.strip():
                continue
            metadata = json.loads(line)["responses_create_params"]["metadata"]
            iid = metadata["instance_id"]
            instance_ids.append(iid)
            id_to_image[iid] = _source_image(json.loads(metadata["instance_dict"]), iid)
    else:
        revision = None
        row_hook = None
        expected_count = None
        if args.dataset == SWE_BENCH_PRO_HF_DATASET:
            revision = SWE_BENCH_PRO_DATASET_REVISION
            row_hook = make_swebench_pro_row_hook()
            expected_count = SWE_BENCH_PRO_EXPECTED_COUNT

        instance_ids, id_to_image = build_dataset(
            args.output,
            args.dataset,
            args.split,
            args.limit,
            args.instance_id,
            sampling,
            revision=revision,
            row_hook=row_hook,
            expected_count=expected_count,
        )

    if not args.no_images:
        build_images(instance_ids, args.sif_dir, args.jobs, args.force, id_to_image=id_to_image)


if __name__ == "__main__":
    main()
