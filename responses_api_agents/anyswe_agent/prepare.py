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
"""Prepare a Hugging Face SWE-bench dataset for anyswe_agent.

    python prepare.py                                                  # SWE-bench Verified (default)
    python prepare.py --dataset-name ScaleAI/SWE-bench_Pro            # SWE-bench Pro
    python prepare.py --limit 5                                        # first 5 instances (smoke test)
    python prepare.py --instance-id astropy__astropy-12907            # single instance
    python prepare.py --dataset-name ScaleAI/SWE-bench_Pro \\
        --build-image --sif-dir /path/to/sifs                         # also build Apptainer SIFs

Image building (--build-image) pre-builds Apptainer SIFs from each instance's Docker image
so Slurm deployments never pull from a registry at runtime. Use the resulting SIF directory
as container_formatter, e.g. ``/path/to/sifs/{instance_id}.sif``.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_HF_DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_SPLIT = "test"

# Docker Hub repository for SWE-bench Pro task images.
_PRO_IMAGE_REPOSITORY = "docker.io/jefzda/sweap-images"

_THIS_DIR = Path(__file__).parent

IMAGE_BUILD_ATTEMPTS = 3
IMAGE_BUILD_RETRY_DELAY_SECONDS = 2


# ---------------------------------------------------------------------------
# Row formatting
# ---------------------------------------------------------------------------


def _to_gym_row(inst: dict, split: str, dataset_name: str) -> dict:
    return {
        "responses_create_params": {
            "input": [],
            "metadata": {
                "instance_id": inst["instance_id"],
                "dataset_name": dataset_name,
                "split": split,
                "problem_statement": inst["problem_statement"],
                "instance_dict": json.dumps(inst),
            },
        },
    }


# ---------------------------------------------------------------------------
# Apptainer SIF building (Slurm deployments)
# ---------------------------------------------------------------------------


def _sif_source_image(inst: dict) -> str:
    """Docker image reference to pull when building a SIF for this instance."""
    if inst.get("dockerhub_tag"):
        return f"{_PRO_IMAGE_REPOSITORY}:{inst['dockerhub_tag']}"
    # Standard SWE-bench Verified image naming convention.
    mangled = inst["instance_id"].replace("__", "_1776_").lower()
    return f"swebench/sweb.eval.x86_64.{mangled}:latest"


def _build_one_sif(instance_id: str, source_image: str, sif_dir: Path, force: bool) -> tuple[str, bool, str]:
    sif_path = sif_dir / f"{instance_id}.sif"
    if sif_path.exists() and not force:
        return instance_id, True, "exists"

    failures: list[str] = []
    for attempt in range(1, IMAGE_BUILD_ATTEMPTS + 1):
        build_dir = Path(tempfile.mkdtemp(prefix=f".{instance_id}-", dir=sif_dir))
        staged_path = build_dir / sif_path.name
        built = False
        try:
            proc = subprocess.run(
                ["apptainer", "build", "--force", str(staged_path), f"docker://{source_image}"],
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
            return instance_id, True, "built" if attempt == 1 else f"built after {attempt} attempts"
        if attempt < IMAGE_BUILD_ATTEMPTS:
            time.sleep(IMAGE_BUILD_RETRY_DELAY_SECONDS * attempt)

    return instance_id, False, "\n".join(failures)


def build_sifs(gym_rows: list[dict], sif_dir: Path, jobs: int, force: bool) -> None:
    if not shutil.which("apptainer"):
        sys.exit("`apptainer` not found on PATH. Install it or omit --build-image to skip image builds.")

    sif_dir.mkdir(parents=True, exist_ok=True)
    id_to_image: dict[str, str] = {}
    for row in gym_rows:
        meta = row.get("responses_create_params", {}).get("metadata", {})
        inst_raw = meta.get("instance_dict")
        inst = json.loads(inst_raw) if isinstance(inst_raw, str) else (inst_raw or {})
        instance_id = meta.get("instance_id") or inst.get("instance_id")
        if instance_id:
            id_to_image[str(instance_id)] = _sif_source_image(inst)

    print(f"Building {len(id_to_image)} SIF(s) into {sif_dir} with {jobs} worker(s)...", flush=True)
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_build_one_sif, iid, src, sif_dir, force): iid for iid, src in id_to_image.items()}
        for fut in as_completed(futures):
            iid, ok, detail = fut.result()
            print(f"  [{'ok' if ok else 'FAIL'}] {iid}: {detail}", flush=True)
            if not ok:
                failures.append(iid)

    if failures:
        print(f"\n{len(failures)} SIF build(s) failed:", flush=True)
        for iid in failures:
            print(f"  - {iid}", flush=True)
        sys.exit(1)
    print(f"All images ready. Use: container_formatter='{sif_dir}/{{instance_id}}.sif'", flush=True)


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------


def build_dataset(
    output: Path,
    split: str,
    limit: int | None,
    instance_id: str | None,
    dataset_name: str,
) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("`datasets` is required for dataset prep: pip install datasets")

    print(f"Loading {dataset_name} [{split}]...", flush=True)
    rows = load_dataset(dataset_name, split=split)

    if instance_id:
        rows = [r for r in rows if r["instance_id"] == instance_id]
        if not rows:
            sys.exit(f"instance_id {instance_id!r} not found in {dataset_name}")
    elif limit:
        rows = rows.select(range(min(limit, len(rows))))

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    gym_rows: list[dict] = []
    with tmp.open("w", encoding="utf-8") as f:
        for inst in rows:
            gym_row = _to_gym_row(dict(inst), split, dataset_name)
            f.write(json.dumps(gym_row) + "\n")
            gym_rows.append(gym_row)
    tmp.replace(output)
    print(f"Wrote {len(gym_rows)} rows -> {output}", flush=True)
    return gym_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-name", default=DEFAULT_HF_DATASET, help="HuggingFace dataset id to load")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: data/<dataset-slug>.jsonl)",
    )
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--limit", type=int, default=None, help="Only the first N instances (default: all)")
    p.add_argument("--instance-id", default=None, help="Only this instance")
    p.add_argument(
        "--build-image",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pre-build Apptainer SIFs (Slurm deployments; requires apptainer on PATH)",
    )
    p.add_argument("--sif-dir", type=Path, default=_THIS_DIR / "data" / "sifs")
    p.add_argument("--jobs", type=int, default=4, help="Parallel SIF build workers (default: 4)")
    p.add_argument("--force", action="store_true", help="Rebuild SIFs that already exist")
    args = p.parse_args()

    if args.output is None:
        slug = args.dataset_name.split("/")[-1].lower().replace("-", "_")
        args.output = _THIS_DIR / "data" / f"{slug}.jsonl"

    gym_rows = build_dataset(args.output, args.split, args.limit, args.instance_id, args.dataset_name)

    if args.build_image:
        build_sifs(gym_rows, args.sif_dir, args.jobs, args.force)


if __name__ == "__main__":
    main()
