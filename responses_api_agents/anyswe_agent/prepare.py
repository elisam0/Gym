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
Prepare the anyswe_agent input dataset from a Hugging Face SWE-bench dataset.

    python prepare.py                                             # download + build dataset (no images)
    python prepare.py --limit 5                                   # first 5 instances (smoke test)
    python prepare.py --instance-id astropy__astropy-12907        # single instance
    python prepare.py --build-image --sandbox-provider PROVIDER   # also prepare provider images
    python prepare.py --build-image --sandbox-provider PROVIDER --image-dir PATH

Image preparation is opt-in via --build-image; it pulls each instance's SWE-bench eval image
through the provider hook (e.g. Apptainer) and records the provider image reference in
`responses_create_params.metadata.image`, which the AnySWE runtime already knows how to use.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from nemo_gym.sandbox.providers import SandboxImagePrepareRequest, create_provider, prepare_provider_image


DEFAULT_HF_DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_SPLIT = "test"
IMAGE_BUILD_ATTEMPTS = 3
IMAGE_BUILD_RETRY_DELAY_SECONDS = 2

_THIS_DIR = Path(__file__).parent


def _mangled_instance_id(instance_id: str) -> str:
    return instance_id.replace("__", "_1776_").lower()


def _source_image(instance_id: str) -> str:
    return f"docker://swebench/sweb.eval.x86_64.{_mangled_instance_id(instance_id)}:latest"


def _parse_provider_config(raw: str) -> dict:
    value = raw.strip()
    if not value:
        raise ValueError("sandbox provider must be a provider name or JSON object")
    if value.startswith("{"):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("sandbox provider JSON must be an object")
        return parsed
    return {value: {}}


def _to_gym_row(inst: dict, split: str, dataset_name: str = DEFAULT_HF_DATASET) -> dict:
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


def build_dataset(
    output: Path,
    split: str,
    limit: int | None,
    instance_id: str | None,
    dataset_name: str,
) -> None:
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
    count = 0
    with output.open("w") as f:
        for inst in rows:
            inst = dict(inst)
            f.write(json.dumps(_to_gym_row(inst, split, dataset_name)) + "\n")
            count += 1
    print(f"Wrote {count} rows -> {output}", flush=True)


def _build_one_image(
    provider: object,
    instance_id: str,
    source_image: str,
    image_dir: Path,
) -> tuple[str, bool, str, str]:
    result = prepare_provider_image(
        provider,
        SandboxImagePrepareRequest(
            image=source_image,
            target_dir=image_dir,
            target_name=instance_id,
            attempts=IMAGE_BUILD_ATTEMPTS,
            retry_delay_s=IMAGE_BUILD_RETRY_DELAY_SECONDS,
        ),
    )
    return instance_id, result.ok, result.detail, result.image


def build_images(rows: list[dict], image_dir: Path, jobs: int, provider: object) -> dict[str, str]:
    """Prepare a provider image per row. Returns {instance_id: provider_image} for all rows,
    exiting the process if any preparation fails."""
    image_dir.mkdir(parents=True, exist_ok=True)
    print(f"Preparing {len(rows)} image(s) into {image_dir} with {jobs} worker(s)...", flush=True)
    images: dict[str, str] = {}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(
                _build_one_image,
                provider,
                r["responses_create_params"]["metadata"]["instance_id"],
                _source_image(r["responses_create_params"]["metadata"]["instance_id"]),
                image_dir,
            ): r
            for r in rows
        }
        for done in as_completed(futures):
            instance_id, ok, detail, image = done.result()
            print(f"  [{'ok' if ok else 'FAIL'}] {instance_id}: {detail}", flush=True)
            if ok:
                images[instance_id] = image
            else:
                failures.append(instance_id)
    if failures:
        print(f"\n{len(failures)} image preparation(s) failed:", flush=True)
        for name in failures:
            print(f"  - {name}", flush=True)
        sys.exit(1)
    print("Image preparation complete.", flush=True)
    return images


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-name", default=DEFAULT_HF_DATASET, help="Hugging Face dataset to load")
    p.add_argument("--output", type=Path, default=_THIS_DIR / "data" / "swebench_verified.jsonl")
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--limit", type=int, default=None, help="Only the first N instances (default: all)")
    p.add_argument("--instance-id", default=None, help="Only this instance")
    p.add_argument("--build-image", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--sandbox-provider",
        default=None,
        help="Provider name or single-key provider config JSON used for image preparation.",
    )
    p.add_argument("--image-dir", type=Path, default=_THIS_DIR / "data" / "images")
    p.add_argument("--jobs", type=int, default=4)
    args = p.parse_args()

    build_dataset(args.output, args.split, args.limit, args.instance_id, args.dataset_name)

    if not args.build_image:
        return

    if args.sandbox_provider is None:
        sys.exit("--sandbox-provider is required when --build-image is set")
    try:
        provider_config = _parse_provider_config(args.sandbox_provider)
    except (ValueError, json.JSONDecodeError) as exc:
        sys.exit(str(exc))
    provider = create_provider(provider_config)

    rows = [json.loads(line) for line in args.output.read_text().splitlines() if line.strip()]
    images = build_images(rows, args.image_dir, args.jobs, provider)

    for row in rows:
        instance_id = row["responses_create_params"]["metadata"]["instance_id"]
        row["responses_create_params"]["metadata"]["image"] = images[instance_id]
    with args.output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} rows with image references -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
