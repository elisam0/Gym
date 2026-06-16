#!/usr/bin/env python3
"""Collect all NemoRelay ATIF traces from a results directory into one JSONL file.

Usage:
    python collect_atif.py <results_dir> [--output traces.jsonl]

Each line in the output is one ATIF trajectory (one task run).
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path, help="Run results directory (anyterminal_results_*)")
    parser.add_argument(
        "--output", "-o", type=Path, default=None, help="Output JSONL file (default: <results_dir>/atif_traces.jsonl)"
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    atif_files = sorted(results_dir.glob("*/nemo_relay/hermes-atif-*.json"))
    if not atif_files:
        print(f"No ATIF files found under {results_dir}", file=sys.stderr)
        sys.exit(1)

    output = args.output or results_dir / "atif_traces.jsonl"

    ok, errors = 0, []
    with output.open("w") as out:
        for path in atif_files:
            try:
                data = json.loads(path.read_text())
                # Annotate with the task dir name for easy filtering
                task_dir = path.parent.parent.name
                data.setdefault("extra", {})["task_dir"] = task_dir
                out.write(json.dumps(data) + "\n")
                ok += 1
            except Exception as e:
                errors.append((path, e))

    print(f"Collected {ok} ATIF traces → {output}")
    if errors:
        print(f"{len(errors)} files failed:", file=sys.stderr)
        for path, e in errors:
            print(f"  {path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
