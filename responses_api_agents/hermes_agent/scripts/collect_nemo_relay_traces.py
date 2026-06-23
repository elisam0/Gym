#!/usr/bin/env python3
"""Collect all NemoRelay ATIF traces from a results directory into one JSONL file.

Usage:
    python collect_nemo_relay_traces.py <results_dir> [--output traces.jsonl]

Each line in the output is one ATIF trajectory (one task run).
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path, help="Run results directory (anyterminal_results_*)")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Output folder (default: <results_dir>)")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    atif_files = sorted(results_dir.glob("*/nemo_relay/*/*.atif.json"))
    atof_files = sorted(results_dir.glob("*/nemo_relay/*/*.atof.jsonl"))
    if not atif_files or not atof_files:
        print(f"No ATIF or ATOF files found under {results_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output or results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    import shutil

    atif_ok, atif_errors = 0, []
    atif_traces_dir = output_dir / "atif-traces"
    atif_traces_dir.mkdir(parents=True, exist_ok=True)
    for path in atif_files:
        try:
            # Copy the ATIF file to the destination with a new name based on the task dir
            task_dir = path.parent.parent.parent.name
            atif_out_path = atif_traces_dir / f"{task_dir}.atif.json"
            shutil.copy2(path, atif_out_path)
            atif_ok += 1
        except Exception as e:
            atif_errors.append((path, e))

    atof_ok, atof_errors = 0, []
    atof_traces_dir = output_dir / "atof-traces"
    atof_traces_dir.mkdir(parents=True, exist_ok=True)
    for path in atof_files:
        try:
            # Copy the ATOF file to the destination with a new name based on the task dir
            task_dir = path.parent.parent.parent.name
            atof_out_path = atof_traces_dir / f"{task_dir}.atof.jsonl"
            shutil.copy2(path, atof_out_path)
            atof_ok += 1
        except Exception as e:
            atof_errors.append((path, e))

    print(f"Collected {atif_ok} ATIF traces → {atif_traces_dir}")
    print(f"Collected {atof_ok} ATOF traces → {atof_traces_dir}")
    if atif_errors:
        print(f"{len(atif_errors)} ATIF files failed:", file=sys.stderr)
        for path, e in atif_errors:
            print(f"  {path}: {e}", file=sys.stderr)
    if atof_errors:
        print(f"{len(atof_errors)} ATOF files failed:", file=sys.stderr)
        for path, e in atof_errors:
            print(f"  {path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
