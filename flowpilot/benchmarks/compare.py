"""
Compare benchmark JSON results across runs and flag regressions.

Usage:  python -m flowpilot.benchmarks.compare benchmarks/results/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


_REGRESSION_THRESHOLD = 1.10  # 10% slower than the previous run triggers a warning


def _load_results(directory: Path) -> list[dict]:
    files = sorted(directory.glob("hardware_*.json"))
    if not files:
        print("No benchmark results found.")
        sys.exit(0)
    return [json.loads(f.read_text()) for f in files], files


def main(directory: Path) -> None:
    results, files = _load_results(directory)

    if len(results) < 2:
        print(f"Only one result in {directory} — nothing to compare yet.")
        print(f"  {files[0].name}: {results[0]['timestamp']}")
        return

    prev = results[-2]
    curr = results[-1]
    print(f"Comparing:\n  prev: {files[-2].name}\n  curr: {files[-1].name}\n")

    prev_map = {(r["dtype"], r["size"]): r for r in prev.get("matmul", [])}
    curr_map = {(r["dtype"], r["size"]): r for r in curr.get("matmul", [])}

    regressions = []
    for key, cr in curr_map.items():
        pr = prev_map.get(key)
        if pr is None:
            print(f"  {key}: new entry — {cr['median_ms']:.2f} ms")
            continue
        ratio = cr["median_ms"] / pr["median_ms"]
        flag = " !! REGRESSION" if ratio > _REGRESSION_THRESHOLD else ""
        print(
            f"  {key[0]} {key[1]}×{key[1]}: "
            f"{pr['median_ms']:.2f} → {cr['median_ms']:.2f} ms  "
            f"({ratio:+.1%}){flag}"
        )
        if ratio > _REGRESSION_THRESHOLD:
            regressions.append(key)

    if regressions:
        print(f"\nFAIL: {len(regressions)} regression(s) detected.")
        sys.exit(1)
    else:
        print("\nOK: no regressions.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m flowpilot.benchmarks.compare <results_dir>")
        sys.exit(1)
    main(Path(sys.argv[1]))
