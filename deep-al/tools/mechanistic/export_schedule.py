#!/usr/bin/env python3
"""Export a proxy-derived hard-switch schedule from a portable trajectory CSV.

Input columns: ``budget`` plus one column per operational proxy.  The output
JSON is the only artifact required by ``--al hard_switch``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mechanistic.phases import analyze_proxy_trajectory, hard_switch_schedule


def read_proxy_csv(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = tuple(rows[0]) if rows else ()
    if "budget" not in fields or len(fields) < 3:
        raise ValueError("input must contain budget and at least two proxy columns")
    budgets = np.asarray([float(row["budget"]) for row in rows], dtype=float)
    proxies = {field: np.asarray([float(row[field]) for row in rows], dtype=float) for field in fields if field != "budget"}
    return budgets, proxies


def export_schedule(input_path: Path, output_path: Path, *, max_breakpoints: int = 2, min_segment_points: int = 3, breakpoint_neighborhood: float = 0.0) -> Path:
    budgets, proxies = read_proxy_csv(input_path)
    analysis = analyze_proxy_trajectory(
        budgets, proxies, max_breakpoints=max_breakpoints, min_segment_points=min_segment_points,
        breakpoint_neighborhood=breakpoint_neighborhood,
    )
    payload = hard_switch_schedule(analysis)
    payload["proxy_columns"] = list(proxies)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a proxy-derived hard-switch schedule.")
    parser.add_argument("--input", required=True, type=Path, help="CSV with budget and proxy columns")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON schedule")
    parser.add_argument("--max-breakpoints", type=int, default=2)
    parser.add_argument("--min-segment-points", type=int, default=3)
    parser.add_argument("--breakpoint-neighborhood", type=float, default=0.0)
    args = parser.parse_args()
    print(export_schedule(args.input, args.output, max_breakpoints=args.max_breakpoints, min_segment_points=args.min_segment_points, breakpoint_neighborhood=args.breakpoint_neighborhood))


if __name__ == "__main__":
    main()
