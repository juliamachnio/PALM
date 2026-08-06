#!/usr/bin/env python3
"""Run the public global mechanism-driven phase analysis."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mechanistic.global_regression import fit_global_phase_model, read_records, write_phase_analysis
from mechanistic.proxies import PAPER_PROXY_COLUMNS


def main():
    parser = argparse.ArgumentParser(description="Fit the global DP/BIC mechanism-driven phase model.")
    parser.add_argument("--inputs", nargs="+", required=True, type=Path, help="Paper-named per-run proxy CSV files")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--features", nargs="+", default=PAPER_PROXY_COLUMNS, choices=PAPER_PROXY_COLUMNS)
    parser.add_argument("--maximum-segments", type=int, default=6)
    parser.add_argument("--minimum-segment-points", type=int, default=8)
    parser.add_argument("--no-standardize-features", action="store_true")
    args = parser.parse_args()
    analysis, pooled = fit_global_phase_model(
        read_records(args.inputs), features=args.features, maximum_segments=args.maximum_segments,
        minimum_segment_points=args.minimum_segment_points, standardize_features=not args.no_standardize_features,
    )
    print(write_phase_analysis(args.output_dir, analysis, pooled))


if __name__ == "__main__":
    main()
