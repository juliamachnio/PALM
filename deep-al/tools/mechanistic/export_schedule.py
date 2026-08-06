#!/usr/bin/env python3
"""Create a fixed hard-switch schedule from two explicitly selected budgets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def export_schedule(output_path: Path, switch_1: float, switch_2: float, phase_analysis: Path | None = None):
    if not 0 < switch_1 < switch_2:
        raise ValueError("switch budgets must be positive and strictly increasing")
    payload = {
        "schema": "mechanistic-hard-switch-v1",
        "source": "mechanistic_global_regression",
        "selection": "predefined_thresholds",
        "thresholds": [float(switch_1), float(switch_2)],
        "stages": ["typiclust", "coreset", "uncertainty"],
    }
    if phase_analysis is not None:
        payload["phase_analysis"] = str(phase_analysis)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Create a fixed three-stage hard-switch schedule.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--switch-1", required=True, type=float, help="TypiClust-to-CoreSet annotation budget")
    parser.add_argument("--switch-2", required=True, type=float, help="CoreSet-to-uncertainty annotation budget")
    parser.add_argument("--phase-analysis", type=Path, help="Optional global_phase_analysis.json used to select these budgets")
    args = parser.parse_args()
    print(export_schedule(args.output, args.switch_1, args.switch_2, args.phase_analysis))

if __name__ == "__main__":
    main()
