"""Synthetic smoke test for the portable ALDA advisor."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from alda.advisor import advise, advise_from_palm_outputs, palm_model


class AdvisorSmokeTest(unittest.TestCase):
    def test_recommends_the_faster_curve(self) -> None:
        budgets = [10, 20, 30, 40, 50, 60, 70, 80]
        curves = {
            "fast": palm_model(__import__("numpy").asarray(budgets, dtype=float), 92, 0.09, 0, 1),
            "slow": palm_model(__import__("numpy").asarray(budgets, dtype=float), 92, 0.025, 0, 1),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "curves.csv"
            with input_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["dataset", "method", "cumulative_budget", "score"])
                writer.writeheader()
                for method, values in curves.items():
                    for budget, score in zip(budgets, values):
                        writer.writerow({"dataset": "synthetic", "method": method, "cumulative_budget": budget, "score": score})
            advise(input_path, root / "out", target=70.0, max_points=None)
            with (root / "out" / "alda_advice.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
        self.assertEqual(row["recommended_method"], "fast")

    def test_reads_palm_output_directly(self) -> None:
        budgets = np.arange(1, 9, dtype=float) * 10
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dirs = []
            for method, delta in {"fast": 0.09, "slow": 0.025}.items():
                palm_dir = root / method
                palm_dir.mkdir()
                (palm_dir / "palm_params.json").write_text(json.dumps({
                    "dataset": "synthetic", "method": method, "budget_size": 10,
                }))
                np.save(palm_dir / "y_avg.npy", palm_model(budgets, 92, delta, 0, 1))
                output_dirs.append(palm_dir)
            advise_from_palm_outputs(output_dirs, root / "out", target=70.0, max_points=None)
            with (root / "out" / "alda_advice.csv").open(newline="") as handle:
                row = next(csv.DictReader(handle))
        self.assertEqual(row["recommended_method"], "fast")


if __name__ == "__main__":
    unittest.main()
