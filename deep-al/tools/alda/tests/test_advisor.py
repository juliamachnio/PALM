"""Synthetic coverage for the ALDA risk-aware decision procedure."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from alda.advisor import (
    Curve,
    PalmFit,
    advise,
    advise_from_palm_outputs,
    budget_at_target,
    deployment_window,
    fit_palm,
    palm_model,
    rank_cost_competitive,
)


class AdvisorTest(unittest.TestCase):
    def _write_csv(self, root: Path, curves: dict[str, np.ndarray], fraction: bool = False) -> Path:
        path = root / "curves.csv"
        budgets = np.arange(1, 9, dtype=float) * 10
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["dataset", "method", "cumulative_budget", "score"])
            writer.writeheader()
            for method, values in curves.items():
                for budget, score in zip(budgets, values):
                    writer.writerow({
                        "dataset": "synthetic", "method": method, "cumulative_budget": budget,
                        "score": score / 100.0 if fraction else score,
                    })
        return path

    def _read_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    def test_b_abs_inverts_palm_curve(self) -> None:
        fit = PalmFit(a_max=90.0, delta=0.1, alpha=0.0, beta=1.0, rmse=0.0, budget_size=10.0)
        budget = budget_at_target(fit, 45.0)
        self.assertIsNotNone(budget)
        self.assertEqual(float(budget) % fit.budget_size, 0.0)
        achieved = float(palm_model(np.array([budget]), 90, 0.1, 0, 1, 10)[0])
        self.assertGreaterEqual(achieved, 45.0)
        previous = float(palm_model(np.array([budget - fit.budget_size]), 90, 0.1, 0, 1, 10)[0])
        self.assertLess(previous, 45.0)

    def test_b_abs_rounds_up_to_next_episode(self) -> None:
        # Eq. (2): B_abs is the smallest budget in Z>0, i.e. a multiple of
        # budget_size, at which the curve reaches the target.
        fit = PalmFit(a_max=90.0, delta=0.1, alpha=0.0, beta=1.0, rmse=0.0, budget_size=10.0)
        budget = budget_at_target(fit, 40.0)
        self.assertIsNotNone(budget)
        self.assertEqual(float(budget) % fit.budget_size, 0.0)

    def test_fit_palm_is_reproducible_with_fixed_seed(self) -> None:
        budgets = np.arange(1, 9, dtype=float) * 10
        curve = Curve(budgets, palm_model(budgets, 92, 0.09, 0, 1, 10), 10.0)
        fit_a = fit_palm(curve, np.random.default_rng(7))
        fit_b = fit_palm(curve, np.random.default_rng(7))
        self.assertIsNotNone(fit_a)
        self.assertIsNotNone(fit_b)
        self.assertEqual(fit_a, fit_b)

    def test_deployment_window_is_positive_and_ceiling_safe(self) -> None:
        fit = PalmFit(a_max=90.0, delta=0.1, alpha=0.0, beta=1.0, rmse=0.0, budget_size=10.0)
        window = deployment_window(fit, target=88.0, delta_target=5.0)
        self.assertIsNotNone(window)
        self.assertGreater(float(window), 0.0)

    def test_deployment_window_uses_paper_asymptote_margin(self) -> None:
        fit = PalmFit(a_max=90.0, delta=0.1, alpha=0.0, beta=1.0, rmse=0.0, budget_size=10.0)
        window = deployment_window(fit, target=85.0, delta_target=5.0)
        expected = budget_at_target(fit, 89.5) - budget_at_target(fit, 80.0)
        self.assertIsNotNone(window)
        self.assertAlmostEqual(float(window), float(expected), places=6)

    def test_cost_competitive_selection_uses_window_not_only_cost(self) -> None:
        candidates = [
            {"method": "lowest_cost", "b_abs": 100.0, "deployment_window": 50.0},
            {"method": "robust_near_optimal", "b_abs": 104.0, "deployment_window": 8.0},
            {"method": "too_costly", "b_abs": 120.0, "deployment_window": 1.0},
        ]
        selected, b_min = rank_cost_competitive(candidates, eta=0.05)
        self.assertEqual(b_min, 100.0)
        self.assertEqual(selected["method"], "robust_near_optimal")
        self.assertTrue(candidates[0]["cost_competitive"])
        self.assertTrue(candidates[1]["cost_competitive"])
        self.assertFalse(candidates[2]["cost_competitive"])
        # Risk is about W relative to the recommendation, not cost: the
        # cheaper-but-wider-window method is risky; the pricier-but-narrower
        # "too_costly" method is not cost-competitive yet still not risky.
        self.assertTrue(candidates[0]["risky"])
        self.assertFalse(candidates[1]["risky"])
        self.assertFalse(candidates[2]["risky"])

    def test_feasible_and_infeasible_methods(self) -> None:
        budgets = np.arange(1, 9, dtype=float) * 10
        curves = {
            "feasible": palm_model(budgets, 92, 0.09, 0, 1, 10),
            "infeasible": palm_model(budgets, 65, 0.09, 0, 1, 10),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            advice = advise(self._write_csv(root, curves), root / "out", 80.0, 5.0, 0.05, None)
            rows = {row["method"]: row for row in self._read_rows(root / "out" / "alda_fits.csv")}
            result = self._read_rows(advice)[0]
        self.assertEqual(rows["feasible"]["feasible"], "True")
        self.assertEqual(rows["infeasible"]["feasible"], "False")
        self.assertNotEqual(rows["feasible"]["A_max"], "")
        self.assertNotEqual(rows["feasible"]["B_abs"], "")
        self.assertNotEqual(rows["feasible"]["W"], "")
        self.assertEqual(rows["feasible"]["in_C_eta"], "True")
        self.assertEqual(result["selected_method"], "feasible")
        self.assertNotEqual(result["selected_B_abs"], "")
        self.assertNotEqual(result["selected_W"], "")
        self.assertEqual(result["decision_reason"], "risk_aware_minimum_window_within_cost_competitive_set")

    def test_no_feasible_method_falls_back_to_highest_ceiling(self) -> None:
        budgets = np.arange(1, 9, dtype=float) * 10
        curves = {
            "lower": palm_model(budgets, 65, 0.09, 0, 1, 10),
            "higher": palm_model(budgets, 72, 0.09, 0, 1, 10),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            advice = advise(self._write_csv(root, curves), root / "out", 80.0, 5.0, 0.05, None)
            result = self._read_rows(advice)[0]
        self.assertEqual(result["selected_method"], "higher")
        self.assertEqual(result["decision_reason"], "infeasible_fallback_highest_estimated_ceiling")

    def test_fractional_csv_scale(self) -> None:
        budgets = np.arange(1, 9, dtype=float) * 10
        curves = {"fast": palm_model(budgets, 92, 0.09, 0, 1, 10)}
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            advice = advise(self._write_csv(root, curves, fraction=True), root / "out", 0.8, None, 0.05, None)
            result = self._read_rows(advice)[0]
        self.assertEqual(result["score_scale"], "fraction")
        self.assertEqual(result["selected_method"], "fast")

    def test_reads_palm_output_interface(self) -> None:
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
                np.save(palm_dir / "y_avg.npy", palm_model(budgets, 92, delta, 0, 1, 10))
                output_dirs.append(palm_dir)
            advice = advise_from_palm_outputs(output_dirs, root / "out", 70.0, 5.0, 0.05, None)
            result = self._read_rows(advice)[0]
        self.assertEqual(result["selected_method"], "fast")


if __name__ == "__main__":
    unittest.main()
