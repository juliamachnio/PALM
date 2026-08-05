"""Synthetic checks for mechanism-driven phase analysis."""

import sys
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mechanistic.phases import PhaseAnalysis, analyze_proxy_trajectory, contiguous_phases, fit_segmented_proxy, hard_switch_schedule, hard_switch_stage, standardize_proxy_values
from mechanistic.export_schedule import export_schedule, read_proxy_csv


class PhaseAnalysisTest(unittest.TestCase):
    def test_standardization_and_contiguous_phases(self) -> None:
        standardized = standardize_proxy_values({"a": [1, 2, 3], "constant": [8, 8, 8]})
        self.assertAlmostEqual(float(np.mean(standardized["a"])), 0.0)
        self.assertTrue(np.array_equal(standardized["constant"], np.zeros(3)))
        phases = contiguous_phases(np.array([10, 20, 30, 40]), ("a", "a", "b", "b"))
        self.assertEqual([(p.start_budget, p.end_budget, p.dominant_proxy) for p in phases], [(10.0, 20.0, "a"), (30.0, 40.0, "b")])

    def test_bic_selects_a_piecewise_slope_change(self) -> None:
        budgets = np.arange(1, 13, dtype=float)
        values = np.where(budgets <= 6, budgets, 6 + 4 * (budgets - 6))
        fit = fit_segmented_proxy(budgets, values, max_breakpoints=1, min_segment_points=3)
        self.assertEqual(fit.breakpoints, (6.0,))
        self.assertGreater(abs(fit.slope_change_t_statistics[0]), 1.96)

    def test_global_transition_needs_two_proxy_changes_and_dominance_change(self) -> None:
        budgets = np.arange(1, 13, dtype=float)
        decreasing = np.where(budgets <= 6, 20 - budgets, 14 - 9 * (budgets - 6))
        increasing = np.where(budgets <= 6, budgets, 6 + 4 * (budgets - 6))
        analysis = analyze_proxy_trajectory(
            budgets,
            {"discrepancy": decreasing, "coverage": increasing, "complexity": np.zeros(12), "risk": np.zeros(12)},
            max_breakpoints=1, min_segment_points=3, breakpoint_neighborhood=2.0,
        )
        self.assertIn(8.0, analysis.transitions)
        self.assertEqual(analysis.dominant_proxies[0], "discrepancy")
        self.assertEqual(analysis.dominant_proxies[-1], "coverage")

    def test_no_transition_without_dominance_change(self) -> None:
        budgets = np.arange(1, 13, dtype=float)
        analysis = analyze_proxy_trajectory(
            budgets, {"always": 2 * budgets, "other": budgets, "flat_a": np.zeros(12), "flat_b": np.zeros(12)},
            max_breakpoints=1, min_segment_points=3, breakpoint_neighborhood=1.0,
        )
        self.assertEqual(analysis.transitions, ())

    def test_hard_switch_schedule_requires_two_proxy_transitions(self) -> None:
        analysis = PhaseAnalysis({}, (), {}, (), (100.0, 300.0))
        self.assertEqual(hard_switch_schedule(analysis)["thresholds"], [100.0, 300.0])
        with self.assertRaisesRegex(ValueError, "at least two"):
            hard_switch_schedule(PhaseAnalysis({}, (), {}, (), (100.0,)))

    def test_hard_switch_dispatches_exactly_at_boundaries(self) -> None:
        thresholds = (100.0, 300.0)
        self.assertEqual(hard_switch_stage(99, thresholds), "typiclust")
        self.assertEqual(hard_switch_stage(100, thresholds), "coreset")
        self.assertEqual(hard_switch_stage(299, thresholds), "coreset")
        self.assertEqual(hard_switch_stage(300, thresholds), "uncertainty")

    def test_portable_csv_reader_and_schedule_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proxies.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["budget", "a", "b"])
                writer.writeheader()
                writer.writerows({"budget": i, "a": i, "b": 20 - i} for i in range(1, 5))
            budgets, proxies = read_proxy_csv(path)
            self.assertEqual(budgets.tolist(), [1.0, 2.0, 3.0, 4.0])
            self.assertEqual(set(proxies), {"a", "b"})
            with self.assertRaisesRegex(ValueError, "at least two proxy-derived"):
                export_schedule(path, Path(directory) / "schedule.json")


if __name__ == "__main__":
    unittest.main()
