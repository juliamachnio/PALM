"""Synthetic checks for the mechanism-driven operational proxies."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mechanistic.proxies import (
    compute_proxy_snapshot,
    confidence_term,
    empirical_risk_reduction,
    feature_discrepancy,
    geometric_coverage,
    label_discrepancy,
    model_complexity,
)


class OperationalProxyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.axes = np.array([[1.0, 0.0], [0.0, 1.0]])

    def test_empirical_risk_reduction(self) -> None:
        self.assertAlmostEqual(empirical_risk_reduction([2.0, 4.0], [1.0, 2.0]), 1.5)

    def test_label_discrepancy_is_total_variation(self) -> None:
        self.assertAlmostEqual(label_discrepancy([0, 0], [0, 1, 1, 1]), 0.75)

    def test_feature_discrepancy_and_geometric_coverage(self) -> None:
        self.assertAlmostEqual(feature_discrepancy(self.axes, self.axes[:1]), 0.5)
        self.assertAlmostEqual(geometric_coverage(self.axes), 1.0)

    def test_model_complexity_and_confidence(self) -> None:
        self.assertAlmostEqual(model_complexity([np.array([3.0, 4.0]), np.array([12.0])]), 13.0)
        self.assertAlmostEqual(confidence_term(25, alpha=2.0), 0.4)

    def test_snapshot_has_all_paper_proxies(self) -> None:
        snapshot = compute_proxy_snapshot(
            budget=4,
            pre_losses=[2.0, 4.0],
            post_losses=[1.0, 2.0],
            labeled_labels=[0, 0],
            reference_labels=[0, 1, 1, 1],
            labeled_features=self.axes,
            reference_features=self.axes,
            parameter_arrays=[np.array([3.0, 4.0])],
        )
        self.assertEqual(snapshot.budget, 4.0)
        self.assertAlmostEqual(snapshot.empirical_risk, 1.5)
        self.assertAlmostEqual(snapshot.label_discrepancy, 0.75)
        self.assertAlmostEqual(snapshot.feature_discrepancy, 0.0)
        self.assertAlmostEqual(snapshot.geometric_coverage, 1.0)
        self.assertAlmostEqual(snapshot.model_complexity, 5.0)
        self.assertAlmostEqual(snapshot.confidence, 0.5)
        self.assertEqual(set(snapshot.as_dict()), {
            "budget", "empirical_risk", "label_discrepancy", "feature_discrepancy",
            "geometric_coverage", "model_complexity", "confidence",
        })

    def test_invalid_geometric_coverage_requires_two_points(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            geometric_coverage(self.axes[:1])


if __name__ == "__main__":
    unittest.main()
