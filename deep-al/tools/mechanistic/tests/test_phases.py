"""Tests for the public global DP/BIC phase-analysis interface."""
import numpy as np
from mechanistic.global_regression import bic_score, dp_best_breaks

def test_dynamic_programming_returns_valid_global_breaks():
    target = np.r_[np.linspace(.8, .6, 10), np.linspace(.4, .1, 10)]
    features = np.column_stack([target + .01, target * 2, np.arange(20), np.ones(20), target * 3, target * 4])
    sse, cuts = dp_best_breaks(features, target, segments=2, minimum_segment_points=8, standardize_features=True)
    assert np.isfinite(sse)
    assert len(cuts) == 1 and 8 <= cuts[0] <= 12

def test_bic_is_finite_for_global_model():
    assert np.isfinite(bic_score(20, 0.5, 14))
