import math
import numpy as np
from mechanistic.proxies import PAPER_PROXY_COLUMNS, compute_proxy_snapshot, confidence_term, empirical_risk_reduction, feature_discrepancy, geometric_coverage, label_discrepancy, model_complexity

def test_paper_named_proxy_values():
    axes = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert empirical_risk_reduction([2, 4], [1, 2]) == 1.5
    assert label_discrepancy([0, 0], [0, 1, 1, 1]) == 0.75
    assert feature_discrepancy(axes, axes[:1]) == 0.5
    assert geometric_coverage(axes) == 1.0
    assert model_complexity([np.array([3., 4.]), np.array([12.])]) == 17.0
    assert math.isclose(confidence_term(25), math.sqrt(2 * math.log(80) / 25))

def test_snapshot_has_only_paper_proxy_columns():
    axes = np.array([[1.0, 0.0], [0.0, 1.0]])
    snapshot = compute_proxy_snapshot(annotation_budget=4, pre_losses=[2, 4], post_losses=[1, 2], labeled_labels=[0, 0], reference_labels=[0, 1, 1, 1], labeled_features=axes, reference_features=axes, parameter_arrays=[np.array([3., 4.])])
    assert tuple(snapshot.as_dict()) == PAPER_PROXY_COLUMNS
