"""Paper-named operational proxies used by the mechanism-driven analysis.

The implementation follows the experiment code: fixed feature-space geometry,
pre/post queried-batch loss reduction, a sum-of-parameter-norms complexity
proxy, and the closed-form confidence term.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

EPS = 1e-12
PAPER_PROXY_COLUMNS = (
    "empirical_risk_reduction", "label_discrepancy", "feature_discrepancy",
    "geometric_coverage", "model_complexity", "confidence_term",
)

def _matrix(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a non-empty finite two-dimensional array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= EPS):
        raise ValueError(f"{name} contains a zero-norm feature vector")
    return values / norms

def empirical_risk_reduction(pre_losses, post_losses):
    pre, post = np.asarray(pre_losses, dtype=float).reshape(-1), np.asarray(post_losses, dtype=float).reshape(-1)
    if not len(pre) or not len(post) or not np.isfinite(pre).all() or not np.isfinite(post).all():
        raise ValueError("pre_losses and post_losses must be non-empty finite arrays")
    return float(pre.mean() - post.mean())

def label_discrepancy(labeled_labels, reference_labels):
    labeled, reference = np.asarray(labeled_labels).reshape(-1), np.asarray(reference_labels).reshape(-1)
    if not len(labeled) or not len(reference):
        raise ValueError("labeled_labels and reference_labels must be non-empty")
    classes = np.union1d(labeled, reference)
    return float(0.5 * sum(abs((labeled == c).mean() - (reference == c).mean()) for c in classes))

def _mean_nearest(query, support, exclude_self=False):
    query, support = _matrix(query, "query_features"), _matrix(support, "support_features")
    if query.shape[1] != support.shape[1]:
        raise ValueError("feature dimensions must match")
    if exclude_self:
        if len(query) < 2 or len(query) != len(support) or not np.array_equal(query, support):
            raise ValueError("geometric coverage needs at least two identical query/support rows")
    result = []
    for start in range(0, len(query), 2048):
        stop = min(start + 2048, len(query))
        distances = 1.0 - np.clip(query[start:stop] @ support.T, -1.0, 1.0)
        if exclude_self:
            distances[np.arange(stop - start), start + np.arange(stop - start)] = np.inf
        result.append(np.min(distances, axis=1))
    return float(np.mean(np.concatenate(result)))

def feature_discrepancy(reference_features, labeled_features):
    return _mean_nearest(reference_features, labeled_features)

def geometric_coverage(labeled_features):
    return _mean_nearest(labeled_features, labeled_features, exclude_self=True)

def model_complexity(parameter_arrays: Iterable[np.ndarray]):
    norms, seen = 0.0, False
    for parameter in parameter_arrays:
        values = np.asarray(parameter, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("model parameters must be finite")
        norms += float(np.linalg.norm(values.ravel(), ord=2))
        seen = True
    if not seen:
        raise ValueError("parameter_arrays must not be empty")
    return norms

def confidence_term(annotation_budget, delta=0.05):
    if annotation_budget <= 0 or not 0 < delta < 1:
        raise ValueError("annotation_budget must be positive and delta must lie in (0, 1)")
    return float(np.sqrt(2.0 * np.log(4.0 / delta) / float(annotation_budget)))

@dataclass(frozen=True)
class ProxySnapshot:
    empirical_risk_reduction: float
    label_discrepancy: float
    feature_discrepancy: float
    geometric_coverage: float
    model_complexity: float
    confidence_term: float
    def as_dict(self): return asdict(self)

def compute_proxy_snapshot(*, annotation_budget, pre_losses, post_losses, labeled_labels, reference_labels, labeled_features, reference_features, parameter_arrays, confidence_delta=0.05, confidence_annotation_budget=None):
    return ProxySnapshot(
        empirical_risk_reduction=empirical_risk_reduction(pre_losses, post_losses),
        label_discrepancy=label_discrepancy(labeled_labels, reference_labels),
        feature_discrepancy=feature_discrepancy(reference_features, labeled_features),
        geometric_coverage=geometric_coverage(labeled_features),
        model_complexity=model_complexity(parameter_arrays),
        confidence_term=confidence_term(annotation_budget if confidence_annotation_budget is None else confidence_annotation_budget, confidence_delta),
    )
