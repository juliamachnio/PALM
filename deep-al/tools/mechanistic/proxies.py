"""Operational proxies from the mechanism-driven active-learning analysis.

These functions implement the observable counterparts of the bound components
in Section 3.6 of the accompanying paper.  They describe a trajectory; they do
not select samples, estimate urgency, allocate a budget, or switch methods.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


EPS = 1e-12


def _as_feature_matrix(features: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty two-dimensional feature matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _normalise_rows(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms <= EPS):
        raise ValueError("feature vectors must have non-zero L2 norm")
    return features / norms


def empirical_risk_reduction(pre_losses: np.ndarray, post_losses: np.ndarray) -> float:
    """Return ER(t) = mean(loss_pre) - mean(loss_post) for the acquired batch."""
    pre = np.asarray(pre_losses, dtype=float).reshape(-1)
    post = np.asarray(post_losses, dtype=float).reshape(-1)
    if len(pre) == 0 or len(post) == 0:
        raise ValueError("pre_losses and post_losses must both be non-empty")
    if not np.isfinite(pre).all() or not np.isfinite(post).all():
        raise ValueError("loss arrays must contain only finite values")
    return float(np.mean(pre) - np.mean(post))


def label_discrepancy(labeled_labels: np.ndarray, reference_labels: np.ndarray) -> float:
    """Return LD(S), the total-variation distance between label frequencies."""
    labeled = np.asarray(labeled_labels).reshape(-1)
    reference = np.asarray(reference_labels).reshape(-1)
    if len(labeled) == 0 or len(reference) == 0:
        raise ValueError("labeled_labels and reference_labels must both be non-empty")
    classes = np.union1d(labeled, reference)
    labeled_frequency = np.array([(labeled == cls).mean() for cls in classes])
    reference_frequency = np.array([(reference == cls).mean() for cls in classes])
    return float(0.5 * np.abs(labeled_frequency - reference_frequency).sum())


def _mean_min_cosine_distance(
    query_features: np.ndarray,
    support_features: np.ndarray,
    *,
    exclude_self: bool,
    chunk_size: int = 2048,
) -> float:
    query = _normalise_rows(_as_feature_matrix(query_features, "query_features"))
    support = _normalise_rows(_as_feature_matrix(support_features, "support_features"))
    if query.shape[1] != support.shape[1]:
        raise ValueError("query_features and support_features must have equal dimensions")
    if exclude_self and (len(query) != len(support) or not np.array_equal(query, support)):
        raise ValueError("exclude_self=True requires the same feature matrix for query and support")
    if exclude_self and len(query) < 2:
        raise ValueError("at least two labeled feature vectors are required for geometric coverage")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    minimums: list[np.ndarray] = []
    for start in range(0, len(query), chunk_size):
        stop = min(start + chunk_size, len(query))
        distances = 1.0 - np.clip(query[start:stop] @ support.T, -1.0, 1.0)
        if exclude_self:
            rows = np.arange(stop - start)
            distances[rows, start + rows] = np.inf
        minimums.append(np.min(distances, axis=1))
    return float(np.mean(np.concatenate(minimums)))


def feature_discrepancy(reference_features: np.ndarray, labeled_features: np.ndarray) -> float:
    """Return FD(S): mean distance from each reference embedding to S."""
    return _mean_min_cosine_distance(reference_features, labeled_features, exclude_self=False)


def geometric_coverage(labeled_features: np.ndarray) -> float:
    """Return GC(S): mean nearest-neighbour cosine distance within S."""
    return _mean_min_cosine_distance(labeled_features, labeled_features, exclude_self=True)


def model_complexity(parameter_arrays: Iterable[np.ndarray]) -> float:
    """Return Comp(t), the joint L2 norm of trained model parameter arrays."""
    squared_norm = 0.0
    seen = False
    for parameter in parameter_arrays:
        values = np.asarray(parameter, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("model parameters must contain only finite values")
        squared_norm += float(np.square(values).sum())
        seen = True
    if not seen:
        raise ValueError("parameter_arrays must contain at least one array")
    return float(np.sqrt(squared_norm))


def confidence_term(budget: int | float, alpha: float = 1.0) -> float:
    """Return Conf(t) = alpha / sqrt(m_t), the paper's vanishing baseline."""
    if budget <= 0:
        raise ValueError("budget must be positive")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    return float(alpha / np.sqrt(float(budget)))


@dataclass(frozen=True)
class ProxySnapshot:
    """Operational-proxy values at one cumulative annotation budget."""

    budget: float
    empirical_risk: float
    label_discrepancy: float
    feature_discrepancy: float
    geometric_coverage: float
    model_complexity: float
    confidence: float

    def as_dict(self) -> dict[str, float]:
        """Return CSV-friendly proxy names used by later trajectory analysis."""
        return asdict(self)


def compute_proxy_snapshot(
    *,
    budget: int | float,
    pre_losses: np.ndarray,
    post_losses: np.ndarray,
    labeled_labels: np.ndarray,
    reference_labels: np.ndarray,
    labeled_features: np.ndarray,
    reference_features: np.ndarray,
    parameter_arrays: Iterable[np.ndarray],
    confidence_alpha: float = 1.0,
) -> ProxySnapshot:
    """Compute the complete paper proxy vector for one AL episode."""
    return ProxySnapshot(
        budget=float(budget),
        empirical_risk=empirical_risk_reduction(pre_losses, post_losses),
        label_discrepancy=label_discrepancy(labeled_labels, reference_labels),
        feature_discrepancy=feature_discrepancy(reference_features, labeled_features),
        geometric_coverage=geometric_coverage(labeled_features),
        model_complexity=model_complexity(parameter_arrays),
        confidence=confidence_term(budget, alpha=confidence_alpha),
    )
