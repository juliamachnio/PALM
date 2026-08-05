"""Segmented-regression phase analysis for operational-proxy trajectories.

Implements Sections 3.4--3.5 of the mechanism-driven theory: standardized
proxy dominance, BIC-selected continuous hinge fits, and global transitions.
It does not select samples or switch methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class SegmentedFit:
    breakpoints: tuple[float, ...]
    coefficients: tuple[float, ...]
    slope_change_t_statistics: tuple[float, ...]
    bic: float
    rss: float
    fitted: np.ndarray

    @property
    def significant_breakpoints(self) -> tuple[float, ...]:
        return tuple(tau for tau, stat in zip(self.breakpoints, self.slope_change_t_statistics) if abs(stat) >= 1.96)


@dataclass(frozen=True)
class Phase:
    start_budget: float
    end_budget: float
    dominant_proxy: str


@dataclass(frozen=True)
class PhaseAnalysis:
    standardized: dict[str, np.ndarray]
    dominant_proxies: tuple[str, ...]
    fits: dict[str, SegmentedFit]
    phases: tuple[Phase, ...]
    transitions: tuple[float, ...]


def _validate(budgets: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(budgets, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    if len(x) != len(y) or len(x) < 4:
        raise ValueError("budgets and values must have the same length of at least four")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x) <= 0):
        raise ValueError("budgets must be finite and strictly increasing; values must be finite")
    return x, y


def standardize_proxy_values(proxies: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Standardize each proxy across budgets for the paper's dominance rule."""
    if not proxies:
        raise ValueError("at least one proxy trajectory is required")
    result: dict[str, np.ndarray] = {}
    length: int | None = None
    for name, raw_values in proxies.items():
        values = np.asarray(raw_values, dtype=float).reshape(-1)
        if not name or not len(values) or not np.isfinite(values).all():
            raise ValueError("proxy names must be non-empty and values finite")
        if length is None:
            length = len(values)
        elif len(values) != length:
            raise ValueError("all proxy trajectories must have the same length")
        std = float(np.std(values))
        result[name] = np.zeros_like(values) if std <= EPS else (values - np.mean(values)) / std
    return result


def dominant_proxies(standardized: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    """Return the maximum standardized-proxy label at each budget."""
    names = tuple(standardized)
    return tuple(names[index] for index in np.argmax(np.vstack([standardized[name] for name in names]), axis=0))


def _design(x: np.ndarray, breakpoints: tuple[float, ...]) -> np.ndarray:
    return np.column_stack([np.ones(len(x)), x, *[np.maximum(0.0, x - tau) for tau in breakpoints]])


def _fit(x: np.ndarray, y: np.ndarray, breakpoints: tuple[float, ...]) -> SegmentedFit:
    design = _design(x, breakpoints)
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residuals = y - fitted
    rss = float(residuals @ residuals)
    n, p = design.shape
    bic = float(n * np.log(max(rss / n, EPS)) + p * np.log(n))
    if not breakpoints or n <= p:
        statistics = tuple(0.0 for _ in breakpoints)
    else:
        covariance = (rss / (n - p)) * np.linalg.pinv(design.T @ design)
        errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        statistics = tuple(float(coefficients[i + 2] / max(errors[i + 2], EPS)) for i in range(len(breakpoints)))
    return SegmentedFit(breakpoints, tuple(float(v) for v in coefficients), statistics, bic, rss, fitted)


def fit_segmented_proxy(
    budgets: np.ndarray, values: np.ndarray, *, max_breakpoints: int = 2, min_segment_points: int = 3
) -> SegmentedFit:
    """Fit continuous Eq. (2) hinge models and select breakpoints by BIC."""
    x, y = _validate(budgets, values)
    if max_breakpoints < 0 or min_segment_points < 2:
        raise ValueError("max_breakpoints must be non-negative and min_segment_points at least two")
    candidates = [_fit(x, y, ())]
    n = len(x)
    positions = range(min_segment_points, n - min_segment_points + 1)
    for count in range(1, max_breakpoints + 1):
        for indices in combinations(positions, count):
            bounds = (0, *indices, n)
            if all(bounds[i + 1] - bounds[i] >= min_segment_points for i in range(count + 1)):
                candidates.append(_fit(x, y, tuple(float(x[index]) for index in indices)))
    return min(candidates, key=lambda fit: fit.bic)


def contiguous_phases(budgets: np.ndarray, labels: tuple[str, ...]) -> tuple[Phase, ...]:
    """Convert dominance labels into maximal contiguous phase intervals."""
    x = np.asarray(budgets, dtype=float).reshape(-1)
    if len(x) != len(labels) or not len(x):
        raise ValueError("budgets and labels must have the same non-zero length")
    result: list[Phase] = []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            result.append(Phase(float(x[start]), float(x[index - 1]), labels[start]))
            start = index
    return tuple(result)


def analyze_proxy_trajectory(
    budgets: np.ndarray,
    proxies: Mapping[str, np.ndarray],
    *,
    max_breakpoints: int = 2,
    min_segment_points: int = 3,
    breakpoint_neighborhood: float = 0.0,
) -> PhaseAnalysis:
    """Apply the paper's standardized dominance and Definition-3 transition rule."""
    x = np.asarray(budgets, dtype=float).reshape(-1)
    if len(x) < 4 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError("budgets must be finite, strictly increasing, and contain at least four values")
    if breakpoint_neighborhood < 0:
        raise ValueError("breakpoint_neighborhood must be non-negative")
    standardized = standardize_proxy_values(proxies)
    if any(len(values) != len(x) for values in standardized.values()):
        raise ValueError("all proxy trajectories must match the number of budgets")
    labels = dominant_proxies(standardized)
    fits = {name: fit_segmented_proxy(x, values, max_breakpoints=max_breakpoints, min_segment_points=min_segment_points)
            for name, values in standardized.items()}
    transitions = []
    for index in range(1, len(x)):
        if labels[index] == labels[index - 1]:
            continue
        support = sum(any(abs(tau - x[index]) <= breakpoint_neighborhood + EPS for tau in fit.significant_breakpoints)
                      for fit in fits.values())
        if support >= 2:
            transitions.append(float(x[index]))
    return PhaseAnalysis(standardized, labels, fits, contiguous_phases(x, labels), tuple(transitions))
