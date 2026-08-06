"""Global DP/BIC segmented regression used for the mechanism analysis."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .proxies import PAPER_PROXY_COLUMNS


def _zscore(values):
    mean, std = np.mean(values, axis=0), np.std(values, axis=0, ddof=1)
    return (values - mean) / np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)


def _ridge_fit(features, target, penalty=1e-6):
    design = np.column_stack([np.ones(len(features)), features])
    regularizer = np.eye(design.shape[1]) * penalty
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ target)
    return coefficients, design @ coefficients


def _segment_sse(features, target, start, stop, standardize_features):
    subset, values = features[start:stop], target[start:stop]
    if standardize_features:
        subset = _zscore(subset)
    _, prediction = _ridge_fit(subset, values)
    return float(np.sum((values - prediction) ** 2))


def dp_best_breaks(features, target, segments, minimum_segment_points, standardize_features):
    """Exact dynamic-programming segmentation from the experiment implementation."""
    n = len(target)
    infinity = 1e30
    sse = np.full((n + 1, n + 1), infinity)
    for start in range(n):
        for stop in range(start + minimum_segment_points, n + 1):
            sse[start, stop] = _segment_sse(features, target, start, stop, standardize_features)
    costs = np.full((segments + 1, n + 1), infinity)
    previous = np.full((segments + 1, n + 1), -1, dtype=int)
    costs[0, 0] = 0.0
    for count in range(1, segments + 1):
        for stop in range(count * minimum_segment_points, n + 1):
            for start in range((count - 1) * minimum_segment_points, stop - minimum_segment_points + 1):
                value = costs[count - 1, start] + sse[start, stop]
                if value < costs[count, stop]:
                    costs[count, stop], previous[count, stop] = value, start
    cuts, stop = [], n
    for count in range(segments, 0, -1):
        start = previous[count, stop]
        if start < 0:
            return infinity, ()
        if start:
            cuts.append(start)
        stop = start
    return float(costs[segments, n]), tuple(sorted(cuts))


def bic_score(n, sse, degrees_of_freedom):
    return float(n * np.log(max(sse, 1e-12) / n) + degrees_of_freedom * np.log(n))


@dataclass(frozen=True)
class GlobalPhaseAnalysis:
    features: tuple[str, ...]
    segment_count: int
    breakpoints: tuple[float, ...]
    bic: float
    sse: float
    coefficients: tuple[tuple[float, ...], ...]

    def as_dict(self):
        return asdict(self)


def _mean_runs(records, features):
    grouped = {}
    for row in records:
        key = (row["method"], float(row["annotation_budget"]))
        grouped.setdefault(key, []).append(row)
    result = []
    for (method, budget), rows in grouped.items():
        result.append({"method": method, "annotation_budget": budget, "true_risk": float(np.mean([1.0 - float(row["test_accuracy"]) / (100.0 if float(row["test_accuracy"]) > 1.5 else 1.0) for row in rows])), **{feature: float(np.mean([float(row[feature]) for row in rows])) for feature in features}})
    return sorted(result, key=lambda row: (row["annotation_budget"], row["method"]))


def fit_global_phase_model(records, *, features=PAPER_PROXY_COLUMNS, maximum_segments=6, minimum_segment_points=8, standardize_features=True):
    """Pool methods, average seeds, then choose global segments by DP and BIC."""
    features = tuple(features)
    required = {"method", "seed", "annotation_budget", "test_accuracy", *features}
    if not records or any(not required.issubset(row) for row in records):
        raise ValueError("records must contain method, seed, annotation_budget, test_accuracy, and all selected paper proxies")
    pooled = _mean_runs(records, features)
    matrix = np.asarray([[row[name] for name in features] for row in pooled], dtype=float)
    target = np.asarray([row["true_risk"] for row in pooled], dtype=float)
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("phase analysis requires finite records; omit the initial undefined empirical-risk row")
    minimum_segment_points = max(int(minimum_segment_points), len(features) + 2)
    candidate = None
    for segments in range(1, int(maximum_segments) + 1):
        if segments * minimum_segment_points > len(target):
            break
        sse, cuts = dp_best_breaks(matrix, target, segments, minimum_segment_points, standardize_features)
        score = bic_score(len(target), sse, segments * (len(features) + 1))
        if candidate is None or score < candidate[0]:
            candidate = (score, sse, segments, cuts)
    if candidate is None:
        raise ValueError("too few pooled records for the selected features and minimum segment size")
    score, sse, segments, cuts = candidate
    boundaries = (0, *cuts, len(target))
    coefficients = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        values = matrix[start:stop]
        if standardize_features:
            values = _zscore(values)
        coefficients.append(tuple(float(value) for value in _ridge_fit(values, target[start:stop])[0]))
    return GlobalPhaseAnalysis(features, segments, tuple(float(pooled[cut]["annotation_budget"]) for cut in cuts), score, sse, tuple(coefficients)), pooled


def read_records(paths):
    records = []
    for path in paths:
        with Path(path).open(newline="") as handle:
            records.extend(csv.DictReader(handle))
    return records


def write_phase_analysis(output_directory, analysis, pooled):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "global_phase_analysis.json").write_text(json.dumps(analysis.as_dict(), indent=2) + "\n")
    columns = ("method", "annotation_budget", "true_risk", *analysis.features)
    with (output / "pooled_phase_records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(pooled)
    return output
