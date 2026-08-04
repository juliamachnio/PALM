#!/usr/bin/env python3
"""ALDA: risk-aware active-learning deployment advice.

The primary workflow is ``active-learning runs -> PALM.py -> ALDA``.  ALDA
fits the multiplicative PALM model to each observed pilot trajectory via
L-BFGS-B with seeded random restarts (paper §3.1), screens methods for
target feasibility, estimates annotation cost (rounded up to the next AL
episode per Eq. 2), and selects the most robust method among cost-competitive
candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import minimize


EPS = 1e-8
# Paper §3.4: do not estimate a deployment window arbitrarily close to the
# fitted asymptote. Scores are normalized to percentage units before this cap.
WINDOW_CEILING_MARGIN_PP = 0.5
# Paper §3.1: number of L-BFGS-B random restarts per curve fit.
N_RESTARTS = 18


@dataclass(frozen=True)
class Curve:
    budgets: np.ndarray
    scores: np.ndarray
    budget_size: float


@dataclass(frozen=True)
class PalmFit:
    a_max: float
    delta: float
    alpha: float
    beta: float
    rmse: float
    budget_size: float


def palm_model(
    budgets: np.ndarray,
    a_max: float,
    delta: float,
    alpha: float,
    beta: float,
    budget_size: float = 1.0,
) -> np.ndarray:
    """Multiplicative PALM curve: A(B)=Amax*[1-(1-delta)^((B/b+alpha)^beta)]."""
    episodes = np.asarray(budgets, dtype=float) / float(budget_size)
    inner = np.clip(episodes + alpha, EPS, None) ** beta
    return a_max * (1.0 - (1.0 - np.clip(delta, EPS, 1.0 - EPS)) ** inner)


def fit_palm(
    curve: Curve,
    rng: np.random.Generator,
    n_restarts: int = N_RESTARTS,
    maxiter: int = 2000,
) -> PalmFit | None:
    """Fit PALM with multiple random restarts (paper §3.1).

    ``rng`` drives the restart initial guesses; a fixed seed makes the fit
    (and therefore B_abs/W) reproducible across runs.
    """
    budgets, scores, budget_size = curve.budgets, curve.scores, curve.budget_size
    if len(budgets) < 4 or len(np.unique(budgets)) < 4:
        return None

    observed_max = float(np.max(scores))
    max_episode = float(np.max(budgets) / budget_size)
    a_max_hi = max(100.0, observed_max * 1.25 + 1.0)
    bounds = [
        (observed_max, a_max_hi),
        (EPS, 1.0 - EPS),
        (-max_episode, max_episode * 10.0 + 1.0),
        (0.05, 10.0),
    ]
    # Restart initial guesses are drawn from a moderate sub-range of the
    # (deliberately wide) optimizer bounds, so restarts explore plausible
    # curve shapes rather than mostly landing on numerically unstable points.
    init_ranges = [
        (observed_max, min(a_max_hi, observed_max * 1.5 + 1.0)),
        (0.005, 0.4),
        (-1.0, max(2.0, max_episode * 0.5)),
        (0.2, 3.0),
    ]

    def objective(params: np.ndarray) -> float:
        a_max, delta, alpha, beta = params
        residual = palm_model(budgets, a_max, delta, alpha, beta, budget_size=budget_size) - scores
        return float(np.dot(residual, residual))

    best: tuple[np.ndarray, float] | None = None
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(n_restarts):
            x0 = np.clip(
                [rng.uniform(lo, hi) for lo, hi in init_ranges],
                [b[0] for b in bounds],
                [b[1] for b in bounds],
            )
            try:
                result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter})
            except (RuntimeError, ValueError, FloatingPointError):
                continue
            if not np.isfinite(result.fun):
                continue
            rmse = math.sqrt(max(result.fun, 0.0) / len(scores))
            if best is None or rmse < best[1]:
                best = (result.x, rmse)
    if best is None:
        return None
    params, rmse = best
    return PalmFit(*(float(value) for value in params), rmse=rmse, budget_size=budget_size)


def budget_at_target(fit: PalmFit, target: float) -> float | None:
    """Invert PALM to estimate labels required to reach a target score.

    Per Eq. (2), the result is rounded up to the next whole AL episode (a
    multiple of ``fit.budget_size``): the smallest budget in Z>0 at which the
    curve reaches ``target``. A target equal to the asymptotic ceiling is
    evaluated infinitesimally below that ceiling to avoid the singular
    infinite-budget case.
    """
    if target <= 0.0 or not 0.0 < fit.delta < 1.0 or fit.beta <= 0.0:
        return 0.0 if target <= 0.0 else None
    safe_target = min(float(target), fit.a_max - EPS)
    if safe_target <= 0.0:
        return None
    ratio = 1.0 - safe_target / fit.a_max
    if ratio <= 0.0:
        return None
    exponent = math.log(ratio) / math.log(1.0 - fit.delta)
    if exponent <= 0.0:
        return None
    raw_episodes = exponent ** (1.0 / fit.beta) - fit.alpha
    episodes = max(0, math.ceil(raw_episodes - EPS))
    return float(episodes * fit.budget_size)


def deployment_window(fit: PalmFit, target: float, delta_target: float) -> float | None:
    """Return W=Babs(tau_hi)-Babs(tau_lo), capped 0.5 pp below Amax.

    The dedicated 0.5-percentage-point cap is the paper's deployment-window
    safeguard, distinct from ``EPS``, which only prevents numerical singularities.
    """
    tau_lo = max(0.0, target - delta_target)
    tau_hi = min(target + delta_target, fit.a_max - WINDOW_CEILING_MARGIN_PP)
    if tau_hi < tau_lo:
        return None
    lower_budget = budget_at_target(fit, tau_lo)
    upper_budget = budget_at_target(fit, tau_hi)
    if lower_budget is None or upper_budget is None:
        return None
    return max(0.0, upper_budget - lower_budget)


def _infer_budget_size(budgets: np.ndarray) -> float:
    differences = np.diff(np.unique(np.sort(budgets)))
    positive = differences[differences > 0]
    if not len(positive):
        raise ValueError("At least two distinct cumulative budgets are required to infer budget_size")
    return float(np.min(positive))


def load_curves(path: Path, budget_size_override: float | None = None) -> dict[tuple[str, str], Curve]:
    """Load a portable curve CSV with dataset, method, cumulative_budget, score."""
    groups: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", "method", "cumulative_budget", "score"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")
        for row in reader:
            groups[(row["dataset"], row["method"])].append(
                (float(row["cumulative_budget"]), float(row["score"]))
            )
    if not groups:
        raise ValueError(f"{path} contains no curve rows")

    curves: dict[tuple[str, str], Curve] = {}
    for key, values in groups.items():
        values.sort()
        budgets, scores = (np.asarray(value, dtype=float) for value in zip(*values))
        budget_size = budget_size_override or _infer_budget_size(budgets)
        if budget_size <= 0.0:
            raise ValueError("budget_size must be positive")
        curves[key] = Curve(budgets, scores, float(budget_size))
    return curves


def load_palm_outputs(paths: Iterable[Path]) -> dict[tuple[str, str], Curve]:
    """Load averaged pilot trajectories from one or more ``PALM.py --out_dir`` directories."""
    curves: dict[tuple[str, str], Curve] = {}
    for path in paths:
        params_path = path / "palm_params.json"
        curve_path = path / "y_avg.npy"
        if not params_path.exists() or not curve_path.exists():
            raise FileNotFoundError(f"{path} must contain palm_params.json and y_avg.npy")
        with params_path.open() as handle:
            metadata = json.load(handle)
        budget_size = metadata.get("budget_size")
        if budget_size is None or float(budget_size) <= 0.0:
            raise ValueError(f"{params_path} has no usable budget_size; re-run PALM.py with this release")
        scores = np.asarray(np.load(curve_path), dtype=float)
        if scores.ndim != 1 or not len(scores):
            raise ValueError(f"{curve_path} must be a non-empty one-dimensional array")
        key = (str(metadata["dataset"]), str(metadata["method"]))
        if key in curves:
            raise ValueError(f"Duplicate PALM output for dataset={key[0]!r}, method={key[1]!r}")
        budgets = np.arange(1, len(scores) + 1, dtype=float) * float(budget_size)
        curves[key] = Curve(budgets, scores, float(budget_size))
    if not curves:
        raise ValueError("At least one PALM output directory is required")
    return curves


def normalize_score_scale(curves: dict[tuple[str, str], Curve]) -> tuple[dict[tuple[str, str], Curve], bool]:
    """Normalize fractional scores to percentage units for stable PALM fitting."""
    largest_score = max(float(np.max(curve.scores)) for curve in curves.values())
    fractional = largest_score <= 1.0
    if not fractional:
        return curves, False
    normalized = {
        key: Curve(curve.budgets, curve.scores * 100.0, curve.budget_size)
        for key, curve in curves.items()
    }
    return normalized, True


def rank_cost_competitive(candidates: list[dict[str, object]], eta: float) -> tuple[dict[str, object], float]:
    """Mark C_eta, select the minimum-window method, and flag deployment-risky alternatives.

    Per 3.4/Algorithm 1, a feasible method is flagged as deployment-risky if
    it is threshold-sensitive relative to the recommendation, i.e. its
    deployment window W exceeds the selected method's W. Cost is judged
    separately via C_eta, so a method can be flagged infeasible or not
    cost-competitive without being "risky", and vice versa.
    """
    b_min = min(float(row["b_abs"]) for row in candidates)
    denominator = max(b_min, EPS)
    competitive = []
    for row in candidates:
        relative_cost = (float(row["b_abs"]) - b_min) / denominator
        row["cost_competitive"] = relative_cost <= eta + EPS
        row["in_C_eta"] = row["cost_competitive"]
        row["risky"] = False
        if bool(row["cost_competitive"]):
            competitive.append(row)
    selected = min(competitive, key=lambda row: (float(row["deployment_window"]), float(row["b_abs"])))
    selected_window = float(selected["deployment_window"])
    for row in candidates:
        if row is not selected and float(row["deployment_window"]) > selected_window:
            row["risky"] = True
    return selected, b_min


def write_rows(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def advise_curves(
    raw_curves: dict[tuple[str, str], Curve],
    output_dir: Path,
    target: float,
    delta_target: float | None,
    eta: float,
    max_points: int | None,
    seed: int = 0,
) -> Path:
    """Run feasibility, cost, and risk-aware selection for all datasets."""
    if eta < 0.0:
        raise ValueError("eta must be non-negative")
    rng = np.random.default_rng(seed)
    curves, fractional_scores = normalize_score_scale(raw_curves)
    target_percent = target * 100.0 if fractional_scores else target
    source_delta = 0.05 if fractional_scores else 5.0
    delta_percent = (delta_target if delta_target is not None else source_delta)
    if fractional_scores:
        delta_percent *= 100.0
    if delta_percent < 0.0:
        raise ValueError("delta_target must be non-negative")

    fit_rows: list[dict[str, object]] = []
    rows_by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for (dataset, method), curve in sorted(curves.items()):
        if max_points is not None:
            curve = Curve(curve.budgets[:max_points], curve.scores[:max_points], curve.budget_size)
        fit = fit_palm(curve, rng)
        row: dict[str, object] = {
            "dataset": dataset,
            "method": method,
            "points_used": len(curve.budgets),
            "budget_size": curve.budget_size,
            "observed_final_score": float(curve.scores[-1]),
            "fit_status": "ok" if fit else "too_few_points_or_fit_failed",
            "a_max": "", "delta": "", "alpha": "", "beta": "", "rmse": "",
            "feasible": False, "b_abs": "", "deployment_window": "",
            "window_over_b_abs": "", "cost_competitive": False, "risky": False,
            "A_max": "", "B_abs": "", "W": "", "W_over_B_abs": "", "in_C_eta": False,
        }
        if fit:
            feasible = fit.a_max >= target_percent
            row.update({
                "a_max": fit.a_max, "delta": fit.delta, "alpha": fit.alpha, "beta": fit.beta,
                "rmse": fit.rmse, "feasible": feasible, "A_max": fit.a_max,
            })
            if feasible:
                b_abs = budget_at_target(fit, target_percent)
                window = deployment_window(fit, target_percent, delta_percent)
                if b_abs is not None and window is not None:
                    row.update({
                        "b_abs": b_abs,
                        "deployment_window": window,
                        "window_over_b_abs": window / max(b_abs, EPS),
                        "B_abs": b_abs,
                        "W": window,
                        "W_over_B_abs": window / max(b_abs, EPS),
                    })
        fit_rows.append(row)
        rows_by_dataset[dataset].append(row)

    advice_rows: list[dict[str, object]] = []
    for dataset, rows in sorted(rows_by_dataset.items()):
        candidates = [
            row for row in rows
            if bool(row["feasible"]) and row["b_abs"] != "" and row["deployment_window"] != ""
        ]
        if candidates:
            selected, b_min = rank_cost_competitive(candidates, eta)
            reason = "risk_aware_minimum_window_within_cost_competitive_set"
            selected_b_abs = selected["b_abs"]
            selected_window = selected["deployment_window"]
        else:
            fitted = [row for row in rows if row["fit_status"] == "ok"]
            if fitted:
                selected = max(fitted, key=lambda row: float(row["a_max"]))
                reason = "infeasible_fallback_highest_estimated_ceiling"
                selected_b_abs = ""
                selected_window = ""
                b_min = ""
            else:
                selected = {"method": ""}
                reason = "no_successful_palm_fits"
                selected_b_abs = ""
                selected_window = ""
                b_min = ""
        advice_rows.append({
            "dataset": dataset,
            "score_scale": "fraction" if fractional_scores else "percent",
            "target": target,
            "delta_target": delta_target if delta_target is not None else source_delta,
            "eta": eta,
            "selected_method": selected["method"],
            "selected_b_abs": selected_b_abs,
            "selected_deployment_window": selected_window,
            "selected_B_abs": selected_b_abs,
            "selected_W": selected_window,
            "b_min": b_min,
            "decision_reason": reason,
        })

    write_rows(
        output_dir / "alda_fits.csv",
        ["dataset", "method", "points_used", "budget_size", "observed_final_score", "fit_status",
         "a_max", "delta", "alpha", "beta", "rmse", "feasible", "b_abs",
         "deployment_window", "window_over_b_abs", "cost_competitive", "risky", "A_max", "B_abs", "W",
         "W_over_B_abs", "in_C_eta"],
        fit_rows,
    )
    advice_path = output_dir / "alda_advice.csv"
    write_rows(
        advice_path,
        ["dataset", "score_scale", "target", "delta_target", "eta", "selected_method",
         "selected_b_abs", "selected_deployment_window", "selected_B_abs", "selected_W", "b_min",
         "decision_reason"],
        advice_rows,
    )
    return advice_path


def advise(
    input_path: Path,
    output_dir: Path,
    target: float,
    delta_target: float | None,
    eta: float,
    max_points: int | None,
    budget_size: float | None = None,
    seed: int = 0,
) -> Path:
    """Run ALDA from a portable external curve CSV."""
    return advise_curves(
        load_curves(input_path, budget_size), output_dir, target, delta_target, eta, max_points, seed
    )


def advise_from_palm_outputs(
    paths: Iterable[Path],
    output_dir: Path,
    target: float,
    delta_target: float | None,
    eta: float,
    max_points: int | None,
    seed: int = 0,
) -> Path:
    """Run ALDA from one PALM output directory per candidate method."""
    return advise_curves(load_palm_outputs(paths), output_dir, target, delta_target, eta, max_points, seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ALDA: risk-aware advice from PALM learning-curve outputs.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="CSV with dataset, method, cumulative_budget, score.")
    source.add_argument("--palm-output", type=Path, nargs="+", metavar="DIR", help="PALM.py --out_dir directories.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for ALDA CSV outputs.")
    parser.add_argument("--target", required=True, type=float, help="Required score in the input score scale.")
    parser.add_argument(
        "--delta-target", type=float, default=None,
        help="Target uncertainty in input score units (default: 5 percentage points or 0.05 for fractional scores).",
    )
    parser.add_argument("--eta", type=float, default=0.05, help="Maximum relative B_abs increase for C_eta (default: 0.05).")
    parser.add_argument("--max-points", type=int, default=None, help="Use only this many early observations per curve.")
    parser.add_argument(
        "--budget-size", type=float, default=None,
        help="Override inferred labels-per-episode for the optional CSV interface only.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for the L-BFGS-B random-restart curve fit, for reproducibility (default: 0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.palm_output:
        advice = advise_from_palm_outputs(
            args.palm_output, args.output_dir, args.target, args.delta_target, args.eta, args.max_points, args.seed
        )
    else:
        advice = advise(
            args.input, args.output_dir, args.target, args.delta_target, args.eta, args.max_points,
            args.budget_size, args.seed,
        )
    print(f"Wrote ALDA recommendation: {advice}")
