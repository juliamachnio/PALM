"""
Author: Julia Machnio (PALM)
License: MIT

PALM curve fitting template

Fits the PALM curve:
    y(x) = A_max * (1 - (1 - δ) ** ((x_eff + α) **β ))

where:
  - A_max  : achievable accuracy (upper bound)
  - δ (delta): coverage efficiency (0..1)
  - α (alpha): early-stage performance / growth shape
  - β (beta): scalability

x_eff can be either:
  (a) episode index x (1..N), or
  (b) cumulative normalised budget [(x*B)/b where B=x*b if b was fixed for all rounds]
        if --x_mode cumulative_normalized is set

The script:
  1) loads multiple runs’ y-values (.npy),
  2) averages them,
  3) fits the PALM curve,
  4) prints & saves parameters + diagnostics,
  5) optionally plots data + fit.
"""

import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
import json
import glob
import re

# -------------------- PALM MODELS -------------------- #
EPS = 1e-8

def _pos(x):
    # ensure base of the power is strictly positive
    return np.clip(x, EPS, None)

def palm_model(x, A_max, delta, alpha, beta):
    return A_max * (1 - (1 - delta) ** (_pos(x + alpha) ** beta))

def palm_model_cum(x, B, b, A_max, delta, alpha, beta):
    x_eff = (x * B)/b
    return A_max * (1 - (1 - delta) ** (_pos(x_eff + alpha) ** beta))


# -------------------- RUN DISCOVERY -------------------- #
def natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def find_run_files(base_path: Path, method: str, num_variants: int, filename="plot_episode_yvalues.npy"):

    base_path = Path(base_path)

    candidates = []

    # 1) Direct-under-base single run
    direct = base_path / filename
    if direct.exists():
        candidates.append(direct)

    # 2) Exact {method}_{i}
    for i in range(1, num_variants + 1):
        p = base_path / f"{method}_{i}" / filename
        if p.exists():
            candidates.append(p)

    # 3) Fuzzy {method}_{i}* (e.g., method_1_seed42)
    if len(candidates) < num_variants:
        pattern = str(base_path / f"{method}_*" / filename)
        for p in sorted(glob.glob(pattern), key=natural_key):
            candidates.append(Path(p))

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for p in candidates:
        if p not in seen:
            uniq.append(p)
            seen.add(p)

    # Limit to num_variants if > 0 and we found too many
    if num_variants > 0:
        uniq = uniq[:num_variants]

    return uniq

# -------------------- MAIN -------------------- #
def main():
    ap = argparse.ArgumentParser(description="Fit PALM curve to averaged runs (no pattern needed)")
    ap.add_argument("--base_path", type=str, required=True,
                    help="Folder that CONTAINS the run folders (e.g., .../output/IMAGENET50/resnet50)")
    ap.add_argument("--dataset", type=str, default="IMAGENET50")  
    ap.add_argument("--model", type=str, default="resnet50")      
    ap.add_argument("--method", type=str, required=True,
                    help="Prefix of run folders, e.g., 'random_ff_byol_b50' for random_ff_byol_b50_1, _2, ...")
    ap.add_argument("--num_variants", type=int, default=1,
                    help="How many run variants to average (1..num_variants). Use what exists.")
    ap.add_argument("--num_samples", type=int, default=20,
                    help="How many episodes to use from each run; -1 uses full length")

    # x-mode: episodes or cumulative-normalized
    ap.add_argument("--x_mode", choices=["episodes", "cumulative_normalized"],
                    default="episodes",
                    help="episodes: fit on x; cumulative_normalized: fit on x*b but plot as 1..N")
    ap.add_argument("--budget_size", "-b", type=float, default=100.0,
                    help="Budget per episode (used iff x_mode=cumulative_normalized)")

    # Initial guesses & bounds
    ap.add_argument("--init_A_max", type=float, default=None)
    ap.add_argument("--init_delta", type=float, default=0.5)
    ap.add_argument("--init_alpha", type=float, default=1.0)
    ap.add_argument("--init_beta",  type=float, default=1.0)
    ap.add_argument("--beta_hi", type=float, default=100.0)
    ap.add_argument("--alpha_lo",  type=float, default=-1000.0)
    ap.add_argument("--alpha_hi",  type=float, default=1000.0)

    # Output / plotting
    ap.add_argument("--out_dir", type=str, default="./palm_fit_out",
                    help="Directory to save parameters/plots")
    ap.add_argument("--plot", action="store_true", help="Show and save a plot")
    ap.add_argument("--save_csv", action="store_true", help="Also save params as CSV")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    num_samples = None if args.num_samples == -1 else args.num_samples

    # Discover run files
    run_files = find_run_files(Path(args.base_path), args.method, args.num_variants)
    print("Discovered runs:")
    for rf in run_files:
        print("  -", rf)
    if not run_files:
        raise FileNotFoundError(
            f"No runs found under {args.base_path} for method prefix '{args.method}'. "
            f"Expected folders like {args.method}_1, {args.method}_2, ... containing plot_episode_yvalues.npy."
        )

    # Load & average
    runs = []
    for f in run_files:
        arr = np.load(f)
        if num_samples is not None:
            arr = arr[:num_samples]
        runs.append(arr)

    # Trim to shortest length (safety)
    min_len = min(len(r) for r in runs)
    y_stack = np.stack([r[:min_len] for r in runs], axis=0)
    y_avg = y_stack.mean(axis=0)

    # x = 1..N iterations (always for plotting)
    x = np.arange(1, len(y_avg) + 1, dtype=float)

    # Fit
    A0 = args.init_A_max if args.init_A_max is not None else float(y_avg.max())
    p0 = [A0, args.init_delta, args.init_alpha, args.init_beta]
    # Bounds: A_max ≥ 0, 0 ≤ δ ≤ 1, α can be negative, β ≥ 0
    alpha_lo = getattr(args, "alpha_lo", -1000.0)                
    bounds_lo = [0.0, 0.0, -1000, 0.0]
    bounds_hi = [100.0, 1.0, args.alpha_hi, args.beta_hi]


    if args.x_mode == "episodes":
        f = lambda xv, A, d, a, b: palm_model(xv, A, d, a, b)
        model_used = "PALM (episodes)"
    else:
        f = lambda xv, A, d, a, b: palm_model_cum(xv, args.budget_size, args.budget_size, A, d, a, b)
        model_used = f"PALM (cumulative-normalized, b={args.budget_size:g})"

    popt, pcov = curve_fit(f, x, y_avg, p0=p0, bounds=(bounds_lo, bounds_hi), maxfev=50000)
    A_max, delta, alpha, beta = popt
    y_fit = f(x, *popt)

    # Print & save params
    print("\n=== PALM fit ===")
    print(f"dataset={args.dataset}  model={args.model}  method={args.method}")
    print(f"variants={len(runs)}  episodes={len(x)}")
    print(f"mode: {model_used}")
    print(f"A_max={A_max:.6f}  delta={delta:.8f}  alpha={alpha:.8f}  beta={beta:.8f}")

    meta = {
        "dataset": args.dataset,
        "model": args.model,
        "method": args.method,
        "episodes": len(x),
        "mode": model_used,
        # Always persist the acquisition budget: ALDA uses it to turn episode
        # indices in y_avg.npy into comparable cumulative label budgets.
        "budget_size": float(args.budget_size),
        "A_max": float(A_max),
        "delta": float(delta),
        "alpha": float(alpha),
        "beta":  float(beta),
        "run_files": [str(p) for p in run_files],
    }
    (out_dir / "palm_params.json").write_text(json.dumps(meta, indent=2))

    if args.save_csv:
        import csv
        with open(out_dir / "palm_params.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset","model","method","episodes","mode","budget_size","A_max","delta","alpha","beta"])
            w.writerow([args.dataset,args.model,args.method,len(x),model_used,
                        (args.budget_size if args.x_mode=="cumulative_normalized" else ""),
                        f"{A_max:.6f}",f"{delta:.8f}",f"{alpha:.8f}",f"{beta:.8f}"])

    # Save series
    np.save(out_dir / "y_avg.npy", y_avg)
    np.save(out_dir / "y_fit.npy", y_fit)

    # Optional plot
    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9,5))
        plt.plot(x*args.budget_size, y_avg, "o", label=f"{args.method} (avg)", markersize=4)
        plt.plot(x*args.budget_size, y_fit, "-", label="PALM fit", linewidth=2)
        plt.title(f"{args.dataset} · {args.model} · {args.method}\n{model_used}")
        plt.xlabel("Cumulative budget")
        plt.ylabel("Accuracy (%)")
        plt.grid(True); plt.legend(); plt.tight_layout()
        fig_path = out_dir / "palm_fit.png"
        plt.savefig(fig_path, dpi=180)
        try: plt.show()
        except Exception: pass
        print(f"Saved plot → {fig_path}")

if __name__ == "__main__":
    main()
