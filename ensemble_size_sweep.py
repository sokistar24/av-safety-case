"""
ensemble_size_sweep.py
Ensemble-size sensitivity: justify the final K rather than choosing it arbitrarily.

Reads the per-member prediction columns written by de_uncertainty.py and recomputes
the ensemble mean, the guard score and the calibration quantities over nested
prefixes K = 2..K_max. No inference is re-run, so this is cheap and can be repeated
as members are added.

Nested prefixes (members 0..K-1) rather than random subsets of each size: prefixes
are what you would actually deploy if you stopped training at K, and they keep the
curve monotone in training cost. With --resample_subsets the script also reports the
spread over random subsets of each size, which separates "K is large enough" from
"this particular prefix happened to be good".

What to look for, in order of importance:
  - u_DE median and the 80th-percentile threshold. These set the guard's operating
    point. If the threshold is still moving at K_max, the calibration is not stable
    and tau_DE would be an artefact of ensemble size.
  - ensemble MAE / discretised accuracy. Usually plateau first.
  - Pearson/Spearman of u_DE against |error|, and P(pass | large error). These are
    the guard premise, not just accuracy, and are the quantities the shifted
    conditions will be judged on.

Run (from av/):
    python ensemble_size_sweep.py
    python ensemble_size_sweep.py --csv results_de/mc_val.csv --fig

Outputs:
    results_de/ensemble_size_sweep.csv
    results_de/figures/ensemble_size_sweep.png   (with --fig)
"""

import os
import csv
import json
import argparse
import numpy as np
import pandas as pd

from cte_dataset import bin_cte, ROAD_EDGE


def rank(a):
    r = np.empty(len(a))
    r[np.argsort(a)] = np.arange(len(a))
    return r


def metrics(trues, preds_subset, pct, err_thresh):
    """Calibration quantities for one subset of members. preds_subset: (k, n)."""
    means = preds_subset.mean(axis=0)
    stds = preds_subset.std(axis=0, ddof=1)
    err = np.abs(means - trues)

    t_states = np.array([bin_cte(x) for x in trues])
    p_states = np.array([bin_cte(x) for x in np.clip(means, -ROAD_EDGE, ROAD_EDGE)])
    onroad = t_states != -1
    correct = t_states == p_states
    big = onroad & (err > err_thresh)

    th = float(np.percentile(stds, pct))
    passing = stds <= th
    pm, fm = passing & onroad, (~passing) & onroad

    return {
        "mae": float(err.mean()),
        "onroad_acc": float(correct[onroad].mean()),
        "u_median": float(np.median(stds)),
        "pearson": float(np.corrcoef(stds, err)[0, 1]),
        "spearman": float(np.corrcoef(rank(stds), rank(err))[0, 1]),
        "threshold": th,
        "beta": float(passing.mean()),
        "acc_pass": float(correct[pm].mean()) if pm.any() else float("nan"),
        "acc_fail": float(correct[fm].mean()) if fm.any() else float("nan"),
        "pass_given_large_err": float(passing[big].mean()) if big.any() else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results_de/mc_val.csv",
                    help="output of de_uncertainty.py, with m0..m{K-1} columns")
    ap.add_argument("--pct", type=float, default=80.0,
                    help="percentile at which the threshold is tracked across K")
    ap.add_argument("--err_thresh", type=float, default=0.8)
    ap.add_argument("--resample_subsets", type=int, default=0,
                    help="if >0, also report spread over this many random subsets "
                         "of each size (K_max choose k, sampled with replacement)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tol_mae", type=float, default=0.02,
                    help="relative MAE tolerance for the K recommendation")
    ap.add_argument("--tol_thr", type=float, default=0.05,
                    help="relative threshold tolerance for the K recommendation")
    ap.add_argument("--out_csv", default="results_de/ensemble_size_sweep.csv")
    ap.add_argument("--fig", action="store_true", help="also write a PNG")
    ap.add_argument("--fig_path", default="results_de/figures/ensemble_size_sweep.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    member_cols = [c for c in df.columns if c.startswith("m") and c[1:].isdigit()]
    member_cols.sort(key=lambda c: int(c[1:]))
    if len(member_cols) < 2:
        raise SystemExit(f"{args.csv} has {len(member_cols)} member columns; need >= 2. "
                         "Re-run de_uncertainty.py (it writes m0..m{K-1}).")

    trues = df["cte_true"].to_numpy()
    preds = df[member_cols].to_numpy().T          # (K_max, n)
    K_max = preds.shape[0]
    print(f"Loaded {len(df)} validation samples, K_max = {K_max} members from {args.csv}\n")

    rows = []
    for k in range(2, K_max + 1):
        r = metrics(trues, preds[:k], args.pct, args.err_thresh)
        r["K"] = k
        rows.append(r)

    ref = rows[-1]
    print(f"Nested prefixes, threshold tracked at the {args.pct:.0f}th percentile:")
    print(f"{'K':>3} {'MAE':>7} {'acc':>7} {'u med':>8} {'thresh':>8} {'beta':>7} "
          f"{'Pear':>6} {'Spear':>6} {'P(pass|big)':>12} {'d thresh':>9}")
    for r in rows:
        d = abs(r["threshold"] - ref["threshold"]) / ref["threshold"]
        print(f"{r['K']:>3} {r['mae']:>7.4f} {r['onroad_acc']*100:>6.2f}% "
              f"{r['u_median']:>8.4f} {r['threshold']:>8.4f} {r['beta']:>7.3f} "
              f"{r['pearson']:>6.3f} {r['spearman']:>6.3f} "
              f"{r['pass_given_large_err']:>12.3f} {d*100:>8.1f}%")

    # ---- optional: spread over random subsets of each size ----------------
    if args.resample_subsets > 0:
        rng = np.random.default_rng(args.seed)
        print(f"\nSpread over {args.resample_subsets} random subsets per size "
              f"(prefix value in brackets):")
        print(f"{'K':>3} {'MAE mean+-sd':>22} {'threshold mean+-sd':>26}")
        for k in range(2, K_max):
            ms, ts = [], []
            for _ in range(args.resample_subsets):
                idx = rng.choice(K_max, size=k, replace=False)
                m = metrics(trues, preds[idx], args.pct, args.err_thresh)
                ms.append(m["mae"]); ts.append(m["threshold"])
            pref = rows[k - 2]
            print(f"{k:>3}   {np.mean(ms):.4f} +- {np.std(ms, ddof=1):.4f} "
                  f"[{pref['mae']:.4f}]      "
                  f"{np.mean(ts):.4f} +- {np.std(ts, ddof=1):.4f} "
                  f"[{pref['threshold']:.4f}]")

    # ---- recommendation (a suggestion, not a decision) --------------------
    rec = None
    for r in rows:
        d_mae = abs(r["mae"] - ref["mae"]) / ref["mae"]
        d_thr = abs(r["threshold"] - ref["threshold"]) / ref["threshold"]
        if d_mae <= args.tol_mae and d_thr <= args.tol_thr:
            rec = r
            break
    print()
    if rec is None:
        print(f"No K below {K_max} is within {args.tol_mae*100:.0f}% MAE and "
              f"{args.tol_thr*100:.0f}% threshold of K = {K_max}.")
        print(f"Nothing has plateaued: use K = {K_max}, and say so in the paper.")
    else:
        print(f"Smallest K within {args.tol_mae*100:.0f}% MAE and "
              f"{args.tol_thr*100:.0f}% threshold of K = {K_max}:  K = {rec['K']}")
        print(f"  MAE {rec['mae']:.4f} vs {ref['mae']:.4f}   "
              f"threshold {rec['threshold']:.4f} vs {ref['threshold']:.4f}")
        print("  Confirm against the correlation and P(pass|big err) columns before "
              "fixing K: the guard premise, not MAE, is what the shifted conditions test.")
    print(f"\nAt the chosen K the guard costs K forward passes per frame against 30 "
          f"for MC-Dropout.")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    fields = ["K", "mae", "onroad_acc", "u_median", "threshold", "beta",
              "pearson", "spearman", "acc_pass", "acc_fail", "pass_given_large_err"]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    print(f"\nSweep -> {args.out_csv}")

    if args.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks = [r["K"] for r in rows]
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
        ax[0].plot(ks, [r["mae"] for r in rows], "o-")
        ax[0].set_xlabel("K"); ax[0].set_ylabel("validation MAE (m)")
        ax[1].plot(ks, [r["threshold"] for r in rows], "o-")
        ax[1].set_xlabel("K"); ax[1].set_ylabel(f"{args.pct:.0f}th-pct threshold")
        ax[2].plot(ks, [r["pearson"] for r in rows], "o-", label="Pearson")
        ax[2].plot(ks, [r["spearman"] for r in rows], "s--", label="Spearman")
        ax[2].set_xlabel("K"); ax[2].set_ylabel("corr(u_DE, |error|)"); ax[2].legend()
        for a in ax:
            a.grid(alpha=0.3)
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.fig_path) or ".", exist_ok=True)
        fig.savefig(args.fig_path, dpi=150)
        print(f"Figure -> {args.fig_path}")


if __name__ == "__main__":
    main()
