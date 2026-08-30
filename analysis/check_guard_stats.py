"""
check_guard_stats.py

Recovers the numbers Section 4 needs from results already on disk. Does not
re-run MC-Dropout: it reads results_cte/mc_val.csv, which mc_uncertainty.py
already wrote.

Prints:
  1. Pearson / Spearman correlation between MC-dropout std and |error|
     (the guard premise: does uncertainty predict error?)
  2. The threshold table: percentile, threshold, beta, accuracy of passing
     vs failing inputs
  3. On-road accuracy computed from the MC mean, for comparison with the
     accuracy reported in confusion_cte.json

Run from av/:
    python check_guard_stats.py
"""

import json
import os
import numpy as np
import pandas as pd

from cte_dataset import bin_cte, ROAD_EDGE

MC_CSV = "results_cte/mc_val.csv"
CONF_JSON = "results_cte/confusion_cte.json"


def rank(a):
    r = np.empty(len(a))
    r[np.argsort(a)] = np.arange(len(a))
    return r


def main():
    df = pd.read_csv(MC_CSV)
    trues = df["cte_true"].to_numpy()
    means = df["mc_mean"].to_numpy()
    stds = df["mc_std"].to_numpy()
    err = np.abs(means - trues)

    print(f"Loaded {len(df)} validation samples from {MC_CSV}\n")

    # ---- 1. does uncertainty predict error? ---------------------------
    pearson = float(np.corrcoef(stds, err)[0, 1])
    spearman = float(np.corrcoef(rank(stds), rank(err))[0, 1])
    print("Guard premise: std vs |error|")
    print(f"  Pearson  r = {pearson:.3f}")
    print(f"  Spearman r = {spearman:.3f}\n")

    # ---- 2. threshold table -------------------------------------------
    t_states = np.array([bin_cte(x) for x in trues])
    p_states = np.array([bin_cte(x) for x in np.clip(means, -ROAD_EDGE, ROAD_EDGE)])
    onroad = t_states != -1
    correct = t_states == p_states

    print("Threshold table (pass if std <= threshold):")
    print(f"{'pct':>5} {'threshold':>10} {'beta':>8} {'acc(pass)':>11} {'acc(fail)':>11}")
    for pct in (60, 70, 80, 85, 90, 95):
        th = float(np.percentile(stds, pct))
        passing = stds <= th
        beta = float(passing.mean())
        pm, fm = passing & onroad, (~passing) & onroad
        accp = correct[pm].mean() * 100 if pm.any() else float("nan")
        accf = correct[fm].mean() * 100 if fm.any() else float("nan")
        print(f"{pct:>5} {th:>10.4f} {beta:>8.3f} {accp:>10.2f}% {accf:>10.2f}%")

    # ---- 3. reconcile the two accuracy figures ------------------------
    acc_mc_mean = correct[onroad].mean() * 100
    print(f"\nOn-road accuracy from the MC-dropout mean: {acc_mc_mean:.2f}%")
    if os.path.exists(CONF_JSON):
        with open(CONF_JSON) as f:
            conf = json.load(f)
        for key in ("accuracy", "acc", "onroad_accuracy", "alpha_accuracy"):
            if key in conf:
                v = conf[key]
                v = v * 100 if v <= 1 else v
                print(f"On-road accuracy in confusion_cte.json ('{key}'): {v:.2f}%")
                break
        else:
            print(f"confusion_cte.json keys: {list(conf.keys())}")
            print("  (no obvious accuracy key - check which one holds it)")
    print("\nIf these differ, the guarded pipeline discretises the MC mean while")
    print("the unguarded abstraction discretises the deterministic prediction.")
    print("State whichever convention the paper uses, consistently.")


if __name__ == "__main__":
    main()
