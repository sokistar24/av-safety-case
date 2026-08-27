"""
mc_uncertainty.py
Validity check for the MC-Dropout run-time guard (Level 1), testing the paper's
premise for run-time checks (Sec. 3.2): "for inputs where the checks pass, the
network is more likely to be accurate."

Runs MC-Dropout (n stochastic forward passes with dropout ACTIVE) over the
held-out validation set and measures whether predictive std actually predicts
error. Reports:
  - correlation between std and |error| (Pearson + Spearman)
  - a threshold table: candidate std thresholds (percentiles) with pass rate
    beta and the discretized accuracy of PASSING vs FAILING inputs
Saves per-image (cte_true, mc_mean, mc_std) to results_cte/mc_val.csv for the
guarded confusion matrix (m2) step that follows.

Run:  python mc_uncertainty.py
"""

import os
import csv
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from cte_dataset import build_cte_datasets, bin_cte, ROAD_EDGE
from cte_model import TaxiNetCTE


def rank(a):
    r = np.empty(len(a))
    r[np.argsort(a)] = np.arange(len(a))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_cte/generated_track")
    ap.add_argument("--ckpt", default="results_cte/cte_model.pth")
    ap.add_argument("--out_dir", default="results_cte")
    ap.add_argument("--n_mc", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    _, val_ds, info = build_cte_datasets(args.data_dir, seed=42)
    print(f"Validation set: {info['val']} images  |  MC passes: {args.n_mc}")
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TaxiNetCTE().to(device)
    with torch.no_grad():
        model(torch.zeros(2, 3, 80, 160, device=device))
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    trues, means, stds = [], [], []
    model.train()  # dropout ACTIVE (net has no batchnorm, so this is safe)
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            preds = torch.stack([model(imgs) for _ in range(args.n_mc)], dim=0)
            means.append(preds.mean(0).cpu().numpy().ravel())
            stds.append(preds.std(0).cpu().numpy().ravel())
            trues.append(targets.numpy().ravel())
    model.eval()

    trues = np.concatenate(trues)
    means = np.concatenate(means)
    stds = np.concatenate(stds)
    err = np.abs(means - trues)

    pear = float(np.corrcoef(stds, err)[0, 1])
    spear = float(np.corrcoef(rank(stds), rank(err))[0, 1])
    print(f"\nDoes uncertainty predict error?")
    print(f"  std vs |error| correlation:  Pearson {pear:.3f}   Spearman {spear:.3f}")

    t_states = np.array([bin_cte(x) for x in trues])
    p_states = np.array([bin_cte(x) for x in np.clip(means, -ROAD_EDGE, ROAD_EDGE)])
    onroad = t_states != -1
    correct = (t_states == p_states)

    print(f"\nThreshold table (guard = 'pass if std <= threshold'):")
    print(f"{'pct':>4} {'threshold':>10} {'beta(pass)':>11} {'acc(pass)':>10} {'acc(fail)':>10}")
    rows = []
    for pct in (60, 70, 80, 85, 90, 95):
        th = float(np.percentile(stds, pct))
        passing = stds <= th
        beta = float(passing.mean())
        pmask = passing & onroad
        fmask = (~passing) & onroad
        accp = float(correct[pmask].mean()) if pmask.any() else float("nan")
        accf = float(correct[fmask].mean()) if fmask.any() else float("nan")
        print(f"{pct:>4} {th:>10.4f} {beta:>11.3f} {accp*100:>9.1f}% {accf*100:>9.1f}%")
        rows.append({"percentile": pct, "threshold": th, "beta": beta,
                     "acc_pass": accp, "acc_fail": accf})

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "mc_val.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cte_true", "mc_mean", "mc_std"])
        for t, m, s in zip(trues, means, stds):
            w.writerow([f"{t:.5f}", f"{m:.5f}", f"{s:.5f}"])
    print(f"\nPer-image MC stats -> {out_csv}")
    print("\nRead the table: the guard premise holds if acc(pass) > acc(fail).")
    print("The paper's guard passed 82.1% of inputs — the 80/85th-percentile rows")
    print("are the comparable choices for beta. Pick one and the next step builds")
    print("the guarded alpha + the m2 PRISM model from it.")


if __name__ == "__main__":
    main()
