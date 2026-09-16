"""
de_shift_eval.py
Evaluate the FIXED Deep-Ensemble guard on a shifted condition. The ensemble analogue
of shift_eval.py, writing the identical output schema so everything downstream
(guarded_model.py, make_prism_model.py, bootstrap_ci.py, condition_summary.py) runs
against results_de/<condition>/ without modification.

The threshold is read from results_de/de_calibration.json and is NOT recalibrated on
the condition's own uncertainty distribution. Same rule as the MC arm: a deployed
system carries its commissioning calibration into conditions it has not met. Pass
--threshold explicitly if the calibration json holds a value you did not freeze.

ONE DELIBERATE DIFFERENCE FROM shift_eval.py. In the MC arm the unguarded abstraction
is built from deterministic single-model predictions while the guarded one is built
from the MC mean, so m1 and m2 rest on slightly different predictors (81.17% vs
81.33% in-distribution). A Deep Ensemble has no separate deterministic mode: yhat_DE
is THE prediction. Both abstractions therefore use the ensemble mean here, which is
internally consistent but means DE-m1 is not the same object as MC-m1. Compare
within-arm (m1 -> m2, beta, pass-rate-on-large-error), not across arms.

DIAGNOSTICS PRINTED BEYOND shift_eval.py, all aimed at the `light` question:
  - signed bias and prediction spread/range, against the in-distribution values.
    Prediction collapse shows up here as a CONTRACTED spread (MC arm under light:
    sd 0.458 against 1.267 in-distribution).
  - P(pass | |err| > err_thresh), the confidently-wrong rate. Under the MC guard this
    separated `light` (0.515) from control (0.075) far more sharply than the
    correlations did.
  - per-member prediction spread. This is the decisive one: if u_DE stays low under
    `light`, this table says whether every member collapsed to the same narrow band
    (disagreement-based uncertainty fails generally) or the members still disagree
    and something else is suppressing the score.

Run (from av/, one condition at a time):
    mkdir results_de\\cond1_light
    python de_shift_eval.py --data_dir data_cte/cond1_light --out_dir results_de/cond1_light

Outputs (same names as the MC arm):
    <out_dir>/confusion_shift.json   alpha_counts / alpha_probs from the ensemble mean
    <out_dir>/mc_shift.csv           cte_true, mc_mean, mc_std, m0..m{K-1}
    <out_dir>/de_shift_summary.json  the diagnostics above
"""

import os
import csv
import json
import glob
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from cte_dataset import load_cte_dataframe, CTEDataset, bin_cte, ROAD_EDGE, STATE_NAMES
from cte_model import TaxiNetCTE

ON_ROAD = [3, 1, 0, 2, 4]


def load_members(ckpt_dir, K, device):
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "member_*.pth")))
    if not paths:
        raise SystemExit(f"No member_*.pth in {ckpt_dir}. Run train_ensemble.py first.")
    if K is not None:
        if K > len(paths):
            raise SystemExit(f"--K {K} but only {len(paths)} members trained.")
        paths = paths[:K]
    models = []
    for p in paths:
        m = TaxiNetCTE().to(device)
        with torch.no_grad():
            m(torch.zeros(2, 3, 80, 160, device=device))
        m.load_state_dict(torch.load(p, map_location=device))
        m.eval()                                  # deterministic: dropout OFF
        models.append(m)
    return models, [os.path.basename(p) for p in paths]


@torch.no_grad()
def ensemble_predict(models, loader, device):
    trues, per_member = [], [[] for _ in models]
    for imgs, targets in loader:
        imgs = imgs.to(device)
        trues.append(targets.numpy().ravel())
        for k, m in enumerate(models):
            per_member[k].append(m(imgs).cpu().numpy().ravel())
    return np.concatenate(trues), np.stack([np.concatenate(p) for p in per_member])


def read_csv_cols(path, cols):
    out = {c: [] for c in cols}
    with open(path) as f:
        for row in csv.DictReader(f):
            for c in cols:
                if c in row:
                    out[c].append(float(row[c]))
    return {c: np.array(v) for c, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="shifted condition data folder")
    ap.add_argument("--ckpt_dir", default="results_de")
    ap.add_argument("--calib_json", default="results_de/de_calibration.json")
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the frozen tau_DE (normally leave unset)")
    ap.add_argument("--K", type=int, default=None,
                    help="use the first K members; default: the K used at calibration")
    ap.add_argument("--id_csv", default="results_de/mc_val.csv",
                    help="in-distribution per-image file, for the side-by-side")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--err_thresh", type=float, default=0.8)
    ap.add_argument("--M", type=int, default=None,
                    help="consecutive failures before handover, for the per-cycle "
                         "abort estimate only (default: from the calibration json)")
    args = ap.parse_args()

    # ---- frozen calibration ------------------------------------------------
    calib = {}
    if os.path.exists(args.calib_json):
        with open(args.calib_json) as f:
            calib = json.load(f)
    thr = args.threshold if args.threshold is not None else calib.get("threshold")
    if thr is None:
        raise SystemExit("No threshold: pass --threshold or point --calib_json at a "
                         "de_calibration.json produced with --pct.")
    thr = float(thr)
    K = args.K if args.K is not None else calib.get("K")
    beta_id = calib.get("beta")
    M = args.M if args.M is not None else 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_cte_dataframe(args.data_dir)
    ds = CTEDataset(df, augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    models, names = load_members(args.ckpt_dir, K, device)
    K = len(models)
    print(f"Condition: {len(ds)} frames from {args.data_dir}  |  K = {K}")
    print(f"Frozen guard: u_DE <= {thr:.4f}"
          + (f"  (nominal beta {beta_id:.3f}, "
             f"{calib.get('selection_mode', 'mode unrecorded')})" if beta_id else ""))
    if calib.get("selection_mode", "").startswith("selection rule") and \
       calib.get("premise_holds_at_all_percentiles"):
        print("  WARNING: this threshold came from the degenerate selection rule. "
              "Re-run de_uncertainty.py --pct before trusting cross-arm comparisons.")

    trues, preds = ensemble_predict(models, loader, device)
    means = preds.mean(axis=0)
    stds = preds.std(axis=0, ddof=1)
    err = np.abs(means - trues)

    # ---- perception: abstraction from the ensemble mean --------------------
    t_states = np.array([bin_cte(x) for x in trues])
    p_states = np.array([bin_cte(x) for x in np.clip(means, -ROAD_EDGE, ROAD_EDGE)])
    i5 = {s: j for j, s in enumerate(ON_ROAD)}
    counts = np.zeros((5, 5), dtype=int)
    onroad = t_states != -1
    for t, p in zip(t_states[onroad], p_states[onroad]):
        counts[i5[t], i5[p]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sums, out=np.zeros_like(counts, float),
                      where=row_sums > 0)
    acc = float(np.trace(counts) / counts.sum()) if counts.sum() else float("nan")
    mae = float(err.mean())
    bias = float((means[onroad] - trues[onroad]).mean())

    print(f"\nPERCEPTION (ensemble mean)")
    print(f"  on-road discretised accuracy : {acc*100:.2f}%")
    print(f"  MAE                          : {mae:.3f}")
    print(f"  signed bias                  : {bias:+.3f}   "
          f"(positive = predicts too far right)")
    print("  alpha (row-normalised):")
    print("        " + "".join(f"{STATE_NAMES[s]:>9s}" for s in ON_ROAD))
    for s in ON_ROAD:
        print(f"{STATE_NAMES[s]:>7s} " + "".join(f"{v:9.3f}" for v in probs[i5[s]])
              + f"   (n={row_sums[i5[s]][0]})")

    # ---- guard response ----------------------------------------------------
    passing = stds <= thr
    beta_shift = float(passing.mean())
    big = onroad & (err > args.err_thresh)
    pass_big = float(passing[big].mean()) if big.any() else float("nan")

    id_stats = None
    if os.path.exists(args.id_csv):
        idc = read_csv_cols(args.id_csv, ["cte_true", "mc_mean", "mc_std"])
        id_stats = {"u_median": float(np.median(idc["mc_std"])),
                    "u_mean": float(idc["mc_std"].mean()),
                    "pred_sd": float(idc["mc_mean"].std(ddof=1)),
                    "pred_min": float(idc["mc_mean"].min()),
                    "pred_max": float(idc["mc_mean"].max())}

    print(f"\nGUARD (frozen tau_DE = {thr:.4f})")
    if id_stats:
        print(f"  u_DE in-distribution : median {id_stats['u_median']:.3f}  "
              f"mean {id_stats['u_mean']:.3f}")
    print(f"  u_DE this condition  : median {np.median(stds):.3f}  "
          f"mean {stds.mean():.3f}")
    if beta_id:
        print(f"  beta in-distribution : {beta_id:.3f}")
    print(f"  beta this condition  : {beta_shift:.3f}   "
          f"-> guard fires on {(1-beta_shift)*100:.1f}% of frames")
    print(f"  per-cycle abort (1-beta)^{M} = {((1-beta_shift)**M):.3g}")
    print(f"  P(pass | |err| > {args.err_thresh}) = {pass_big:.3f}   "
          f"[{int(big.sum())} of {int(onroad.sum())} on-road frames exceed the cut]")
    if not np.isnan(pass_big) and pass_big > 0.3 and acc < 0.7:
        print("  -> CONFIDENTLY WRONG: the guard admits most of the badly-wrong "
              "frames while accuracy is degraded. This is the light regime.")

    # ---- prediction collapse check ----------------------------------------
    pred_sd = float(means.std(ddof=1))
    print(f"\nPREDICTION SPREAD (collapse check)")
    line = f"  ensemble mean : sd {pred_sd:.3f}  range [{means.min():+.2f}, {means.max():+.2f}]"
    if id_stats:
        line += (f"   vs in-distribution sd {id_stats['pred_sd']:.3f} "
                 f"range [{id_stats['pred_min']:+.2f}, {id_stats['pred_max']:+.2f}]")
    print(line)
    if id_stats and pred_sd < id_stats["pred_sd"]:
        print(f"  -> CONTRACTED to {pred_sd/id_stats['pred_sd']*100:.0f}% of the "
              f"in-distribution spread.")
    print("  per-member prediction sd (do the members collapse together?):")
    member_sd = [float(preds[k].std(ddof=1)) for k in range(K)]
    for k, nm in enumerate(names):
        print(f"    {nm:>16s}  sd {member_sd[k]:.3f}  "
              f"range [{preds[k].min():+.2f}, {preds[k].max():+.2f}]")
    print(f"    {'across members':>16s}  mean sd {np.mean(member_sd):.3f}  "
          f"sd of sds {np.std(member_sd, ddof=1):.3f}")

    # ---- write -------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    shift_json = {
        "val_samples": int(len(ds)),
        "on_road_samples": int(counts.sum()),
        "on_road_disc_accuracy": acc,
        "state_order": ON_ROAD,
        "alpha_counts": counts.tolist(),
        "alpha_probs": probs.tolist(),
        "shifted_from": args.data_dir,
        "shifted_mae": mae,
        "beta_shift": beta_shift,
        "estimator": "deep_ensemble",
        "K": K, "members": names, "threshold": thr,
        "predictor": "ensemble mean (used for BOTH m1 and m2)",
    }
    with open(os.path.join(args.out_dir, "confusion_shift.json"), "w") as f:
        json.dump(shift_json, f, indent=2)

    cp = os.path.join(args.out_dir, "mc_shift.csv")
    with open(cp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cte_true", "mc_mean", "mc_std"] + [f"m{k}" for k in range(K)])
        for i in range(len(trues)):
            w.writerow([f"{trues[i]:.5f}", f"{means[i]:.5f}", f"{stds[i]:.5f}"] +
                       [f"{preds[k, i]:.5f}" for k in range(K)])

    summary = {
        "condition": args.data_dir, "K": K, "threshold": thr,
        "accuracy": acc, "mae": mae, "signed_bias": bias,
        "u_median": float(np.median(stds)), "u_mean": float(stds.mean()),
        "beta": beta_shift, "beta_id": beta_id,
        "err_thresh": args.err_thresh,
        "frac_large_err": float(big.sum() / onroad.sum()) if onroad.any() else None,
        "pass_given_large_err": pass_big,
        "pred_sd": pred_sd,
        "pred_range": [float(means.min()), float(means.max())],
        "member_pred_sd": member_sd,
        "id_reference": id_stats,
    }
    with open(os.path.join(args.out_dir, "de_shift_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved -> {args.out_dir}/confusion_shift.json, mc_shift.csv, "
          f"de_shift_summary.json")
    print("\nThen, for this condition:")
    print(f"  python make_prism_model.py --json {args.out_dir}/confusion_shift.json "
          f"--out_dir {args.out_dir}")
    print(f"  python guarded_model.py --mc_csv {cp} --threshold {thr:.4f} "
          f"--out_dir {args.out_dir}")


if __name__ == "__main__":
    main()
