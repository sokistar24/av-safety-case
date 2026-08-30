"""
condition_summary.py

Assembles the conditions x measures table for Section 5 from the per-condition
outputs written by shift_eval.py and shift_distance.py.

Expects, for each condition <c>:
    results_cte/<c>/confusion_shift.json
    results_cte/<c>/mc_shift.csv
    results_cte/<c>/shift_distance.json

Adds one measure those scripts do not compute: the SIGNED bias of the prediction
error. MAE says how wrong the model is; signed bias says whether it is wrong in a
consistent direction. Appearance shift should scatter roughly symmetrically;
geometry shift may acquire a systematic sign.

Run from av/:
    python condition_summary.py
    python condition_summary.py --conditions cond1_trees cond1_light
"""

import argparse
import csv
import json
import os

import numpy as np

from cte_dataset import bin_cte

DEFAULT_CONDITIONS = ["cond1_trees", "cond1_light", "cond1_cones", "generated_road"]


def load_mc(path):
    trues, means, stds = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            trues.append(float(row["cte_true"]))
            means.append(float(row["mc_mean"]))
            stds.append(float(row["mc_std"]))
    return np.array(trues), np.array(means), np.array(stds)


def signed_bias(trues, means):
    """Mean signed error, on-road samples only. Positive = predicts too far right."""
    onroad = np.array([bin_cte(x) != -1 for x in trues])
    err = means[onroad] - trues[onroad]
    return float(err.mean()), float(err.std()), int(onroad.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    ap.add_argument("--results_root", default="results_cte")
    ap.add_argument("--id_mc_csv", default="results_cte/mc_val.csv")
    ap.add_argument("--guard_json", default="results_cte/guarded_alpha.json")
    ap.add_argument("--out_csv", default="results_cte/condition_summary.csv")
    args = ap.parse_args()

    # ---- in-distribution reference row -------------------------------
    rows = []
    if os.path.exists(args.id_mc_csv):
        t, m, s = load_mc(args.id_mc_csv)
        bias, bias_sd, n_on = signed_bias(t, m)
        guard = json.load(open(args.guard_json))
        rows.append({
            "condition": "in-distribution",
            "n": len(t), "n_onroad": n_on,
            "accuracy": None, "mae": None,
            "signed_bias": bias, "bias_sd": bias_sd,
            "mc_std_median": float(np.median(s)),
            "beta": guard["beta"],
            "pixel_js": 0.0, "maha_median": None, "auroc": 0.5, "sym_kl": 0.0,
        })

    # ---- one row per condition ---------------------------------------
    for c in args.conditions:
        d = os.path.join(args.results_root, c)
        conf_p = os.path.join(d, "confusion_shift.json")
        mc_p = os.path.join(d, "mc_shift.csv")
        dist_p = os.path.join(d, "shift_distance.json")

        missing = [p for p in (conf_p, mc_p, dist_p) if not os.path.exists(p)]
        if missing:
            print(f"SKIP {c}: missing {', '.join(os.path.basename(p) for p in missing)}")
            continue

        conf = json.load(open(conf_p))
        dist = json.load(open(dist_p))
        t, m, s = load_mc(mc_p)
        bias, bias_sd, n_on = signed_bias(t, m)

        rows.append({
            "condition": c,
            "n": conf["val_samples"], "n_onroad": conf["on_road_samples"],
            "accuracy": conf["on_road_disc_accuracy"],
            "mae": conf["shifted_mae"],
            "signed_bias": bias, "bias_sd": bias_sd,
            "mc_std_median": float(np.median(s)),
            "beta": conf["beta_shift"],
            "pixel_js": dist["pixel_js_bits"],
            "maha_median": dist["mahalanobis"]["shift_median"],
            "auroc": dist["mahalanobis"]["auroc"],
            "sym_kl": dist["feature_sym_kl_nats"],
        })

    if not rows:
        print("Nothing to summarise.")
        return

    # ---- print ---------------------------------------------------------
    def fmt(v, spec=".3f"):
        return "  --  " if v is None else format(v, spec)

    print(f"\n{'condition':<18} {'n_on':>6} {'acc':>7} {'MAE':>7} {'bias':>8} "
          f"{'std_med':>8} {'beta':>6} {'JS':>7} {'AUROC':>7} {'symKL':>8}")
    print("-" * 96)
    for r in rows:
        acc = "  --  " if r["accuracy"] is None else f"{r['accuracy']*100:6.2f}%"
        print(f"{r['condition']:<18} {r['n_onroad']:>6} {acc:>7} "
              f"{fmt(r['mae']):>7} {fmt(r['signed_bias'], '+.3f'):>8} "
              f"{fmt(r['mc_std_median']):>8} {fmt(r['beta'], '.3f'):>6} "
              f"{fmt(r['pixel_js']):>7} {fmt(r['auroc']):>7} "
              f"{fmt(r['sym_kl'], '.2f'):>8}")

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved -> {args.out_csv}")

    print("\nReading the table:")
    print("  acc / MAE     how far perception degrades")
    print("  bias          mean signed error, on-road only; sign indicates a")
    print("                systematic direction rather than scatter")
    print("  std_med       does the model know it is degrading?")
    print("  beta          fraction passing the FIXED in-distribution threshold;")
    print("                1-beta is how often the guard fires")
    print("  JS            pixel-space distance")
    print("  AUROC/symKL   feature-space distance")


if __name__ == "__main__":
    main()
