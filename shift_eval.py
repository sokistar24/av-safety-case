"""
shift_eval.py
Distribution-shift evaluation: run the generated_track-trained model on a
DIFFERENT track's data and measure (a) how perception degrades and (b) whether
the MC-Dropout guard detects the shift.

Everything is measured against the IN-DISTRIBUTION calibration: the guard
threshold comes from results_cte/guarded_alpha.json (fitted on generated_track),
exactly as a deployed system would carry its calibration into new conditions.

Reports:
  1. Shifted confusion matrix + on-road discretized accuracy (vs ~81% ID)
  2. MC-dropout std distribution: shifted vs in-distribution (mc_val.csv)
  3. beta_shift = fraction of shifted inputs PASSING the ID threshold
     -> (1 - beta_shift) is how often the guard/handover fires under shift
Writes (schema-compatible with the existing generators):
  results_cte/confusion_shift.json   -> make_prism_model.py --json ... (shifted m1)
  results_cte/mc_shift.csv           -> guarded_model.py --mc_csv ... (shifted m2)

Run:  python shift_eval.py --data_dir data_cte/mini_monaco
"""

import os
import csv
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from cte_dataset import load_cte_dataframe, CTEDataset, bin_cte, ROAD_EDGE, STATE_NAMES
from cte_model import TaxiNetCTE

ORDER6 = [3, 1, 0, 2, 4, -1]
ON_ROAD = [3, 1, 0, 2, 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="shifted track data folder")
    ap.add_argument("--ckpt", default="results_cte/cte_model.pth")
    ap.add_argument("--guard_json", default="results_cte/guarded_alpha.json")
    ap.add_argument("--id_mc_csv", default="results_cte/mc_val.csv")
    ap.add_argument("--out_dir", default="results_cte")
    ap.add_argument("--n_mc", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_cte_dataframe(args.data_dir)
    ds = CTEDataset(df, augment=False)
    print(f"Shifted dataset: {len(ds)} frames from {args.data_dir}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TaxiNetCTE().to(device)
    with torch.no_grad():
        model(torch.zeros(2, 3, 80, 160, device=device))
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    # ---- pass 1: deterministic predictions (for the shifted confusion matrix) ----
    model.eval()
    trues, det_preds = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            out = model(imgs.to(device))
            trues.append(targets.numpy().ravel())
            det_preds.append(out.cpu().numpy().ravel())
    trues = np.concatenate(trues)
    det_preds = np.concatenate(det_preds)

    # ---- pass 2: MC-dropout (for the guard) ----
    model.train()
    mc_means, mc_stds = [], []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            preds = torch.stack([model(imgs) for _ in range(args.n_mc)], dim=0)
            mc_means.append(preds.mean(0).cpu().numpy().ravel())
            mc_stds.append(preds.std(0).cpu().numpy().ravel())
    model.eval()
    mc_means = np.concatenate(mc_means)
    mc_stds = np.concatenate(mc_stds)

    # ---- shifted confusion matrix (deterministic, mirrors confusion_cte.py) ----
    t_states = np.array([bin_cte(x) for x in trues])
    p_clip = np.array([bin_cte(x) for x in np.clip(det_preds, -ROAD_EDGE, ROAD_EDGE)])
    i5 = {s: j for j, s in enumerate(ON_ROAD)}
    counts = np.zeros((5, 5), dtype=int)
    mask = t_states != -1
    for t, p in zip(t_states[mask], p_clip[mask]):
        counts[i5[t], i5[p]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sums, out=np.zeros_like(counts, float),
                      where=row_sums > 0)
    acc = float(np.trace(counts) / counts.sum()) if counts.sum() else float("nan")

    print(f"\nSHIFTED on-road discretized accuracy: {acc*100:.2f}%   "
          f"(in-distribution was ~81%)")
    print("Shifted alpha (row-normalized):")
    print("        " + "".join(f"{STATE_NAMES[s]:>9s}" for s in ON_ROAD))
    for s in ON_ROAD:
        print(f"{STATE_NAMES[s]:>7s} " + "".join(f"{v:9.3f}" for v in probs[i5[s]])
              + f"   (n={row_sums[i5[s]][0]})")

    mae = float(np.abs(det_preds - trues).mean())
    print(f"Shifted continuous MAE: {mae:.3f}   (in-distribution val MAE was ~0.19)")

    # ---- guard under shift ----
    with open(args.guard_json) as f:
        guard = json.load(f)
    thr = guard["threshold"]
    beta_id = guard["beta"]
    beta_shift = float((mc_stds <= thr).mean())

    id_stds = []
    if os.path.exists(args.id_mc_csv):
        with open(args.id_mc_csv) as f:
            for row in csv.DictReader(f):
                id_stds.append(float(row["mc_std"]))
    id_stds = np.array(id_stds)

    print(f"\nGUARD UNDER SHIFT (ID-calibrated threshold {thr:.4f}):")
    if len(id_stds):
        print(f"  MC std, in-distribution : median {np.median(id_stds):.3f}  "
              f"mean {id_stds.mean():.3f}")
    print(f"  MC std, shifted         : median {np.median(mc_stds):.3f}  "
          f"mean {mc_stds.mean():.3f}")
    print(f"  beta in-distribution    : {beta_id:.3f}")
    print(f"  beta under shift        : {beta_shift:.3f}")
    print(f"  -> guard fires on {(1-beta_shift)*100:.1f}% of shifted frames "
          f"(vs {(1-beta_id)*100:.1f}% in-distribution)")
    m = guard.get("M", 10)
    p_abort_cycle = (1 - beta_shift) ** m if beta_shift < 1 else 0.0
    print(f"  -> per-cycle abort prob (1-beta)^{m} = {p_abort_cycle:.3g}")

    # ---- save schema-compatible outputs for the PRISM regeneration ----
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
    }
    jp = os.path.join(args.out_dir, "confusion_shift.json")
    with open(jp, "w") as f:
        json.dump(shift_json, f, indent=2)
    cp = os.path.join(args.out_dir, "mc_shift.csv")
    with open(cp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cte_true", "mc_mean", "mc_std"])
        for t, m_, s in zip(trues, mc_means, mc_stds):
            w.writerow([f"{t:.5f}", f"{m_:.5f}", f"{s:.5f}"])
    print(f"\nSaved shifted alpha -> {jp}")
    print(f"Saved shifted MC stats -> {cp}")
    print("\nRegenerate the models under shift:")
    print("  python make_prism_model.py --json results_cte/confusion_shift.json --out_dir results_cte/shift")
    print(f"  python guarded_model.py --mc_csv results_cte/mc_shift.csv --threshold {thr:.4f} --out_dir results_cte/shift")


if __name__ == "__main__":
    main()
