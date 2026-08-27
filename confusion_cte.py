"""
confusion_cte.py
Build the perception confusion matrix and the probabilistic abstraction (alpha)
— the paper's Table 1/2 analog and Equation 3 — from the trained CTE model.

What it does:
  1. Runs the trained model over the HELD-OUT validation split (same seed as
     training, so these images were never trained on), single deterministic
     predictions (dropout off) — the paper's baseline m1 setting.
  2. Bins true and predicted cte with the shared bin_cte().
  3. Reports:
       - full 6x6 matrix (incl. error state) for transparency
       - the 5x5 on-road ALPHA: rows = true state, columns = estimated state,
         row-normalized to transition probabilities (paper Eq. 3). Predictions
         are clipped to the road edge first, mirroring the paper's note that the
         classifier never outputs -1; the error state enters via DYNAMICS in the
         DTMC, not via perception.
  4. Emits a ready-to-paste PRISM transition block in the paper's DTMC syntax
     (results_cte/prism_perception.txt) and saves all numbers to JSON.

Run:  python confusion_cte.py
"""

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from cte_dataset import build_cte_datasets, bin_cte, STATE_NAMES, ROAD_EDGE
from cte_model import TaxiNetCTE

ORDER6 = [3, 1, 0, 2, 4, -1]          # display order: far-L ... far-R, error
ON_ROAD = [3, 1, 0, 2, 4]             # alpha rows/cols (paper states 0..4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_cte/generated_track")
    ap.add_argument("--ckpt", default="results_cte/cte_model.pth")
    ap.add_argument("--out_dir", default="results_cte")
    ap.add_argument("--batch_size", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # exact same seeded split as training -> val set is truly held out
    _, val_ds, info = build_cte_datasets(args.data_dir, seed=42)
    print(f"Validation set: {info['val']} images (of {info['total']})")
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TaxiNetCTE().to(device)
    with torch.no_grad():
        model(torch.zeros(2, 3, 80, 160, device=device))
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    print(f"Loaded {args.ckpt}")

    trues, preds = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            out = model(imgs.to(device))
            trues.append(targets.numpy().ravel())
            preds.append(out.cpu().numpy().ravel())
    trues = np.concatenate(trues)
    preds = np.concatenate(preds)

    t_states = np.array([bin_cte(x) for x in trues])
    p_states = np.array([bin_cte(x) for x in preds])

    # ---------- full 6x6 (transparency) ----------
    i6 = {s: i for i, s in enumerate(ORDER6)}
    full = np.zeros((6, 6), dtype=int)
    for t, p in zip(t_states, p_states):
        full[i6[t], i6[p]] += 1
    print("\nFULL 6x6 confusion matrix (rows=true, cols=predicted):")
    hdr = "        " + "".join(f"{STATE_NAMES[s]:>9s}" for s in ORDER6)
    print(hdr)
    for s in ORDER6:
        row = full[i6[s]]
        print(f"{STATE_NAMES[s]:>7s} " + "".join(f"{v:9d}" for v in row))

    # ---------- alpha: on-road rows, predictions clipped to the road ----------
    p_clip_states = np.array([bin_cte(x) for x in np.clip(preds, -ROAD_EDGE, ROAD_EDGE)])
    i5 = {s: i for i, s in enumerate(ON_ROAD)}
    counts = np.zeros((5, 5), dtype=int)
    mask = t_states != -1
    for t, p in zip(t_states[mask], p_clip_states[mask]):
        counts[i5[t], i5[p]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sums, out=np.zeros_like(counts, float), where=row_sums > 0)

    on_road_acc = np.trace(counts) / counts.sum()
    print(f"\nALPHA (on-road, {counts.sum()} samples). "
          f"On-road discretized accuracy: {on_road_acc*100:.2f}%")
    print("Paper-style table (true state -> estimated state probabilities):")
    print("        " + "".join(f"{STATE_NAMES[s]:>9s}" for s in ON_ROAD))
    for s in ON_ROAD:
        r = probs[i5[s]]
        print(f"{STATE_NAMES[s]:>7s} " + "".join(f"{v:9.3f}" for v in r)
              + f"   (n={row_sums[i5[s]][0]})")

    # per-state accuracy like the paper's Table 2
    print("\nPer-state accuracy:")
    for s in ON_ROAD:
        r = counts[i5[s]]
        acc = r[i5[s]] / r.sum() if r.sum() else 0
        print(f"  true {s} ({STATE_NAMES[s]:6s}): {acc*100:5.1f}%  ({r.sum()} samples)")

    # ---------- PRISM transition block (paper Appendix B syntax) ----------
    # probabilities rounded then residual-fixed so each row sums to exactly 1
    lines = []
    for s in ON_ROAD:
        r = probs[i5[s]].copy()
        nz = [(ON_ROAD[j], r[j]) for j in range(5) if r[j] > 0]
        vals = [round(v, 4) for _, v in nz]
        resid = 1.0 - sum(vals)
        k = int(np.argmax(vals))
        vals[k] = round(vals[k] + resid, 4)          # largest entry absorbs residual
        terms = " +\n            ".join(
            f"{v}: (cte_est'={st}) & (pc'=2)" for (st, _), v in zip(nz, vals))
        lines.append(f"[] cte={s} & v=1 & pc=1 -> {terms};")
    prism_block = "\n".join(lines)
    prism_path = os.path.join(args.out_dir, "prism_perception.txt")
    with open(prism_path, "w") as f:
        f.write("// Perception abstraction alpha — auto-generated from the confusion matrix\n")
        f.write("// (rows sum to exactly 1; zero-probability branches omitted)\n")
        f.write(prism_block + "\n")
    print(f"\nPRISM perception block -> {prism_path}")

    # ---------- save everything ----------
    out = {
        "val_samples": int(info["val"]),
        "on_road_samples": int(counts.sum()),
        "on_road_disc_accuracy": float(on_road_acc),
        "state_order": ON_ROAD,
        "alpha_counts": counts.tolist(),
        "alpha_probs": probs.tolist(),
        "full_order": ORDER6,
        "full_counts": full.tolist(),
    }
    json_path = os.path.join(args.out_dir, "confusion_cte.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"All matrices saved -> {json_path}")


if __name__ == "__main__":
    main()
