"""
grab_camera_pair.py  (v2)

Finds a control/light camera-frame pair for the two-panel figure in Section 5.

v1 matched on position within a lap, which fails here: baseline_run was appended
across sessions so its lap counter resets and no clean lap can be extracted.

v2 matches on (cte, steer_cmd) instead. The seeded conditions share identical road
geometry, so the steering command the autopilot issued is a proxy for where on the
circuit the vehicle is - the same corner demands the same steer. Matching on both
gives the same place on the track with the vehicle in the same pose, without
relying on lap numbering at all.

Run from av/:
    python grab_camera_pair.py
    python grab_camera_pair.py --cte_tol 0.05 --steer_tol 0.05
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data_cte")
    ap.add_argument("--control", default="baseline_run")
    ap.add_argument("--light", default="cond1_light")
    ap.add_argument("--cte_tol", type=float, default=0.08)
    ap.add_argument("--steer_tol", type=float, default=0.08)
    ap.add_argument("--cte_lo", type=float, default=-0.9)
    ap.add_argument("--cte_hi", type=float, default=-0.1)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out_dir", default="paper_frames/pairs")
    args = ap.parse_args()

    def load(folder):
        d = pd.read_csv(os.path.join(args.data_root, folder, "cte_log.csv"))
        return d[d["cte"].between(args.cte_lo, args.cte_hi)].reset_index(drop=True)

    c = load(args.control)
    l = load(args.light)
    print(f"control: {len(c)} candidate frames   light: {len(l)} candidate frames")
    print(f"matching on |dcte| <= {args.cte_tol} and |dsteer| <= {args.steer_tol}\n")

    lc, ls = l["cte"].to_numpy(), l["steer_cmd"].to_numpy()
    pairs = []
    for i, cr in c.iterrows():
        dc = np.abs(lc - cr["cte"])
        ds = np.abs(ls - cr["steer_cmd"])
        ok = np.where((dc <= args.cte_tol) & (ds <= args.steer_tol))[0]
        if ok.size == 0:
            continue
        j = ok[np.argmin(dc[ok] + ds[ok])]
        pairs.append((float(dc[j] + ds[j]), cr, l.iloc[j]))

    if not pairs:
        print("No pairs within tolerance. Loosen --cte_tol / --steer_tol.")
        return

    # spread the saved candidates over different steer values rather than
    # returning eight near-identical straight-line frames
    pairs.sort(key=lambda t: t[0])
    chosen, used = [], []
    for d, cr, lr in pairs:
        if all(abs(cr["steer_cmd"] - u) > 0.12 for u in used):
            chosen.append((d, cr, lr))
            used.append(cr["steer_cmd"])
        if len(chosen) >= args.n:
            break
    if len(chosen) < args.n:
        chosen = pairs[:args.n]

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"{'#':>2} {'cte ctrl':>9} {'cte light':>10} {'steer':>7} {'score':>7}")
    print("-" * 40)
    for i, (d, cr, lr) in enumerate(chosen):
        for tag, row, folder in (("control", cr, args.control),
                                 ("light", lr, args.light)):
            shutil.copy(os.path.join(args.data_root, folder, row["image"]),
                        os.path.join(args.out_dir, f"pair{i}_{tag}.jpg"))
        print(f"{i:>2} {cr['cte']:>9.2f} {lr['cte']:>10.2f} "
              f"{cr['steer_cmd']:>7.2f} {d:>7.3f}")

    print(f"\nSaved to {args.out_dir}/ - pick the pair where the only visible")
    print("difference is the illumination, then rename to cam_control.jpg /")
    print("cam_light.jpg. A gently curving frame usually reads better than a")
    print("dead-straight one, since the lane markings are more informative.")


if __name__ == "__main__":
    main()
