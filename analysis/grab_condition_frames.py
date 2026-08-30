"""
grab_condition_frames.py

Pulls one representative camera frame per condition for the Section 5 figure.

Frames are matched by POSITION WITHIN A LAP, not by CTE value. This matters:
matching on CTE says nothing about where on the circuit a frame was taken, so a
straight-section frame and a bend frame can share a CTE and look misleadingly
different. Matching on lap fraction puts the vehicle at approximately the same
place on the track.

The four generated-track conditions share a terrain seed, so their frames are
genuinely comparable - same corner, same geometry, different surroundings. The two
other circuits cannot be matched to them and are shown as representative frames.

Output: paper_frames/<name>.jpg

Run from av/:
    python grab_condition_frames.py
    python grab_condition_frames.py --lap_frac 0.35
"""

import argparse
import os
import shutil

import pandas as pd

# (output name, data folder, matched to the seeded track?)
CONDITIONS = [
    ("control", "baseline_run",    True),
    ("light",   "cond1_light",     True),
    ("trees",   "cond1_trees",     True),
    ("cones",   "cond1_cones",     True),
    ("road",    "generated_road",  False),
    ("circuit", "mini_monaco",     False),
]


def pick_frame(df, lap_frac):
    """Frame at lap_frac through the first complete lap, if lap data allows."""
    if "lap" in df.columns and df["lap"].max() >= 1:
        lap1 = df[df["lap"] == 1]
        if len(lap1) > 10:
            return lap1.iloc[int(len(lap1) * lap_frac)]
    return df.iloc[int(len(df) * lap_frac)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lap_frac", type=float, default=0.35,
                    help="fraction through the lap to sample (0-1)")
    ap.add_argument("--data_root", default="data_cte")
    ap.add_argument("--out_dir", default="paper_frames")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Sampling at {args.lap_frac:.0%} through lap 1\n")
    print(f"{'name':10s} {'folder':16s} {'frame':>8s} {'cte':>8s}  matched")
    print("-" * 56)

    for name, folder, matched in CONDITIONS:
        csv_path = os.path.join(args.data_root, folder, "cte_log.csv")
        if not os.path.exists(csv_path):
            print(f"{name:10s} {folder:16s}  MISSING")
            continue
        df = pd.read_csv(csv_path)
        row = pick_frame(df, args.lap_frac)
        src = os.path.join(args.data_root, folder, row["image"])
        dst = os.path.join(args.out_dir, f"{name}.jpg")
        shutil.copy(src, dst)
        print(f"{name:10s} {folder:16s} {os.path.basename(src):>8s} "
              f"{row['cte']:+8.2f}  {'yes' if matched else 'no'}")

    print(f"\nSaved to {args.out_dir}/ - upload these for the figure.")
    print("If a frame is unrepresentative (mid-corner, obscured), re-run with a")
    print("different --lap_frac; the four seeded conditions move together.")


if __name__ == "__main__":
    main()
