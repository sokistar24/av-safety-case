"""
check_startup.py
Is the guard's early-run behaviour a startup transient rather than a response to the
operating condition?

Motivation: in the Deep-Ensemble arm the guard handed over at frame 9 - the earliest
possible under M = 10 - in light, trees, road and circuit. Tested against each run's
own observed pass rate, three of those are unremarkable (P(10 straight fails) = 0.41,
1.0, 0.92) but TREES is a 1-in-700 event, so trees fired early for some reason other
than its steady-state guard behaviour. The obvious candidate is the opening frames:
the vehicle is stationary or barely moving, and the view from the start line (the
checkered marking) is under-represented in training data collected while driving.

If u is systematically elevated over the first W frames across conditions, then any
"handover at frame 9" row is an artefact of the run's beginning and should be reported
with a warm-up excluded, not as evidence that the guard detected the condition.

The script compares the first W frames against the rest of the same run:
median u, guard pass rate, and mean speed. It also reports the first frame at which
speed exceeds a threshold, so a transient can be attributed to the vehicle being
stationary rather than to the scene.

Run (from av/):
    python analysis/check_startup.py --logs "results_de/*/live/*unguarded*.csv" --threshold 0.1923
    python analysis/check_startup.py --logs "results_cte/*/live/*.csv" --threshold 0.2849
    python analysis/check_startup.py --logs "results_cte/guarded_drive_*_log.csv" --threshold 0.2849

Prefer UNGUARDED logs: in a guarded run that hands over early, the later frames are
driven by the fail-safe, so "steady state" is a different system.
"""

import os
import csv
import glob
import argparse
import statistics as st


def read_log(path):
    rows = []
    with open(path, newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            u = r.get("u_de", r.get("mc_std"))
            if u is None:
                continue
            try:
                rows.append({
                    "frame": int(r["frame"]) if r.get("frame") else i,
                    "u": float(u),
                    "cte": float(r.get("cte_true", "nan")),
                    "speed": float(r.get("speed", "nan")),
                    "mode": r.get("mode", "MODEL"),
                })
            except ValueError:
                continue
    return rows


def summarise(rows, thr):
    if not rows:
        return None
    us = [r["u"] for r in rows]
    sp = [r["speed"] for r in rows if r["speed"] == r["speed"]]
    return {
        "n": len(rows),
        "median_u": st.median(us),
        "beta": sum(1 for u in us if u <= thr) / len(us) if thr else None,
        "mean_speed": (sum(sp) / len(sp)) if sp else None,
        "modes": ",".join(sorted({r["mode"] for r in rows})),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True,
                    help="glob for live drive logs (quote it)")
    ap.add_argument("--threshold", type=float, required=True,
                    help="the arm's frozen guard threshold")
    ap.add_argument("--W", type=int, default=20,
                    help="number of opening frames treated as the startup window")
    ap.add_argument("--speed_eps", type=float, default=0.05,
                    help="speed above which the vehicle counts as moving")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.logs))
    if not paths:
        raise SystemExit(f"No logs matched {args.logs}")

    print(f"threshold {args.threshold}   startup window = first {args.W} frames\n")
    print(f"{'log':>44} {'early n':>8} {'med u':>8} {'beta':>7} {'speed':>7}   "
          f"{'late n':>7} {'med u':>8} {'beta':>7} {'speed':>7}   {'u ratio':>8} {'moves@':>7}")

    out = []
    for p in paths:
        rows = read_log(p)
        if len(rows) < args.W + 10:
            print(f"{os.path.basename(p)[:44]:>44}  (only {len(rows)} frames, skipped)")
            continue
        early, late = rows[:args.W], rows[args.W:]
        e, l = summarise(early, args.threshold), summarise(late, args.threshold)
        ratio = e["median_u"] / l["median_u"] if l["median_u"] else float("nan")
        moving = next((r["frame"] for r in rows if r["speed"] > args.speed_eps), None)
        name = os.path.basename(p)
        print(f"{name[:44]:>44} {e['n']:>8} {e['median_u']:>8.3f} {e['beta']:>7.3f} "
              f"{(e['mean_speed'] or 0):>7.3f}   {l['n']:>7} {l['median_u']:>8.3f} "
              f"{l['beta']:>7.3f} {(l['mean_speed'] or 0):>7.3f}   {ratio:>8.2f}x "
              f"{str(moving):>7}")
        out.append({"log": name, "early_n": e["n"], "early_median_u": e["median_u"],
                    "early_beta": e["beta"], "early_speed": e["mean_speed"],
                    "late_n": l["n"], "late_median_u": l["median_u"],
                    "late_beta": l["beta"], "late_speed": l["mean_speed"],
                    "u_ratio": ratio, "first_moving_frame": moving,
                    "late_modes": l["modes"]})

    if out:
        ratios = [r["u_ratio"] for r in out if r["u_ratio"] == r["u_ratio"]]
        med = st.median(ratios)
        print(f"\nMedian early/late uncertainty ratio across {len(ratios)} runs: {med:.2f}x")
        if med > 1.3:
            print("  -> STARTUP TRANSIENT PRESENT. The opening frames are systematically")
            print("     more uncertain than the rest of the run. Any 'handover at frame")
            print("     ~M' result is then partly an artefact of the run's beginning.")
            print("     Report a warm-up: start the failure counter after the vehicle is")
            print("     moving, and state the warm-up length in the methodology.")
        elif med < 0.77:
            print("  -> the opening frames are LESS uncertain than the rest; early")
            print("     handovers are not a startup effect.")
        else:
            print("  -> no systematic startup effect. Early handovers reflect the")
            print("     condition, not the run's beginning.")
        stat = [r for r in out if r["first_moving_frame"] is not None
                and r["first_moving_frame"] >= 5]
        if stat:
            print(f"\n  Note: in {len(stat)} of {len(out)} runs the vehicle is still "
                  f"stationary past frame 5 (first moving frame up to "
                  f"{max(r['first_moving_frame'] for r in stat)}). Training data was "
                  f"collected while driving, so stationary frames are out of "
                  f"distribution for reasons unrelated to the evaluated condition.")

    if args.out_csv and out:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            for r in out:
                w.writerow(r)
        print(f"\nSaved -> {args.out_csv}")


if __name__ == "__main__":
    main()
