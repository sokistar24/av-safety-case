"""
compare_live_offline.py
Check whether a live episode is the same distribution as the offline dataset used to
build the abstraction for that condition.

This exists because of a specific failure: the verified m1 for the Deep-Ensemble arm
under `light` reported the unguarded system as essentially safe, while two live runs
left the road at frame 134. Two explanations compete, and they need separating:

  (a) CONDITION MISMATCH. The live episode is a harsher draw than the collected
      dataset. `randomLight` re-randomises per episode, so "light" is a family of
      conditions, not one condition. If live u is systematically higher than offline
      u AT COMPARABLE LATERAL POSITIONS, the abstraction was built on easier data and
      the optimism is a dataset problem.

  (b) STATE-VISITATION MISMATCH. Offline data is collected by the autopilot, which
      holds the vehicle near nominal; the live system under test drives itself into
      states the autopilot never visits. Then even a correct abstraction is being
      applied outside the region it was estimated on, and the empirical zeros in the
      extreme rows are systematic rather than accidental.

The two leave different fingerprints, which is why every comparison here is reported
BOTH pooled and restricted to matched |cte| bands. A gap that survives band-matching
is (a). A gap that disappears under band-matching, but with very different |cte|
histograms, is (b). Both can be present.

Run (from av/):
    python analysis/compare_live_offline.py \\
        --live results_de/cond1_light/live/de_cond1_light_unguarded_<stamp>.csv \\
        --offline results_de/cond1_light/mc_shift.csv --threshold 0.1923

    # MC arm, for the same treatment
    python analysis/compare_live_offline.py \\
        --live results_cte/guarded_drive_light_pd_log.csv \\
        --offline results_cte/cond1_light/mc_shift.csv --threshold 0.2849
"""

import os
import csv
import json
import argparse
import numpy as np

BANDS = [(0.0, 0.8, "on target"), (0.8, 1.4, "near"), (1.4, 2.0, "far"),
         (2.0, np.inf, "off road")]


def read_cols(path, candidates):
    """Pull the first matching column name from each candidate group."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} is empty.")
    out = {}
    for key, names in candidates.items():
        col = next((n for n in names if n in rows[0]), None)
        if col is None:
            raise SystemExit(f"{path}: none of {names} present. "
                             f"Columns are {list(rows[0])}")
        out[key] = np.array([float(r[col]) for r in rows])
    return out


def describe(u, thr):
    return {"n": int(len(u)), "median": float(np.median(u)),
            "q25": float(np.percentile(u, 25)), "q75": float(np.percentile(u, 75)),
            "mean": float(u.mean()),
            "beta": float((u <= thr).mean()) if thr else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", required=True, help="live drive log csv")
    ap.add_argument("--offline", required=True,
                    help="offline per-frame csv for the same condition (mc_shift.csv)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="guard threshold, for pass-rate comparison")
    ap.add_argument("--pre_offroad_only", action="store_true",
                    help="restrict the live run to frames before it first left the "
                         "road; isolates behaviour during nominal-ish operation")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    live = read_cols(args.live, {"u": ["u_de", "mc_std", "std"],
                                 "cte": ["cte_true"], "pred": ["cte_pred"]})
    off = read_cols(args.offline, {"u": ["mc_std", "u_de", "std"],
                                   "cte": ["cte_true"], "pred": ["mc_mean", "cte_pred"]})

    label = "live"
    if args.pre_offroad_only:
        bad = np.where(np.abs(live["cte"]) > 2.0)[0]
        if len(bad):
            cut = int(bad[0])
            live = {k: v[:cut] for k, v in live.items()}
            label = f"live (first {cut} frames, before departure)"

    thr = args.threshold
    print(f"live    : {args.live}")
    print(f"offline : {args.offline}")
    if thr:
        print(f"threshold: {thr}")

    # ---- pooled -----------------------------------------------------------
    print(f"\nPOOLED uncertainty")
    print(f"{'set':>34} {'n':>7} {'median':>9} {'IQR':>19} {'beta':>7}")
    for name, d in ((label, live), ("offline", off)):
        s = describe(d["u"], thr)
        print(f"{name:>34} {s['n']:>7} {s['median']:>9.3f} "
              f"[{s['q25']:.3f}, {s['q75']:.3f}]".rjust(19 + 52 - 52) +
              (f" {s['beta']:>7.3f}" if thr else ""))

    # ---- matched lateral bands -------------------------------------------
    print(f"\nBY |cte| BAND  (a gap that SURVIVES band-matching is a condition "
          f"mismatch;\n               a gap that vanishes, with different band "
          f"occupancies, is state visitation)")
    print(f"{'band':>12} {'|cte| range':>14} "
          f"{'live n':>8} {'live med u':>11} {'live beta':>10}   "
          f"{'off n':>7} {'off med u':>10} {'off beta':>9}")
    rows = {}
    for lo, hi, name in BANDS:
        lm = (np.abs(live["cte"]) >= lo) & (np.abs(live["cte"]) < hi)
        om = (np.abs(off["cte"]) >= lo) & (np.abs(off["cte"]) < hi)
        rng = f"[{lo:.1f}, {hi:.1f})" if np.isfinite(hi) else f">= {lo:.1f}"
        lu, ou = live["u"][lm], off["u"][om]
        lmed = np.median(lu) if len(lu) else float("nan")
        omed = np.median(ou) if len(ou) else float("nan")
        lb = (lu <= thr).mean() if (thr and len(lu)) else float("nan")
        ob = (ou <= thr).mean() if (thr and len(ou)) else float("nan")
        print(f"{name:>12} {rng:>14} {len(lu):>8} {lmed:>11.3f} {lb:>10.3f}   "
              f"{len(ou):>7} {omed:>10.3f} {ob:>9.3f}")
        rows[name] = {"live_n": int(len(lu)), "live_median_u": float(lmed),
                      "live_beta": float(lb), "offline_n": int(len(ou)),
                      "offline_median_u": float(omed), "offline_beta": float(ob)}

    # ---- state occupancy --------------------------------------------------
    print(f"\nLATERAL OCCUPANCY (where each set spends its time)")
    print(f"{'band':>12} {'live %':>9} {'offline %':>11}")
    for lo, hi, name in BANDS:
        lp = ((np.abs(live["cte"]) >= lo) & (np.abs(live["cte"]) < hi)).mean() * 100
        op = ((np.abs(off["cte"]) >= lo) & (np.abs(off["cte"]) < hi)).mean() * 100
        print(f"{name:>12} {lp:>9.1f} {op:>11.1f}")

    # ---- verdict ----------------------------------------------------------
    onroad = [b for b in ("on target", "near", "far") if rows[b]["live_n"] >= 10
              and rows[b]["offline_n"] >= 10]
    print()
    if not onroad:
        print("Too few matched on-road frames to compare; run a longer episode.")
    else:
        ratios = [rows[b]["live_median_u"] / rows[b]["offline_median_u"]
                  for b in onroad if rows[b]["offline_median_u"] > 0]
        r = float(np.median(ratios))
        print(f"Median live/offline uncertainty ratio across matched on-road bands: "
              f"{r:.2f}x  (bands: {', '.join(onroad)})")
        if r > 1.5:
            print("  -> CONDITION MISMATCH: at the same lateral positions the live "
                  "episode is markedly more uncertain than the collected data. The "
                  "abstraction was estimated on an easier draw of this condition, so "
                  "its optimism is partly a dataset-representativeness problem, not "
                  "only a modelling one.")
        elif r < 0.67:
            print("  -> the live episode is EASIER than the collected data at matched "
                  "positions; look elsewhere for the source of any discrepancy.")
        else:
            print("  -> conditions look comparable at matched positions. A "
                  "verified-vs-live discrepancy is then more likely to come from "
                  "state visitation or from the abstract dynamics than from the data.")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({"live": args.live, "offline": args.offline,
                       "threshold": thr, "bands": rows}, f, indent=2)
        print(f"\nSaved -> {args.out_json}")


if __name__ == "__main__":
    main()
