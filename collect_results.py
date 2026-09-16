"""
collect_results.py
Assemble every result from both arms into paper-ready tables, read from disk.

Nothing here recomputes anything. It walks results_cte/ (MC-Dropout) and results_de/
(Deep Ensemble), pulls the numbers out of the json/csv artefacts each pipeline stage
already wrote, and emits one CSV per table plus a markdown digest. The point is that
the numbers in the paper come from files rather than from transcription, and that
re-running a stage and re-running this script is enough to refresh everything.

Tables produced:
  table_perception.csv   per arm x condition: accuracy, MAE, bias, median u, beta,
                         and the distributional distances (JS, Mahalanobis AUROC,
                         symmetrised KL) where shift_distance.json exists
  table_verified.csv     per arm x condition: P_off(m1), P_off(m2), P_abort, each with
                         its bootstrap median and 95% interval, plus guard row support
  table_live.csv         per arm x condition x policy: handover frame and |cte|,
                         first off-road frame, per-mode certified %, off-road %, laps
  table_identifiability.csv  beta, admitted on-road frames, per-row admitted counts,
                         and how many rows were prior-determined or smoothed
  RESULTS.md             the four tables rendered, with the caveats that must travel
                         with specific cells (structural zeros, prior-dominated rows,
                         P_abort being a function of beta)

Run (from av/):
    python collect_results.py
    python collect_results.py --out_dir paper_tables --N 30
"""

import os
import csv
import json
import glob
import argparse

CONDS = ["baseline_run", "cond1_cones", "cond1_light", "cond1_trees",
         "generated_road", "mini_monaco"]
NICE = {"baseline_run": "control", "cond1_cones": "cones", "cond1_light": "light",
        "cond1_trees": "trees", "generated_road": "road", "mini_monaco": "circuit"}
ARMS = [("MC", "results_cte"), ("DE", "results_de")]
STATE_NAMES = {3: "far-L", 1: "near-L", 0: "LANE", 2: "near-R", 4: "far-R"}


def frame_stats(path, thr, err_thresh=0.8, road_edge=2.0):
    """Median u, signed bias, prediction spread and the confidently-wrong rate,
    computed from a per-frame csv. Used when no *_shift_summary.json exists, so the
    MC and DE arms report the same quantities rather than only the arm whose stage
    happened to write a summary."""
    if not os.path.exists(path):
        return {}
    us, preds, trues = [], [], []
    try:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                us.append(float(r["mc_std"]))
                preds.append(float(r["mc_mean"]))
                trues.append(float(r["cte_true"]))
    except Exception:
        return {}
    if not us:
        return {}
    import statistics as st
    on = [i for i, t in enumerate(trues) if abs(t) <= road_edge]
    big = [i for i in on if abs(preds[i] - trues[i]) > err_thresh]
    out = {"median_u": st.median(us),
           "pred_sd": st.stdev(preds) if len(preds) > 1 else None}
    if on:
        out["bias"] = sum(preds[i] - trues[i] for i in on) / len(on)
    if big and thr is not None:
        out["pass_given_large_err"] = sum(1 for i in big if us[i] <= thr) / len(big)
    return out


def jload(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def prism_at(path, N):
    """Read a prism curve csv and return the value at horizon N."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                if int(float(row[list(row)[0]])) == N:
                    return float(row[list(row)[1]])
    except Exception:
        return None
    return None


def boot(root, cond, kind, N):
    """Find a bootstrap json however it was tagged."""
    pats = [f"bootstrap_{kind}_{cond}_N{N}.json", f"bootstrap_{kind}_N{N}.json",
            f"bootstrap_{kind}*_N{N}.json"]
    for p in pats:
        hits = sorted(glob.glob(os.path.join(root, cond, p)))
        if hits:
            return jload(hits[0])
    return None


def fmt(x, sig=3):
    if x is None:
        return ""
    if x == 0:
        return "0"
    if abs(x) >= 0.01:
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return f"{x:.{sig}g}"


def ci(b):
    if not b or "ci95" not in b:
        return ""
    lo, hi = b["ci95"]
    return f"[{fmt(lo)}, {fmt(hi)}]"


# ---------------------------------------------------------------- perception
def perception_rows(N):
    rows = []
    for arm, root in ARMS:
        for c in CONDS:
            d = os.path.join(root, c)
            conf = jload(os.path.join(d, "confusion_shift.json"))
            if not conf:
                continue
            dist = jload(os.path.join(d, "shift_distance.json")) or {}
            de = jload(os.path.join(d, "de_shift_summary.json")) or {}
            guard = jload(os.path.join(d, "guarded_alpha.json")) or {}
            if not de:      # MC arm: derive the same quantities from the frame csv
                de = frame_stats(os.path.join(d, "mc_shift.csv"),
                                 guard.get("threshold"))
            maha = dist.get("mahalanobis") or {}
            rows.append({
                "arm": arm, "condition": NICE[c],
                "frames": conf.get("val_samples"),
                "on_road_frames": conf.get("on_road_samples"),
                "accuracy": conf.get("on_road_disc_accuracy"),
                "mae": conf.get("shifted_mae"),
                "bias": de.get("signed_bias", de.get("bias")),
                "median_u": de.get("u_median", de.get("median_u")),
                "beta": conf.get("beta_shift", guard.get("beta")),
                "pass_given_large_err": de.get("pass_given_large_err"),
                "pred_sd": de.get("pred_sd"),
                "pixel_js": dist.get("pixel_js_bits"),
                "maha_auroc": maha.get("auroc") if isinstance(maha, dict) else None,
                "sym_kl_nats": dist.get("feature_sym_kl_nats"),
            })
    return rows


# ----------------------------------------------------------------- verified
def verified_rows(N):
    rows = []
    for arm, root in ARMS:
        for c in CONDS:
            d = os.path.join(root, c)
            if not os.path.isdir(d):
                continue
            guard = jload(os.path.join(d, "guarded_alpha.json")) or {}
            b1 = boot(root, c, "m1", N)
            b2 = boot(root, c, "m2", N)
            ba = boot(root, c, "m2abort", N)
            rows.append({
                "arm": arm, "condition": NICE[c],
                "beta": guard.get("beta"),
                "m1_point": prism_at(os.path.join(d, "prism_m1.csv"), N),
                "m1_median": (b1 or {}).get("median"), "m1_ci95": ci(b1),
                "m2_point": prism_at(os.path.join(d, "prism_m2_offroad.csv"), N),
                "m2_median": (b2 or {}).get("median"), "m2_ci95": ci(b2),
                "abort_point": prism_at(os.path.join(d, "prism_m2_abort.csv"), N),
                "abort_median": (ba or {}).get("median"), "abort_ci95": ci(ba),
                "n_passing_onroad": guard.get("n_passing_onroad"),
            })
    return rows


# ----------------------------------------------------------- identifiability
def identifiability_rows():
    rows = []
    for arm, root in ARMS:
        for c in CONDS:
            g = jload(os.path.join(root, c, "guarded_alpha.json"))
            if not g:
                continue
            counts = g.get("guarded_counts") or []
            order = g.get("state_order") or [3, 1, 0, 2, 4]
            per_row = {STATE_NAMES.get(s, str(s)): (sum(counts[i]) if i < len(counts) else None)
                       for i, s in enumerate(order)}
            status = g.get("row_status") or {}
            mc = g.get("min_count") or 0
            ns = [v.get("n") for v in status.values() if isinstance(v, dict)] or \
                 [sum(r) for r in counts]
            empty = sum(1 for n in ns if n == 0)
            smoothed = sum(1 for n in ns if 0 < n < mc)
            rows.append({
                "arm": arm, "condition": NICE[c],
                "beta": g.get("beta"),
                "admitted_onroad": g.get("n_passing_onroad") or sum(
                    sum(r) for r in counts) if counts else None,
                **{f"n_{k}": v for k, v in per_row.items()},
                "rows_empty": empty, "rows_smoothed": smoothed,
                "min_count": g.get("min_count"),
                "acc_all": g.get("acc_all"), "acc_guarded": g.get("acc_guarded"),
            })
    return rows


# --------------------------------------------------------------------- live
def live_rows():
    rows = []
    for arm, root in ARMS:
        for c in CONDS:
            for sp in sorted(glob.glob(os.path.join(root, c, "live", "*_summary.json"))):
                s = jload(sp)
                if not s:
                    continue
                fh, fo, cf = (s.get("first_handover"), s.get("first_offroad"),
                              s.get("counterfactual_handover"))
                md, fb = s.get("MODEL") or {}, s.get("FALLBACK") or {}
                rows.append({
                    "arm": arm, "condition": NICE[c],
                    "policy": ("unguarded" if s.get("unguarded")
                               else f"M={s.get('M')}" +
                                    (f",hb={s['handback']}" if s.get("handback") else "")),
                    "frames": s.get("frames"), "laps": s.get("laps_completed"),
                    "handovers": s.get("handovers"),
                    "handover_frame": (fh or {}).get("frame"),
                    "handover_abs_cte": (fh or {}).get("abs_cte"),
                    "cf_handover_frame": (cf or {}).get("frame"),
                    "cf_handover_abs_cte": (cf or {}).get("abs_cte"),
                    "first_offroad_frame": (fo or {}).get("frame"),
                    "first_offroad_mode": (fo or {}).get("mode"),
                    "model_frames": md.get("frames"),
                    "model_certified_pct": md.get("certified_pct"),
                    "model_offroad_pct": md.get("offroad_pct"),
                    "model_mean_abs_cte": md.get("mean_abs_cte"),
                    "fallback_frames": fb.get("frames"),
                    "fallback_offroad_pct": fb.get("offroad_pct"),
                    "fallback_mean_abs_cte": fb.get("mean_abs_cte"),
                    "log": os.path.basename(s.get("log", "")),
                })
    return rows


def exclusion_reason(r):
    """Why a live run must not be used as evidence. Returns None if it is valid.

    Rule 1 - pre-fix unguarded logging. Early versions of the drive scripts forced
    the guard verdict to PASS whenever the guard was disabled, so the run reports
    100% certified and no counterfactual handover frame. The steering and the
    trajectory are still correct, but the guard columns are meaningless, and the
    counterfactual (the whole point of an unguarded run) was never recorded.

    Rule 2 - the vehicle was already off-road at frame 0. The episode starts outside
    the road boundary, so 'first off-road frame' is a spawn artefact and the run
    says nothing about when the system left the road.
    """
    if r.get("policy") == "unguarded" and r.get("model_certified_pct") is not None \
            and abs(r["model_certified_pct"] - 100.0) < 1e-9 \
            and r.get("cf_handover_frame") is None:
        return "pre-fix unguarded logging: guard verdict forced to pass"
    if r.get("first_offroad_frame") == 0:
        return "off-road at frame 0: spawn artefact"
    return None


def write_csv(path, rows):
    if not rows:
        print(f"  (no rows) {path}")
        return
    keys = list(rows[0])
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  {len(rows):3d} rows -> {path}")


def md_table(rows, cols, headers=None):
    headers = headers or cols
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append(fmt(v) if isinstance(v, float) else ("" if v is None else str(v)))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="paper_tables")
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--keep_all", action="store_true",
                    help="do not apply the live-run exclusion rules")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    N = args.N

    print("Collecting results...")
    perc = perception_rows(N)
    ver = verified_rows(N)
    iden = identifiability_rows()
    live = live_rows()

    write_csv(os.path.join(args.out_dir, "table_perception.csv"), perc)
    write_csv(os.path.join(args.out_dir, "table_verified.csv"), ver)
    write_csv(os.path.join(args.out_dir, "table_identifiability.csv"), iden)
    live_ok, live_bad = [], []
    for r in live:
        why = None if args.keep_all else exclusion_reason(r)
        if why:
            live_bad.append(dict(r, excluded_reason=why))
        else:
            live_ok.append(r)
    write_csv(os.path.join(args.out_dir, "table_live.csv"), live_ok)
    if live_bad:
        write_csv(os.path.join(args.out_dir, "table_live_excluded.csv"), live_bad)
        print(f"\n  {len(live_bad)} live run(s) excluded:")
        for r in live_bad:
            print(f"    {r['arm']}/{r['condition']:8s} {r['policy']:10s} "
                  f"{r['log'][:46]:46s}  {r['excluded_reason']}")
        print("  These are written to table_live_excluded.csv rather than deleted, so "
              "the exclusion is visible and reversible (--keep_all).")
    live = live_ok

    # ---- markdown digest ----
    md = [f"# Results digest (horizon N = {N})", "",
          "Generated by `collect_results.py` from the artefacts on disk. "
          "Re-run any pipeline stage and re-run this script to refresh.", ""]

    md += ["## 1. Perception and guard response", "",
           md_table(perc, ["arm", "condition", "accuracy", "mae", "bias", "median_u",
                           "beta", "pass_given_large_err", "pred_sd"],
                    ["arm", "cond", "acc", "MAE", "bias", "med u", "beta",
                     "P(pass|big err)", "pred sd"]), "",
           "`beta` is the fraction of frames the fixed guard admits. "
           "`P(pass|big err)` is the confidently-wrong rate: the share of frames with "
           "absolute error above one on-target bin width that the guard still admits. "
           "`pred sd` should be read against each arm's own control row, not against "
           "the in-distribution validation set, whose spread is inflated by the "
           "training distribution's excursions.", ""]

    md += ["## 2. Verified closed-loop probabilities", "",
           md_table(ver, ["arm", "condition", "beta", "m1_point", "m1_median", "m1_ci95",
                          "m2_point", "m2_median", "m2_ci95", "abort_point"],
                    ["arm", "cond", "beta", "m1", "m1 med", "m1 95%",
                     "m2", "m2 med", "m2 95%", "P_abort"]), "",
           "CAVEATS THAT MUST TRAVEL WITH THESE CELLS:", "",
           "- A point estimate of 0 is an unobserved transition in a finite sample, "
           "not an impossibility. Several are *structural*: a single zero-count cell in "
           "a well-populated row can make an entire state unreachable, so the zero "
           "rests on the absence of evidence in one cell rather than on the model "
           "being safe. Report the bootstrap median and interval for these rows.", "",
           "- Where `beta` is very small the guarded rows are prior-determined "
           "(see table 4). The guarded off-road probability there is not an empirical "
           "estimate and its narrow interval reflects prior sampling, not precision.", "",
           "- `P_abort` is reproduced to within a few percent by "
           "`1 - (1 - (1-beta)^M)^N`, i.e. it is a function of beta, M and N with the "
           "abstraction contributing almost nothing. The bootstrap resamples alpha and "
           "holds beta fixed, so its interval on P_abort is spuriously narrow. Report "
           "P_abort with a beta sensitivity band instead.", ""]

    md += ["## 3. Guard selectivity vs identifiability", "",
           md_table(iden, ["arm", "condition", "beta", "admitted_onroad", "n_far-L",
                           "n_near-L", "n_LANE", "n_near-R", "n_far-R",
                           "rows_empty", "rows_smoothed"],
                    ["arm", "cond", "beta", "admitted", "far-L", "near-L", "LANE",
                     "near-R", "far-R", "empty rows", "smoothed"]), "",
           "As the guard becomes more selective the guarded abstraction loses the "
           "observations it would be estimated from. Rows marked empty are determined "
           "entirely by the Jeffreys prior; smoothed rows fall below the "
           "`--min_count` threshold in `guarded_model.py` and are prior-shrunk in the "
           "POINT estimate as well as in the bootstrap. That smoothing is a deviation "
           "from plain row-normalisation and must be stated in the methodology.", ""]

    md += ["## 4. Live closed-loop runs", "",
           md_table(live, ["arm", "condition", "policy", "handover_frame",
                           "handover_abs_cte", "cf_handover_frame", "first_offroad_frame",
                           "first_offroad_mode", "model_certified_pct",
                           "model_offroad_pct", "fallback_offroad_pct", "laps"],
                    ["arm", "cond", "policy", "HO frame", "|cte| at HO", "cf HO frame",
                     "1st off-road", "under", "model cert %", "model off %",
                     "fallback off %", "laps"]), "",
           ("" if not live_bad else
            f"{len(live_bad)} live run(s) were excluded by the rules in "
            f"`exclusion_reason()` and are listed in `table_live_excluded.csv`. "
            f"No claim in the paper rests on them: the excluded unguarded runs are "
            f"duplicated by valid repeats of the same condition, and the excluded "
            f"spawn-artefact runs report a first-off-road frame of 0.\n\n") +
           "`cf HO frame` is the counterfactual: in an unguarded run the guard verdict "
           "is still scored, so this is the frame at which it would have handed over. "
           "Comparing it with `1st off-road` gives the detection margin directly.", ""]

    path = os.path.join(args.out_dir, "RESULTS.md")
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nDigest -> {path}")

    # ---- gaps ----
    missing = []
    for arm, root in ARMS:
        for c in CONDS:
            d = os.path.join(root, c)
            if not os.path.isdir(d):
                missing.append(f"{arm}/{NICE[c]}: condition folder absent")
                continue
            for f_, what in (("confusion_shift.json", "offline eval"),
                             ("guarded_alpha.json", "guarded abstraction"),
                             ("prism_m1.csv", "m1 model check"),
                             ("prism_m2_abort.csv", "abort model check")):
                if not os.path.exists(os.path.join(d, f_)):
                    missing.append(f"{arm}/{NICE[c]}: missing {what} ({f_})")
            if not boot(root, c, "m1", N):
                missing.append(f"{arm}/{NICE[c]}: no m1 bootstrap at N={N}")
            if not boot(root, c, "m2abort", N):
                missing.append(f"{arm}/{NICE[c]}: no P_abort bootstrap at N={N}")
            if not glob.glob(os.path.join(d, "live", "*_summary.json")):
                missing.append(f"{arm}/{NICE[c]}: no live runs")
    if missing:
        print(f"\nGAPS ({len(missing)}):")
        for m in missing:
            print(f"  - {m}")
    else:
        print("\nNo gaps: every arm x condition has offline, verified, bootstrap and live results.")


if __name__ == "__main__":
    main()
