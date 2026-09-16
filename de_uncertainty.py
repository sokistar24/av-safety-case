"""
de_uncertainty.py
Nominal calibration of the Deep-Ensemble run-time guard. The ensemble analogue of
mc_uncertainty.py, and deliberately the same shape: same validation split, same
percentile sweep, same selection rule, same output schema.

Members are evaluated DETERMINISTICALLY (model.eval(), dropout off). Disagreement
therefore comes only from independently trained parameter solutions (paper Eq. 8-9),
not from stochastic masks.

    yhat_DE(x) = (1/K) sum_k f_k(x)          ensemble mean, used by the controller
    u_DE(x)    = sd over members, ddof=1     guard score; pass iff u_DE <= tau_DE

Calibration uses ONLY the held-out nominal validation set. Two selection modes:

  default      the tau_MC rule - the loosest candidate percentile at which admitted
               inputs still have higher discretised on-road accuracy than rejected
               inputs. This rule assumes the ordering eventually inverts; if it never
               does, the rule degenerates to "the loosest percentile offered" and the
               selected beta is an artefact of the candidate list, not of the data.
  --pct P      freeze the operating point at percentile P regardless of the rule. Use
               this to MATCH the nominal pass rate of the arm being compared against
               (--pct 80 -> beta = 0.800, the MC-Dropout operating point), so that any
               difference under shift is attributable to the score rather than to
               different selectivity.

Either way the rule's own verdict is recorded in the json under "rule_selected", so the
provenance of the frozen threshold is explicit. Do NOT reuse 0.2849 - that is the
MC-Dropout operating point on a different score with a different scale.

OUTPUT SCHEMA. The CSV keeps the legacy column names cte_true, mc_mean, mc_std so the
rest of the pipeline (guarded_model.py, bootstrap_ci.py, condition_summary.py) reads
it unmodified: keep the DE arm in its own results_de/ tree and every downstream script
works via --mc_csv / --results_root. Per-member predictions m0..m{K-1} are appended as
extra columns; DictReader ignores them. Those columns are the point of the exercise -
they let you ask whether a low u_DE under `light` is members agreeing on a collapsed
band (disagreement-based uncertainty fails generally) or members disagreeing but the
sd being small for another reason.

Run (from av/):
    python de_uncertainty.py --out_name mc_val_K10.csv          # survey all members
    python de_uncertainty.py --K 10 --pct 80 --out_name mc_val.csv   # freeze at beta=0.800

Outputs:
    results_de/mc_val.csv            cte_true, mc_mean, mc_std, m0..m{K-1}
    results_de/de_calibration.json   sweep table, selected tau_DE, correlations
"""

import os
import csv
import json
import glob
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from cte_dataset import build_cte_datasets, bin_cte, ROAD_EDGE
from cte_model import TaxiNetCTE

SPLIT_SEED = 42
PERCENTILES = (60, 70, 80, 85, 90, 95)


def rank(a):
    r = np.empty(len(a))
    r[np.argsort(a)] = np.arange(len(a))
    return r


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
        with torch.no_grad():                    # initialise LazyLinear before load
            m(torch.zeros(2, 3, 80, 160, device=device))
        m.load_state_dict(torch.load(p, map_location=device))
        m.eval()                                 # deterministic: dropout OFF
        models.append(m)
    return models, [os.path.basename(p) for p in paths]


@torch.no_grad()
def ensemble_predict(models, loader, device):
    """Returns (trues, member_preds) with member_preds shape (K, n)."""
    trues, per_member = [], [[] for _ in models]
    for imgs, targets in loader:
        imgs = imgs.to(device)
        trues.append(targets.numpy().ravel())
        for k, m in enumerate(models):
            per_member[k].append(m(imgs).cpu().numpy().ravel())
    trues = np.concatenate(trues)
    preds = np.stack([np.concatenate(p) for p in per_member], axis=0)
    return trues, preds


def sweep(trues, means, stds, err_thresh, percentiles=PERCENTILES):
    """Threshold table. Adds P(pass | large error), the confidently-wrong rate."""
    err = np.abs(means - trues)
    t_states = np.array([bin_cte(x) for x in trues])
    p_states = np.array([bin_cte(x) for x in np.clip(means, -ROAD_EDGE, ROAD_EDGE)])
    onroad = t_states != -1
    correct = t_states == p_states
    big = onroad & (err > err_thresh)

    rows = []
    for pct in percentiles:
        th = float(np.percentile(stds, pct))
        passing = stds <= th
        pm, fm = passing & onroad, (~passing) & onroad
        rows.append({
            "percentile": pct,
            "threshold": th,
            "beta": float(passing.mean()),
            "acc_pass": float(correct[pm].mean()) if pm.any() else float("nan"),
            "acc_fail": float(correct[fm].mean()) if fm.any() else float("nan"),
            "mae_pass": float(err[pm].mean()) if pm.any() else float("nan"),
            "mae_fail": float(err[fm].mean()) if fm.any() else float("nan"),
            "pass_given_large_err": float(passing[big].mean()) if big.any() else float("nan"),
        })
    return rows, float(big.mean())


def select_threshold(rows):
    """Loosest percentile at which acc(pass) > acc(fail) - the paper's rule."""
    ok = [r for r in rows if r["acc_pass"] > r["acc_fail"]]
    if not ok:
        return None
    return max(ok, key=lambda r: r["percentile"])


def row_at_percentile(rows, stds, pct, trues, means, err_thresh):
    """Fetch (or compute) the sweep row for an arbitrary percentile."""
    for r in rows:
        if abs(r["percentile"] - pct) < 1e-9:
            return r
    extra, _ = sweep(trues, means, stds, err_thresh, percentiles=(pct,))
    return extra[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_cte/generated_track")
    ap.add_argument("--ckpt_dir", default="results_de")
    ap.add_argument("--out_dir", default="results_de")
    ap.add_argument("--out_name", default="mc_val.csv",
                    help="legacy name kept so guarded_model.py reads it unmodified")
    ap.add_argument("--K", type=int, default=None,
                    help="use the first K members (default: all trained)")
    ap.add_argument("--split_seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--pct", type=float, default=None,
                    help="freeze the operating point at this percentile instead of "
                         "applying the selection rule (--pct 80 matches the "
                         "MC-Dropout nominal pass rate beta = 0.800)")
    ap.add_argument("--err_thresh", type=float, default=0.8,
                    help="'large error' cut for the confidently-wrong rate, in metres "
                         "(0.8 = one on-target bin width)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    _, val_ds, info = build_cte_datasets(args.data_dir, seed=args.split_seed)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    models, names = load_members(args.ckpt_dir, args.K, device)
    K = len(models)
    print(f"Validation set: {info['val']} images  |  ensemble members: K = {K}")
    print(f"Members: {', '.join(names)}")

    trues, preds = ensemble_predict(models, loader, device)
    means = preds.mean(axis=0)
    stds = preds.std(axis=0, ddof=1)          # Eq. 9: K-1 denominator
    err = np.abs(means - trues)

    # ---- per-member quality, for the manifest cross-check ------------------
    print("\nPer-member validation MAE (deterministic eval):")
    member_mae = []
    for k, nm in enumerate(names):
        mae_k = float(np.abs(preds[k] - trues).mean())
        member_mae.append(mae_k)
        print(f"  {nm:>16s}  MAE {mae_k:.4f}")
    print(f"  {'ensemble mean':>16s}  MAE {err.mean():.4f}   "
          f"(best member {min(member_mae):.4f})")

    # ---- guard premise: does u_DE predict error? ---------------------------
    pear = float(np.corrcoef(stds, err)[0, 1])
    spear = float(np.corrcoef(rank(stds), rank(err))[0, 1])
    print(f"\nGuard premise: u_DE vs |error|   Pearson {pear:.3f}   Spearman {spear:.3f}")
    print(f"  (MC-Dropout on the same split: Pearson 0.501, Spearman 0.260)")
    print(f"u_DE distribution: median {np.median(stds):.4f}  "
          f"mean {stds.mean():.4f}  range [{stds.min():.4f}, {stds.max():.4f}]")

    # ---- threshold sweep ---------------------------------------------------
    rows, frac_big = sweep(trues, means, stds, args.err_thresh)
    print(f"\nThreshold table (guard = 'pass if u_DE <= threshold'); "
          f"{frac_big*100:.1f}% of on-road frames have |err| > {args.err_thresh}:")
    print(f"{'pct':>4} {'threshold':>10} {'beta':>7} {'acc pass':>9} {'acc fail':>9} "
          f"{'|e| pass':>9} {'|e| fail':>9} {'P(pass|big err)':>16}")
    for r in rows:
        print(f"{r['percentile']:>4} {r['threshold']:>10.4f} {r['beta']:>7.3f} "
              f"{r['acc_pass']*100:>8.2f}% {r['acc_fail']*100:>8.2f}% "
              f"{r['mae_pass']:>9.3f} {r['mae_fail']:>9.3f} "
              f"{r['pass_given_large_err']:>16.3f}")

    rule_sel = select_threshold(rows)
    never_inverts = all(r["acc_pass"] > r["acc_fail"] for r in rows)

    if rule_sel is None:
        print("\nNO candidate threshold satisfies acc(pass) > acc(fail).")
        print("The guard premise fails in-distribution; that is itself a result, but")
        print("it must be resolved before the shifted conditions are run.")
    elif never_inverts:
        print(f"\nThe discretised ordering holds at EVERY candidate percentile "
              f"({PERCENTILES[0]}-{PERCENTILES[-1]}).")
        print("The selection rule therefore does not bind: it returns the loosest "
              "percentile offered, which is a property of the candidate list rather "
              "than of the data. Freeze the operating point with --pct instead, "
              "matched to the arm you are comparing against (--pct 80 -> beta 0.800).")
        print("The absence of an inversion is itself reportable: the MC-Dropout score "
              "inverted above the 80th percentile on this same split.")
    else:
        print(f"\nSelection rule: loosest percentile with acc(pass) > acc(fail) is "
              f"{rule_sel['percentile']} (tau = {rule_sel['threshold']:.4f}).")
        print("Check the continuous-error columns separately: if |e| pass < |e| fail "
              "holds past that point, the reversal is a discretisation effect (the "
              "far-L composition argument in Appendix B), not a loss of signal.")

    if args.pct is not None:
        sel = row_at_percentile(rows, stds, args.pct, trues, means, args.err_thresh)
        mode = f"frozen at the {args.pct:g}th percentile (matched nominal beta)"
    else:
        sel = rule_sel
        mode = "selection rule (loosest percentile at which the premise holds)"
    tau = sel["threshold"] if sel else None

    if sel is not None:
        print(f"\nOPERATING POINT [{mode}]")
        print(f"  percentile {sel['percentile']:g}   tau_DE = {tau:.4f}   "
              f"beta = {sel['beta']:.4f}")
        print(f"  acc pass {sel['acc_pass']*100:.2f}%  vs  acc fail "
              f"{sel['acc_fail']*100:.2f}%   (gap {(sel['acc_pass']-sel['acc_fail'])*100:.2f} pts)")
        if args.pct is None and never_inverts:
            print("  WARNING: this came from the degenerate rule above. Re-run with "
                  "--pct to freeze a defensible operating point.")

    # ---- write ------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, args.out_name)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cte_true", "mc_mean", "mc_std"] + [f"m{k}" for k in range(K)])
        for i in range(len(trues)):
            w.writerow([f"{trues[i]:.5f}", f"{means[i]:.5f}", f"{stds[i]:.5f}"] +
                       [f"{preds[k, i]:.5f}" for k in range(K)])

    payload = {
        "estimator": "deep_ensemble",
        "K": K, "members": names, "member_val_mae": member_mae,
        "eval_mode": "deterministic (dropout off)",
        "data_dir": args.data_dir, "split_seed": args.split_seed,
        "n_val": int(len(trues)),
        "ensemble_mae": float(err.mean()),
        "u_median": float(np.median(stds)), "u_mean": float(stds.mean()),
        "pearson_u_err": pear, "spearman_u_err": spear,
        "err_thresh": args.err_thresh, "frac_large_err": frac_big,
        "sweep": rows,
        "selection_mode": mode,
        "rule_selected": ({"percentile": rule_sel["percentile"],
                           "threshold": rule_sel["threshold"],
                           "beta": rule_sel["beta"]} if rule_sel else None),
        "premise_holds_at_all_percentiles": never_inverts,
        "selected_percentile": sel["percentile"] if sel else None,
        "threshold": tau, "beta": sel["beta"] if sel else None,
        "acc_pass": sel["acc_pass"] if sel else None,
        "acc_fail": sel["acc_fail"] if sel else None,
    }
    out_json = os.path.join(args.out_dir, "de_calibration.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nPer-image predictions -> {out_csv}")
    print(f"Calibration           -> {out_json}")
    if tau is not None:
        print("\nFreeze tau_DE from here on; pass it explicitly to every condition:")
        print(f"  python guarded_model.py --mc_csv {out_csv} \\")
        print(f"         --threshold {tau:.4f} --out_dir {args.out_dir}")
    print("\nNext:  python ensemble_size_sweep.py   (then re-run with --K and --pct)")
    print("       python de_shift_eval.py --data_dir data_cte/<condition> \\")
    print("              --out_dir results_de/<condition>")


if __name__ == "__main__":
    main()
