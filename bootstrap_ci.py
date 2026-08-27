"""
bootstrap_ci.py
Parametric bootstrap confidence intervals for the PRISM safety probabilities.

WHY: PRISM computes an exact answer for the alpha it is GIVEN, but alpha is
estimated from finite counts (far-L: 200 samples, far-R: 96, and the single
misperception in the far-L row carries most of the off-road risk). PRISM reports
no error bars, so the headline numbers are point estimates. This script attaches
intervals to them.

METHOD: for each row of the confusion matrix, resample the transition
probabilities from a Dirichlet posterior with Jeffreys' prior (alpha0 = 0.5) over
the observed counts. Jeffreys gives ZERO-count cells small positive mass rather
than pinning them at exactly 0 - the honest treatment of the zero-observation
transitions (the paper's Appendix-E caveat). Each resampled alpha is written to a
PRISM model, checked, and the percentiles of the resulting distribution form the
confidence interval.

For the guarded (m2) model, alpha is resampled but beta is held fixed: beta is
estimated from ~2180 samples (s.e. ~0.008) while the alpha rows rest on 96-200,
so the intervals are attributable to perception sampling.

This is used INSTEAD OF FACT (the paper's Sec. 3.3 approach). FACT gives formal
(1-delta) intervals via parametric model checking, but the authors report it
scaling poorly - no results beyond N=4 within a two-hour timeout, and
zero-observation transitions had to be dropped to make it tractable. The
bootstrap has no such ceiling, reuses the existing toolchain, and handles the
zero cells naturally. FACT remains the more formal option for interested authors.

Run (after the point-estimate pipeline):
    python bootstrap_ci.py --json results_cte/confusion_cte.json --N 30 --n_boot 300
    python bootstrap_ci.py --json results_cte/confusion_shift.json --N 30 --n_boot 300 --tag shift
    python bootstrap_ci.py --json results_cte/guarded_alpha.json --guarded --N 30 --n_boot 300
"""

import os
import csv
import json
import argparse
import tempfile
import numpy as np

from make_prism_model import perception_lines, ON_ROAD, MODEL_TEMPLATE
from run_prism import run_one, DEFAULT_LIB

try:
    from guarded_model import M2_TEMPLATE
except Exception:
    M2_TEMPLATE = None


def dirichlet_resample(counts, rng, prior=0.5):
    """One Dirichlet draw per row (Jeffreys prior: zero cells get small mass)."""
    counts = np.asarray(counts, dtype=float)
    out = np.zeros_like(counts)
    for i, row in enumerate(counts):
        out[i] = rng.dirichlet(row + prior)
    return out


def write_model(probs, path, guarded=False, beta=None, M=10):
    perception = perception_lines(probs.tolist(), ON_ROAD)
    if guarded:
        if M2_TEMPLATE is None:
            raise RuntimeError("M2_TEMPLATE unavailable; is guarded_model.py present?")
        b = round(float(beta), 4)
        text = (M2_TEMPLATE.replace("{BETA}", f"{b}")
                           .replace("{BETAC}", f"{round(1.0-b,4)}")
                           .replace("{M}", str(M))
                           .replace("{PERCEPTION}", perception))
    else:
        text = MODEL_TEMPLATE.replace("{PERCEPTION}", perception)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results_cte/confusion_cte.json")
    ap.add_argument("--guarded", action="store_true",
                    help="input json is guarded_alpha.json; build the m2 model")
    ap.add_argument("--N", type=int, default=30, help="horizon to bootstrap at")
    ap.add_argument("--n_boot", type=int, default=300)
    ap.add_argument("--prop", default=None,
                    help="property file (default: off-road for the chosen model)")
    ap.add_argument("--lib", default=DEFAULT_LIB)
    ap.add_argument("--java", default="java")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None, help="label for the output files")
    ap.add_argument("--out_dir", default="results_cte")
    args = ap.parse_args()

    with open(args.json, encoding="utf-8") as f:
        data = json.load(f)
    if args.guarded:
        counts = np.array(data["guarded_counts"], dtype=float)
        beta, M = float(data["beta"]), int(data.get("M", 10))
        tag = args.tag or "m2"
    else:
        counts = np.array(data["alpha_counts"], dtype=float)
        beta, M = None, 10
        tag = args.tag or "m1"

    os.makedirs(args.out_dir, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="boot_")
    model_path = os.path.join(tmpdir, "boot.prism")
    if args.prop:
        prop_path = args.prop
    else:
        prop_path = os.path.join(tmpdir, "boot.props")
        with open(prop_path, "w", encoding="utf-8") as f:
            f.write("P=? [ F cte=-1 ]\n")

    # point estimate first
    row_sums = counts.sum(axis=1, keepdims=True)
    point_probs = np.divide(counts, row_sums, out=np.zeros_like(counts),
                            where=row_sums > 0)
    write_model(point_probs, model_path, args.guarded, beta, M)
    point = run_one(args.java, args.lib, model_path, prop_path, args.N)
    print(f"Point estimate (N={args.N}): {point:.6g}")
    print(f"Bootstrapping {args.n_boot} replicates "
          f"(Dirichlet, Jeffreys prior; {'alpha only, beta fixed' if args.guarded else 'alpha'})...")

    rng = np.random.default_rng(args.seed)
    vals = []
    for b in range(args.n_boot):
        probs = dirichlet_resample(counts, rng)
        write_model(probs, model_path, args.guarded, beta, M)
        vals.append(run_one(args.java, args.lib, model_path, prop_path, args.N))
        if (b + 1) % 25 == 0:
            print(f"  {b+1}/{args.n_boot}", end="\r")
    vals = np.array(vals)

    lo95, hi95 = np.percentile(vals, [2.5, 97.5])
    lo90, hi90 = np.percentile(vals, [5, 95])
    print(f"\n\nBootstrap distribution of P[F cte=-1] at N={args.N}:")
    print(f"  point estimate : {point:.6g}")
    print(f"  median         : {np.median(vals):.6g}")
    print(f"  90% CI         : [{lo90:.6g}, {hi90:.6g}]")
    print(f"  95% CI         : [{lo95:.6g}, {hi95:.6g}]")
    print(f"  min / max      : {vals.min():.6g} / {vals.max():.6g}")
    frac_zero = float((vals == 0).mean())
    if frac_zero > 0:
        print(f"  replicates at exactly 0: {frac_zero*100:.1f}%")

    out = {"json": args.json, "guarded": args.guarded, "N": args.N,
           "n_boot": args.n_boot, "point": point,
           "median": float(np.median(vals)),
           "ci90": [float(lo90), float(hi90)], "ci95": [float(lo95), float(hi95)],
           "min": float(vals.min()), "max": float(vals.max()),
           "frac_zero": frac_zero}
    jp = os.path.join(args.out_dir, f"bootstrap_{tag}_N{args.N}.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    cp = os.path.join(args.out_dir, f"bootstrap_{tag}_N{args.N}_samples.csv")
    with open(cp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["replicate", "p"])
        for i, v in enumerate(vals):
            w.writerow([i, f"{v:.10g}"])
    print(f"\nSaved -> {jp}\nSaved -> {cp}")


if __name__ == "__main__":
    main()
