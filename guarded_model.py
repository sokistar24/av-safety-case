"""
guarded_model.py
Build the GUARDED perception abstraction (the paper's alpha^true, Sec. 3.2) from
the MC-Dropout uncertainty check, and generate the m2 PRISM model (Appendix B.2
structure): the closed loop WITH the run-time guard — a beta-weighted
certify/repeat loop that aborts to a fail-safe (the handover) after M
consecutive failed checks.

Guard: an input PASSES if its MC-dropout std <= threshold, where the threshold
is a percentile of the validation std distribution (--pct 80 -> beta ~= 0.80,
comparable to the paper's 82.1% pass rate).

Reads:   results_cte/mc_val.csv          (from mc_uncertainty.py)
Writes:  results_cte/lanekeep_m2.prism
         results_cte/lanekeep_m2_offroad.props    P=? [ F cte=-1 ]
         results_cte/lanekeep_m2_abort.props      P=? [ F i=M ]   (Property 3)
         results_cte/guarded_alpha.json

Run:     python guarded_model.py --pct 80
Then sweep both properties with the existing runner:
    python run_prism.py --model results_cte/lanekeep_m2.prism --props results_cte/lanekeep_m2_offroad.props --out results_cte/prism_m2_offroad.csv
    python run_prism.py --model results_cte/lanekeep_m2.prism --props results_cte/lanekeep_m2_abort.props --out results_cte/prism_m2_abort.csv
"""

import os
import csv
import json
import argparse
import numpy as np

from cte_dataset import bin_cte, ROAD_EDGE, STATE_NAMES
from make_prism_model import perception_lines, ON_ROAD

M2_TEMPLATE = """// Auto-generated lane-keeping DTMC WITH run-time guard (paper Appendix B.2, CTE-only)
// Guard: MC-dropout uncertainty check; passes w.p. beta={BETA}; abort after M fails.
// Perception probabilities measured on inputs that PASS the check (guarded alpha).
dtmc

const int N;        // finite time horizon (steps)
const int M = {M};  // max consecutive failed checks before abort (fail-safe handover)

module lanekeep

  cte : [-1..4] init 0;
  cte_est : [0..4] init 0;
  v : [0..1] init 1;
  a : [0..2] init 0;
  pc : [0..5] init 0;
  k : [1..N] init 1;
  i : [0..M] init 0;   // consecutive failed-check counter

  // ---- run-time guard: certify w.p. beta, else repeat the sensor reading ----
  [] pc=0 & i<M -> {BETA}: (v'=1) & (pc'=1) & (i'=0) + {BETAC}: (v'=0) & (i'=i+1);
  // abort to fail-safe (the HANDOVER) after M consecutive failures - absorbing
  [] pc=0 & i=M -> true;

  // ---- perception: GUARDED alpha (inputs that pass the uncertainty check) ----
{PERCEPTION}

  // ---- controller: discretized P-controller (same as m1) ----
  [] cte_est=0 & pc=2                 -> 1: (a'=0) & (pc'=3);
  [] (cte_est=1 | cte_est=3) & pc=2   -> 1: (a'=2) & (pc'=3);
  [] (cte_est=2 | cte_est=4) & pc=2   -> 1: (a'=1) & (pc'=3);

  // ---- dynamics (same deterministic dynamics as m1) ----
  [] pc=3 & k<N & a=0 -> 1: (pc'=0) & (k'=k+1);
  [] pc=3 & k<N & a=1 & cte=4 -> 1: (cte'=2) & (pc'=0) & (k'=k+1);
  [] pc=3 & k<N & a=1 & cte=2 -> 1: (cte'=0) & (pc'=0) & (k'=k+1);
  [] pc=3 & k<N & a=1 & cte=0 -> 1: (cte'=1) & (pc'=0) & (k'=k+1);
  [] pc=3 & k<N & a=1 & cte=1 -> 1: (cte'=3) & (pc'=0) & (k'=k+1);
  [] pc=3 & k<N & a=1 & cte=3 -> 1: (cte'=-1) & (pc'=5);
  [] pc=3 & k<N & a=2 & cte=3 -> 1: (cte'=1) & (pc'=0) & (k'=k+1);
  [] pc=3 & k<N & a=2 & cte=1 -> 1: (cte'=0) & (pc'=0) & (k'=k+1);
  [] pc=3 & k<N & a=2 & cte=0 -> 1: (cte'=2) & (pc'=0) & (k'=k+1);
  [] pc=3 & k<N & a=2 & cte=2 -> 1: (cte'=4) & (pc'=0) & (k'=k+1);
  [] pc=3 & k<N & a=2 & cte=4 -> 1: (cte'=-1) & (pc'=5);

  // ---- termination (absorbing) ----
  [] pc=3 & k=N -> 1: (pc'=4);
  [] pc=4 -> true;
  [] pc=5 -> true;

endmodule
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc_csv", default="results_cte/mc_val.csv")
    ap.add_argument("--pct", type=float, default=80.0,
                    help="std percentile for the guard threshold (80 -> beta~0.80)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="absolute std threshold; overrides --pct (use the "
                         "ID-calibrated value for shift studies)")
    ap.add_argument("--M", type=int, default=10,
                    help="consecutive failed checks before abort (paper: 10)")
    ap.add_argument("--out_dir", default="results_cte")
    args = ap.parse_args()

    trues, means, stds = [], [], []
    with open(args.mc_csv) as f:
        for row in csv.DictReader(f):
            trues.append(float(row["cte_true"]))
            means.append(float(row["mc_mean"]))
            stds.append(float(row["mc_std"]))
    trues, means, stds = map(np.array, (trues, means, stds))

    if args.threshold is not None:
        threshold = float(args.threshold)
        label = "ID-calibrated"
    else:
        threshold = float(np.percentile(stds, args.pct))
        label = f"{args.pct:.0f}th pct of this file"
    passing = stds <= threshold
    beta = float(passing.mean())
    print(f"Guard threshold: std <= {threshold:.4f}  ({label})")
    print(f"beta (pass rate): {beta:.4f}   (paper: 0.821)")

    t_states = np.array([bin_cte(x) for x in trues])
    p_states = np.array([bin_cte(x) for x in np.clip(means, -ROAD_EDGE, ROAD_EDGE)])
    onroad = t_states != -1

    acc_all = float((t_states == p_states)[onroad].mean())
    acc_guarded = float((t_states == p_states)[passing & onroad].mean())
    print(f"On-road discretized accuracy: all inputs {acc_all*100:.2f}%  ->  "
          f"passing inputs {acc_guarded*100:.2f}%")

    # guarded alpha from PASSING on-road samples only
    i5 = {s: j for j, s in enumerate(ON_ROAD)}
    counts = np.zeros((5, 5), dtype=int)
    for t, p in zip(t_states[passing & onroad], p_states[passing & onroad]):
        counts[i5[t], i5[p]] += 1
    probs = np.zeros((5, 5))
    print("\nGuarded alpha (row-normalized):")
    print("        " + "".join(f"{STATE_NAMES[s]:>9s}" for s in ON_ROAD))
    for s in ON_ROAD:
        r = counts[i5[s]]
        if r.sum() == 0:
            print(f"  WARNING: state {s} ({STATE_NAMES[s]}) has no passing samples; "
                  f"using identity row.")
            probs[i5[s], i5[s]] = 1.0
        else:
            probs[i5[s]] = r / r.sum()
        print(f"{STATE_NAMES[s]:>7s} " + "".join(f"{v:9.3f}" for v in probs[i5[s]])
              + f"   (n={r.sum()})")

    # generate m2 model
    beta_r = round(beta, 4)
    beta_c = round(1.0 - beta_r, 4)
    perception = perception_lines(probs.tolist(), ON_ROAD)
    model = (M2_TEMPLATE
             .replace("{BETA}", f"{beta_r}")
             .replace("{BETAC}", f"{beta_c}")
             .replace("{M}", str(args.M))
             .replace("{PERCEPTION}", perception))

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "lanekeep_m2.prism")
    with open(model_path, "w") as f:
        f.write(model)
    with open(os.path.join(args.out_dir, "lanekeep_m2_offroad.props"), "w") as f:
        f.write("// Property 1 with guard: probability the car ever leaves the road\n")
        f.write("P=? [ F cte=-1 ]\n")
    with open(os.path.join(args.out_dir, "lanekeep_m2_abort.props"), "w") as f:
        f.write("// Property 3: probability the guard aborts to the fail-safe (handover)\n")
        f.write("P=? [ F i=M ]\n")
    with open(os.path.join(args.out_dir, "guarded_alpha.json"), "w") as f:
        json.dump({"threshold": threshold, "percentile": args.pct, "beta": beta,
                   "M": args.M, "acc_all": acc_all, "acc_guarded": acc_guarded,
                   "state_order": ON_ROAD, "guarded_counts": counts.tolist(),
                   "guarded_probs": probs.tolist()}, f, indent=2)

    print(f"\nm2 model  -> {model_path}")
    print(f"props     -> lanekeep_m2_offroad.props, lanekeep_m2_abort.props")
    print(f"json      -> guarded_alpha.json")


if __name__ == "__main__":
    main()
