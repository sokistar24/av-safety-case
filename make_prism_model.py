"""
make_prism_model.py
Generate the complete PRISM DTMC for the lane-keeping closed loop from the
measured perception abstraction (confusion_cte.json).

Mirrors the paper's Appendix B.1 model m1 (no run-time guard), CTE-only:

  states:  cte true state 0..4 (0 LANE, 1 near-L, 3 far-L, 2 near-R, 4 far-R),
           -1 = off-road error (absorbing)
  loop:    pc=0 guard (certify all, v=1)  ->  pc=1 perception (ALPHA, measured)
           -> pc=2 controller -> pc=3 dynamics -> back to pc=0, k+1
  controller (deterministic, our discretized P-controller):
           est=0 -> a=0 straight ; est in {1,3} -> a=2 steer-right ;
           est in {2,4} -> a=1 steer-left
  dynamics (deterministic; stochasticity comes ONLY from perception, as in paper):
           a=0 stays; a=1 moves one state left  (4->2->0->1->3->ERROR)
                      a=2 moves one state right (3->1->0->2->4->ERROR)
  horizon: const int N steps; property  P=? [ F cte=-1 ]  (paper Property 1)

Usage:
    python make_prism_model.py                       # reads results_cte/confusion_cte.json
Outputs:
    results_cte/lanekeep.prism    the model
    results_cte/lanekeep.props    the property file
"""

import os
import json
import argparse
import numpy as np

ON_ROAD = [3, 1, 0, 2, 4]


def perception_lines(probs, order):
    """Rows -> PRISM guarded commands; rounded, residual-fixed to sum to 1."""
    i5 = {s: i for i, s in enumerate(order)}
    lines = []
    for s in order:
        r = np.array(probs[i5[s]], dtype=float)
        nz = [(order[j], r[j]) for j in range(len(order)) if r[j] > 0]
        vals = [round(v, 4) for _, v in nz]
        resid = round(1.0 - sum(vals), 4)
        k = int(np.argmax(vals))
        vals[k] = round(vals[k] + resid, 4)
        terms = " + ".join(f"{v}: (cte_est'={st}) & (pc'=2)"
                           for (st, _), v in zip(nz, vals))
        lines.append(f"  [] cte={s} & v=1 & pc=1 -> {terms};")
    return "\n".join(lines)


MODEL_TEMPLATE = """// Auto-generated lane-keeping DTMC (paper Appendix B.1 structure, CTE-only)
// Perception probabilities measured from the trained model's confusion matrix.
dtmc

const int N; // finite time horizon (steps)

module lanekeep

  // true state: 0 LANE, 1 near-left, 3 far-left, 2 near-right, 4 far-right, -1 off-road
  cte : [-1..4] init 0;
  cte_est : [0..4] init 0;
  v : [0..1] init 1;        // run-time check result (m1: always certified)
  a : [0..2] init 0;        // 0 straight, 1 steer-left, 2 steer-right
  pc : [0..5] init 0;       // 0 guard, 1 perception, 2 controller, 3 dynamics, 4 done, 5 error-halt
  k : [1..N] init 1;

  // run-time guard placeholder (m1: certify every input)
  [] pc=0 -> 1: (v'=1) & (pc'=1);

  // ---- perception: measured probabilistic abstraction (alpha) ----
{PERCEPTION}

  // ---- controller: discretized P-controller ----
  [] cte_est=0 & pc=2                 -> 1: (a'=0) & (pc'=3);
  [] (cte_est=1 | cte_est=3) & pc=2   -> 1: (a'=2) & (pc'=3);
  [] (cte_est=2 | cte_est=4) & pc=2   -> 1: (a'=1) & (pc'=3);

  // ---- dynamics (deterministic; one state per step) ----
  // a=0: hold position
  [] pc=3 & k<N & a=0 -> 1: (pc'=0) & (k'=k+1);
  // a=1: steer LEFT   (4->2->0->1->3->ERROR)
  [] pc=3 & k<N & a=1 & cte=4 -> 1: (cte'=2) & (pc'=0) & (k'=k+1);
  [] pc=3 & k<N & a=1 & cte=2 -> 1: (cte'=0) & (pc'=0) & (k'=k+1);
  [] pc=3 & k<N & a=1 & cte=0 -> 1: (cte'=1) & (pc'=0) & (k'=k+1);
  [] pc=3 & k<N & a=1 & cte=1 -> 1: (cte'=3) & (pc'=0) & (k'=k+1);
  [] pc=3 & k<N & a=1 & cte=3 -> 1: (cte'=-1) & (pc'=5);          // off-road LEFT
  // a=2: steer RIGHT  (3->1->0->2->4->ERROR)
  [] pc=3 & k<N & a=2 & cte=3 -> 1: (cte'=1) & (pc'=0) & (k'=k+1);
  [] pc=3 & k<N & a=2 & cte=1 -> 1: (cte'=0) & (pc'=0) & (k'=k+1);
  [] pc=3 & k<N & a=2 & cte=0 -> 1: (cte'=2) & (pc'=0) & (k'=k+1);
  [] pc=3 & k<N & a=2 & cte=2 -> 1: (cte'=4) & (pc'=0) & (k'=k+1);
  [] pc=3 & k<N & a=2 & cte=4 -> 1: (cte'=-1) & (pc'=5);          // off-road RIGHT

  // ---- termination (absorbing, avoids deadlocks) ----
  [] pc=3 & k=N -> 1: (pc'=4);   // horizon reached, safe halt
  [] pc=4 -> true;
  [] pc=5 -> true;               // error halt (cte stays -1)

endmodule
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results_cte/confusion_cte.json")
    ap.add_argument("--out_dir", default="results_cte")
    args = ap.parse_args()

    with open(args.json) as f:
        data = json.load(f)
    assert data["state_order"] == ON_ROAD, "state order mismatch"

    os.makedirs(args.out_dir, exist_ok=True)

    perception = perception_lines(data["alpha_probs"], ON_ROAD)
    model = MODEL_TEMPLATE.replace("{PERCEPTION}", perception)

    model_path = os.path.join(args.out_dir, "lanekeep.prism")
    with open(model_path, "w") as f:
        f.write(model)

    props_path = os.path.join(args.out_dir, "lanekeep.props")
    with open(props_path, "w") as f:
        f.write('// Property 1 (paper): probability the car ever leaves the road\n')
        f.write('P=? [ F cte=-1 ]\n')

    print(f"PRISM model  -> {model_path}")
    print(f"Properties   -> {props_path}")
    print("\nRun in PRISM (CLI):")
    print("  prism results_cte/lanekeep.prism results_cte/lanekeep.props -const N=30")
    print("Or sweep the horizon like the paper's Fig. 3:")
    print("  prism results_cte/lanekeep.prism results_cte/lanekeep.props -const N=5:5:50")


if __name__ == "__main__":
    main()
