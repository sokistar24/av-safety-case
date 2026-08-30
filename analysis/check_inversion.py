"""
check_inversion.py

Section 4 reports that above the 85th percentile the inputs the guard REJECTS are
more accurate than the ones it admits - the opposite of the guard premise. This
script tests the obvious explanation: that the highest-uncertainty inputs
concentrate in wide state bins, where even a large regression error still
discretises to the correct state.

For each candidate threshold it reports, for passing and failing inputs
separately: the distribution over true states, the discretised accuracy, and the
mean absolute regression error. If the explanation holds, failing inputs at high
percentiles should sit disproportionately in the wide central bin and carry a
LARGER regression error while still scoring a HIGHER discretised accuracy.

Run from av/:
    python check_inversion.py
"""

import numpy as np
import pandas as pd

from cte_dataset import bin_cte, ROAD_EDGE, STATE_NAMES

ON_ROAD = [3, 1, 0, 2, 4]
WIDTHS = {3: 0.6, 1: 0.6, 0: 0.8, 2: 0.8, 4: 1.2}   # bin width in metres


def main():
    d = pd.read_csv("results_cte/mc_val.csv")
    trues = d["cte_true"].to_numpy()
    means = d["mc_mean"].to_numpy()
    stds = d["mc_std"].to_numpy()

    t_state = np.array([bin_cte(x) for x in trues])
    p_state = np.array([bin_cte(x) for x in np.clip(means, -ROAD_EDGE, ROAD_EDGE)])
    onroad = t_state != -1
    correct = t_state == p_state
    abserr = np.abs(means - trues)

    print(f"{len(d)} validation samples, {onroad.sum()} on-road\n")
    print("Bin widths: " + ", ".join(
        f"{STATE_NAMES[s]} {WIDTHS[s]}m" for s in ON_ROAD) + "\n")

    for pct in (60, 70, 80, 85, 90, 95):
        th = float(np.percentile(stds, pct))
        passing = stds <= th
        pm, fm = passing & onroad, (~passing) & onroad

        print(f"--- {pct}th percentile, threshold {th:.4f} "
              f"({pm.sum()} pass / {fm.sum()} fail, on-road) ---")
        print(f"{'':10s} {'pass %':>8s} {'fail %':>8s} {'acc pass':>9s} {'acc fail':>9s}")
        for s in ON_ROAD:
            sp, sf = pm & (t_state == s), fm & (t_state == s)
            fp = sp.sum() / pm.sum() * 100 if pm.sum() else 0
            ff = sf.sum() / fm.sum() * 100 if fm.sum() else 0
            ap = correct[sp].mean() * 100 if sp.sum() else float("nan")
            af = correct[sf].mean() * 100 if sf.sum() else float("nan")
            print(f"{STATE_NAMES[s]:>10s} {fp:>7.1f}% {ff:>7.1f}% "
                  f"{ap:>8.1f}% {af:>8.1f}%")
        print(f"{'TOTAL':>10s} {'':>8s} {'':>8s} "
              f"{correct[pm].mean()*100:>8.1f}% {correct[fm].mean()*100:>8.1f}%")
        print(f"{'mean |err|':>10s} {'':>8s} {'':>8s} "
              f"{abserr[pm].mean():>8.3f}  {abserr[fm].mean():>8.3f}\n")

    print("Reading it: if the inversion is a bin-width artefact, the failing set at")
    print("90-95 should be dominated by the wide central bin and show a LARGER mean")
    print("regression error alongside its HIGHER discretised accuracy.")


if __name__ == "__main__":
    main()
