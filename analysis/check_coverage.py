import glob, os
import pandas as pd
from cte_dataset import bin_cte, STATES, STATE_NAMES
for p in sorted(glob.glob("data_cte/*/cte_log.csv")):
    t = os.path.basename(os.path.dirname(p))
    d = pd.read_csv(p)
    s = d["cte"].apply(bin_cte).value_counts()
    err = s.get(-1, 0) / len(d) * 100
    print(f"\n{t}  (n={len(d)}, off-road {err:.1f}%)")
    for st in STATES:
        print(f"  {st:+d} {STATE_NAMES[st]:8s} {s.get(st,0):5d}")
