import numpy as np, pandas as pd
for c in ["cond1_trees","cond1_light","cond1_cones","generated_road"]:
    d = pd.read_csv(f"results_cte/{c}/mc_shift.csv")
    print(f"{c:16s} pred sd {d['mc_mean'].std():.3f}  "
          f"range [{d['mc_mean'].min():+.2f}, {d['mc_mean'].max():+.2f}]  "
          f"mc_std med {d['mc_std'].median():.3f}")
d = pd.read_csv("results_cte/mc_val.csv")
print(f"{'in-distribution':16s} pred sd {d['mc_mean'].std():.3f}  "
      f"range [{d['mc_mean'].min():+.2f}, {d['mc_mean'].max():+.2f}]  "
      f"mc_std med {d['mc_std'].median():.3f}")
