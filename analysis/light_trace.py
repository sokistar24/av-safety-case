import pandas as pd
d = pd.read_csv("results_cte/guarded_drive_light_pd_log.csv")
d["err"] = d["cte_pred"] - d["cte_true"]
print(d[["time_s","cte_true","cte_pred","err","mc_std","certified","fail_count","mode"]]
      .iloc[::5].to_string(index=False))
print("\nframes certified while |cte_true|>2:",
      int(((d.cte_true.abs()>2) & (d.certified==1)).sum()))
print("first frame |cte_true|>2:", int((d.cte_true.abs()>2).idxmax()))
print("first frame std>thr     :", int((d.mc_std>0.2849).idxmax()))
