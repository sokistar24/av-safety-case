import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
THR = 0.2849
sets = [("in-distribution", "results_cte/mc_val.csv"),
        ("light",  "results_cte/cond1_light/mc_shift.csv"),
        ("road",   "results_cte/generated_road/mc_shift.csv")]
fig, ax = plt.subplots(figsize=(6.2, 3.0))
for label, path in sets:
    s = pd.read_csv(path)["mc_std"].to_numpy()
    ax.hist(s, bins=60, range=(0, 1.6), density=True, alpha=0.55, label=label)
ax.axvline(THR, color="k", ls="--", lw=1.2, label="guard threshold")
ax.set_xlabel("MC-dropout standard deviation"); ax.set_ylabel("density")
ax.legend(frameon=False); fig.tight_layout()
fig.savefig("paper_figures/fig_mcstd_three.pdf")
print("written paper_figures/fig_mcstd_three.pdf")
