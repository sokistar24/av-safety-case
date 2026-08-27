"""
make_report.py
Collect every result in results_cte/ into the paper's figures and a structured
results report. Robust to missing pieces: anything not yet generated is listed
as PENDING with the command that produces it.

Reads (whatever exists):
  cte_history.json, confusion_cte.json, guarded_alpha.json, shift_distance.json,
  confusion_shift.json, mc_val.csv, mc_shift.csv,
  prism_results.csv, prism_m2_offroad.csv, prism_m2_abort.csv,
  shift/prism_m1_shift.csv, shift/prism_m2_offroad_shift.csv, shift/prism_m2_abort_shift.csv,
  guarded_drive_*_log.csv, drive_*_log.csv

Writes:
  results_cte/figures/*.png
  results_cte/report.md

Run:  python make_report.py
"""

import os
import csv
import json
import glob
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results_cte"
STATE_NAMES = {0: "LANE", 1: "near-L", 3: "far-L", 2: "near-R", 4: "far-R"}
ORDER = [3, 1, 0, 2, 4]


def jload(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def csvload(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def md_alpha_table(probs, counts=None):
    hdr = "| true \\ est | " + " | ".join(STATE_NAMES[s] for s in ORDER) + " | n |"
    sep = "|" + "---|" * (len(ORDER) + 2)
    lines = [hdr, sep]
    for i, s in enumerate(ORDER):
        n = int(sum(counts[i])) if counts is not None else ""
        row = " | ".join(f"{probs[i][j]:.3f}" for j in range(len(ORDER)))
        lines.append(f"| {STATE_NAMES[s]} | {row} | {n} |")
    return "\n".join(lines)


def parse_drive_log(path):
    rows = csvload(path)
    if not rows:
        return None
    modes = [r["mode"] for r in rows]
    cte = np.array([float(r["cte_true"]) for r in rows])
    cert = np.array([int(r["certified"]) for r in rows]) if "certified" in rows[0] else None
    handovers = sum(1 for a, b in zip(modes, modes[1:]) if a == "MODEL" and b == "FALLBACK")
    handbacks = sum(1 for a, b in zip(modes, modes[1:]) if a == "FALLBACK" and b == "MODEL")
    out = {"frames": len(rows), "handovers": handovers, "handbacks": handbacks}
    for m in ("MODEL", "FALLBACK"):
        idx = np.array([i for i, x in enumerate(modes) if x == m])
        if len(idx):
            out[m] = {"n": int(len(idx)),
                      "mean_abs_cte": float(np.abs(cte[idx]).mean()),
                      "offroad_pct": float((np.abs(cte[idx]) > 2.0).mean() * 100),
                      "certified_pct": float(cert[idx].mean() * 100) if cert is not None else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(R, "report.md"))
    args = ap.parse_args()
    figdir = os.path.join(R, "figures")
    os.makedirs(figdir, exist_ok=True)
    pend = []
    S = []  # report lines

    S.append("# Closed-loop safety analysis of a vision-based lane-keeping system\n")
    S.append("Auto-generated results report. Figures in `results_cte/figures/`.\n")

    # ---------- 1. training ----------
    hist = jload(f"{R}/cte_history.json")
    S.append("## 1. Perception model (in-distribution)\n")
    if hist:
        S.append(f"- Best val MAE: **{hist['best_val_mae']:.4f}** "
                 f"(TaxiNet reference: 1.185 on a ±8 m range)")
        S.append(f"- Final discretized state accuracy: "
                 f"**{hist['val_disc_acc'][-1]*100:.1f}%**\n")
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        ax[0].plot(hist["train_loss"], label="train loss")
        ax[0].plot(hist["val_mse"], label="val MSE")
        ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].set_title("Loss")
        ax[1].plot(hist["val_mae"], label="val MAE")
        ax[1].plot(hist["val_disc_acc"], label="disc. accuracy")
        ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].set_title("Validation")
        fig.tight_layout(); fig.savefig(f"{figdir}/fig_training.png", dpi=140)
        plt.close(fig)
        S.append("![](figures/fig_training.png)\n")
    else:
        pend.append(("cte_history.json", "python train_cte.py"))

    # ---------- 2. alpha (ID) ----------
    conf = jload(f"{R}/confusion_cte.json")
    if conf:
        S.append("## 2. Perception abstraction α (in-distribution)\n")
        S.append(f"- On-road discretized accuracy: "
                 f"**{conf['on_road_disc_accuracy']*100:.2f}%** "
                 f"({conf['on_road_samples']} held-out samples)\n")
        S.append(md_alpha_table(conf["alpha_probs"], conf["alpha_counts"]) + "\n")
    else:
        pend.append(("confusion_cte.json", "python confusion_cte.py"))

    # ---------- 3. guard ----------
    g = jload(f"{R}/guarded_alpha.json")
    if g:
        S.append("## 3. MC-Dropout run-time guard (in-distribution calibration)\n")
        S.append(f"- Threshold: std ≤ **{g['threshold']:.4f}** "
                 f"({g.get('percentile','?')}th pct of val stds), "
                 f"β = **{g['beta']:.3f}** (paper: 0.821), M = {g.get('M',10)}")
        S.append(f"- Accuracy all inputs {g['acc_all']*100:.2f}% → passing inputs "
                 f"{g['acc_guarded']*100:.2f}%\n")
        S.append("Guarded α (passing inputs only):\n")
        S.append(md_alpha_table(g["guarded_probs"], g["guarded_counts"]) + "\n")
    else:
        pend.append(("guarded_alpha.json", "python guarded_model.py --pct 80"))

    # ---------- 4. MC std distributions ----------
    mcv, mcs = csvload(f"{R}/mc_val.csv"), csvload(f"{R}/mc_shift.csv")
    if mcv:
        sv = np.array([float(r["mc_std"]) for r in mcv])
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.hist(sv, bins=50, alpha=0.6, density=True, label="in-distribution")
        if mcs:
            ss = np.array([float(r["mc_std"]) for r in mcs])
            ax.hist(ss, bins=50, alpha=0.6, density=True, label="shifted")
        if g:
            ax.axvline(g["threshold"], color="k", ls="--", label="guard threshold")
        ax.set_xlabel("MC-dropout std"); ax.set_ylabel("density"); ax.legend()
        fig.tight_layout(); fig.savefig(f"{figdir}/fig_mcstd.png", dpi=140)
        plt.close(fig)
        S.append("## 4. Uncertainty under shift\n")
        line = f"- MC std median: ID {np.median(sv):.3f}"
        if mcs:
            line += f" → shifted {np.median(ss):.3f}"
        S.append(line + "\n\n![](figures/fig_mcstd.png)\n")

    # ---------- 5. shift measurements ----------
    cs = jload(f"{R}/confusion_shift.json")
    sd = jload(f"{R}/shift_distance.json")
    S.append("## 5. Distribution shift\n")
    if cs:
        S.append(f"- Shifted on-road accuracy: **{cs['on_road_disc_accuracy']*100:.2f}%** "
                 f"(vs ~81% ID); shifted MAE **{cs.get('shifted_mae', float('nan')):.2f}** "
                 f"(vs 0.19 ID); β under ID threshold: **{cs.get('beta_shift', float('nan')):.3f}**\n")
        S.append("Shifted α:\n")
        S.append(md_alpha_table(cs["alpha_probs"], cs["alpha_counts"]) + "\n")
    else:
        pend.append(("confusion_shift.json",
                     "python shift_eval.py --data_dir data_cte/generated_road"))
    if sd:
        S.append(f"- Pixel-level JS divergence: **{sd['pixel_js_bits']:.3f}** bits "
                 f"(low: the tracks look similar)")
        m = sd["mahalanobis"]
        S.append(f"- Feature-space Mahalanobis: median {m['id_median']:.2f} → "
                 f"{m['shift_median']:.2f}; **AUROC {m['auroc']:.3f}** as OOD detector")
        S.append(f"- Feature-space symmetrized KL: **{sd['feature_sym_kl_nats']:.1f} nats**")
        if sd.get("corr_maha_mcstd"):
            c = sd["corr_maha_mcstd"]
            S.append(f"- corr(Mahalanobis, MC std): "
                     + ", ".join(f"{k} {v:.2f}" for k, v in c.items()))
        S.append("")
    else:
        pend.append(("shift_distance.json",
                     "python shift_distance.py --shift_dir data_cte/generated_road"))

    # ---------- 6. PRISM curves ----------
    curves = {
        "m1 baseline (off-road)": f"{R}/prism_results.csv",
        "m2 baseline (off-road)": f"{R}/prism_m2_offroad.csv",
        "m2 baseline (abort)": f"{R}/prism_m2_abort.csv",
        "m1 shifted (off-road)": f"{R}/shift/prism_m1_shift.csv",
        "m2 shifted (off-road)": f"{R}/shift/prism_m2_offroad_shift.csv",
        "m2 shifted (abort)": f"{R}/shift/prism_m2_abort_shift.csv",
    }
    have = {}
    for name, path in curves.items():
        rows = csvload(path)
        if rows:
            key = "p_offroad" if "p_offroad" in rows[0] else list(rows[0].keys())[1]
            have[name] = ([int(r["N"]) for r in rows], [float(r[key]) for r in rows])
        else:
            pend.append((os.path.relpath(path, R), "run_prism.py sweep (see report §6)"))
    S.append("## 6. Formal safety (PRISM)\n")
    if have:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for name, (ns, ps) in have.items():
            style = "--" if "abort" in name else "-"
            ax.plot(ns, [max(p, 1e-12) for p in ps], style, marker="o", ms=3, label=name)
        ax.set_yscale("log"); ax.set_xlabel("horizon N (control steps)")
        ax.set_ylabel("probability"); ax.legend(fontsize=7)
        ax.set_title("P[F cte=-1] and P[F abort] vs horizon")
        fig.tight_layout(); fig.savefig(f"{figdir}/fig_prism.png", dpi=140)
        plt.close(fig)
        S.append("![](figures/fig_prism.png)\n")
        rows = []
        for name, (ns, ps) in have.items():
            if 30 in ns:
                rows.append(f"| {name} | {ps[ns.index(30)]:.3g} |")
        if rows:
            S.append("| curve | value at N=30 |\n|---|---|")
            S.extend(rows)
            S.append("")
        S.append("Note: probabilities of exactly 0 are zero-observation point estimates "
                 "(no such misperception among the passing samples), bounded by sample "
                 "size rather than proven — the paper's Appendix-E caveat.\n")

    # ---- 6b. bootstrap confidence intervals ----
    boots = [b for b in sorted(glob.glob(f"{R}/bootstrap_*_N*.json")) if "samples" not in b]
    if boots:
        S.append("### 6b. Bootstrap confidence intervals\n")
        S.append("PRISM computes an exact answer for the alpha it is given, but alpha is "
                 "estimated from finite counts (the far-L row rests on ~200 samples, far-R on "
                 "~96, and a single misperception carries most of the off-road risk). Each row "
                 "was resampled from a Dirichlet posterior with Jeffreys' prior and re-checked "
                 "in PRISM; percentiles give the intervals below. Zero-count cells receive "
                 "small positive mass rather than being asserted impossible.\n")
        S.append("| model | N | point | median | 90% CI | 95% CI |")
        S.append("|---|---|---|---|---|---|")
        for b in boots:
            d = jload(b)
            if not d:
                continue
            name = os.path.basename(b).replace("bootstrap_", "").replace(".json", "")
            def _f(v):
                # keep precision for values very close to 1 (e.g. 0.999999)
                return f"{v:.6f}" if v > 0.99 else f"{v:.3g}"
            S.append(f"| {name} | {d['N']} | {_f(d['point'])} | {_f(d['median'])} | "
                     f"[{_f(d['ci90'][0])}, {_f(d['ci90'][1])}] | "
                     f"[{_f(d['ci95'][0])}, {_f(d['ci95'][1])}] |")
        S.append("")
        S.append("Bootstrap medians sit above the plug-in point estimates because the map from "
                 "alpha to the reaching probability is convex in the small critical cells "
                 "(Jensen), and Jeffreys' prior adds half a count to rare cells - expected, "
                 "not an error.\n")
        S.append("**Interpretation.** In-distribution, the m1 and m2 intervals overlap almost "
                 "entirely: with ~2,000 validation samples the guard's effect on off-road "
                 "probability is *not resolvable*, and the point-estimate contrast (3.5e-5 vs 0) "
                 "was within sampling noise. Under shift the unguarded estimate is both extreme "
                 "and tight, because that alpha rests on thousands of samples with large "
                 "probabilities. The supported claim is therefore: in-distribution the guard is "
                 "neither needed nor harmful; under shift the unguarded system fails with "
                 "probability ~1 while the guarded system hands over.\n")
        S.append("**Why bootstrap rather than FACT.** The reference study obtains formal "
                 "(1-delta) intervals with FACT via parametric model checking over the raw "
                 "counts. The authors report it scaling poorly - no results beyond N=4 within a "
                 "two-hour timeout, with zero-observation transitions dropped to make it "
                 "tractable. The bootstrap reuses the existing PRISM toolchain, has no horizon "
                 "ceiling, and handles zero cells naturally, at the cost of being empirical "
                 "rather than formally guaranteed. FACT remains the more rigorous option for "
                 "authors who want it.\n")

    # ---------- 7. live closed-loop runs ----------
    S.append("## 7. Live closed-loop runs\n")
    logs = sorted(glob.glob(f"{R}/guarded_drive_*_log.csv"))
    if logs:
        S.append("| run | frames | handovers | hand-backs | MODEL: n / cert% / mean\\|cte\\| / off-road% | FALLBACK: n / mean\\|cte\\| / off-road% |")
        S.append("|---|---|---|---|---|---|")
        for p in logs:
            d = parse_drive_log(p)
            name = os.path.basename(p).replace("guarded_drive_", "").replace("_log.csv", "")
            mm = d.get("MODEL"); ff = d.get("FALLBACK")
            mstr = (f"{mm['n']} / {mm['certified_pct']:.0f}% / {mm['mean_abs_cte']:.2f} / "
                    f"{mm['offroad_pct']:.1f}%") if mm else "—"
            fstr = (f"{ff['n']} / {ff['mean_abs_cte']:.2f} / {ff['offroad_pct']:.1f}%") if ff else "—"
            S.append(f"| {name} | {d['frames']} | {d['handovers']} | {d['handbacks']} | {mstr} | {fstr} |")
        S.append("")
    else:
        pend.append(("guarded_drive_*_log.csv", "python donkey_drive_guarded.py ..."))

    # ---------- 8. limitations & further work ----------
    S.append("## 8. Limitations and further work\n")
    S.append("- Zero-observation transitions: measured 0-probabilities are point "
             "estimates bounded by sample size; section 6b attaches bootstrap intervals, "
             "and FACT remains the formal alternative.")
    S.append("- Interval asymmetry: the *safe* in-distribution estimates carry wide "
             "intervals (thin far-L/far-R rows) while the *unsafe* shifted estimate is "
             "precise. Reassuring safety claims need more data - specifically more samples "
             "in the extreme lane states, not more data overall.")
    S.append("- Deterministic dynamics in the DTMC isolate perception-induced risk; "
             "environmental drift (curves) is unmodeled.")
    S.append("- Procedurally generated tracks regenerate per load unless useSeed is set; "
             "live cross-session comparisons on generated tracks require a fixed seed. "
             "Dataset-level results are unaffected.")
    S.append("- Augmentation ablation under shift: training already used flip + "
             "brightness jitter and still failed on the shifted instance; a heavier "
             "recipe (rotation, shadows, stronger photometric jitter) should narrow — "
             "but cannot close — the gap. Measure how much, and how the guard "
             "fire-rate responds.")
    S.append("- Static extreme-shift dose (mini_monaco), transient shift via randomLight "
             "(the live hand-back demonstration), and resumption-safety modeling for "
             "the bidirectional guard.\n")

    if pend:
        S.append("## Pending artifacts\n")
        for name, cmd in pend:
            S.append(f"- `{name}` — generate with: `{cmd}`")
        S.append("")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(S))
    print(f"Report -> {args.out}")
    print(f"Figures -> {figdir}/")
    if pend:
        print(f"Pending items listed in the report: {len(pend)}")


if __name__ == "__main__":
    main()
