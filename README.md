# Vision-based lane-keeping: perception model, run-time guard, and closed-loop verification

Code for a safety analysis of a CNN perception component in a simulated lane-keeping system
(`gym-donkeycar`). The pipeline trains a CTE regression model, abstracts it into a probabilistic
perception model, adds an MC-Dropout run-time guard with handover to a fail-safe controller,
verifies the closed loop in PRISM, and evaluates all of it under distribution shift.

Results are not documented here. Running the pipeline generates
[`results_cte/report.md`](results_cte/report.md) with all numbers, tables and figures.

---

## Requirements

```powershell
conda env create -f environment.yml
conda activate av
```

Two external dependencies not installable via conda:

- **Donkey simulator** — the Unity build must be running and listening before any collection or
  driving script starts. Environments used: `donkey-generated-track-v0`,
  `donkey-generated-roads-v0`.
- **PRISM** — must be installed and on `PATH` for `run_prism.py`.

---

## Repository layout

```
av/
  README.md
  environment.yml

  cte_dataset.py            # dataset loader; defines bin_cte()
  cte_model.py              # dropout CNN, CTE regression
  collect_cte_data.py       # simulator -> frames + cte_log.csv
  train_cte.py              # training loop

  confusion_cte.py          # perception abstraction (alpha)
  mc_uncertainty.py         # MC-Dropout std distribution, threshold selection
  guarded_model.py          # guarded abstraction (beta)

  shift_eval.py             # alpha / beta re-estimated on shifted data
  shift_distance.py         # pixel-space vs feature-space distance

  make_prism_model.py       # DTMC + property files
  run_prism.py              # sweeps over N
  bootstrap_ci.py           # confidence intervals over PRISM outputs

  donkey_drive_cte.py       # closed loop, no guard
  donkey_drive_guarded.py   # closed loop with bidirectional handover
  donkey_manual_test.py     # manual-control sanity check

  make_report.py            # assembles report.md + figures from results_cte/

  data_cte/                 # collected frames, one folder per track (gitignored)
  results_cte/              # all outputs
    figures/
    shift/
```

### Shared state discretisation

`bin_cte()` in `cte_dataset.py` is imported by everything that needs to discretise CTE. It is
defined in one place so training, evaluation and the confusion matrices cannot drift apart. Any
change to the bin edges invalidates every downstream artefact and requires re-running from
step 3.

| State | Meaning | CTE range (m) |
|---|---|---|
| `3` | far-left | [−2.0, −1.4] |
| `1` | near-left | [−1.4, −0.8] |
| `0` | lane / centre | [−0.8, 0] |
| `2` | near-right | [0, 0.8] |
| `4` | far-right | [0.8, 2.0] |
| `−1` | error (off-road) | \|CTE\| > 2.0 |

---

## Running the pipeline

Stages are ordered by dependency. Each writes into `results_cte/` and can be re-run on its own
provided its inputs exist.

### 1. Collect data

```powershell
python collect_cte_data.py --track generated_track --env donkey-generated-track-v0 --laps 10
python collect_cte_data.py --track generated_road  --env donkey-generated-roads-v0 --laps 5
```

Drives the ground-truth autopilot and logs frames plus true CTE to `data_cte/<track>/`. The
first track is used for training; the second is evaluation-only.

### 2. Train

```powershell
python train_cte.py
```
→ `cte_model.pth`, `cte_history.json`

### 3. Perception abstraction

```powershell
python confusion_cte.py
```
→ `confusion_cte.json`

### 4. Run-time guard

```powershell
python mc_uncertainty.py
python guarded_model.py
```
→ `mc_val.csv`, `guarded_alpha.json`

`mc_uncertainty.py` runs M stochastic forward passes per image and selects the uncertainty
threshold from a percentile of the in-distribution validation distribution. `guarded_model.py`
rebuilds the abstraction over passing inputs only.

### 5. Distribution shift

```powershell
python shift_eval.py --data_dir data_cte/generated_road
python shift_distance.py
```
→ `confusion_shift.json`, `mc_shift.csv`, `shift_distance.json`

### 6. Formal verification

```powershell
mkdir results_cte\shift
python make_prism_model.py --json results_cte/confusion_cte.json   --out_dir results_cte
python make_prism_model.py --json results_cte/confusion_shift.json --out_dir results_cte/shift
python run_prism.py
```
→ `lanekeep.prism`, `lanekeep_m2.prism`, `*.props`, `prism_results.csv`,
`prism_m2_offroad.csv`, `prism_m2_abort.csv`, `prism_perception.txt`

`make_prism_model.py` does not create `--out_dir`; make it first.

### 7. Confidence intervals

```powershell
python bootstrap_ci.py
```
→ `bootstrap_m1_N30.json`, `bootstrap_m2_N30.json`, `bootstrap_m1_shift_N30.json`,
each with a `_samples.csv`

### 8. Closed-loop driving

```powershell
python donkey_drive_cte.py
python donkey_drive_guarded.py
```
→ `drive_<track>_log.csv`, `guarded_drive_<track>_log.csv`

Requires the simulator running. The guarded script logs per-frame uncertainty, which controller
holds control, and every handover and hand-back.

### 9. Report

```powershell
python make_report.py
```
→ `report.md`, `figures/`

Reads whatever result files are present and regenerates the report from scratch. Safe to re-run
after any single stage.

---

## Notes

- `cte_model.pth` (~1.5 MB) is committed, so stages 3–7 can be reproduced without retraining or
  running the simulator. Stages 1 and 8 need the simulator.
- `data_cte/` is gitignored — regenerate with stage 1.
- Procedurally generated tracks regenerate on each load unless a seed is fixed. Dataset-level
  results are unaffected, but live cross-session comparisons need `useSeed` set.

## Related repository

`av-attention-study` — the steering-angle architecture comparison (PilotNet, ConvNeXt,
EfficientNetV2, ViT, Swin) on the Udacity behavioural-cloning dataset.
