# Vision-based lane-keeping: perception model, run-time guard, and closed-loop verification

Code for a safety analysis of a CNN perception component in a simulated lane-keeping system
(`gym-donkeycar`). The pipeline trains a cross-track-error regression model, abstracts it into a
probabilistic perception model, adds an MC-Dropout run-time guard with handover to a fail-safe
controller, verifies the closed loop in PRISM with bootstrap confidence intervals, and evaluates
all of it across several operating conditions.

Results are not documented here. Per-condition results, the provenance of every dataset and the
simulator settings each was collected under are recorded in
[`conditions.md`](conditions.md).

---

## Requirements

```powershell
conda env create -f environment.yml
conda activate av
```

Two external dependencies not installable via conda:

- **Donkey simulator** — the Unity build must be running and listening before any collection or
  driving script starts. Environments used: `donkey-generated-track-v0`,
  `donkey-generated-roads-v0`, `donkey-minimonaco-track-v0`.
- **PRISM** — installed and reachable; `run_prism.py` invokes it through Java and expects the
  lib path in `--lib` (default `C:\prism-4.10.1\lib`).

---

## Repository layout

```
av/
  README.md
  conditions.md             # dataset provenance, sim settings, per-condition results
  environment.yml

  cte_dataset.py            # dataset loader; defines bin_cte()
  cte_model.py              # dropout CNN, CTE regression
  collect_cte_data.py       # simulator -> frames + cte_log.csv
  train_cte.py              # training loop

  confusion_cte.py          # perception abstraction (alpha)
  mc_uncertainty.py         # MC-Dropout std distribution, threshold selection
  guarded_model.py          # guarded abstraction (beta) + m2 DTMC

  shift_eval.py             # alpha / beta re-estimated on another condition
  shift_distance.py         # pixel-space vs feature-space distance
  condition_summary.py      # assembles the conditions x measures table

  make_prism_model.py       # m1 DTMC + property files
  run_prism.py              # sweeps over the horizon N
  bootstrap_ci.py           # confidence intervals over PRISM outputs

  donkey_drive_cte.py       # closed loop, no guard
  donkey_drive_guarded.py   # closed loop with guard, handover and optional hand-back
  donkey_manual_test.py     # manual-control sanity check

  analysis/                 # one-off checks whose output appears in the paper
    check_coverage.py       # state counts per dataset
    check_inversion.py      # why guard accuracy inverts at high thresholds
    check_uniformity.py     # near-field contrast and left-right asymmetry
    check_collapse.py       # prediction spread per condition
    check_guard_stats.py    # std vs error correlation, threshold table
    light_trace.py          # frame-by-frame trace of a failing run
    grab_condition_frames.py  # one matched frame per condition, for figures
    grab_camera_pair.py     # control/light frame pair matched on cte and steer

  data_cte/<condition>/     # collected frames + cte_log.csv (gitignored)
  results_cte/              # in-distribution outputs
    <condition>/            # per-condition outputs, same file names
    figures/
```

Per-condition results live in their own subdirectory because `shift_eval.py`,
`shift_distance.py` and `guarded_model.py` write to fixed file names. Running a second condition
without `--out_dir` overwrites the first.

### Shared state discretisation

`bin_cte()` in `cte_dataset.py` is imported by everything that discretises CTE. It is defined in
one place so training, evaluation and the confusion matrices cannot drift apart. Changing the bin
edges invalidates every downstream artefact.

| State | Meaning | CTE range (m) |
|---|---|---|
| `3` | far-left | [−2.0, −1.4) |
| `1` | near-left | [−1.4, −0.8) |
| `0` | on target — the nominal driving band | [−0.8, 0.0] |
| `2` | near-right | (0.0, 0.8] |
| `4` | far-right | (0.8, 2.0] |
| `−1` | error (off-road) | \|CTE\| > 2.0 |

CTE is measured from the **road centre line**, and the vehicle drives the left-hand lane, so the
bins are asymmetric about zero by design: state `0` contains the nominal driving position of
roughly −0.5 m, not the geometric centre of the road.

---

## Running the pipeline

### 1. Collect

```powershell
python collect_cte_data.py --track generated_track --env donkey-generated-track-v0 --laps 10
```

Drives the ground-truth autopilot and logs frames plus true CTE to `data_cte/<track>/`.
**Appends** across runs, so reuse a `--track` name to accumulate and use a new one for a new
condition. Record the simulator settings in `conditions.md` at collection time — they are not
captured in the CSV.

### 2. Train

```powershell
python train_cte.py
```
→ `results_cte/cte_model.pth`, `cte_history.json`

### 3. Perception abstraction

```powershell
python confusion_cte.py
```
→ `results_cte/confusion_cte.json`

### 4. Run-time guard

```powershell
python mc_uncertainty.py          # n_mc = 30 stochastic passes per image
python guarded_model.py --pct 80  # threshold at the 80th percentile of val stds
```
→ `mc_val.csv`, `guarded_alpha.json`, `lanekeep_m2.prism`, `lanekeep_m2_*.props`

`mc_uncertainty.py` runs `--n_mc` forward passes with dropout active and tabulates candidate
thresholds. `guarded_model.py` rebuilds the abstraction over passing inputs only and emits the
guarded DTMC. Two distinct parameters are easily confused: `--n_mc` is the number of stochastic
passes (30); `--M` is the number of consecutive failed checks before handover (10).

Rows with fewer than `--min_count` passing samples (default 30) are smoothed towards the Jeffreys
prior rather than reported as measured, and rows with none become uniform. Which rows were
affected is recorded under `row_status` in `guarded_alpha.json`. This matters under severe shift,
where a correctly-firing guard can leave almost nothing to estimate from.

### 5. Evaluate another condition

```powershell
mkdir results_cte\<condition>
python shift_eval.py     --data_dir  data_cte/<condition> --out_dir results_cte/<condition>
python shift_distance.py --shift_dir data_cte/<condition> --out_dir results_cte/<condition> `
                         --shift_mc_csv results_cte/<condition>/mc_shift.csv
```
→ `confusion_shift.json`, `mc_shift.csv`, `shift_distance.json`

`shift_eval.py` reads the guard threshold from the in-distribution `guarded_alpha.json` and does
not recalibrate. That is deliberate: a deployed system carries its commissioning calibration into
conditions it has not met.

Then across all conditions:

```powershell
python condition_summary.py --conditions cond_a cond_b cond_c
```
→ `results_cte/condition_summary.csv` — accuracy, MAE, signed bias, uncertainty, pass rate, and
the three distance measures, one row per condition.

### 6. Formal verification

```powershell
python make_prism_model.py --json results_cte/<condition>/confusion_shift.json `
                           --out_dir results_cte/<condition>
python guarded_model.py --mc_csv results_cte/<condition>/mc_shift.csv `
                        --threshold 0.2849 --out_dir results_cte/<condition>
python run_prism.py --model results_cte/<condition>/lanekeep.prism `
                    --props results_cte/<condition>/lanekeep.props `
                    --out   results_cte/<condition>/prism_m1.csv
```

Pass `--threshold` explicitly for every condition. Omitting it makes `guarded_model.py` pick a
percentile of the condition's own uncertainty distribution, which silently recalibrates the guard
and invalidates the comparison.

`make_prism_model.py` does not create `--out_dir`; make it first.

### 7. Confidence intervals

```powershell
python bootstrap_ci.py --json results_cte/<condition>/confusion_shift.json `
                       --N 30 --n_boot 200 --tag m1_<condition> --out_dir results_cte/<condition>
python bootstrap_ci.py --json results_cte/<condition>/guarded_alpha.json --guarded `
                       --N 30 --n_boot 200 --tag m2_<condition> --out_dir results_cte/<condition>
```
→ `bootstrap_<tag>_N30.json` + `_samples.csv`

Each replicate launches a JVM, so 200 replicates take several minutes per model.

### 8. Closed-loop driving

```powershell
python donkey_drive_guarded.py --env <env-id> --track <name> --n_mc 30 `
       --handback 0 --fallback oracle --fallback_kp 5 --fallback_kd 5 --laps 3 --show
```
→ `results_cte/guarded_drive_<track>_log.csv`

Set `--track` explicitly, or runs on the same environment overwrite each other's logs. Use
`--n_mc 30` to match the calibration. `--handback 0` makes handover absorbing, matching the
verified m2 model; a positive value enables hand-back after that many consecutive passes.
`--fallback_kp` / `--fallback_kd` set the fail-safe's gains independently of the model
controller's; a proportional-only fail-safe oscillates even with perfect state knowledge.

---

## Notes

- Scripts in `analysis/` are one-off checks rather than pipeline stages. They read only from
  `results_cte/` and `data_cte/` and write nothing back, so they can be run in any order. They
  are committed because their output appears in the write-up and would otherwise be
  unreproducible. Run them from the repository root, not from inside `analysis/`.
- `cte_model.pth` (~1.5 MB) is committed, so stages 3–7 can be reproduced without retraining or
  running the simulator. Stages 1 and 8 need the simulator.
- `data_cte/` is gitignored — regenerate with stage 1.
- Procedurally generated tracks regenerate on each load unless `useSeed` is set. Dataset-level
  results are unaffected, but comparisons across conditions require a fixed seed.
- `randomLight` re-randomises per episode, so that condition is not reproducible run to run.

## Related repository

`av-attention-study` — the steering-angle architecture comparison (PilotNet, ConvNeXt,
EfficientNetV2, ViT, Swin) on the Udacity behavioural-cloning dataset.
