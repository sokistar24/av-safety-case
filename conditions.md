# Data collection conditions

Provenance for every folder under `data_cte/`. Sim settings are not captured in
`cte_log.csv`, so this file is the only record of them. Update it at collection
time, not afterwards.

Simulator: Donkey, accessed through `gym-donkeycar`. Camera 160×120.

---

## Summary

| track | env | seed | trees | light | cones | mode | excursion_mag | laps | frames | off-road | use |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `generated_track` | donkey-generated-track-v0 | 20432814 | off | off | off | autopilot_lane | mixed: 0.3 then 0.0, proportions not recorded | multiple sessions | 10,896 | 12.0% | **training** |
| `baseline_run` | donkey-generated-track-v0 | 20432814 | off | off | off | autopilot_lane | 0.0 | ? | 5,046 | 7.2% | **condition-1 reference** |
| `cond1_trees` | donkey-generated-track-v0 | 20432814 | **ON** | off | off | autopilot_lane | 0.0 | 13 | 8,429 | 6.5% | condition C1a |
| `cond1_light` | donkey-generated-track-v0 | 20432814 | off | **ON** | off | autopilot_lane | 0.0 | 3 + 10 | 6,879 | 6.9% | condition C1b |
| `cond1_cones` | donkey-generated-track-v0 | 20432814 | off | off | **ON** | autopilot_lane | 0.0 | 3 + 10 | 6,922 | 6.7% | condition C1c |
| `generated_road` | donkey-generated-roads-v0 | n/a | off | off | off | autopilot_lane | ? | ? | 4,714 | 7.3% | condition C2 |
| `mini_monaco` | donkey-minimonaco-track-v0 | n/a | off | off | off | autopilot_lane | 0.0 | 3 | 3,965 | 2.1% | condition C3 |
| `mountain_track` | ? | ? | ? | ? | ? | autopilot_lane | ? | ? | 1,777 | 75.0% | **unusable** |

Collection flags not listed above were left at their defaults: `--kp 0.95`,
`--target_speed 0.1`, `--lane_offset -0.3`, `--excursion_every 180`.

---

## State coverage

Counts from `bin_cte()` over the full folder (not a train/val split).

| track | far-L (3) | near-L (1) | on-target (0) | near-R (2) | far-R (4) | error (−1) |
|---|---|---|---|---|---|---|
| `generated_track` | 999 | 2,191 | 3,594 | 2,340 | 469 | 1,303 |
| `baseline_run` | 361 | 958 | 2,293 | 1,016 | 53 | 365 |
| `cond1_trees` | 547 | 1,805 | 3,337 | 2,092 | 101 | 547 |
| `cond1_light` | 454 | 1,427 | 2,730 | 1,728 | 67 | 473 |
| `cond1_cones` | 442 | 1,531 | 2,650 | 1,759 | 76 | 464 |
| `generated_road` | 83 | 836 | 1,951 | 1,419 | 82 | 343 |
| `mini_monaco` | 207 | 966 | 1,395 | 1,129 | 186 | 82 |
| `mountain_track` | 105 | 68 | 80 | 79 | 113 | 1,332 |

Far-right is the sparsest on-road state in every dataset **except `mini_monaco`**,
where three laps gave 186 far-right samples — more than thirteen laps produced on
any generated-track condition. The circuit has genuine left- and right-hand corners
rather than a predominantly left-handed loop, so the vehicle crosses the centre line
in normal driving. `mini_monaco` is also the cleanest dataset collected, at 2.1%
off-road.

Far-right is the sparsest on-road state in every dataset. This is structural: the
vehicle drives left of the centre line, so reaching far-right requires crossing it,
which correct behaviour rarely does. Additional laps fill it sub-linearly —
`cond1_light` went from 20 to 67 far-right samples over a 4.2× increase in frames.

---

## Notes per dataset

### `generated_track` — training set
Pooled across several sessions with `--excursion_mag` varied between them (0.3
early, 0.0 later) to fill the extreme states. The per-session split is not
recorded and cannot be recovered exactly from the CSV.

The final ~1,400 frames (index ≳9,500) are a session in which the autopilot failed
repeatedly: off-road fraction 31–43% per 500-frame block against 8% earlier, CTE
reaching ±7.8, and lap counter stuck at 0. That session contributes 41% of all
error-state frames but only 6–13% of each on-road state, so it inflates the
error-state count without distorting the on-road abstraction.

At 12.0% off-road this dataset is the outlier; every other set sits at 6.5–7.3%.
Comparisons between the baseline and the conditions therefore carry a confound:
the baseline contains more chaotic frames. Condition-versus-condition comparisons
do not.

### `baseline_run` — condition-1 reference
Collected on `donkey-generated-track-v0` with seed 20432814, all three toggles off,
`autopilot_lane` mode with no excursion. Same environment, same geometry and same
collection regime as the `cond1_*` runs, differing only in that no toggle is set.

**This, not `generated_track`, is the correct comparator for condition 1.** The
training set pools sessions with varying excursion settings and contains a session
in which the autopilot failed repeatedly, putting it at 12.0% off-road against
6.5–7.3% everywhere else. Comparing a condition against `generated_track` therefore
confounds the visual change with a difference in collection regime; comparing
against `baseline_run` does not.

`generated_track` remains the training distribution and the reference for the
feature-space and pixel-space distance measures, since those compare against the
data the model was fitted on. But accuracy, MAE, signed bias and $\beta$ under
condition 1 should be read against `baseline_run`.

**Outstanding: the Stage 2 battery has not been run on `baseline_run`.** Until it
is, the condition-1 comparisons in `condition_summary.csv` are against the training
distribution and carry the confound above.

### `cond1_*` — intended as a controlled visual arm
`useSeed` on with seed 20432814 throughout, one toggle changed per condition,
everything else off. Lap times sat in a 25.2–28.2 s band across all three
conditions and both sessions, so the visual condition did not measurably change
how the car drove.

**However, inspection of the frames shows the conditions are not purely visual,
and the labels should not be taken at face value in the paper.**

`cond1_cones` — `generateRandomCones` places cones **on the driving surface**,
including straddling the centre line, and they are large relative to the vehicle.
Note that cones at the *roadside* are part of the default scene and appear in every
condition including the control; the toggle adds them to the driving surface only.
This is therefore an obstruction condition, not an appearance one. Perception is
unaffected (80.7% accuracy, MAE 0.165) because lateral position remains estimable
with a cone in view; what the abstraction cannot express is an obstacle in the path.
The safety property (|cte| > 2.0) says nothing about collisions.

`cond1_light` — `randomLight` flattens the directional structure of the
illumination rather than changing its overall level. Measured over 300 random
frames per condition:

| condition | near-field within-frame contrast | left–right asymmetry | lower-left mean (sd) |
|---|---|---|---|
| `generated_track` | 33.0 | 17.8 | 150.4 (22.9) |
| `cond1_trees` | 40.1 | 36.6 | 143.7 (29.7) |
| `cond1_cones` | 32.9 | 18.9 | 148.8 (23.6) |
| `cond1_light` | **25.5** | **11.1** | 155.6 (**8.9**) |

Mean intensity is essentially unchanged, so the global histogram is unchanged
(pixel JS 0.024). What changes is modulation: less contrast within each frame,
less left–right gradient, and a third of the baseline's frame-to-frame variability
in the near field. Trees moves the opposite way on both measures, as tree shadows
*add* directional structure; cones is indistinguishable from baseline.

The working account, consistent with four converging statistics but **not
demonstrated by ablation**: the model learned to use directional illumination
structure in the near field, which varies with heading and lane position and so
correlates with the target. Training augmentation applies a global multiplicative
brightness scale, which rescales contrast rather than removing it, so nothing in
training simulated flatter light. Deprived of the cue, predictions contract toward
the conditional mean (sd 0.458 against 1.267 in-distribution) — and low spread
across dropout masks is exactly what MC-dropout reports as confidence.

An earlier reading of this condition as a *geometry* shift was wrong. It came from
comparing frames matched on CTE value, which says nothing about position on the
lap; a baseline frame on a straight and a `cond1_light` frame on a bend can share a
CTE. Frames sampled by position in the run show curvature in both sets. Do not
repeat the geometry claim.

`randomLight` also re-randomises per episode, so the condition is not reproducible
run to run: two live runs under identical flags gave different outcomes (guard never
fired in one, fired at frame 89 in the other).

`cond1_trees` — frames not inspected directly, but the contrast statistics above
place it on the opposite side of the baseline from `cond1_light`, consistent with
added shadow structure rather than a geometric change.

### `generated_road` — condition C2
Different road layout and desert surroundings: appearance and geometry both change.
Lane convention is unchanged — the vehicle drives left of the yellow centre line,
as in the baseline.

**TODO: record the laps and excursion setting used.**

### `mini_monaco` — condition C3
Geometry-dominant: same asphalt-and-grass palette as the baseline, different circuit
layout, no toggles enabled. Lap times ~65 s against ~27 s on the generated track, so
a considerably longer circuit. The collection autopilot drove it without crashing
(2.1% off-road, the lowest of any dataset), unlike `mountain_track`.

Collected at 3 laps for the go/no-go check. Extend to 13 laps if the conditions are
to be matched on collection effort, though the state coverage is already adequate.

### `mountain_track` — not usable
75% of frames off-road, with only 80 in the on-target state. The autopilot did not
successfully drive this circuit. An abstraction estimated from it would describe a
vehicle that is not lane-keeping. Collected early and never used.

If a geometry-dominant condition is wanted, this needs re-collecting — and it may
be that the autopilot cannot drive the track at all, which is worth establishing
before spending time on it.

---

## Live closed-loop runs (Stage 4)

All runs: `donkey_drive_guarded.py`, `--n_mc 30`, `--fallback oracle`, 3 laps
requested, ID-calibrated threshold 0.2849. `--track` set explicitly so logs do not
collide. Logs at `results_cte/guarded_drive_<track>_log.csv`.

`M` = consecutive failed checks before handover. `R` = consecutive passes before
hand-back (0 = absorbing, matching the m2 DTMC). Fail-safe gains are the
`--fallback_kp` / `--fallback_kd` pair; the original runs used the model's own
`--kp 0.95` with no derivative term.

| run | condition | M | R | fail-safe | handovers | hand-backs | model off-road | fallback off-road | laps |
|---|---|---|---|---|---|---|---|---|---|
| `road_absorbing` | generated_road | 10 | 0 | P 0.95 | 1 | 0 | 3.6% | 11.3% | 3 |
| `road_bidir` | generated_road | 10 | 30 | P 0.95 | 3 | 2 | 14.2% | 18.2% | 1 (crash) |
| `road_pd` | generated_road | 10 | 0 | PD 5/5 | 1 | 0 | 0.0% | **0.4%** | 0 (4253 fr) |
| `cones_absorbing` | cones | 10 | 0 | P 0.95 | 1 | 0 | 2.0% | 11.0% | 3 |
| `cones_bidir` | cones | 10 | 30 | P 0.95 | 5 | 4 | 2.2% | 19.1% | 3 |
| `cones_pd` | cones | 10 | 0 | PD 5/5 | 1 | 0 | 0.0% | **0.0%** | 3 |
| `trees_absorbing` | trees | 10 | 0 | P 0.95 | 1 | 0 | 0.0% | 12.0% | 3 |
| `trees_bidir` | trees | 10 | 30 | P 0.95 | 1 | 0 | 4.8% | 8.9% | 3 |
| `trees_pd` | trees | 10 | 0 | PD 5/5 | 1 | 0 | 2.4% | **0.0%** | 3 |
| `light_absorbing` (run 1) | light | 10 | 0 | P 0.95 | 0 | 0 | 32.2% | — | 0 (crash) |
| `light_absorbing` (run 2) | light | 10 | 0 | P 0.95 | 1 | 0 | 30.3% | 100% | 0 (crash) |
| `light_bidir` | light | 10 | 30 | P 0.95 | 1 | 0 | 30.3% | 100% | 0 (crash) |
| `light_pd` (run 1) | light | 10 | 0 | PD 5/5 | 1 | 0 | 30.0% | 79.7% | 0 (crash) |
| `light_pd` (run 2) | light | 10 | 0 | PD 5/5 | 1 | 0 | 29.5% | 100% | 0 (crash) |
| `light_M3` | light | **3** | 0 | PD 5/5 | 1 | 0 | 0.0% | **0.0%** | **3** |
| `light_M3_bidir` | light | 3 | 30 | PD 5/5 | 19 | 18 | 1.8% | 12.3% | 2 (crash) |

Note the `light_absorbing` log was overwritten by the second run; only run 2
survives on disk. The two runs differed because `randomLight` re-randomises.

### Three findings from this matrix

**Fail-safe competence.** The P-only oracle drives badly despite perfect state
knowledge: 11–12% off-road, because a proportional controller without damping
oscillates. With PD gains (5, 5) the same fail-safe reaches 0.0–0.4%. Earlier
readings of these runs as "handover transfers risk rather than eliminating it"
were an artifact of an under-designed fail-safe and should not be repeated.

**Detection latency.** Under `light`, M=10 never fires in time — the vehicle is
already at |cte| ≈ 7.9 when handover triggers. At M=3, everything else identical,
the guard fires at frame 55 with |cte| ≈ 0.3 and the run completes 3 laps at 0%
off-road. The consecutive-failure debounce filters exactly the intermittent signal
that indicated failure: in the M=10 log, std first exceeds threshold at frame 53
but the road edge is crossed at frame 62, and 18 frames were certified while the
vehicle was already off-road.

**Hand-back is unsafe on this signal.** Absorbing runs complete; bidirectional runs
crash, in both `generated_road` (3 laps vs 1) and `light` at M=3 (3 laps vs 2, with
19 handovers in 82 s). The reason is visible in the logs: during fallback the PD
controller holds the vehicle near lane centre, where the model is accurate, so the
hand-back criterion is satisfied on a trajectory the fail-safe produced rather than
one the model would produce. Control returns, the model's error grows, and the guard
fires again within 5–20 frames. The absorbing policy the m2 DTMC encodes is
therefore the safer design, not merely the simpler one.

**Caveat for the paper:** the `oracle` fail-safe steers from ground-truth CTE, so
it is an upper bound on what handover can achieve. A deployed fail-safe would not
have ground truth.

---

## Results by condition (Stages 2–3)

Guard threshold fixed at the in-distribution value 0.2849 throughout; never
recalibrated on condition data.

### Perception and detection (Stage 2)

| condition | on-road acc | MAE | signed bias | MC std median | β | pixel JS | Mahalanobis AUROC | sym KL |
|---|---|---|---|---|---|---|---|---|
| in-distribution (val) | 81.2% | 0.19 | −0.009 | 0.171 | 0.800 | 0 | 0.500 | 0 |
| `baseline_run` | 83.5% | 0.223 | −0.001 | 0.156 | 0.874 | 0.014 | 0.424 | 0.77 |
| `cond1_cones` | 80.7% | 0.165 | +0.026 | 0.162 | 0.874 | 0.001 | 0.458 | 3.67 |
| `cond1_light` | 56.0% | 0.736 | +0.389 | 0.163 | **0.852** | 0.024 | **0.441** | 88.04 |
| `cond1_trees` | 49.5% | 0.618 | +0.237 | 0.236 | 0.622 | 0.044 | 0.655 | 2.21 |
| `generated_road` | 5.2% | 3.279 | −3.072 | 0.630 | 0.043 | 0.180 | 0.875 | 10.37 |
| `mini_monaco` | 8.7% | 2.572 | −2.418 | 0.775 | 0.001 | 0.235 | 0.954 | 30.21 |

β and AUROC track accuracy loss in every condition **except `cond1_light`**, where
accuracy falls 28 points while β stays near the clean-baseline value and AUROC sits
below chance. Note that 0.874 — not the calibrated 0.800 — is the natural pass rate
on a clean unshifted run, so `cond1_light`'s 0.852 should be read against 0.874.

Symmetrised KL is the one measure that flags `cond1_light` (88.04 nats, the largest
of any condition). Mahalanobis measures displacement from the ID mean under the ID
covariance; symmetrised KL compares full distributions including covariance. A
prediction collapse changes covariance drastically while staying near the mean, so
KL sees it and Mahalanobis cannot.

### Verified probabilities (Stage 3, N = 30, 95% CI from 200 bootstrap replicates)

| condition | β | m1 off-road | m1 95% CI | m2 off-road | m2 95% CI | m2 abort |
|---|---|---|---|---|---|---|
| `baseline_run` | 0.874 | 2.3e-05 | [1.5e-05, 3.7e-04] | 0 | [5.3e-06, 3.7e-04] | 2.9e-08 |
| `cond1_cones` | 0.874 | 0 | [6.2e-08, 5.2e-05] | 0 | [5.4e-07, 9.4e-05] | 3.0e-08 |
| `cond1_light` | 0.852 | 0.971 | [0.961, 0.982] | 0.864 | [0.801, 0.907] | 7.8e-08 |
| `cond1_trees` | 0.622 | 0.0041 | [0.0024, 0.0114] | 0 | [5.9e-05, 0.0016] | 0.0018 |
| `generated_road` | 0.043 | 1.000 | [0.99998, 1] | 0.0018 | [4.2e-05, 0.0072] | 0.998 |
| `mini_monaco` | 0.001 | 0.996 | [0.987, 0.999] | 2.5e-07 | [4.5e-08, 5.8e-07] | 1.000 |

Four regimes:

- **Nothing to catch** — `baseline_run`, `cond1_cones`. m1 and m2 intervals overlap
  almost entirely; the guard's effect is not resolvable, and the system is safe
  either way. `baseline_run` is the control: an unshifted condition verifies as
  unshifted through the full pipeline.
- **Guard works** — `cond1_trees`. Non-overlapping intervals, roughly an order of
  magnitude improvement.
- **Guard rescues** — `generated_road`, `mini_monaco`. Near-certain failure becomes
  near-certain handover.
- **Guard resolvable but useless** — `cond1_light`. The intervals separate, so the
  effect is real, but it moves off-road probability from 97% to 86% while abort sits
  at 7.8e-08. The system almost never hands over and leaves the road anyway.

### Two reporting hazards these results expose

**Zero point estimates.** Three conditions produced a PRISM point estimate of
exactly 0 (`cond1_cones` m1 and m2, `cond1_trees` m2) with bootstrap medians of
7.4e-06, 1.6e-05 and 5.0e-04. Reporting those zeros unqualified would claim
impossibility from finite samples. No zero should appear without its interval.

**Narrow intervals from absent data.** `mini_monaco` m2 gives 2.5e-07 with an
apparently tight interval of [4.5e-08, 5.8e-07] — seemingly the safest guarded
system in the study. It is not. β = 0.001 left four passing frames out of 3,883:
two rows had no passing samples and became uniform priors, the other three rest on
n = 1, 1 and 2. The interval is narrow *because* the Jeffreys prior dominates when
counts are near zero, so every replicate draws from nearly the same distribution.
The abort probability of 1.000 is the number that describes this system.

Both extremes show the same self-limiting property: **the guarded abstraction can
only be estimated from inputs the guard admits, so the more effective the guard, the
less identifiable the guarded model.** `generated_road` at β = 0.043 gives a 170-fold
interval; `mini_monaco` at β = 0.001 gives no estimate at all, disguised as a precise
one.
