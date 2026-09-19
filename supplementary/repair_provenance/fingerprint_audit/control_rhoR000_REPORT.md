# ρ_R=0 Zero-Control — Closed-Loop Foundation Check (2026-06-24)

Run: `mainb_oxeonly_rhoR000_control` (OXE-only continuation from V1 300k — optimizer reset,
flat LR, +10k steps, **NO eggplant teacher data**). 20 checkpoints (500…10000, every 500).
Pulled to `~/octo_repair_ckpts/mainb_oxeonly_rhoR000_control/` (state/ excluded).

**Purpose:** isolate the cause of the ρ_R=0.5 repair. Did the closed loop move because of the
*eggplant teacher data*, or just because of the *training mechanics* (reset optimizer + flat LR +
10k more steps)? This control has the identical mechanics but no eggplant data. If it also fixes
the closed loop → attribution fails. If it stays at the V1 floor → the eggplant data is the cause.

**Closed loop is the only judge. The control's training loss necessarily decreases (it keeps
training on OXE); loss was NOT read as success — V1 itself is loss-perfect and closed-loop 0.**

## Part A — download + verify (PASS)
- 20 step dirs + root config/dataset_statistics/example_batch; `state/` excluded (13 GB). No MANIFEST
  on A100 → sha256 -c skipped; integrity confirmed by load test.
- step 500 & 10000 load via `OctoModel.load_pretrained` (per-step staging dirs) as **8L**
  (config num_layers=8, 8 param encoderblocks, 174,107,148 params — identical to V1/repair).
- Normalization: control `dataset_statistics` == V1 for all 52 action mean/std entries,
  max|Δ| = **0.000e+00**. De-normalization behaves exactly like V1 and the repair run → the
  control vs repair comparison is clean (same de-norm, same rig, same protocol).

## Part B — rig calibration (HARD GATE: PASS)
New session = re-calibrated. Both known baselines reproduced (eps 0–23, rng0, threshold075):

| control | success | moved | grasp | consec | target |
|---|---|---|---|---|---|
| teacher15 (octo-base-1.5) | **15/24** | 20 | 20 | 19 | 15 |
| V1 (octo-distill 300k)    | **0/24**  | 12 | 11 | 10 | 0  |

teacher15 15/24 (last session 14/24, within ±1) and V1 0/24 → scene/robot/threshold075/normalization
sound. (Note: the teacher result dir held 9 stale MP4s from the prior identically-named session;
counted from today's 24 fresh files only. The control ladder uses fresh `eval_results_control_*`
dirs, so it is uncontaminated.)

## Part C — control closed-loop ladder
**Identical protocol to the repair ladder** (same gripper/wrapper/env/rng/+45-excluded; only
checkpoint source + labels differ). in-dist eids {0,1,2,9–14,21–23} × rng{0,1,2}, n=36/model.
Held-out eids {3–8,15–20} NOT touched (foundation check needs only in-dist). All n=36 verified.

| step | success | moved | grasp | consec | **target** | n |
|---|---|---|---|---|---|---|
| **V1 anchor** | 0 | 26 | 16 | 14 | **0** | 36 |
| 500   | 0 | 27 | 14 | 13 | **0** | 36 |
| 1000  | 0 | 28 | 15 | 14 | **0** | 36 |
| 2000  | 0 | 26 | 17 | 13 | **0** | 36 |
| 4000  | 0 | 28 | 14 | 13 | **0** | 36 |
| 6000  | 0 | 27 | 14 | 14 | **0** | 36 |
| 8000  | 1 | 31 | 12 | 11 | **1** | 36 |
| 10000 | 0 | 32 | 16 | 13 | **0** | 36 |

CSV: `control_ladder_results.csv`.

## Part D — point-by-point: control vs repair vs V1 anchor (success / target)

| step | control succ | control target | repair succ | repair target |
|---|---|---|---|---|
| V1 anchor | 0 | 0 | 1  | 1  |
| 500   | 0 | 0 | 15 | 15 |
| 1000  | 0 | 0 | 19 | 19 |
| 2000  | 0 | 0 | 18 | 18 |
| 4000  | 0 | 0 | 20 | 20 |
| 6000  | 0 | 0 | 20 | 20 |
| 8000  | **1** | **1** | **24** | **24** |
| 10000 | 0 | 0 | 20 | 20 |

## Part D — judgement: **FOUNDATION STABLE. Attribution holds.**
- **Control closed-loop ≈ 0 at every checkpoint.** target = 0 for 7 of 8 ladder points; the single
  c8000 success (1/36, target 1/36) is below the pre-registered ≥3 "control also fixed" threshold →
  **noise / V1 floor** (the repair's V1 anchor likewise hit 1 once). The control never leaves the
  floor that V1 occupies.
- **The repair, on the identical mechanics, went 0→15–24 at every checkpoint.** The *only*
  difference between control and repair is the eggplant teacher data. Therefore:
  **"reset optimizer + flat LR + 10k more steps" by itself does NOT fix the closed loop — the
  eggplant teacher data is the cause.** This is the desired (hoped-for) outcome; the paper's core
  attribution for the ρ_R=0.5 repair is sound.
- **Grasp corroborates.** Control grasp 12–17 ≈ V1's 16 (calib V1 11) — unchanged base capability.
  The repair's grasp 31–36 is far above. So even the grasp improvement is attributable to the
  eggplant data, not to extra OXE training. Control did not move grasp *or* placement.
- **moved** ticks up slightly on the control (26→32) but this is the object being nudged, not
  placed — target (the post-grasp carry-into-basket criterion the repair actually fixed) stays 0.

## Bottom line
The ρ_R=0 OXE-only control **stays at the V1 closed-loop floor (target ≈ 0) across the entire
ladder**, while the ρ_R=0.5 repair — same optimizer reset, same flat LR, same 10k steps, same rig,
same de-normalization — moved target to 15–24. **The repair effect is caused by the eggplant
teacher data, not by the training mechanics.** Foundation is stable; the repair-run claims hold.

CSV: `control_ladder_results.csv` (control ladder); `repair_ladder_results.csv` (repair, for the
point-by-point comparison).
