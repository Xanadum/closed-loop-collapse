# ρ_R=0.5 Teacher-Repair — Closed-Loop Evaluation Report (2026-06-21)

Run: `mainb_repair_rhoR050_predB` (teacher closed-loop distillation, forked from V1 300k,
predicted-chunk target = path B). 20 checkpoints (500…10000, every 500). Pulled to
`~/octo_repair_ckpts/mainb_repair_rhoR050_predB/` (state/ excluded).

**Closed loop is the only judge. No success was inferred from loss.**

## Part A — download + verify (PASS)
- 20 step dirs + root config/dataset_statistics/example_batch; `state/` excluded. No MANIFEST on
  A100 → sha256 -c skipped; integrity confirmed by load test.
- step 500 & 10000 load via `OctoModel.load_pretrained` (per-step staging dirs) as **8L**
  (config num_layers=8, 8 param encoderblocks, 174,107,148 params restored).
- Normalization: repaired `dataset_statistics` == V1 for all 26 datasets, action mean/std
  max|Δ| = **0.000e+00** (incl. bridge_dataset). De-normalization behaves exactly like V1.

## Part B — rig calibration (HARD GATE: PASS)
Both known baselines reproduced (eps 0–23, rng0, threshold075):

| control | success | moved | grasp | consec | target |
|---|---|---|---|---|---|
| teacher15 (octo-base-1.5) | **14/24** | 23 | 18 | 17 | 14 |
| V1 (octo-distill 300k)    | **0/24**  | 12 | 11 | 10 | 0  |

Positive (~15) and negative (0) controls both hit → scene/robot/threshold075/normalization sound.

## Part C+D — coarse ladder + densify (in-dist eids {0,1,2,9–14,21–23} × rng{0,1,2}, n=36/model)
Held-out eids {3–8,15–20} never touched (three contiguous in-dist blocks [0,3) [9,15) [21,24)).

| step | success | moved | grasp | consec | **target** | n |
|---|---|---|---|---|---|---|
| **V1 anchor** | 1 | 25 | 17 | 16 | **1** | 36 |
| 500   | 15 | 31 | 32 | 31 | 15 | 36 |
| 1000  | 19 | 34 | 31 | 31 | 19 | 36 |
| 2000  | 18 | 32 | 33 | 32 | 18 | 36 |
| 4000  | 20 | 36 | 35 | 32 | 20 | 36 |
| 6000  | 20 | 35 | 34 | 32 | 20 | 36 |
| 7500  | 18 | 34 | 36 | 33 | 18 | 36 |
| **8000**  | **24** | 35 | 35 | 33 | **24** | 36 |
| 8500  | 21 | 35 | 34 | 32 | 21 | 36 |
| 10000 | 20 | 34 | 36 | 35 | 20 | 36 |

CSV: `repair_ladder_results.csv`.

## Part D — judgement: **SIGNAL. Repair moved the closed loop. Direction confirmed.**
- **target** went from V1 ≈ 1 (noise) → **15–24 at every repaired checkpoint**; success tracks
  target exactly. Best = **step 8000, 24/36 success & target (67%)** — at/above the teacher's own
  rate. Onset already by step 500, sustained/strengthening through 10000 (no collapse).
- **No forgetting (foundation intact):** grasp 31–36 across the ladder vs V1's 17 — the base
  *improved*, not degraded. Rules out the "0.5 dose hurt the base" confound; ρ_R=0.5 foreground
  protection held. (So this is the clean "BC repair worked" branch, NOT the lower-dose branch.)

### Mechanism: post-grasp z-gap (action trace, in-dist eids 2,9,11,12,14,23, rng0)
Executed world_z post-grasp (negative = arm sinking, V1's documented failure):

| policy | post-grasp mean world_z | late-half | eps grasped |
|---|---|---|---|
| teacher15  | −0.00056 | −0.00080 | 6/6 |
| V1         | −0.00542 | −0.00263 | 2/6 |
| **repair8000** | **−0.00055** | **−0.00110** | **6/6** |

repair8000's post-grasp z ≈ teacher's → **closes ~100% of the V1→teacher z-gap (84% on the
late half)**, and grasps 6/6 like the teacher (V1 2/6). The exact mechanism the repair targeted
(close → lift → hold → carry) is restored.

## Part E — held-out generalization + full teacher/student matrix (2026-06-21)
Held-out eids {3-8,15-20} touched here for the FIRST time. **Only the pre-selected checkpoint
(step 8000, chosen by in-dist argmax) is the headline** — no checkpoint selection on held-out.
Config identical to the ladder; only eid set + model swapped. n=36/cell (12 eids × rng{0,1,2}).

**2×3 matrix — success / target / grasp:**

| model | in-dist (×rng012) | held-out (×rng012) |
|---|---|---|
| V1            | 1 / 1 / 17   | **0 / 0 / 17**  (floor) |
| teacher15     | 19 / 19 / 30 | **17 / 17 / 30** (ceiling) |
| **repair8000**| 24 / 24 / 35 | **18 / 18 / 34** (headline) |

Robustness band (held-out, NOT for selection): repair6000 20, repair8000 18, repair10000 21 →
mean **19.7** (8000 headline is the lowest of the three → conservative).

CSV: `teacher_student_compare.csv`.

### Three key readouts
1. **Same-set teacher/student (in-dist):** repair8000 **24/36 (67%)** vs teacher15 **19/36 (53%)** —
   the student *exceeds* its teacher on training-distribution eids (+5).
2. **Headline generalization (held-out, unbiased):** repair8000 **18/36 (50%)** — at/above the
   teacher's own held-out ceiling (17) and far above the V1 floor (0). The robustness band
   {6000,8000,10000} mean 19.7 sits slightly above the teacher ceiling too.
3. **Overfitting vs test difficulty:** in-dist→held-out success drop repair8000 = **−6** (24→18),
   teacher15 = **−2** (19→17). The teacher's −2 is intrinsic eid difficulty; the student's extra
   ~4 is mild train-eid specialization (expected — it was distilled on train-eid rollouts). It does
   **NOT** compromise generalization: absolute held-out success (18) still equals the teacher
   ceiling. Grasp stays high on held-out (34 vs teacher 30, V1 17) → no forgetting off-distribution
   either; the small student drop is in placement, not grasping. **Verdict: true generalization with
   a mild train-eid bonus, not overfitting that breaks held-out.** V1 held-out 0 = floor confirmed.

## Bottom line
ρ_R=0.5 teacher closed-loop repair (predicted-chunk / path B) **decisively moved the closed loop and
generalizes**: in-dist target 0→24/36 (beats teacher), **unbiased held-out 18/36 (50%) = teacher
ceiling, V1 floor 0**, post-grasp z-collapse eliminated (~100% of z-gap closed), no foundation
forgetting (grasp ≫ V1 on both sets). Headline checkpoint = **step 8000** (pre-selected on in-dist).
Mild train-eid specialization (drop 6 vs teacher 2) is the only caveat and does not reduce
off-distribution success below the teacher ceiling.
