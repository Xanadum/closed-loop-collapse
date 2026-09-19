# Provenance of the 72-trial diagnostic counts

`teacher_72_counts.csv` and `student_72_counts.csv` are **derived records**. Each was
produced by parsing the filenames of the rollout videos written by the evaluator; the
videos themselves are not included in this package (they total 296 MB). No evaluation
was re-run to produce them.

- teacher (`octo-base-1.5`): seed 0 = 15/24, seed 1 = 14/24, seed 2 = 11/24 → **40/72**
- compressed policy (8-layer student): 0/24 on every seed → **0/72**

---

## 1. Where the videos live

One leaf directory per (seed × policy), under the local evaluation tree:

```
~/SimplerEnv_stable/eval_results_threshold075_teacher15_student_rng<SEED>/
  <MODEL>/bridge_table_1_v2/
    arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos/
      PutEggplantInBasketScene-v0_<TAG>_eggplant_threshold075_rng0_24eps/
        rob_0.127_0.06_rot_0.000_-0.000_3.142_rgb_overlay_bridge_sink/
          *.mp4
```

with `<SEED>` ∈ {0,1,2} and

| policy | `<MODEL>` | `<TAG>` |
|---|---|---|
| teacher | `octo-base-1.5` | `teacher15` |
| compressed student | `octo-distill-8l` | `student8l` |

The sibling `bridge_table_1_v1` subtree under the same roots holds the *carrot* task and
is not part of this protocol; it was excluded by selecting `bridge_table_1_v2` /
`PutEggplantInBasketScene-v0` only.

---

## 2. Filename encoding, and the cross-check used

The evaluator encodes the full outcome tuple into the filename:

```
<success|failure>_obj_episode_<EID>_moved_correct_obj_<B>_moved_wrong_obj_<B>
_is_src_obj_grasped_<B>_consecutive_grasp_<B>_src_on_target_<B>.mp4
```

Two fields independently encode the graded outcome:

- the leading token, `success_` or `failure_`
- the trailing field, `src_on_target_True` or `src_on_target_False`

For this task these must agree, because the environment defines
`success = src_on_target` (see `../../../offline_eval/METRIC_DEFINITIONS.md`).
**The extraction asserts equality of the two on every file and aborts on any mismatch.**
All 168 parsed files (72 teacher + 96 student) passed; the `success` and
`src_on_target` columns in both CSVs are therefore identical by verification, not by
construction.

Counting reduces to:

```bash
find "$LEAF" -name 'success_*.mp4'          | wc -l   # successes
find "$LEAF" -name '*src_on_target_True.mp4' | wc -l  # independent cross-check
```

---

## 3. Completeness and de-duplication

| policy | seed | files | unique episode IDs | duplicated IDs |
|---|---|---|---|---|
| teacher | 0 | 24 | 24 | none |
| teacher | 1 | 24 | 24 | none |
| teacher | 2 | 24 | 24 | none |
| student | 0 | **36** | 24 | 12 (IDs 1, 8, 9, 10, 12, 15, 17, 18, 19, 20, 21, 23) |
| student | 1 | 24 | 24 | none |
| student | 2 | 24 | 24 | none |

Every cell covers episode IDs 0–23 exactly once after de-duplication; the extraction
asserts this.

The student seed-0 directory holds 36 files for 24 episodes because a later run was
written over an earlier one in the same directory (its file timestamps span 18:01 to
21:26, whereas the other five cells are each a single contiguous 6–8 minute batch).
**All 12 duplicated IDs carry identical outcomes in both copies** — the extraction
asserts this too — so de-duplication is unambiguous and cannot change the count. Every
duplicate pair is `failure` / `src_on_target_False`, consistent with the 0/24 result.

**Scope of that assertion.** The equality asserted above covers the two graded outcome
fields, `success` and `src_on_target`, on the duplicated files within this directory. It
does not extend to the intermediate stage predicates. Across the two separate evaluation
sessions that both cover the compressed student at seed 0 — the instrumented diagnostic
session and a later rig-calibration re-run — `success` (0/24) and `is_src_obj_grasped`
(11) agree, while `moved_correct_obj` reads 18 and 12 and `consecutive_grasp` reads 11
and 10. The `moved` predicate is a 3 cm displacement threshold that the student's
marginal contacts straddle, so it re-runs less stably than the latched grasp and success
flags. Table I of the paper reports the instrumented session, and pairs its teacher and
student rows from that same session; no causal claim rests on the `moved` count. See
"Package notes" in the top-level `README.md`.

---

## 4. The leaf directory is labelled `rng0` for all three seeds

Note the `..._rng0_24eps` component in the path template above: it is literally `rng0`
in **all three** seed trees. This is a stale hard-coded `--additional-env-save-tags`
value in the driver script, not the seed that was actually used. The real seed is
carried by the parent directory name (`..._student_rng0` / `_rng1` / `_rng2`).

Four independent lines of evidence confirm the three trees are genuinely seeds 0, 1
and 2, and not copies of one run:

**(a) The recorded argument.** The evaluator logs its parsed argument namespace. In
`~/SimplerEnv_stable/eval_logs_threshold075_teacher15_student_rng<SEED>/teacher15_eggplant_threshold075_rng0_24eps.log`:

| directory | recorded `octo_init_rng` | recorded `ckpt_path` |
|---|---|---|
| `..._rng0` | `0` | `hf://rail-berkeley/octo-base-1.5` |
| `..._rng1` | `1` | `hf://rail-berkeley/octo-base-1.5` |
| `..._rng2` | `2` | `hf://rail-berkeley/octo-base-1.5` |

The student logs in the same three directories likewise record `0`, `1`, `2`.

**(b) The success sets differ**, and are not nested or shifted versions of each other:

| seed | successful episode IDs |
|---|---|
| 0 | 0, 2, 3, 6, 7, 8, 9, 11, 14, 16, 17, 20, 21, 22, 23 |
| 1 | 0, 2, 3, 6, 9, 11, 13, 14, 17, 18, 20, 21, 22, 23 |
| 2 | 0, 2, 5, 6, 8, 9, 10, 11, 14, 20, 23 |

**(c) The videos differ byte-wise.** MD5 of the episode-0 teacher video (first 8 hex
digits): seed 0 = `b96a66af`, seed 1 = `ab833d41`, seed 2 = `513fa03b`.

**(d) Three separate execution batches.** File modification times, all on 2026-05-24:

| batch | teacher | student |
|---|---|---|
| seed 0 | 21:09 – 21:16 | 18:01 – 21:26 (overlaid; see §3) |
| seed 1 | 21:31 – 21:38 | 21:42 – 21:48 |
| seed 2 | 21:53 – 22:00 | 22:04 – 22:10 |

The six batches are sequential and non-overlapping (apart from the seed-0 student
overlay), consistent with one driver invocation per seed.

---

## 5. Relation to the per-seed summary CSVs in this directory

`teacher15_rng0_24eps_teacher_family_summary.csv` and
`v1_step0_step1_rng0_24eps_teacher_family_summary.csv` are the evaluator's **own**
summary output, and exist for seed 0 only — the driver did not write them for seeds 1
and 2. Their seed-0 eggplant rows (teacher `15` successes of `24`; student `0` of `24`)
agree exactly with the seed-0 block of the CSVs here, which is the check that validates
the filename-parsing method against the evaluator's independent accounting.

The absence of those summary files for seeds 1 and 2 is what previously made these two
counts look unrecoverable. The underlying rollouts were always present.
