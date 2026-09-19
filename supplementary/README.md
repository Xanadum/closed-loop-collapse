# Supplementary material

Anonymous supplementary package. All artefacts are copies of files produced by the
original runs; nothing here was regenerated or re-derived except the eight clearly
marked summary documents listed under "Documents written for this package" below.

Host names, user names, e-mail addresses, absolute filesystem paths, interpreter paths
and experiment-tracker run identifiers have been replaced with neutral placeholders.
**No seed, episode ID, count, hyper-parameter, configuration value or timestamp was
altered.** Every file's origin is recorded line-by-line in `PROVENANCE.txt`.

---

## MISSING

Nothing is missing. Every requested item is present.

The two headline diagnostic counts — `40/72` (teacher) and `0/72` (compressed policy) —
are **verified at the level of individual trials**, in
`closed_loop/counted_outcomes/diagnostic_72_protocol/`:

| file | contents |
|---|---|
| `teacher_72_counts.csv` | 72 rows, one per (seed, episode ID): seed 0 = 15/24, seed 1 = 14/24, seed 2 = 11/24, **total 40/72** |
| `student_72_counts.csv` | 72 rows, same structure, 0 successes on every seed, **total 0/72** |
| `PROVENANCE_72.md` | how both were derived, and the evidence that the three seed directories are genuinely distinct runs |

Both CSVs are derived records, obtained by parsing the outcome tuple that the evaluator
encodes into each rollout video's filename. The videos themselves are not included
(296 MB). Two fields in each filename independently encode the graded outcome, and
their agreement is asserted on every one of the 168 parsed files. The seed-0 block of
each CSV reproduces the evaluator's own summary output for seed 0, which is retained
alongside them as an independent check on the parsing method.

---

## Where the paper's four commitments are satisfied

| paper location | commitment | file in this package |
|---|---|---|
| §III-B | per-dataset offline records | `offline_eval/eval_results_v3.json` |
| §III-C (footnote) | metric definitions | `offline_eval/METRIC_DEFINITIONS.md` |
| §V-C | micro-displacement measurements included | `micro_displacement/` (whole directory) |
| §VI-B | execution records for the two training runs | `repair_provenance/execution_records/` |

---

## Directory guide

### `offline_eval/`
The offline evaluation the compressed policy was accepted on: six datasets scored for
both the teacher and the student, plus parameter counts and inference latency
(`eval_results_v3.json`), and the script-level definition of every graded metric read
straight out of the simulator's own `evaluate()` (`METRIC_DEFINITIONS.md`).
`variants/` holds five further offline evaluations of *other* checkpoints, retained so
the reported run can be seen in the context of its siblings rather than in isolation.

### `closed_loop/counted_outcomes/`
Every counted closed-loop outcome in the paper, as trial counts out of a stated `n`:
the repair and zero-data-control checkpoint ladders, the teacher/student in-distribution
and held-out comparison, the deployment-time offset and clamp sweeps, the dose-response
curves for three checkpoints, and the second-testbed triage. The third candidate Bridge
task reported in Sec. VII is recorded in
`carrot_threshold_sweep_quick_8eps_teacher_family_summary.csv` (teacher 0/8 at the
threshold-0.75 calibration, and 0/8 at every other threshold swept).
`diagnostic_72_protocol/`
holds the original 72-trial diagnostic: per-trial outcome records for the teacher
(40/72) and the compressed policy (0/72), the evaluator's own seed-0 summaries, and a
provenance note covering the derivation.

### `closed_loop/trace_panel/`
The paired per-step action traces behind the phase- and dimension-resolved signature
panel, plus the panel CSVs themselves. Traces are per-episode and per-model, recorded
under an identical protocol so that student and teacher can be differenced episode by
episode.

### `micro_displacement/`
The full micro-displacement measurement set: every z-trace rollout log and per-step
action trace for the offset sweep, the clamp sweep and the three-checkpoint dose-response
suite, together with the absorption tables computed from them and the two analysis
scripts that compute those tables. These are sub-millimetre, non-paired differences and
are included so that the measurement can be re-derived rather than taken on trust.

### `repair_provenance/execution_records/`
Logged configuration, environment metadata and full training log for both training runs
— the one that received repair data and the one that received none — plus the same for
the original distillation run that produced the compressed policy.
`CONFIG_DIFF.md` is a field-by-field comparison of the two: 57 fields compared,
46 identical, 11 differing, **0 unexplained**. The differing fields are the data-mixing
ratio and its mechanical downstream (stream batch sizes, episode/frame counts, output
directory), plus run identity, launch timestamp and disk-bytes-at-launch.

### `repair_provenance/trainer_config/`
The trainer source used by both runs, and `REPAIR_FRAMES_CONFIRMATION.md`, which
confirms the repair-stream size (324 episodes → 38,880 frames → 607.5 steps/epoch)
from three independent places in the record, and shows that total batch size is
identical across the two runs.

### `repair_provenance/collection/`
The data-collection script and the dataset builder, plus `BUILDER_GUARD_EXCERPT.md`,
which quotes the hard-coded isolation guards with their original line numbers: which
episode IDs and RNG seeds were collected on, and the code that refuses to ingest any
held-out episode ID or any evaluation seed.

### `repair_provenance/fingerprint_audit/`
The integrity audit of the constructed training set: round-trip verification against the
original collected files, the two pre-upload data gates, the trajectory-fingerprint
uniqueness check, and the normalisation-statistics comparison. Also the two run reports
that record the normalisation result (maximum absolute difference 0).

---

## Documents written for this package

Eight files are derived or composed here rather than copied verbatim from a run. Each
cites the material it is derived from, and each is marked
`<GENERATED FOR THIS PACKAGE>` in `PROVENANCE.txt`. The count below and the count of
`<GENERATED FOR THIS PACKAGE>` markers in `PROVENANCE.txt` agree exactly:

- `closed_loop/counted_outcomes/diagnostic_72_protocol/teacher_72_counts.csv`
- `closed_loop/counted_outcomes/diagnostic_72_protocol/student_72_counts.csv`
- `closed_loop/counted_outcomes/diagnostic_72_protocol/PROVENANCE_72.md`
- `repair_provenance/execution_records/CONFIG_DIFF.md`
- `repair_provenance/trainer_config/REPAIR_FRAMES_CONFIRMATION.md`
- `repair_provenance/collection/BUILDER_GUARD_EXCERPT.md`
- `PROVENANCE.txt`
- this `README.md`

The two `*_72_counts.csv` files are the only derived files that carry numbers not
present verbatim in a packaged source file; `PROVENANCE_72.md` documents exactly how
they were obtained.

---

## Package notes

Points where the package holds more than one file bearing on the same quantity, or
where a naive reading could mislead. Each is stated so that a reader need not infer it.

**Training log completeness.** The packaged `v1_distill_300k.log` is the complete
training log: 3,000 telemetry records at 100-step intervals covering steps
100–300,000. Step numbers below 100,000 are space-padded (`Step    100/300000`), so
searches should use a whitespace-tolerant pattern such as `Step\s+[0-9]+/300000`; a
pattern assuming a single space silently matches only the 2,001 records from step
100,000 onward.

**Inference latency: which of six measurements the paper reports.** Six latency
measurements of the same architecture pair are included. The main text uses
`offline_eval/eval_results_v3.json` — the only n=200 run, measured on its primary
real-batch path — which is also the most conservative of the six (1.08×; the variants
report 1.15–1.18×), consistent with the paper's practice of reporting conservative
values.

**Two staged-count summaries for the student at seed 0.** Two staged-count summaries
exist for the compressed student at seed 0. Table I reports the instrumented diagnostic
session (moved 18, grasp 11, sustained 11, success 0/24); a later rig-calibration re-run
yielded moved 12, grasp 11, sustained 10, success 0/24. Success and grasp agree across
runs; the moved predicate depends on a displacement threshold that the student's
marginal contacts straddle, so it carries larger re-run variability than the latched
grasp/success flags. No causal claim in the paper rests on the moved count; Table I
pairs teacher and student rows from the same session.

**Triage and convenience-slice files.** `T2_triage_summary.csv` is the raw record of the
second-testbed triage reported in Sec. VII (spoon 1/12, stack 0/12). The
`e1_absorption_d2rows_*.csv` files are per-checkpoint convenience slices of
`E1_absorption_suite.csv`.

**Third candidate task (Sec. VII, 0/8).** The third candidate Bridge task is the carrot
task (`PutCarrotOnPlateInScene-v0`), recorded in
`closed_loop/counted_outcomes/carrot_threshold_sweep_quick_8eps_teacher_family_summary.csv`.
The teacher scores 0/8 there under the threshold-0.75 calibration used throughout the
paper, and 0/8 at each of the six thresholds swept (0.45–0.90); the sweep used 8
episodes at 60 steps per episode. This is a separate, shorter run from the 24-episode
carrot evaluation whose rows appear in the seed-0 evaluator summary under
`diagnostic_72_protocol/`, where the teacher likewise scores 0/24.

**Duplicate trace files across sweep directories.** Baseline rollouts are shared across
sweeps by design: the offset-0, clamp-baseline, and panel traces are the same underlying
episodes under different labels, so byte-identical files across those directories are
expected. Where traces for two clamp settings coincide on an episode, the commanded z
never crossed the clamp floor in that episode (the clamp was a no-op there).

---

## Note on residual strings

Automated screening for identifying strings returns zero hits for user names, host
names, institution names, e-mail addresses, IP addresses, repository URLs, commit
identifiers and experiment-tracker URLs.

Sixty-three lines across fifty-six log files still contain the substring `huggingface`.
All are one of two non-identifying forms: a path into the local model cache for a
public pre-trained checkpoint, or the name of a Python library module appearing in a
deprecation warning. They were deliberately left intact rather than edited, because
rewriting a library name inside a runtime log would alter the technical record without
improving anonymity.
