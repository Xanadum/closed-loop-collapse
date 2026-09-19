# Phase B — RLDS Dataset Construction REPORT

Lossless format translation of collected teacher rollouts into an Octo-trainable
RLDS dataset. No new numbers were produced; every value was read verbatim off disk
and verified by a round-trip equality check against the original collected files.

## Success markers

### ① tfds build OK
- Builder: `sim_eggplant_teacher_dataset_builder.py` (`SimEggplantTeacher`, TFDS `GeneratorBasedBuilder`).
- Built via `build_teacher_rlds.py` → `download_and_prepare()`.
- Dataset path: `tfds_data/sim_eggplant_teacher/1.0.0/` (3.1 GB, 32 train shards).
- Split: `train` = **324 examples** (one RLDS episode per success rollout; key `seed{n}_ep{eid}`).
- Features:
  - `observation/image_primary`: `Image(256,256,3) uint8` (lossless PNG)
  - `language_instruction`: `Text`
  - `action_exec`: `Tensor(7,) float32`  (= `norm_exec`, executed normalized action)
  - `teacher_chunk`: `Tensor(4,7) float32`  (= `norm_chunk` = `norm_raw_actions[0]`, the training target)
  - `episode_metadata`: `{seed:int32, eid:int32, file_path:Text}`
- Action `mean`/`std` (bridge stats) stored in `metadata.json` for traceability.

### ② visualize OK
- `visualize_dataset.py` → `phase_b_viz/`:
  - `episode_{0,1,2}_frames.png` — confirmed eggplant scene (purple eggplant, yellow basket,
    sink); strips show approach → grasp → lift → carry-to-basket.
  - `action_histograms.png` — per-dim `teacher_chunk[h=0]` over 7200 steps from 60 episodes.
- Distributions are sane:
  - dims 0–5 (xyz, rot) unimodal, near-zero mean; **z mean −0.291** (the documented
    post-grasp downward bias — expected, not a bug).
  - dim 6 (gripper) **clean bimodal**: sharp peaks at 0.0 (close) and 1.0 (open). Gripper is
    stored raw in [0,1] (un-normalized, `action_normalization_mask[6]=False`), as Octo's head emits.

### ③ round-trip assertion (decisive)
- `verify_roundtrip.py`, `CHUNK_SOURCE=predicted`.
- Identity: each pipeline-output trajectory matched to its source episode by a **full
  image-stack fingerprint** (324/324 unique). (Step-0 image alone is shared across seeds with
  the same eid — config-determined — so a step-0-only fingerprint was non-discriminative; fixed.)
- **N = 24 (episode, step) checks across 8 episodes**, each episode contributing step 0
  (window padding), one interior step, and the last step. **step0(pad) covered: 8, last: 8.**
- Compared against the **ORIGINAL collected files** (`chunk_*.npy`, `obs/*.npy`, `actions.csv`):
  - (a) batch `action[t,-1]` reshaped (4,7) vs `chunk_{t}.npy`: **max abs err = 0.000e+00** (≤ 1e-5)
  - (b) batch `image_primary[t,-1]` vs `obs/{t}.npy`: **byte-exact (True)**
  - (c) `language_instruction[t]` vs csv: **match (True)**
- **RESULT: PASS.**

### ④ episode count + isolation
- Episodes = **324** (all `success=true`).
- **Zero test-eid contamination** (eids ⊆ `{0,1,2,9,10,11,12,13,14,21,22,23}`).
- **Zero eval-seed contamination** (seeds ∈ 100..179, never `{0,1,2}`).
- Builder enforces both as defense-in-depth (`_list_success_episodes` filters on eid/seed/success).

### ⑤ imports OK
- `tensorflow_datasets` 4.9.4, `tensorflow` 2.15.0.
- `octo.data.dataset.make_single_dataset`, `octo.data.oxe.oxe_standardization_transforms`,
  chunking transform `octo.data.traj_transforms.chunk_act_obs` — all import cleanly (offline).

## Path-B injection mechanism

`octo.data.traj_transforms.chunk_act_obs` **natively supports pre-chunked actions**: if
`traj["action"]` arrives 3-D `[traj_len, N≥action_horizon, dim]`, it slices `[:, :action_horizon]`
and adds only the history axis (it does NOT re-derive a chunk from future timesteps). We exploit
this entirely inside our dataset-specific `standardize_fn` (`teacher_pipeline.py`):

- `CHUNK_SOURCE=predicted` (default, path B) → `traj["action"] = teacher_chunk` shape `[T,4,7]`.
  `chunk_act_obs(window_size=2, action_horizon=4)` → `[T, 2, 4, 7]`; the current window slot
  (`[t,-1]`) carries exactly `teacher_chunk[t]`. **No edit to any shared/OXE pipeline code** — the
  fuse (modify shared pipeline that touches the OXE preserve flow) was never tripped.
- `CHUNK_SOURCE=executed` (path A, config switch, **not run this round**) → `traj["action"] =
  action_exec` shape `[T,7]`; standard `range(action_horizon)` gather of executed actions.

Loader config (`make_single_dataset`): `skip_norm=True` (chunks already normalized),
`window_size=2`, `action_horizon=4`, `image_obs_keys={"primary":"image_primary"}`,
`language_key="language_instruction"`, `action_normalization_mask=[T,T,T,T,T,T,F]` (moot under
skip_norm), `dataset_statistics` passed explicitly (bridge mean/std) so `get_dataset_statistics`
is never invoked.

## Scope / what was NOT done
- No upload to A100. No full smoke train locally (RTX3070 8GB OOMs the trainer; round-trip uses
  only the data pipeline — no model forward — so it ran CPU/RAM-only).
- Collected data untouched. No edits to `octo_model.py`, no env/package changes.

## Pre-upload local data gates (2026-06-17)

Two final local gates before A100 upload/training. Both use only the data pipeline
(no model forward), CPU/RAM only.

### Gate 1 — action scale/format alignment (local Property B) — PASS
`gate1_action_scale.py`, `CHUNK_SOURCE=predicted`. Batch built by flattening one trajectory
to frames (B=16).

Batch structure (matches `train_distill.py` dummy self-check: window_size=2, action_horizon=4,
resize primary→(256,256)):
- `observation.image_primary`: `(16, 2, 256, 256, 3)` uint8 ✓ (= [B,2,256,256,3])
- `action`: `(16, 2, 4, 7)` float32 ✓ (= [B,2,4,7])
- `observation.timestep_pad_mask`: `(16, 2)` bool ✓
- `action_pad_mask`: `(16, 2, 4, 7)` bool ✓
- `task.language_instruction` (after `process_text` + t5-base tokenizer, the same octo-base-1.5
  uses): `{input_ids (16,64), attention_mask (16,64)}` ✓ — "put eggplant into yellow basket" →
  ids `[474, 30898, 139, 4459, 8235, 1, 0, …]`.

Per-dim action stats (teacher_chunk[h=0], 40 episodes / 4800 steps):

| dim | name | min | max | mean | std |
|----|----|----|----|----|----|
| 0 | x | −3.66 | 5.00 | −0.04 | 0.47 |
| 1 | y | −2.89 | 4.54 | +0.07 | 0.52 |
| 2 | z | −2.82 | 3.16 | **−0.27** | 0.66 |
| 3 | rot_x | −4.58 | 3.09 | +0.06 | 0.67 |
| 4 | rot_y | −4.96 | 3.46 | +0.07 | 0.77 |
| 5 | rot_z | −3.35 | 3.37 | +0.04 | 0.47 |
| 6 | gripper | −0.10 | 1.20 | +0.72 | 0.44 |

- dims 0–5 on **standard normalized scale** (~0 mean, std ~0.5–0.8; slightly small/shifted because
  this is a single task — expected, not an error). Not the raw ~0.001 micro-deltas, not hundreds.
- dim 6 (gripper) **raw/unscaled in ~[0,1], bimodal** (frac in [−0.3,1.3] = 1.000; frac within 0.25
  of 0 or 1 = 0.990). A z-normalized gripper would instead center ~0 with std~1 spanning ±3 — it does not.
- **Reverse cross-check** (un-normalize dims 0–5 with bridge mean/std → physical deltas): position
  deltas |max| = **0.061 m** (mm–cm scale), rotation deltas |max| = **0.256 rad** (small radians). Sane.
- **GATE 1: PASS.**

### Gate 2 — full 324-episode coverage (bijection) — PASS
`gate2_full_coverage.py`. Iterates ALL built episodes across both pipeline splits
(`train[:95%]` = 308 + `train[95%:]` = 16), full-stack fingerprint match to originals:
- built episodes iterated: **324**
- matched to a **distinct** original: **324** (bijection)
- unmatched: **0**; originals matched >1× (dup): **0**; originals never matched: **0**
- step-0 action == original `chunk_0000.npy` for all **324** episodes, max abs err **0.000e+00**
- **GATE 2: PASS.** (No silent drop / corruption / duplication; all 324 images byte-faithful;
  all 324 step-0 actions exact.) Why not 324×all-steps: action injection is one uniform code path
  with no per-episode branch — the 8-episode exact-zero round-trip already proves it lossless; this
  gate closes the all-episodes existence + image-fidelity + bijection gap.

**Both local gates PASS → dataset is format/scale-aligned and fully accounted for.**

## Files added
- `sim_eggplant_teacher_dataset_builder.py` — RLDS builder
- `build_teacher_rlds.py` — build driver
- `teacher_pipeline.py` — standardize_fn (CHUNK_SOURCE switch) + `make_loader`
- `verify_roundtrip.py` — Phase 3 round-trip verification
- `visualize_dataset.py` — Phase 4 visualization
- `gate1_action_scale.py` — Gate 1 (action scale/format alignment)
- `gate2_full_coverage.py` — Gate 2 (full 324-episode bijection)
- `tfds_data/sim_eggplant_teacher/1.0.0/` — built dataset (3.1 GB)
- `phase_b_viz/` — frame strips + action histograms
