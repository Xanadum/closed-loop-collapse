# METRIC_DEFINITIONS.md — script-level definitions of the graded metrics (CL-29 / D7)

All definitions are the ENVIRONMENT's own `evaluate()` logic. Primary source:
`ManiSkill2_real2sim/mani_skill2_real2sim/envs/custom_scenes/put_on_in_scene.py`
(base `PutOnBridgeInSceneEnv`), with the eggplant override in `PutEggplantInBasketScene`.
The four boolean fields are written into each rollout's `.mp4` filename and parsed by
`count_repair_metrics.py:19-24` (regex) → `:37-43` (success=g1, moved=g3, grasp=g5, consecutive=g6, target=g7).

## moved  (filename field `moved_correct_obj`)
`put_on_in_scene.py:60-62`:
```
moved_correct_obj = (source_obj_xy_move_dist > 0.03) and (all([x < source_obj_xy_move_dist for x in other_obj_xy_move_dist]))
```
Source object moved > **3 cm** in xy AND moved more than every other object.

## grasp  (filename field `is_src_obj_grasped`)
`put_on_in_scene.py:68`:
```
is_src_obj_grasped = self.agent.check_grasp(self.episode_source_obj)
```
Episode-level OR-accumulated across steps at `:126-128` (grasped at any step ⇒ True).

## consecutive / sustained  (filename field `consecutive_grasp`)
`put_on_in_scene.py:69-73`:
```
if is_src_obj_grasped: self.consecutive_grasp += 1
else:                  self.consecutive_grasp = 0
consecutive_grasp = self.consecutive_grasp >= 5
```
Grasp held ≥ **5 consecutive** steps. OR-accumulated at `:129-131`.

## target  (filename field `src_on_target`) — also the success criterion
`put_on_in_scene.py:92`: `src_on_target = xy_flag and z_flag`
- **xy_flag** `:84-87`: `np.linalg.norm(offset[:2]) <= np.linalg.norm(tgt_obj_half_length_bbox[:2]) + 0.003`
  → source-target xy within **target half-bbox + 3 mm**.
- **z_flag** `:88-91`: `(offset[2] > 0) and (offset[2] - tgt_half_bbox_z - src_half_bbox_z <= z_flag_required_offset)`
  → source **above** target (offset z > 0) AND within the z band.

## success
`put_on_in_scene.py:121`: `success = src_on_target` (i.e. success == target field).

## Eggplant-specific band (PutEggplantInBasketScene override)
`put_on_in_scene.py:452-454` (env registered `:375-376`):
```
return super().evaluate(success_require_src_completely_on_target=False,
                        z_flag_required_offset=0.06, ...)
```
→ **z band = 6 cm** (`z_flag_required_offset=0.06`), and the strict on-target contact check is
skipped (`success_require_src_completely_on_target=False`). Base-class defaults are 0.02 / True (`:43`).

**Net success criterion (eggplant):** source xy within target-basket half-bbox **+ 3 mm** AND source
**above** target within a **6 cm** z band. (Matches the paper's stated criterion exactly.)

## Gripper convention (CL-31), for completeness
`octo_model.py:595-596` (threshold_075): `gripper_cmd = 1.0 if open_val > 0.75 else -1.0`; env action
**+1 = open, −1 = close** (comment `:556-557`). The trace column `env_gripper` is this scalar.
"Closed" detection: `analyze_zgap.py:35` and the z-clamp phase gate use `< 0.5`; panel `close_frac`
(`analyze_action_traces_nopandas.py:122`) uses `< 0`. On the two-valued {−1,+1} signal both select
exactly the −1 (closed) rows — consistent.
