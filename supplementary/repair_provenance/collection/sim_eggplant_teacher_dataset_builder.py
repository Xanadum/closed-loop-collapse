"""
TFDS GeneratorBasedBuilder for the teacher-rollout eggplant dataset (Phase B).

Lossless format translation: wraps the collected teacher rollouts
(rollout_teacher_collect_v1/seed_<n>/ep_<eid>/) into an Octo-trainable RLDS
dataset. We produce NO new numbers; every value is read verbatim off disk.

Per-step features:
  observation/image_primary : (256,256,3) uint8   (= obs/{step:04d}.npy, lossless PNG)
  language_instruction      : text                (= per-step instruction; constant per episode)
  action_exec               : (7,)   float32      (= norm_exec, the executed normalized action)
  teacher_chunk             : (4,7)  float32      (= norm_chunk = norm_raw_actions[0], the training target)

Only success==true episodes are included. eids are train-only
{0,1,2,9,10,11,12,13,14,21,22,23}; seeds are 100..179 (never the eval seeds 0/1/2).

Build with build_teacher_rlds.py (calls download_and_prepare directly).
"""

import csv
import glob
import json
import os

import numpy as np
import tensorflow_datasets as tfds

# --- collection layout / isolation constants -------------------------------
COLLECT_DIR = os.environ.get(
    "TEACHER_COLLECT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "rollout_teacher_collect_v1"),
)
TRAIN_EIDS = {0, 1, 2, 9, 10, 11, 12, 13, 14, 21, 22, 23}
EVAL_SEEDS = {0, 1, 2}
PRED_ACTION_HORIZON = 4
ACTION_DIM = 7
IMG_HW = (256, 256, 3)


def _list_success_episodes(collect_dir):
    """Return [(seed:int, eid:int, ep_dir:str), ...] for success episodes only."""
    out = []
    for ep_dir in sorted(glob.glob(os.path.join(collect_dir, "seed_*", "ep_*"))):
        parts = ep_dir.split(os.sep)
        seed = int(parts[-2].replace("seed_", ""))
        eid = int(parts[-1].replace("ep_", ""))
        # isolation guards (defense in depth; builder must never ingest test data)
        if eid not in TRAIN_EIDS:
            continue
        if seed in EVAL_SEEDS:
            continue
        oc = os.path.join(ep_dir, "episode_outcome.json")
        if not os.path.exists(oc):
            continue
        try:
            outcome = json.load(open(oc))
        except Exception:
            continue
        if not outcome.get("success"):
            continue
        out.append((seed, eid, ep_dir))
    return out


def _load_norm_exec_and_instruction(ep_dir):
    """Read actions.csv -> (instructions[list[str]], norm_exec[np.ndarray (T,7)])."""
    instrs = []
    rows = []
    with open(os.path.join(ep_dir, "actions.csv")) as f:
        reader = csv.DictReader(f)
        for r in reader:
            instrs.append(r["instruction"])
            rows.append([float(r[f"norm_exec_{i}"]) for i in range(ACTION_DIM)])
    return instrs, np.asarray(rows, dtype=np.float32)


class SimEggplantTeacher(tfds.core.GeneratorBasedBuilder):
    """Teacher-rollout eggplant RLDS dataset for closed-loop distillation."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial: 324 success episodes, teacher_chunk + action_exec."}

    def _info(self) -> tfds.core.DatasetInfo:
        # Bridge normalization stats (recorded so loaders can pass them explicitly).
        meta = tfds.core.MetadataDict(
            action_mean=[
                0.0002175864647142589, 0.00012508298095781356, -0.00017108325846493244,
                -0.00016171138850040734, -0.0002524856827221811, 0.00025157874915748835,
                0.5879484415054321,
            ],
            action_std=[
                0.009632384404540062, 0.013500643894076347, 0.012510603293776512,
                0.028145212680101395, 0.030282432213425636, 0.07585602253675461,
                0.4877190887928009,
            ],
            gripper_mode="threshold_075",
            pred_action_horizon=PRED_ACTION_HORIZON,
            source="rollout_teacher_collect_v1 (teacher = octo-base-1.5 + threshold075)",
            note="actions stored ALREADY NORMALIZED (skip_norm=True at load time)",
        )
        return tfds.core.DatasetInfo(
            builder=self,
            description="Teacher closed-loop rollouts on PutEggplantInBasketScene-v0 "
            "(success episodes only). teacher_chunk = norm_raw_actions[0].",
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image_primary": tfds.features.Image(
                                        shape=IMG_HW, dtype=np.uint8, encoding_format="png"
                                    ),
                                }
                            ),
                            "language_instruction": tfds.features.Text(),
                            "action_exec": tfds.features.Tensor(
                                shape=(ACTION_DIM,), dtype=np.float32
                            ),
                            "teacher_chunk": tfds.features.Tensor(
                                shape=(PRED_ACTION_HORIZON, ACTION_DIM), dtype=np.float32
                            ),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "seed": np.int32,
                            "eid": np.int32,
                            "file_path": tfds.features.Text(),
                        }
                    ),
                }
            ),
            metadata=meta,
        )

    def _split_generators(self, dl_manager):
        episodes = _list_success_episodes(COLLECT_DIR)
        return {"train": self._generate_examples(episodes)}

    def _generate_examples(self, episodes):
        for seed, eid, ep_dir in episodes:
            chunk_files = sorted(glob.glob(os.path.join(ep_dir, "chunk_*.npy")))
            instrs, norm_exec = _load_norm_exec_and_instruction(ep_dir)
            # trajectory length = min of the three sources (they match in practice)
            T = min(len(chunk_files), norm_exec.shape[0])
            steps = []
            for t in range(T):
                img = np.load(os.path.join(ep_dir, "obs", f"{t:04d}.npy"))
                assert img.shape == IMG_HW and img.dtype == np.uint8, (ep_dir, t, img.shape, img.dtype)
                chunk = np.load(os.path.join(ep_dir, f"chunk_{t:04d}.npy")).astype(np.float32)
                assert chunk.shape == (PRED_ACTION_HORIZON, ACTION_DIM), (ep_dir, t, chunk.shape)
                steps.append(
                    {
                        "observation": {"image_primary": img},
                        "language_instruction": instrs[t],
                        "action_exec": norm_exec[t].astype(np.float32),
                        "teacher_chunk": chunk,
                    }
                )
            key = f"seed{seed}_ep{eid}"
            yield key, {
                "steps": steps,
                "episode_metadata": {
                    "seed": np.int32(seed),
                    "eid": np.int32(eid),
                    "file_path": ep_dir,
                },
            }
