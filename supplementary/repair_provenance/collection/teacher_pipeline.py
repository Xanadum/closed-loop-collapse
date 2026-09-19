"""
Phase B loader + path-B injection for the SimEggplantTeacher RLDS dataset.

The standardize_fn here is the ONLY place the teacher_chunk is wired into the
action slot. It is a dataset-specific (non-shared) function, so path B touches
no shared/OXE pipeline code.

CHUNK_SOURCE env var:
  "predicted" (default) -> traj["action"] = teacher_chunk, shape [T, 4, 7] (PRE-CHUNKED).
                           octo.data.traj_transforms.chunk_act_obs detects the 3-D
                           action and only adds the history axis -> [T, win, 4, 7].
  "executed"            -> traj["action"] = action_exec, shape [T, 7] (standard chunking).
"""

import os

import numpy as np
import tensorflow as tf

from octo.data.dataset import make_single_dataset
from octo.utils.spec import ModuleSpec

PRED_ACTION_HORIZON = 4
ACTION_DIM = 7

# Bridge normalization stats (chunks are already normalized; skip_norm=True at load).
ACTION_MEAN = np.array(
    [
        0.0002175864647142589, 0.00012508298095781356, -0.00017108325846493244,
        -0.00016171138850040734, -0.0002524856827221811, 0.00025157874915748835,
        0.5879484415054321,
    ],
    dtype=np.float32,
)
ACTION_STD = np.array(
    [
        0.009632384404540062, 0.013500643894076347, 0.012510603293776512,
        0.028145212680101395, 0.030282432213425636, 0.07585602253675461,
        0.4877190887928009,
    ],
    dtype=np.float32,
)


def sim_eggplant_teacher_standardize(traj: dict) -> dict:
    """Map raw RLDS step fields into the (observation, action, language) form
    that make_dataset_from_rlds.restructure expects. CHUNK_SOURCE picks the action."""
    src = os.environ.get("CHUNK_SOURCE", "predicted")
    if src == "predicted":
        # pre-chunked: [T, 4, 7] -> chunk_act_obs adds only the history axis
        action = tf.cast(traj["teacher_chunk"], tf.float32)
    elif src == "executed":
        action = tf.cast(traj["action_exec"], tf.float32)  # [T, 7] standard chunking
    else:
        raise ValueError(f"Unknown CHUNK_SOURCE={src!r} (expected 'predicted' or 'executed')")

    return {
        "observation": {"image_primary": traj["observation"]["image_primary"]},
        "language_instruction": traj["language_instruction"],
        "action": action,
    }


def _dataset_statistics():
    """Explicit stats so get_dataset_statistics is never invoked (moot under skip_norm)."""
    return {
        "action": {
            "mean": ACTION_MEAN,
            "std": ACTION_STD,
            "max": np.full(ACTION_DIM, 10.0, dtype=np.float32),
            "min": np.full(ACTION_DIM, -10.0, dtype=np.float32),
            "p99": np.full(ACTION_DIM, 5.0, dtype=np.float32),
            "p01": np.full(ACTION_DIM, -5.0, dtype=np.float32),
        },
        "num_transitions": 1,
        "num_trajectories": 1,
    }


def make_loader(data_dir, train=True, shuffle=False):
    """Returns an Octo single-dataset of trajectories with path-A/B action injection."""
    dataset_kwargs = dict(
        name="sim_eggplant_teacher",
        data_dir=data_dir,
        image_obs_keys={"primary": "image_primary"},
        depth_obs_keys={},
        proprio_obs_key=None,
        language_key="language_instruction",
        standardize_fn=ModuleSpec.create(sim_eggplant_teacher_standardize),
        shuffle=shuffle,
        skip_norm=True,
        action_normalization_mask=[True, True, True, True, True, True, False],
        dataset_statistics=_dataset_statistics(),
    )
    traj_transform_kwargs = dict(
        window_size=2,
        action_horizon=PRED_ACTION_HORIZON,
    )
    frame_transform_kwargs = dict(
        resize_size={"primary": (256, 256)},
    )
    return make_single_dataset(
        dataset_kwargs,
        train=train,
        traj_transform_kwargs=traj_transform_kwargs,
        frame_transform_kwargs=frame_transform_kwargs,
    )
