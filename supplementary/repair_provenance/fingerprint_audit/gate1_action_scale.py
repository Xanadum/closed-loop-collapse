"""
Gate 1 (local Property B): confirm #2 (teacher RLDS) through the SAME loader pipeline
lands in the model's expected format + standard scale.

No model forward. Reuses teacher_pipeline.make_loader. Tokenizer = t5-base (offline cache),
the same text_processor octo-base-1.5 uses, applied via the trainer's process_text.
"""

import os

import numpy as np

from teacher_pipeline import make_loader, ACTION_MEAN, ACTION_STD

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tfds_data")
DIM_NAMES = ["x", "y", "z", "rot_x", "rot_y", "rot_z", "gripper"]
B = 16


def make_batch(traj, b):
    """Flatten a trajectory [T, win, ...] into frames and take the first b as a batch [b, win, ...]."""
    img = traj["observation"]["image_primary"]          # [T, win, H, W, 3]
    act = traj["action"]                                 # [T, win, 4, 7]
    tpm = traj["observation"]["timestep_pad_mask"]       # [T, win]
    apm = traj["action_pad_mask"]                        # [T, win, 4, 7]
    instr = traj["task"]["language_instruction"]         # [T]
    n = min(b, img.shape[0])
    return {
        "observation": {
            "image_primary": img[:n],
            "timestep_pad_mask": tpm[:n],
        },
        "action": act[:n],
        "action_pad_mask": apm[:n],
        "task": {"language_instruction": instr[:n]},
    }


def main():
    os.environ["CHUNK_SOURCE"] = "predicted"
    ds = make_loader(DATA_DIR, train=True, shuffle=False)

    # ---- structure of one batch ----
    first = next(ds.as_numpy_iterator())
    batch = make_batch(first, B)

    print("=== BATCH STRUCTURE (CHUNK_SOURCE=predicted) ===")
    img = batch["observation"]["image_primary"]
    act = batch["action"]
    tpm = batch["observation"]["timestep_pad_mask"]
    apm = batch["action_pad_mask"]
    print(f"observation.image_primary : shape {img.shape} dtype {img.dtype}   "
          f"(expect [B,2,256,256,3] uint8)")
    print(f"action                    : shape {act.shape} dtype {act.dtype}   "
          f"(expect [B,2,4,7] float32)")
    print(f"observation.timestep_pad_mask: shape {tpm.shape} dtype {tpm.dtype}")
    print(f"action_pad_mask           : shape {apm.shape} dtype {apm.dtype}")

    # ---- tokenize language exactly like the trainer (process_text + t5-base) ----
    from octo.data.utils.text_processing import HFTokenizer
    from octo.utils.train_utils import process_text

    tp = HFTokenizer("t5-base")
    btxt = {"task": {"language_instruction": [s for s in batch["task"]["language_instruction"]]}}
    btxt = process_text(btxt, tp)
    lang = btxt["task"]["language_instruction"]
    print(f"task.language_instruction (tokenized) keys: {list(lang.keys())}")
    print(f"  input_ids shape {np.asarray(lang['input_ids']).shape}, "
          f"attention_mask shape {np.asarray(lang['attention_mask']).shape}")
    sample_txt = batch["task"]["language_instruction"][0]
    sample_txt = sample_txt.decode() if isinstance(sample_txt, bytes) else str(sample_txt)
    print(f"  sample instruction: {sample_txt!r}  -> first ids {np.asarray(lang['input_ids'])[0][:8]}")

    # compare with train_distill dummy self-check
    print("\n  [train_distill.py dummy self-check uses window_size=2, action_horizon=4,")
    print("   resize {'primary':(256,256)}; obs keys include image_primary + timestep_pad_mask;")
    print("   action shape (B,2,4,7) -> MATCHES above.]")

    # ---- per-dim action stats over >=30 episodes ----
    print("\n=== PER-DIM ACTION STATS (teacher_chunk[h=0], current window slot) ===")
    vals = []
    n_ep = 0
    for traj in ds.as_numpy_iterator():
        vals.append(traj["action"][:, -1, 0, :])   # [T,7]
        n_ep += 1
        if n_ep >= 40:
            break
    A = np.concatenate(vals, axis=0)
    print(f"episodes aggregated: {n_ep}, steps: {A.shape[0]}")
    print(f"{'dim':>3} {'name':8} {'min':>9} {'max':>9} {'mean':>9} {'std':>9}")
    for d in range(7):
        print(f"{d:>3} {DIM_NAMES[d]:8} {A[:,d].min():9.4f} {A[:,d].max():9.4f} "
              f"{A[:,d].mean():9.4f} {A[:,d].std():9.4f}")

    # gripper stored RAW/UNSCALED in ~[0,1] (continuous diffusion-head output clustered near 0/1),
    # NOT z-normalized (a normalized gripper would center ~0, std~1, span ~+/-3).
    g = A[:, 6]
    in_unit = float(((g > -0.3) & (g < 1.3)).mean())                 # raw [0,1] scale
    bimodal = float(((np.abs(g) < 0.25) | (np.abs(g - 1) < 0.25)).mean())  # mass near 0 or 1
    uniq_near = (in_unit > 0.99) and (bimodal > 0.9)
    print(f"\ngripper RAW/UNSCALED in ~[0,1] & bimodal: {uniq_near}")
    print(f"  range [{g.min():.3f}, {g.max():.3f}], mean {g.mean():.3f}, std {g.std():.3f}")
    print(f"  frac in [-0.3,1.3]: {in_unit:.3f}; frac within 0.25 of 0 or 1: {bimodal:.3f}")
    print(f"  (z-normalized gripper would instead center ~0, std~1, span ~+/-3 -> NOT the case)")

    # ---- reverse cross-check: un-normalize dims 0-5 to physical scale ----
    print("\n=== REVERSE UN-NORMALIZE (dims 0-5) -> physical action deltas ===")
    phys = A[:, :6] * ACTION_STD[:6] + ACTION_MEAN[:6]   # normalized -> raw delta
    print(f"{'dim':>3} {'name':8} {'phys_min':>11} {'phys_max':>11} {'phys_mean':>11} {'phys_std':>11}")
    for d in range(6):
        print(f"{d:>3} {DIM_NAMES[d]:8} {phys[:,d].min():11.5f} {phys[:,d].max():11.5f} "
              f"{phys[:,d].mean():11.5f} {phys[:,d].std():11.5f}")
    pos_ok = np.max(np.abs(phys[:, :3])) < 0.5    # position deltas: mm-cm scale, well under 0.5 m
    rot_ok = np.max(np.abs(phys[:, 3:6])) < 1.5   # rotation deltas: small radians
    print(f"\nposition deltas (dims 0-2) |max| = {np.max(np.abs(phys[:,:3])):.4f} m  (sane if < ~0.5)")
    print(f"rotation deltas (dims 3-5) |max| = {np.max(np.abs(phys[:,3:6])):.4f} rad (sane if < ~1.5)")

    # ---- verdict ----
    scale_ok = (np.max(np.abs(A[:, :6])) < 50) and (np.max(np.abs(A[:, :6])) > 0.1)
    ok = bool(img.shape[1:] == (2, 256, 256, 3) and img.dtype == np.uint8
              and act.shape[1:] == (2, 4, 7) and act.dtype == np.float32
              and uniq_near and pos_ok and rot_ok and scale_ok)
    print(f"\n=== GATE 1 RESULT: {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
