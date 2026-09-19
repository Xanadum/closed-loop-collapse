"""
Phase 3 round-trip verification (decisive).

For each pipeline-output trajectory (out of make_single_dataset, CHUNK_SOURCE=predicted),
we identify its source episode by a content fingerprint of its step-0 current image,
then assert against the ORIGINAL collected files (never the dataset itself):
  (a) batch action[t, -1] (current window slot, shape 4x7) ~= chunk_{t}.npy   (atol 1e-5)
  (b) batch image_primary[t, -1] ~= obs/{t}.npy                                (exact, PNG lossless)
  (c) task language_instruction[t] == actions.csv instruction
We sample N>=20 (episode, step) pairs and ALWAYS include each sampled episode's
step 0 (window padding) and last step.
"""

import csv
import glob
import hashlib
import os

import numpy as np
import tensorflow as tf

from teacher_pipeline import make_loader

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tfds_data")
COLLECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rollout_teacher_collect_v1")
TRAIN_EIDS = {0, 1, 2, 9, 10, 11, 12, 13, 14, 21, 22, 23}
EVAL_SEEDS = {0, 1, 2}


def _img_fp(arr):
    return hashlib.sha1(np.ascontiguousarray(arr, dtype=np.uint8).tobytes()).hexdigest()


def _stack_fp(ep_dir):
    """Fingerprint the FULL image stack (unique per seed; step-0 alone is shared across seeds
    with the same eid because the initial frame is determined by object config, not seed)."""
    files = sorted(glob.glob(os.path.join(ep_dir, "obs", "*.npy")))
    h = hashlib.sha1()
    for fp in files:
        h.update(np.ascontiguousarray(np.load(fp), dtype=np.uint8).tobytes())
    return h.hexdigest()


def _list_success_eps():
    out = []
    for d in sorted(glob.glob(os.path.join(COLLECT_DIR, "seed_*", "ep_*"))):
        parts = d.split(os.sep)
        seed = int(parts[-2].replace("seed_", ""))
        eid = int(parts[-1].replace("ep_", ""))
        if eid not in TRAIN_EIDS or seed in EVAL_SEEDS:
            continue
        import json
        oc = os.path.join(d, "episode_outcome.json")
        if os.path.exists(oc) and json.load(open(oc)).get("success"):
            out.append((seed, eid, d))
    return out


def build_step0_fingerprint_index(eps):
    """sha1(full obs stack) -> (seed, eid, ep_dir). Unique per (seed, eid)."""
    idx = {}
    for seed, eid, d in eps:
        idx[_stack_fp(d)] = (seed, eid, d)
    return idx


def load_orig_instructions(ep_dir):
    instrs = []
    with open(os.path.join(ep_dir, "actions.csv")) as f:
        for r in csv.DictReader(f):
            instrs.append(r["instruction"])
    return instrs


def main():
    os.environ["CHUNK_SOURCE"] = "predicted"
    eps = _list_success_eps()
    fp_index = build_step0_fingerprint_index(eps)
    print(f"original success episodes: {len(eps)}; step0 fingerprints: {len(fp_index)} "
          f"(unique={len(fp_index)==len(eps)})")

    ds = make_loader(DATA_DIR, train=True, shuffle=False)

    rng = np.random.RandomState(0)
    # collect a handful of trajectories from the pipeline, identify each, verify boundaries+random steps
    n_traj_target = 8       # >=20 (episode,step) pairs comes from these x (2 boundaries + 1 random)
    checks = []             # (kind, seed, eid, step, max_err_action, img_ok, instr_ok)
    seen_keys = set()
    matched = 0

    for traj in ds.as_numpy_iterator():
        img = traj["observation"]["image_primary"]      # [T, win, H, W, 3]
        act = traj["action"]                              # [T, win, 4, 7]
        instr = traj["task"]["language_instruction"]      # [T]
        T = img.shape[0]
        # fingerprint the full current-window image stack (unique per seed)
        h = hashlib.sha1()
        for t in range(T):
            h.update(np.ascontiguousarray(img[t, -1], dtype=np.uint8).tobytes())
        fp = h.hexdigest()
        if fp not in fp_index:
            continue
        seed, eid, ep_dir = fp_index[fp]
        if (seed, eid) in seen_keys:
            continue
        seen_keys.add((seed, eid))
        matched += 1

        orig_instrs = load_orig_instructions(ep_dir)
        # steps to check: boundaries (0, T-1) + one random interior
        steps = {0, T - 1}
        if T > 2:
            steps.add(int(rng.randint(1, T - 1)))
        for t in steps:
            cur_img = img[t, -1]
            cur_act = act[t, -1]                          # [4,7]
            orig_chunk = np.load(os.path.join(ep_dir, f"chunk_{t:04d}.npy")).astype(np.float32)
            orig_img = np.load(os.path.join(ep_dir, "obs", f"{t:04d}.npy"))
            a_err = float(np.max(np.abs(cur_act.reshape(4, 7) - orig_chunk)))
            img_ok = bool(np.array_equal(cur_img, orig_img))
            instr_txt = instr[t].decode() if isinstance(instr[t], bytes) else str(instr[t])
            instr_ok = bool(instr_txt == orig_instrs[t])
            kind = "step0(pad)" if t == 0 else ("last" if t == T - 1 else "interior")
            checks.append((kind, seed, eid, t, a_err, img_ok, instr_ok))

        if matched >= n_traj_target:
            break

    print(f"\nmatched/identified trajectories: {matched}")
    print(f"total (episode,step) checks: {len(checks)}")
    print(f"{'kind':12s} {'seed':>5s} {'eid':>4s} {'step':>5s} {'act_max_err':>12s} {'img_ok':>7s} {'instr_ok':>9s}")
    worst_act = 0.0
    all_img = all_instr = True
    nb0 = nlast = 0
    for kind, seed, eid, t, a_err, img_ok, instr_ok in checks:
        print(f"{kind:12s} {seed:5d} {eid:4d} {t:5d} {a_err:12.3e} {str(img_ok):>7s} {str(instr_ok):>9s}")
        worst_act = max(worst_act, a_err)
        all_img = all_img and img_ok
        all_instr = all_instr and instr_ok
        if kind == "step0(pad)":
            nb0 += 1
        if kind == "last":
            nlast += 1

    print("\n=== ROUND-TRIP SUMMARY ===")
    print(f"compared against ORIGINAL collected files: chunk_*.npy / obs/*.npy / actions.csv")
    print(f"checks: {len(checks)}  | episodes: {matched}  | step0(pad) covered: {nb0}  last covered: {nlast}")
    print(f"action max abs err: {worst_act:.3e}  (threshold 1e-5)")
    print(f"images byte-exact: {all_img}")
    print(f"instructions match: {all_instr}")
    ok = (worst_act <= 1e-5) and all_img and all_instr and len(checks) >= 20 and nb0 >= 1 and nlast >= 1
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
