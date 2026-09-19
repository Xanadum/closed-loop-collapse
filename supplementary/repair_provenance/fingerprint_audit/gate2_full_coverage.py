"""
Gate 2: full 324-episode coverage. Confirms every built episode maps to a DISTINCT real
collected success episode (bijection -> no silent drop / corruption / duplication), every
built episode's images are byte-faithful (full image-stack fingerprint), and every built
episode's step-0 action == original chunk_0000.npy.

Why not compare 324 x all-steps: action injection is ONE uniform code path (no per-episode
branch); the 8-episode exact-zero round-trip already proves it lossless. This gate closes the
"all-episodes existence + image fidelity + bijection" gap.

Reuses verify_roundtrip's fingerprint + original-file readers. Streaming hash, low memory.
"""

import hashlib
import os

import numpy as np

from teacher_pipeline import make_loader
from verify_roundtrip import _list_success_eps, build_step0_fingerprint_index

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tfds_data")


def main():
    os.environ["CHUNK_SOURCE"] = "predicted"
    eps = _list_success_eps()
    fp_index = build_step0_fingerprint_index(eps)   # full-stack fp -> (seed, eid, ep_dir)
    print(f"original success episodes: {len(eps)}; unique full-stack fingerprints: {len(fp_index)}")

    matched_orig = {}        # (seed,eid) -> count matched (must end == 1 each)
    n_built = 0
    unmatched = 0
    step0_errs = []          # max abs err of step0 action vs chunk_0000
    step0_fail = []

    # make_single_dataset splits train[:95%] (train=True) / train[95%:] (train=False);
    # iterate BOTH to cover all 324 episodes.
    for split_train in (True, False):
        ds = make_loader(DATA_DIR, train=split_train, shuffle=False)
        for traj in ds.as_numpy_iterator():
            n_built += 1
            img = traj["observation"]["image_primary"]   # [T, win, H, W, 3]
            act = traj["action"]                           # [T, win, 4, 7]
            T = img.shape[0]
            h = hashlib.sha1()
            for t in range(T):
                h.update(np.ascontiguousarray(img[t, -1], dtype=np.uint8).tobytes())
            fp = h.hexdigest()
            if fp not in fp_index:
                unmatched += 1
                continue
            seed, eid, ep_dir = fp_index[fp]
            matched_orig[(seed, eid)] = matched_orig.get((seed, eid), 0) + 1

            # step0 action == original chunk_0000
            chunk0 = np.load(os.path.join(ep_dir, "chunk_0000.npy")).astype(np.float32)
            err = float(np.max(np.abs(act[0, -1].reshape(4, 7) - chunk0)))
            step0_errs.append(err)
            if err > 1e-5:
                step0_fail.append((seed, eid, err))

    dup = {k: v for k, v in matched_orig.items() if v > 1}
    n_distinct = len(matched_orig)

    print(f"\nbuilt episodes iterated:        {n_built}")
    print(f"matched to a distinct original: {n_distinct}")
    print(f"unmatched built episodes:       {unmatched}")
    print(f"originals matched >1 time (dup): {len(dup)} {dict(list(dup.items())[:5])}")
    print(f"originals never matched:        {len(eps) - n_distinct}")
    print(f"step0 action checks:            {len(step0_errs)}  max abs err {max(step0_errs):.3e}")
    print(f"step0 failures (>1e-5):         {len(step0_fail)} {step0_fail[:5]}")

    bijection = (n_built == 324 and n_distinct == 324 and unmatched == 0
                 and len(dup) == 0 and (len(eps) - n_distinct) == 0)
    step0_ok = (len(step0_errs) == 324 and max(step0_errs) <= 1e-5)
    ok = bijection and step0_ok
    print(f"\n=== GATE 2 RESULT: {'PASS' if ok else 'FAIL'} ===")
    print(f"  bijection 324<->324: {bijection}; all step0 actions exact: {step0_ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
