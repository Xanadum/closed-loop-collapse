"""Part A normalization consistency: control dataset_statistics action mean/std must EXACTLY
match V1 (control was forked from V1, inherits same stats). Pure-python (no numpy needed)."""
import json

CONTROL = "~/octo_repair_ckpts/mainb_oxeonly_rhoR000_control/dataset_statistics.json"
V1 = "~/hf_models/octo-distill-8l/dataset_statistics.json"


def load(p):
    with open(p) as f:
        return json.load(f)


def walk_action_stats(d, prefix=""):
    """Yield (path, key, list) for every action.mean / action.std found at any depth."""
    if isinstance(d, dict):
        if "action" in d and isinstance(d["action"], dict):
            a = d["action"]
            for k in ("mean", "std"):
                if k in a:
                    yield (prefix, k, a[k])
        for kk, vv in d.items():
            yield from walk_action_stats(vv, prefix + "/" + str(kk))


c = load(CONTROL)
v = load(V1)

cmap = {(p, k): vals for p, k, vals in walk_action_stats(c)}
vmap = {(p, k): vals for p, k, vals in walk_action_stats(v)}

shared = sorted(set(cmap) & set(vmap))
print(f"control action-stat entries: {len(cmap)}  V1: {len(vmap)}  shared: {len(shared)}")

maxd = 0.0
nbad = 0
for key in shared:
    cv, vv_ = cmap[key], vmap[key]
    if len(cv) != len(vv_):
        print(f"LENGTH MISMATCH at {key}: {len(cv)} vs {len(vv_)}")
        nbad += 1
        continue
    d = max(abs(a - b) for a, b in zip(cv, vv_))
    maxd = max(maxd, d)
    if d > 0:
        print(f"DIFF at {key}: max|Δ|={d}")
        nbad += 1

only_c = sorted(set(cmap) - set(vmap))
only_v = sorted(set(vmap) - set(cmap))
if only_c:
    print(f"only in control: {only_c}")
if only_v:
    print(f"only in V1: {only_v}")

print(f"\nmax|Δ| over {len(shared)} shared action mean/std entries = {maxd:.3e}")
if nbad == 0 and maxd == 0.0 and not only_c and not only_v:
    print("NORM CHECK PASS: control dataset_statistics == V1 (de-normalization identical)")
else:
    print("NORM CHECK FAIL: STOP — control stats differ from V1 (this is a bug)")
