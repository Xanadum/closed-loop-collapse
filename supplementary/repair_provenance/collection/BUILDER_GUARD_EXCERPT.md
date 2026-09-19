# Seed / episode-ID isolation: hard-coded guards

Excerpts are verbatim; the leading numbers are the **original line numbers** in the
un-excerpted source files, which are included whole in this same directory.

---

## A. Collection script — `collect_teacher_v1.sh`

Which episode IDs and which RNG seeds the teacher rollouts were collected on.

```bash
  11    EIDS="0 1 2 9 10 11 12 13 14 21 22 23"
  12    echo "COLLECT_START $(date +%s) base=$BASE eids=[$EIDS] seeds=100..179"
  13    for seed in $(seq 100 179); do
  14      export OCTO_ROLLOUT_RECORD_DIR="$BASE/seed_$seed"
```

and the episode-selection flag actually passed to the evaluator:

```bash
  24        --obj-variation-mode episode --obj-episode-list $EIDS \
```

with the episode length that fixes frames-per-episode:

```bash
  19        --octo-init-rng "$seed" --control-freq 5 --sim-freq 500 --max-episode-steps 120 \
```

---

## B. Dataset builder — `sim_eggplant_teacher_dataset_builder.py`

Docstring statement of the isolation contract:

```python
  14    Only success==true episodes are included. eids are train-only
  15    {0,1,2,9,10,11,12,13,14,21,22,23}; seeds are 100..179 (never the eval seeds 0/1/2).
```

The constants:

```python
  33    TRAIN_EIDS = {0, 1, 2, 9, 10, 11, 12, 13, 14, 21, 22, 23}
  34    EVAL_SEEDS = {0, 1, 2}
```

The enforcing guards — an episode is skipped unless its EID is a training EID *and*
its seed is not an evaluation seed:

```python
  40    def _list_success_episodes(collect_dir):
  41        """Return [(seed:int, eid:int, ep_dir:str), ...] for success episodes only."""
  42        out = []
  43        for ep_dir in sorted(glob.glob(os.path.join(collect_dir, "seed_*", "ep_*"))):
  44            parts = ep_dir.split(os.sep)
  45            seed = int(parts[-2].replace("seed_", ""))
  46            eid = int(parts[-1].replace("ep_", ""))
  47            # isolation guards (defense in depth; builder must never ingest test data)
  48            if eid not in TRAIN_EIDS:
  49                continue
  50            if seed in EVAL_SEEDS:
  51                continue
```

and the success filter that reduces the collected rollouts to the 324 training episodes:

```python
  59            if not outcome.get("success"):
  60                continue
  61            out.append((seed, eid, ep_dir))
  62        return out
```

---

## C. Resulting disjointness

| set | members |
|---|---|
| collection EIDs (training) | 0, 1, 2, 9, 10, 11, 12, 13, 14, 21, 22, 23 |
| held-out EIDs (never collected, never trained on) | 3–8, 15–20 |
| collection seeds | 100 … 179 (80 seeds) |
| evaluation seeds | 0, 1, 2 |

Training EIDs ∩ held-out EIDs = ∅. Collection seeds ∩ evaluation seeds = ∅.
Both guards are enforced in code, not only by convention.

Round-trip verification that the built dataset contains exactly those 324 episodes and
nothing else: `../fingerprint_audit/`.
