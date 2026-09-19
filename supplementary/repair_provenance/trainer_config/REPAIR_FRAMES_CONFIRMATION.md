# Trainer configuration record: repair-stream size (38,880 frames)

Confirms the size of the repair data stream that the rho_R = 0.5 run consumed, and that
the rho_R = 0 control consumed none of it.

## Source 1 — logged config of the repair run

`../execution_records/rhoR050/config.yaml`, verbatim (desensitised copy in this package):

```yaml
repair_episodes:
    value: 324
repair_frames:
    value: 38880
repair_steps_per_epoch:
    value: 607.5
rho_R:
    value: 0.5
B_P:
    value: 64
B_R:
    value: 64
```

## Source 2 — same values echoed at runtime

`../execution_records/rhoR050/output.log`, line 2 (paths already relativised):

```
I0620 21:01:24.239600 ... train_repair_mainb.py:262] repair epoch = 38880 frames / 64 per step = 607.5 steps/epoch
```

## Source 3 — the control run, same fields

`../execution_records/rhoR000/config.yaml`:

```yaml
repair_episodes:
    value: 0
repair_frames:
    value: 0
repair_steps_per_epoch:
    value: 0
rho_R:
    value: 0
B_P:
    value: 128
B_R:
    value: 0
```

## Arithmetic

- 324 episodes (all `success == true`; see `../collection/`) → 38,880 frames → 120 frames/episode,
  matching the `--max-episode-steps 120` used at collection time
  (`../collection/collect_teacher_v1.sh`, line 19).
- 38,880 frames / 64 per step = 607.5 optimiser steps per repair epoch.
- Total batch is 128 on both runs (64 + 64 vs 128 + 0): the repair stream **replaces**
  preserve-stream capacity, it does not add to it. Compute per step is identical.

Trainer source for both runs: `train_repair_mainb.py` in this directory
(`codePath` field of both execution records).

Field-by-field diff of the two runs: `../execution_records/CONFIG_DIFF.md`.
