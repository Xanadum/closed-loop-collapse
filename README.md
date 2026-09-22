# Anatomy of a Closed-Loop Collapse

Supplementary records for the paper *Anatomy of a Closed-Loop Collapse: A Causal Case Study of a Compressed VLA Policy* (IROS 2026 ScaleInfra workshop, poster).

**Author:** Fengze Jia, The Ohio State University, Columbus, OH, USA ([fengzejia.com](https://fengzejia.com))

**Paper:** [arXiv:2609.23048](https://arxiv.org/abs/2609.23048) (a copy is included as `paper.pdf`)

## Where the paper's references point

| Paper location | Content | Path |
|---|---|---|
| Sec. III-B | Per-dataset offline evaluation records (six datasets, teacher and student) | `supplementary/offline_eval/eval_results_v3.json` |
| Sec. III-C (footnote) | Metric definitions (the environment evaluator's predicates) | `supplementary/offline_eval/METRIC_DEFINITIONS.md` |
| Sec. V-C | Micro-displacement trace measurements under z-offset injection | `supplementary/micro_displacement/` (z-offset sweep: `eval_logs_zoffset_ztrace/`, `z_offset_absorption.csv`) |
| Sec. VI-B | Execution records of the two minimal-pair runs (tracking config + command-line metadata; rho_R = 0.5 and rho_R = 0) | `supplementary/repair_provenance/execution_records/rhoR050/`, `supplementary/repair_provenance/execution_records/rhoR000/` |

The package's own index is `supplementary/README.md`; per-file origins are listed in `supplementary/PROVENANCE.txt`.

Model checkpoints and the teacher-rollout dataset are not part of this package.

## License

Code in this repository is released under the MIT License (see `LICENSE`).
Data and documents are released under CC BY 4.0 (see `LICENSE-DATA`).

## Citation

```bibtex
@misc{jia2026anatomy,
  title         = {Anatomy of a Closed-Loop Collapse: A Causal Case Study of a Compressed VLA Policy},
  author        = {Jia, Fengze},
  year          = {2026},
  eprint        = {2609.23048},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2609.23048}
}
```
