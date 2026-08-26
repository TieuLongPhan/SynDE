# Result records

These files are committed evidence, not a work directory. Temporary caches,
logs, SQLite databases, and partial training outputs belong under `/tmp` or
`data/` and are not copied here.

## Active publication records

| File | Produced by | Used by |
|---|---|---|
| `global_comparators/synde.json` | `07_train_energy.py` | publication validator, tables, figures |
| `global_comparators/benchmark.json` | `08_compare_models.py` | publication validator, tables, figures |
| `calibration_results.json` | `05_calibrate.py` | coefficient table and diagnostics |
| `external_results.json` | `06_evaluate.py` | held-out stratification figure |
| `synde_baselines_ablation.json` | `10_baselines.py` | ablation and comparison figures |
| `synde_atom_count_baseline.json` | `11_atom_counts.py` | comparison figure |
| `inference_runtime_benchmark.json` | `12_runtime.py` | comparison figure |
| `diagnostics.json` | `09_diagnostics.py` | ablation figure and coefficient table |

`amended_xtb_recovery_audit.json` is retained because it defines the 149-molecule
runtime subset recorded in `inference_runtime_benchmark.json`. The old label-
seeding run and its reuse audit were removed; they depended on unavailable
legacy inputs and are not part of the active workflow.
