# Experiment steps

Run these scripts from the repository root in the `synde` Conda environment.
The numeric prefixes are the execution order; underscore-prefixed files are
internal modules and are not standalone workflow steps.

For an independent rerun, create a fresh environment from `env.yml`; its core
numerical, xTB, RDKit, Chemprop, and Torch versions match the publication
records.

| Step | Script | Output |
|---:|---|---|
| 01 | `01_prepare_ord.py` | canonical ORD molecular inventory |
| 02 | `02_select_cohorts.py` | frozen training and test manifests |
| 03 | `03_run_xtb.py` | xTB result database and energy CSVs |
| 04 | `04_build_cache.py` | training feature cache |
| 05 | `05_calibrate.py` | frozen connectivity model and nested-CV record |
| 06 | `06_evaluate.py` | held-out connectivity evaluation |
| 07 | `07_train_energy.py` | calibrated total-energy model and SynDE benchmark |
| 08 | `08_compare_models.py` | fingerprint and Chemprop comparison |
| 09 | `09_diagnostics.py` | post-fit diagnostics (optional analysis) |
| 10 | `10_baselines.py` | descriptor-block ablation (optional analysis) |
| 11 | `11_atom_counts.py` | atom-count baseline (optional analysis) |
| 12 | `12_runtime.py` | inference timing (optional analysis) |
| 13 | `13_validate.py` | read-only publication validation |

`_fit_energy.py` is the fitting implementation called by step 07.
`_helpers.py` contains shared data and metric functions.

The committed labels and frozen connectivity model let the active benchmark
start at step 07:

```bash
bash Experiment/run_global_comparators.sh
```

That launcher executes steps 07, 08, and 13 and writes the benchmark records
and summary tables. To rebuild the data from the pinned ORD source, begin at
step 01 and follow the commands in [`data/README.md`](../../data/README.md).
