# Experiments

This directory contains the training, held-out evaluation, and comparator
workflows used for the SynDE artifact. Run commands from the repository root in
the `synde` Conda environment.

## Environment

```bash
conda env create -f env.yml
conda activate synde
python -m pip install -e '.[experiment]'
```

`env.yml` pins the numerical and model-training versions recorded by the
benchmark. Use a fresh environment for an independent rerun; updating an
existing environment can leave files from replaced packages.

## Inputs and cohorts

The active workflow requires:

- `data/ord_training_xtb.csv` for fitting and model selection;
- `data/ord_test_xtb.csv` for held-out evaluation;
- `synde/models/synde_frozen_model.json` for the fixed connectivity weights;
- `Experiment/calibration_seed.json` for the frozen feature support.

Model fitting uses only `data/ord_training_xtb.csv`. Final held-out evaluation
uses only `data/ord_test_xtb.csv`; formula and connectivity identities must not
cross the split. The validator requires 78,513 training molecules in 11,993
groups and 19,940 test molecules in 3,005 groups.

The two xTB CSVs and their resumable SQLite database are stored as recovered
data artifacts. Verify that the database covers every tracked label-blind
manifest row without launching xTB:

```bash
python Experiment/scripts/03_run_xtb.py --workers 16 --dry-run
```

If an artifact is intentionally replaced, rebuild the CSV snapshots or resume
calculations from the tracked candidate manifests:

```bash
python Experiment/scripts/03_run_xtb.py --workers 16
```

The runner is resumable through `data/ord_xtb_results.sqlite3`. See
[`data/README.md`](../data/README.md) for the pinned ORD snapshot and cohort
construction steps.

## Validate the committed benchmark

The read-only validator checks the release without fitting:

```bash
python Experiment/scripts/13_validate.py
```

It verifies cohort counts, split firewalls, frozen connectivity weights, the
packaged energy model, and committed comparator artifacts without loading
caches or labels.

## Fit SynDE

```bash
python Experiment/scripts/07_train_energy.py \
  --training-source data/ord_training_xtb.csv \
  --external-source data/ord_test_xtb.csv \
  --work-dir /tmp/synde-energy-external-validation \
  --rebuild-training-cache \
  --rebuild-external-cache
```

This stage keeps the 633 connectivity coefficients fixed and fits the
elemental calibration on training data. It writes `training.joblib`,
`external.joblib`, `synde_energy_model.json`, and `synde.json` to the work
directory.

## Fit comparators

Rebuild the SynDE, fingerprint, and five-fold Chemprop comparison:

```bash
bash Experiment/run_global_comparators.sh
```

Every representation receives a training-only elemental calibration. Global
and within-formula metrics use the same held-out prediction vector. The
launcher writes machine-readable global and local summary tables and validates
the completed work directory.

Generated caches, logs, and intermediate models are written to
`/tmp/synde-external-global-comparators-v1` by default. Set
`SYNDE_COMPARATOR_WORK_DIR` to use another location.

## Outputs

The committed evidence is under `Experiment/results/`. The canonical SynDE and
comparator records are:

- `Experiment/results/global_comparators/synde.json`;
- `Experiment/results/global_comparators/benchmark.json`;
- `synde/models/synde_energy_model.json`.

Any change to the target protocol, descriptor definitions, supported domain,
split, or connectivity weights defines a new model and requires a new held-out
evaluation. [`scripts/README.md`](scripts/README.md) lists every stage in
execution order.
