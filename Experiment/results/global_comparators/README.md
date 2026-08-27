# Published global/local comparator benchmark

This directory is the canonical repository snapshot of the completed
training/external-validation comparison. The two JSON records retain absolute
`/tmp/synde-external-global-comparators-v1` paths as immutable workstation-run
provenance; published artifacts are organized here instead:

| Content | Canonical path |
|---|---|
| SynDE global/local record | `Experiment/results/global_comparators/synde.json` |
| Comparator record | `Experiment/results/global_comparators/benchmark.json` |
| Fingerprint models | `Experiment/results/global_comparators/artifacts/fingerprints/` |
| Chemprop checkpoints and calibration | `Experiment/results/global_comparators/artifacts/chemprop/` |
| Machine-readable summary tables | `Experiment/results/global_comparators/tables/` |
| Packaged SynDE predictor | `synde/models/synde_energy_model.json` |

The active evidence chain contains only 78,513 training molecules in 11,993
groups and 19,940 external-validation molecules in 3,005 formula- and
connectivity-disjoint groups. No external label was used for fitting or model
selection. The same external prediction vector supplies global and
within-formula metrics for each model.

Validate cohort counts, split firewalls, frozen SynDE coefficients, model
hashes, and comparator artifacts without retraining:

```bash
python Experiment/scripts/13_validate.py
```

Rebuild the fitted records and machine-readable summary tables:

```bash
bash Experiment/run_global_comparators.sh
```
