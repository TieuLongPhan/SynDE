# SynDE

[![Documentation Status](https://readthedocs.org/projects/synde/badge/?version=latest)](https://synde.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://img.shields.io/pypi/v/synde.svg)](https://pypi.org/project/synde/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/tieulongphan/synde.svg)](https://github.com/tieulongphan/synde/blob/main/LICENSE)
[![Release](https://img.shields.io/github/v/release/tieulongphan/synde.svg)](https://github.com/tieulongphan/synde/releases)
[![Last Commit](https://img.shields.io/github/last-commit/tieulongphan/synde.svg)](https://github.com/tieulongphan/synde/commits)
[![Zenodo](https://zenodo.org/badge/1296441287.svg)](https://zenodo.org/badge/latestdoi/1296441287)
[![CI](https://github.com/tieulongphan/synde/actions/workflows/test-and-lint.yml/badge.svg?branch=main)](https://github.com/tieulongphan/synde/actions/workflows/test-and-lint.yml)
[![Stars](https://img.shields.io/github/stars/tieulongphan/synde.svg?style=social&label=Star)](https://github.com/tieulongphan/synde/stargazers)

Interpretable 2D prediction of protocol-defined GFN2-xTB molecular energies,
with constitutional-isomer ranking as the connectivity stress test.

Online documentation is hosted at [synde.readthedocs.io](https://synde.readthedocs.io/en/latest/).

**Contents** — [Overview](#overview) · [Command-line example](#command-line-example) ·
[Installation](#installation) · [Usage](#usage) · [Scope and domain constraints](#scope-and-domain-constraints) ·
[Reproducibility](#reproducibility) · [Package structure](#package-structure) · [Citation](#citation)

---

## Overview

SynDE predicts a protocol-defined molecular-energy coordinate from a
two-dimensional molecular graph without conformer generation or xTB execution
at inference time. The validated 633-term connectivity equation ranks
constitutional isomers. The active energy workflow fits an extensive
element-count baseline around those unchanged weights using training data only;
the resulting complete predictor is then evaluated globally and within formula
on one external cohort.

- **Two complementary tasks**: Predict total energies across formulas or rank
  constitutional isomers within one formula.
- **Connectivity stress test**: Same-formula ranking removes atom-count signal
  exactly and tests whether a model resolves bonding and topology.
- **Interpretable terms**: Each prediction decomposes into exact signed linear components that sum to the total score.
- **Self-contained ranking**: The validated connectivity weights are bundled;
  no external semiempirical quantum binary is required for graph scoring.
- **Provenance tracking**: Includes model cards, validation records, and feature-distance diagnostic warnings.
- **Usable from the shell**: A `synde` console script predicts, explains, and
  ranks, with JSON and CSV output, parallel scoring, and shell completion.
- **Lazy imports**: `import synde` does not load RDKit, NetworkX, or NumPy until
  an exported symbol needs one of them.

Every prediction is reported as the sum of its composition and connectivity
blocks:

```text
predicted_energy = composition_total + connectivity_total
```

---

## Command-line example

Predicting across formulas, and ranking constitutional isomers within one:

```console
$ synde predict CCO 'CC(=O)O' c1ccccc1

structure  formula  energy (eV)  composition  connectivity
---------  -------  -----------  -----------  ------------
CCO        C2H6O      -310.0180    -304.0719       -5.9462
CC(=O)O    C2H4O2     -393.3796    -386.3469       -7.0327
c1ccccc1   C6H6       -432.0563    -422.7000       -9.3563

$ synde rank CCCCC 'CC(C)CC' 'CC(C)(C)C'

C5H12   3 candidates, lowest predicted energy first
──────────────────────────────────────────────────────────────
  #  structure  energy (eV)  Δ vs best  connectivity
  -  ---------  -----------  ---------  ------------
  1  CC(C)(C)C    -458.0003    +0.0000      -11.0020
  2  CC(C)CC      -457.9810    +0.0193      -10.9827
  3  CCCCC        -457.9317    +0.0685      -10.9335
```

Install below, then see [Usage](#usage) for the full command set and the Python API.

---

## Installation

### Standard installation

```bash
git clone https://github.com/TieuLongPhan/SynDE.git
cd SynDE
python -m pip install -e .
```

This installs the `synde` command and the packaged predictor. The default
frozen 2D scorer needs no `thermo`, no `tblite`, and no xTB executable.

Verify the install:

```bash
synde --version
synde predict CCO
```

### Development dependencies

The supplied Conda environment includes optional empirical, semiempirical,
experiment, test, and documentation dependencies, including the xTB executable:

```bash
conda env create -f env.yml
conda activate synde
python -m pip install -e .
```

Optional Python backends can instead be installed individually:

```bash
python -m pip install -e '.[empirical]'       # Joback terms via thermo
python -m pip install -e '.[semiempirical]'   # GFN2 single points via tblite
python -m pip install -e '.[experiment,dev]'  # calibration and developer tools
```

`--jobs` parallel scoring uses `joblib` from the `experiment` extra. Without it
SynDE scores serially and says so; nothing fails.

---

## Usage

### Which entry point do I need?

| Goal | Python | Command line |
| :--- | :--- | :--- |
| Energy of one molecule | `predictor.predict_smiles(s)` | `synde predict SMILES` |
| Energies of many molecules | `predictor.predict_many_smiles([...])` | `synde predict --input FILE` |
| Why is this value what it is? | `prediction.summary()` | `synde explain SMILES` |
| Order constitutional isomers | `predictor.rank_smiles([...])` | `synde rank SMILES...` |
| What can this model accept? | `predictor.summary()` | `synde card` |

The `*_smiles` helpers parse with `GraphBuilder` and then call the graph-level
`predict()`, `predict_many()`, and `rank_group()` methods, which remain
available when a normalized graph is already in hand.

### From the command line

```bash
synde predict CCO 'CC(=O)O'                  # energies across formulas
synde explain 'CC(=O)NC' --top 5             # signed contribution breakdown
synde rank CCCCC 'CC(C)CC' 'CC(C)(C)C'       # order isomers
synde card                                   # provenance and domain limits
```

Reading input and writing machine-readable output:

```bash
synde predict --input molecules.smi --format csv > energies.csv
synde predict --input big.smi --jobs 8 --format json
cat molecules.smi | synde predict --input -
synde completion zsh >> ~/.zshrc
```

`--input` skips blank lines and `#` comments and reads only the first
whitespace-separated field, so ordinary `.smi` files work unchanged. By default
a structure outside the model domain stops the run; `--keep-going` scores the
rest and reports skips on stderr, still exiting non-zero. See the
[CLI documentation](doc/cli.rst) for the full option list.

### 1. Predict energy across formula groups

```python
from synde.energy import SynDEEnergyPredictor
from synde.graph import GraphBuilder

predictor = SynDEEnergyPredictor.load_default()
molecules = [
    GraphBuilder.from_smiles("CCO"),
    GraphBuilder.from_smiles("CC(=O)O"),
]

for molecule, output in zip(molecules, predictor.predict_many(molecules)):
    print(molecule.canonical_smiles, output.predicted_energy, output.units)
    print(output.composition_total, output.connectivity_total)
```

The packaged default artifact was generated by
`bash Experiment/run_global_comparators.sh` from the active training cohort.
These predictions estimate the raw optimized total energy produced by the
model's declared GFN2-xTB reference protocol. They are comparable across
formula groups only within that same protocol and chemical domain; they are
not experimental energies, free energies, or conformer-ensemble energies.

### 2. Rank isomer groups with the same model

```python
ranking = predictor.rank_smiles(["CCCCC", "CC(C)CC", "CC(C)(C)C"])

for position, (input_index, output) in enumerate(ranking, start=1):
    print(position, output.canonical_smiles, output.predicted_energy)

print(ranking.summary())  # or just `ranking` in a Jupyter cell
```

Lower predictions indicate lower model energy. All candidates passed to
`rank_group()` and `rank_smiles()` must share the exact same molecular formula
and formal charge. Their composition totals are identical, so only connectivity
changes the ordering.

### 3. Inspect a prediction

```python
output = predictor.predict_smiles("CC(=O)NC")

print(output)                      # one-line headline
print(output.summary())            # full signed breakdown
output.top_contributions(5)        # largest active connectivity terms

print("Status:", output.status)
print("Energy:", output.predicted_energy, output.units)
print("Composition:", output.composition_contributions)
print("Connectivity:", output.connectivity_contributions)
print("Warnings:", output.warnings)
print("Provenance:", output.provenance)

data_dict = output.to_dict()
```

Predictions, rankings, and the predictor itself render as HTML tables in
Jupyter. `summary()` takes `color=False` for log files and `precision` for the
displayed digits.

### 4. Understand a rejection

Structures outside the applicability domain raise `SynDEDomainError`, which
names the input, the rule it violated, and a concrete next step:

```python
>>> predictor.predict_smiles("[Na+].[Cl-]")
SynDEDomainError: SynDE energy prediction requires one connected molecule;
this input has 2 disconnected fragments. (input: [Cl-].[Na+])
  Hint: Split the input on '.' and score each neutral component separately;
  salts and solvates are not single molecules.
  Model domain: elements [B Br C Cl F H I N O P S Si]; total formal charge [0];
  connected, closed-shell, non-isotopic structures
```

Every SynDE exception subclasses `ValueError`, so existing `except ValueError`
handlers keep working. `error.details` carries the same facts in
machine-readable form.

---

## Scope and domain constraints

- **Supported model elements**: determined from the active training cohort;
  external molecules containing unseen elements are rejected. Run `synde card`
  to print the list carried by the artifact you have installed.
- **Electronic domain**: Connected, neutral, closed-shell structures.
- **Isomer Class**: Constitutional isomers (same formula, different atom connectivity).
- **Cross-formula target**: Single-conformer, gas-phase GFN2-xTB 6.7.1
  optimized total energy in eV under the declared reference protocol.
- **Ranking target**: Formula-relative ordering under the same declared target
  protocol.
- **Interpretation**: Both are statistical graph projections, not physical
  conformer populations, experimental energies, or free energies.

Use `SynDEEnergyPredictor` for both tasks: `predict()`/`predict_many()` globally
and `predict_group()`/`rank_group()` locally. `SynDEScorer` remains available
only as a compatibility interface to the original connectivity-validation
record; its raw subtotal must not be compared across formulas.

---

## Reproducibility

Use the `synde` Conda environment for every command below:

```bash
conda activate synde
```

Validate the published records and packaged model without retraining:

```bash
python Experiment/scripts/13_validate.py
```

Regenerate the fitted artifacts and external-evaluation outputs:

```bash
bash Experiment/run_global_comparators.sh
```

Run the repository quality gates and strict documentation build:

```bash
bash scripts/lint.sh
bash scripts/pytest.sh -q
bash scripts/build_doc.sh
```

Build and inspect the installable distributions:

```bash
python -m build
python scripts/check_package_artifacts.py dist
```

---

## Package structure

| Directory | Description |
| :--- | :--- |
| `synde/graph/` | Graph normalization, topological invariants, and $\pi$-system assignments. |
| `synde/energy/` | Cross-formula predictor, frozen ranking scorer, model cards, attribution, and result dataclasses. |
| `synde/geometry/` | Conformer generation and semiempirical xTB workflow utilities. |
| `synde/integration/` | Workflow adapters and reaction/ITS scoring tools. |
| `synde/models/` | Bundled default model resources and weights. |
| `synde/cli.py` | The `synde` console script. |
| `synde/report.py` | Rendering shared by the CLI, `__repr__`, and Jupyter output. |
| `synde/formatting.py` | Dependency-free table, colour, and number formatting. |
| `synde/errors.py` | Structured, actionable exception types. |
| `doc/` | Sphinx documentation source files. |
| `scripts/` | Repository lint, test, documentation, and package checks. |

---

## Citation

If SynDE contributes to work you publish, please cite the software:

```bibtex
@software{phan_synde,
  author  = {Phan, Tieu Long},
  title   = {SynDE: Interpretable 2D GFN2-xTB energy prediction and isomer ranking},
  version = {0.5.0},
  url     = {https://github.com/TieuLongPhan/SynDE},
  license = {MIT}
}
```

Please also report the `model_name` and `model_sha256` from the model card of
the artifact you used, which `synde card` prints, so results stay traceable to
an exact set of weights.

---

## Documentation

- [Documentation](https://synde.readthedocs.io/en/latest/)
- [Getting Started](doc/getting_started.rst)
- [Command Line Interface](doc/cli.rst)
- [Changelog](doc/changelog.rst)
- [Tutorials and Examples](doc/tutorial.rst)
- [API Reference](doc/api.rst)

---

## License

SynDE is distributed under the MIT License. See [LICENSE](LICENSE).

## Acknowledgments

This project received funding from the European Union's Horizon Europe Doctoral Network programme under Marie Skłodowska-Curie grant agreement No. 101072930 ([TACsy](https://tacsy.eu/)).
