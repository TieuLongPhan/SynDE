# SynDE

[![Documentation Status](https://readthedocs.org/projects/synde/badge/?version=latest)](https://synde.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Interpretable 2D ranking of constitutional isomers by GFN2-xTB energy.

Online documentation is hosted at [synde.readthedocs.io](https://synde.readthedocs.io/en/latest/).

---

## Overview

SynDE ranks constitutional isomers using a coordinate-free linear model over two-dimensional molecular-graph descriptors. The default model is packaged directly with its trained weights, enabling isomer ranking without conformer generation or xTB execution at inference time.

- **Inference speed**: Ranks constitutional isomers in milliseconds from 2D molecular graphs.
- **Interpretable terms**: Each prediction decomposes into exact signed linear components that sum to the total score.
- **Self-contained**: Pre-trained model weights are bundled into the package; no external semiempirical quantum binaries are required for graph scoring.
- **Provenance tracking**: Includes model cards, validation records, and feature-distance diagnostic warnings.

---

## Navigation

- [Documentation](https://synde.readthedocs.io/en/latest/)
- [Getting Started](doc/getting_started.rst)
- [Tutorials and Examples](doc/tutorial.rst)
- [API Reference](doc/api.rst)

---

## Installation

### Standard Installation

```bash
git clone https://github.com/TieuLongPhan/SynDE.git
cd SynDE
python -m pip install -e .
```

### Development Dependencies

The supplied Conda environment includes optional empirical, semiempirical,
benchmark, test, and documentation dependencies, including the xTB executable:

```bash
conda env create -f env.yml
conda activate synde
python -m pip install -e .
```

Optional Python backends can instead be installed individually:

```bash
python -m pip install -e '.[empirical]'       # Joback terms via thermo
python -m pip install -e '.[semiempirical]'   # GFN2 single points via tblite
python -m pip install -e '.[benchmark,dev]'   # calibration and developer tools
```

---

## Quickstart

### 1. Rank Isomer Groups

```python
from synde.energy import SynDEScorer
from synde.graph import GraphBuilder

scorer = SynDEScorer.load_default()

molecules = [
    GraphBuilder.from_smiles("CCCCC"),
    GraphBuilder.from_smiles("CC(C)CC"),
    GraphBuilder.from_smiles("CC(C)(C)C"),
]

outputs = scorer.score_group(molecules)

ranking = sorted(zip(molecules, outputs), key=lambda item: item[1].score)
for position, (molecule, output) in enumerate(ranking, start=1):
    print(f"{position} {molecule.canonical_smiles} {output.score:.4f} {output.status}")
```

*Note*: Lower scores indicate lower predicted relative energy within the input group. All candidates passed to `score_group()` must share the exact same molecular formula and formal charge.

### 2. Single Molecule Scoring

```python
molecule = GraphBuilder.from_smiles("CC(=O)NC")
output = scorer.score(molecule)
print(output.score)
```

### 3. Inspect Output Attributes

```python
output = outputs[0]

print("Status:", output.status)
print("Score:", output.score)
print("Units:", output.units)
print("Components:", output.components)
print("Descriptors:", output.descriptors)
print("Warnings:", output.warnings)
print("Provenance:", output.provenance)

data_dict = output.to_dict()
```

---

## Scope and Domain Constraints

- **Supported Elements**: Connected, neutral, closed-shell structures containing B, C, N, O, F, Si, S, Cl, Br, or I.
- **Isomer Class**: Constitutional isomers (same formula, different atom connectivity).
- **Target Metric**: Model scores correspond to GFN2-xTB ranking targets. They represent relative statistical predictions rather than physical conformer populations or free energies.

---

## Package Structure

| Directory | Description |
| :--- | :--- |
| `synde/graph/` | Graph normalization, topological invariants, and $\pi$-system assignments. |
| `synde/energy/` | `SynDEScorer`, model cards, feature attribution, and score dataclasses. |
| `synde/geometry/` | Conformer generation and semiempirical xTB workflow utilities. |
| `synde/integration/` | Workflow adapters and reaction/ITS scoring tools. |
| `synde/models/` | Bundled default model resources and weights. |
| `doc/` | Sphinx documentation source files. |

---

## Documentation

Hosted documentation is available at [synde.readthedocs.io](https://synde.readthedocs.io/en/latest/).

To build HTML documentation locally:

```bash
sphinx-build -b html doc docs
```

The output will be rendered in `docs/index.html`.

---

## Testing

```bash
pytest -q
bash lint.sh
```

---

## License

SynDE is distributed under the MIT License. See [LICENSE](LICENSE).

## Acknowledgments

This project received funding from the European Union's Horizon Europe Doctoral Network programme under Marie Skłodowska-Curie grant agreement No. 101072930 ([TACsy](https://tacsy.eu/)).
