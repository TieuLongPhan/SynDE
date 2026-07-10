# SynDE

**SynDE** is a fast, graph-first heuristic for ranking molecular and reaction
candidates. It combines local graph features with π-system, generalized
Hückel, frontier-orbital, and HSAB descriptors.

SynDE is designed for transparent screening and prioritization. Its default
outputs are model `score` units—not kcal/mol, absolute molecular energies, or
activation barriers. A physical-energy claim requires a named external method
or a calibrated model with held-out validation.

## What it provides

- Molecule-level graph scores.
- Product-minus-reactant reaction-state scores.
- Atom-pair ranking using FMO, HSAB, charge complementarity, and graph
  accessibility.
- Atom-mapped ITS (imaginary transition-state) reaction-centre scoring.
- Optional RDKit conformer refinement and xTB shortlist workflows.
- Explicit, inspectable components and initial heuristic weights for future
  calibration.

## Install

### Conda

```bash
conda env create -f env.yml
conda activate synde
python -m pip install -e .
```

### Pip

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

SynDE requires Python 3.11 or newer, RDKit, NumPy, and Synkit 1.5 or newer.

## Quick start

```python
from synde.energy import GraphEnergy

energy = GraphEnergy()

molecule = energy.score_molecule_from_smiles("c1ccccc1")
print(molecule.score, molecule.units)  # model score, not kcal/mol

reaction = energy.score_reaction(
    "[CH2:1]=[CH2:2]>>[CH3:1][CH3:2]"
)
print(reaction.reaction_delta_score)
```

For a fully atom-mapped reaction, inspect the reaction centre with the ITS
heuristic:

```python
its = energy.score_its("[CH2:1]=[CH2:2]>>[CH3:1][CH3:2]")

print(its.state_delta_score)
print(its.components)
print(its.its_score)
```

`score_its()` requires complete atom mapping. It reports an explicit
unsupported result for missing or inconsistent maps rather than guessing a
reaction centre.

## Package layout

```text
synde.graph       graph construction and electronic descriptors
synde.energy      molecule, reaction-state, calibration, and ITS scoring
synde.geometry    optional conformer-aware refinement
synde.external    external backends, including xTB
synde.integration workflow adapters and graph-to-xTB cascade
```

## Scientific scope

SynDE currently supports a graph-only, closed-shell organic chemistry model.
It does not replace quantum chemistry, transition-state search, or experimental
validation. Treat scores as ranking features and calibrate them separately for
each target, such as reaction ΔE, activation barrier, yield, or selectivity.

## Development checks

```bash
conda run -n synde python -m pytest -q
conda run -n synde bash lint.sh
conda run -n synde sphinx-build -W -b html doc /tmp/synde-docs
```

## Documentation

Build the local documentation with `build_doc.sh`, or start with
[`doc/getting_started.rst`](doc/getting_started.rst).

## License

SynDE is licensed under the [MIT License](LICENSE).

## Acknowledgements

This project has received funding from the European Union's Horizon Europe
Doctoral Network programme under Marie Skłodowska-Curie grant agreement
No. 101072930 ([TACsy](https://tacsy.eu/)).
