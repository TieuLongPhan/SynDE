Getting started
===============

SynDE is a graph-first heuristic for molecular and reaction ranking. Its
default output is a dimensionless model ``score``; it is not an absolute
molecular energy, kcal/mol prediction, or activation barrier.

Scope and interpretation
------------------------

``GraphEnergy`` provides molecule scores, products-minus-reactants state
scores, and candidate-pair rankings. The graph-only model uses normalized
molecular graphs, π-system/Hückel descriptors, frontier-orbital and HSAB
features, and transparent local topology terms.

For fully atom-mapped reactions, ``score_its()`` also evaluates an imaginary
transition-state graph. It exposes state-delta, bond-edit, local
reorganization, and formed-bond interaction terms separately. The resulting
``its_score`` is a barrier-like ranking heuristic, not a physical
transition-state energy.

The initial supported domain is closed-shell organic chemistry with common
main-group elements. Metals, radicals, unusual valence states, ambiguous
π-electron assignments, and intermolecular complexes without a pose are
reported as structured warnings or unsupported results.

Installation
------------

Create the project environment from the supplied Conda definition:

.. code-block:: bash

   conda env create -f env.yml
   conda activate synde
   python -m pip install -e .

For a pip-only installation:

.. code-block:: bash

   python -m pip install -r requirements.txt
   python -m pip install -e .

Quick graph scoring
-------------------

.. code-block:: python

   from synde.energy import GraphEnergy

   energy = GraphEnergy()
   molecule = energy.score_molecule_from_smiles("c1ccccc1")
   reaction = energy.score_reaction(
       "[CH2:1]=[CH2:2]>>[CH3:1][CH3:2]"
   )

   print(molecule.score, molecule.units)
   print(reaction.reaction_delta_score, reaction.units)

Use :mod:`synde.graph` when you need normalized graph construction or
descriptor-level access, and :mod:`synde.geometry` only when conformer-aware
information is explicitly required.

Mapped reaction-centre scoring
------------------------------

For a completely atom-mapped reaction, use the ITS heuristic:

.. code-block:: python

   its = energy.score_its("[CH2:1]=[CH2:2]>>[CH3:1][CH3:2]")
   print(its.state_delta_score)
   print(its.its_score)

``its_score`` is a barrier-like ranking feature. It keeps state-delta,
bond-edit, local reorganization, and FMO/HSAB/charge interaction terms
separate. An unmapped reaction returns an explicit unsupported ITS result;
SynDE does not silently infer a reaction centre.
