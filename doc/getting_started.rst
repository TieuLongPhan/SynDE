Getting Started
===============

SynDE provides coordinate-free prediction of a protocol-defined molecular
energy from normalized 2D graphs. One packaged predictor supports both
cross-formula total-energy prediction and formula-relative constitutional-
isomer ranking. It does not generate conformers or run xTB at
inference time.

Quick Start
-----------

After installing, score a molecule without writing any Python:

.. code-block:: console

   synde predict CCO
   synde explain 'CC(=O)NC'
   synde rank CCCCC 'CC(C)CC' 'CC(C)(C)C'

The equivalent three lines in Python:

.. code-block:: python

   from synde.energy import SynDEEnergyPredictor

   predictor = SynDEEnergyPredictor.load_default()
   print(predictor.predict_smiles("CCO").summary())

Choosing an Entry Point
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Goal
     - Python
     - Command line
   * - Energy of one molecule
     - ``predictor.predict_smiles(s)``
     - ``synde predict SMILES``
   * - Energies of many molecules
     - ``predictor.predict_many_smiles(...)``
     - ``synde predict --input FILE``
   * - Why is this value what it is?
     - ``prediction.summary()``
     - ``synde explain SMILES``
   * - Order constitutional isomers
     - ``predictor.rank_smiles([...])``
     - ``synde rank SMILES...``
   * - What can this model accept?
     - ``predictor.summary()``
     - ``synde card``

The ``*_smiles`` helpers parse with :class:`~synde.graph.GraphBuilder` and then
call the graph-level ``predict()``, ``predict_many()``, and ``rank_group()``
methods, which remain available when a normalized graph is already in hand.

Installation
------------

Standard Installation
^^^^^^^^^^^^^^^^^^^^^

Install SynDE from the repository. The default frozen 2D scorer does not
require ``thermo``, ``tblite``, or an xTB executable:

.. code-block:: console

   python -m pip install -e .

Development Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^

The supplied Conda environment includes optional empirical, semiempirical,
experiment, test, and documentation dependencies:

.. code-block:: console

   conda env create -f env.yml
   conda activate synde
   python -m pip install -e .

Install optional Python backends individually when needed:

.. code-block:: console

   python -m pip install -e '.[empirical]'
   python -m pip install -e '.[semiempirical]'
   python -m pip install -e '.[experiment,dev]'

Run the repository checks inside the same environment:

.. code-block:: console

   bash scripts/lint.sh
   bash scripts/pytest.sh -q
   bash scripts/build_doc.sh

Basic Usage: Cross-Formula Energy Prediction
--------------------------------------------

The validated training-only composition calibration is packaged with SynDE.
Load the default predictor directly:

.. code-block:: python

   from synde.energy import SynDEEnergyPredictor
   from synde.graph import GraphBuilder

   predictor = SynDEEnergyPredictor.load_default()
   molecules = [
       GraphBuilder.from_smiles("CCO"),
       GraphBuilder.from_smiles("CC(=O)O"),
   ]
   for molecule, output in zip(molecules, predictor.predict_many(molecules)):
       print(molecule.canonical_smiles, output.predicted_energy, output.units)
       print("composition:", output.composition_total)
       print("connectivity:", output.connectivity_total)

The prediction is the sum of an intercept, elemental atom-count terms, and
named connectivity terms. It estimates the raw optimized total energy from
the declared single-conformer gas-phase GFN2-xTB reference protocol. It is not
an experimental energy, free energy, atomization energy, or conformer-
ensemble energy.

Basic Usage: Ranking Isomers
----------------------------

Pass candidate isomers with the same molecular formula to ``rank_group()`` on
the predictor already loaded above:

.. code-block:: python

   candidates = [
       GraphBuilder.from_smiles("CCCCC"),
       GraphBuilder.from_smiles("CC(C)CC"),
       GraphBuilder.from_smiles("CC(C)(C)C"),
   ]

   ranking = predictor.rank_group(candidates)
   for position, (index, output) in enumerate(ranking, start=1):
       molecule = candidates[index]
       print(f"Rank {position}: {molecule.canonical_smiles} -> Energy: {output.predicted_energy:.4f}")

.. note::

   All candidates passed to ``rank_group()`` must share the exact same molecular formula and formal charge. Lower predictions indicate lower model energy.

.. warning::

   Composition is necessary for global calibration but is constant within a
   formula group. Therefore the same artifact ranks locally through its
   connectivity block without defining a second model.

Single-Molecule Scoring
-----------------------

Applications can predict individual normalized graphs via ``predict()``:

.. code-block:: python

   molecule = GraphBuilder.from_smiles("CC(=O)NC")
   output = predictor.predict(molecule)
   print(f"Energy: {output.predicted_energy} | Status: {output.status}")

Refining Weights on External Data
---------------------------------

Applications with reference energies from the same target protocol can refine
the packaged coefficients without modifying the installed model:

.. code-block:: python

   from synde.energy import EnergyRefinementRecord

   records = [
       EnergyRefinementRecord.from_smiles("sample-1", "CCO", -12.34),
       EnergyRefinementRecord.from_smiles("sample-2", "COC", -12.10),
   ]
   refined, report = predictor.refine(
       records,
       dataset_name="laboratory-reference-v1",
       alpha=10.0,
   )
   refined.save("synde-refined.json")
   print(report.baseline_metrics, report.refined_metrics)

``alpha`` controls how strongly the fitted coefficients remain anchored to the
packaged values. The intercept, elemental-composition weights, and connectivity
weights can be enabled independently with ``refine_intercept``,
``refine_composition``, and ``refine_connectivity``.

Use a development subset for ``refine()`` and reserve an independent subset
for evaluation. See :doc:`refinement` for dataset requirements, block selection,
artifact persistence, and validation guidance.

.. warning::

   Refinement creates a new model whose evaluation status is ``independent
   validation required``. It does not inherit the packaged model's external-
   validation claim. Targets must use the units and reference protocol declared
   by the base predictor's model card.

Input Domain & Constraints
--------------------------

Supported Chemical Domain
^^^^^^^^^^^^^^^^^^^^^^^^^

- **Energy-model elements**: H, B, C, N, O, F, Si, P, S, Cl, Br, and I.
  Supported elements are determined by the active fitted composition
  calibration; run ``synde card`` to print the list carried by the artifact
  you actually have installed.
- **Electronic domain**: Connected, neutral, closed-shell structures.
- **Cross-formula task**: Neutral molecules may have different formulas;
  predictions are comparable only under the same declared target protocol.
- **Ranking task**: Constitutional isomers share a formula but differ in atom
  connectivity.
- **Stereochemistry**: Stereoisomers with identical normalized 2D graph identities share the same 2D descriptor output.

Reading Results Interactively
-----------------------------

Predictions, rankings, and the predictor itself render themselves.  In a
terminal, ``summary()`` prints an aligned report; in Jupyter, the same objects
display as HTML tables automatically:

.. code-block:: python

   prediction = predictor.predict_smiles("CC(=O)NC")

   print(prediction)            # one-line headline
   print(prediction.summary())  # full signed breakdown
   prediction.top_contributions(5)

   ranking = predictor.rank_smiles(["CCCCC", "CC(C)CC", "CC(C)(C)C"])
   print(ranking.summary())

   print(predictor.summary())   # model card and domain limits

``summary()`` accepts ``color=False`` to suppress ANSI escapes when writing to
a log file, and ``precision`` to control the displayed digits.

Understanding Rejections
------------------------

Structures outside the model's applicability domain raise
:class:`~synde.errors.SynDEDomainError`, which names the offending input, the
rule it violated, and a concrete next step:

.. code-block:: python

   from synde.errors import SynDEDomainError

   try:
       predictor.predict_smiles("[Na+].[Cl-]")
   except SynDEDomainError as error:
       print(error)
       print(error.details)

.. code-block:: text

   SynDE energy prediction requires one connected molecule; this input has 2
   disconnected fragments. (input: [Cl-].[Na+])
     Hint: Split the input on '.' and score each neutral component separately;
     salts and solvates are not single molecules.
     Model domain: elements [B Br C Cl F H I N O P S Si]; total formal charge
     [0]; connected, closed-shell, non-isotopic structures

Every SynDE exception subclasses :class:`ValueError`, so existing
``except ValueError`` handlers keep working.  ``error.details`` carries the
same facts in machine-readable form.

Output Diagnostics & JSON Export
--------------------------------

``SynDEEnergyPrediction`` provides exact contributions and diagnostics:

.. code-block:: python

   output = predictor.predict(GraphBuilder.from_smiles("CCO"))

   print("Status:", output.status)
   print("Energy:", output.predicted_energy)
   print("Units:", output.units)
   print("Composition:", output.composition_contributions)
   print("Connectivity:", output.connectivity_contributions)
   print("Warnings:", output.warnings)
   print("Provenance:", output.provenance)

   json_data = output.to_dict()

- ``composition_contributions`` contains the elemental baseline terms.
- ``connectivity_contributions`` contains the named graph terms.
- ``warnings`` reports domain diagnostics without silently changing the value.

``SynDEEnergyPrediction`` similarly exposes ``predicted_energy``, exact
``composition_contributions`` and ``connectivity_contributions``, model
provenance, and applicability warnings. The sum identity is

.. code-block:: text

   predicted_energy = composition_total + connectivity_total

Local Documentation Building
----------------------------

Use the strict helper so warnings fail the build:

.. code-block:: console

   bash scripts/build_doc.sh

The compiled HTML files are written to ``docs/``. Online documentation is
hosted at `synde.readthedocs.io
<https://synde.readthedocs.io/en/latest/>`_.
