SynDE Documentation
===================

Documentation for **SynDE** version |version| (release |release|).

Online documentation is hosted at `synde.readthedocs.io <https://synde.readthedocs.io/en/latest/>`_.

SynDE predicts protocol-defined molecular energy and ranks constitutional
isomers using one interpretable, coordinate-free linear model. The packaged
predictor does not require conformer generation or xTB calculations at
inference time.

Key Features
------------

- **One global/local model**: predict across formulas and rank isomers with
  the same equation.
- **No conformer bottleneck**: inference uses linear weights over normalized
  2D graph descriptors.
- **Attributable predictions**: exact signed contributions sum to every
  reported prediction.
- **Refinable weights**: fit an anchored update on compatible external labels
  without changing the installed model.
- **Explicit provenance**: model cards, artifact hashes, fit reports, and
  applicability warnings travel with predictions.
- **Usable from the shell**: a ``synde`` console script predicts, explains,
  and ranks, with ``--json`` output for pipelines.

Quick Example
-------------

From the command line, no Python required:

.. code-block:: console

   synde predict CCO 'CC(=O)O'
   synde rank CCCCC 'CC(C)CC' 'CC(C)(C)C'
   synde explain 'CC(=O)NC'

From Python:

.. code-block:: python

   from synde.energy import SynDEEnergyPredictor
   from synde.graph import GraphBuilder

   predictor = SynDEEnergyPredictor.load_default()

   isomers = [
       GraphBuilder.from_smiles("CCCCC"),
       GraphBuilder.from_smiles("CC(C)CC"),
       GraphBuilder.from_smiles("CC(C)(C)C"),
   ]

   for position, (index, output) in enumerate(
       predictor.rank_group(isomers), start=1
   ):
       print(
           position,
           isomers[index].canonical_smiles,
           output.predicted_energy,
           output.units,
       )

Contents
--------

.. toctree::
   :maxdepth: 2

   Getting Started <getting_started>
   Command Line Interface <cli>
   Tutorials and Examples <tutorial>
   External Weight Refinement <refinement>
   API Reference <api>
   Changelog <changelog>
   References <reference>
