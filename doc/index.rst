SynDE Documentation
===================

Documentation for **SynDE** version |version| (release |release|).

Online documentation is hosted at `synde.readthedocs.io <https://synde.readthedocs.io/en/latest/>`_.

SynDE ranks constitutional isomers using an interpretable, coordinate-free linear model over two-dimensional molecular-graph descriptors. The default scorer is distributed with model weights and does not require conformer generation or xTB calculations at inference time.

Key Features
------------

- **2D Isomer Ranking**: Ranks constitutional isomers directly from SMILES or normalized graph descriptors.
- **No Conformer Bottleneck**: Inference uses linear weights over 2D graph descriptors.
- **Attributable Predictions**: Output decomposes into exact signed linear components summing to the final score.
- **Model Provenance**: Built-in model cards, validation records, and feature-distance diagnostic warnings.

Quick Example
-------------

.. code-block:: python

   from synde.energy import SynDEScorer
   from synde.graph import GraphBuilder

   scorer = SynDEScorer.load_default()

   isomers = [
       GraphBuilder.from_smiles("CCCCC"),
       GraphBuilder.from_smiles("CC(C)CC"),
       GraphBuilder.from_smiles("CC(C)(C)C"),
   ]

   outputs = scorer.score_group(isomers)

   ranking = sorted(zip(isomers, outputs), key=lambda item: item[1].score)
   for pos, (mol, output) in enumerate(ranking, start=1):
       print(pos, mol.canonical_smiles, output.score, output.status)

Contents
--------

.. toctree::
   :maxdepth: 2

   Getting Started <getting_started>
   Tutorials and Examples <tutorial>
   API Reference <api>
   References <reference>
