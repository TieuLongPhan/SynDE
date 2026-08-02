Getting Started
===============

SynDE provides rapid, interpretable ranking of constitutional isomers from normalized 2D molecular graphs. The canonical model is packaged directly within the library, allowing graph scoring without conformer generation or xTB execution.

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
benchmark, test, and documentation dependencies:

.. code-block:: console

   conda env create -f env.yml
   conda activate synde
   python -m pip install -e .

Install optional Python backends individually when needed:

.. code-block:: console

   python -m pip install -e '.[empirical]'
   python -m pip install -e '.[semiempirical]'
   python -m pip install -e '.[benchmark,dev]'

Basic Usage: Ranking Isomers
----------------------------

Construct normalized molecular graphs from SMILES, load the default scorer, and pass candidate isomers with the same molecular formula to ``score_group()``:

.. code-block:: python

   from synde.energy import SynDEScorer
   from synde.graph import GraphBuilder

   scorer = SynDEScorer.load_default()
   candidates = [
       GraphBuilder.from_smiles("CCCCC"),
       GraphBuilder.from_smiles("CC(C)CC"),
       GraphBuilder.from_smiles("CC(C)(C)C"),
   ]

   outputs = scorer.score_group(candidates)

   ranking = sorted(zip(candidates, outputs), key=lambda row: row[1].score)
   for position, (molecule, output) in enumerate(ranking, start=1):
       print(f"Rank {position}: {molecule.canonical_smiles} -> Score: {output.score:.4f}")

.. note::

   All candidates passed to ``score_group()`` must share the exact same molecular formula and formal charge. Lower scores indicate lower predicted relative energy within the group. Do not compare raw scores across different formula groups.

Single-Molecule Scoring
-----------------------

Applications that handle grouping externally can score individual normalized graphs via ``score()``:

.. code-block:: python

   molecule = GraphBuilder.from_smiles("CC(=O)NC")
   output = scorer.score(molecule)
   print(f"Score: {output.score} | Status: {output.status}")

Input Domain & Constraints
--------------------------

Supported Chemical Domain
^^^^^^^^^^^^^^^^^^^^^^^^^

- **Elements**: Connected, neutral, closed-shell structures containing B, C, N, O, F, Si, S, Cl, Br, or I.
- **Isomer Type**: Constitutional isomers (same formula, different atom connectivity).
- **Stereochemistry**: Stereoisomers with identical normalized 2D graph identities share the same 2D descriptor output.

Output Diagnostics & JSON Export
--------------------------------

The ``MoleculeScoreResult`` dataclass provides rich diagnostic fields:

.. code-block:: python

   output = outputs[0]

   print("Status:", output.status)
   print("Score:", output.score)
   print("Units:", output.units)
   print("Components:", output.components)
   print("Warnings:", output.warnings)
   print("Provenance:", output.provenance)

   # Convert to JSON-serializable dictionary
   json_data = output.to_dict()

- **components**: Dictionary of signed terms in the linear model that sum to the score.
- **warnings**: Diagnostic flags (e.g., feature-distance boundaries) that do not alter the score.

Local Documentation Building
----------------------------

To render these docs locally:

.. code-block:: console

   sphinx-build -b html doc docs

Or use the helper script:

.. code-block:: console

   bash build_doc.sh

The compiled HTML files will be available in the ``docs/`` directory. Online documentation is hosted at `synde.readthedocs.io <https://synde.readthedocs.io/en/latest/>`_.
