Command Line Interface
======================

SynDE installs a ``synde`` console script.  Every scoring subcommand accepts
SMILES as positional arguments, from a file with ``--input``, or on standard
input, and can emit a table, JSON, or CSV with ``--format``.

.. code-block:: console

   synde --help
   synde info

Predicting Energies
-------------------

``synde predict`` scores any collection of structures.  Formulas may differ
freely, because the composition baseline is extensive:

.. code-block:: console

   synde predict CCO 'CC(=O)O' c1ccccc1

.. code-block:: text

   structure  formula  energy (eV)  composition  connectivity
   ---------  -------  -----------  -----------  ------------
   CCO        C2H6O      -310.0180    -304.0719       -5.9462
   CC(=O)O    C2H4O2     -393.3796    -386.3469       -7.0327
   c1ccccc1   C6H6       -432.0563    -422.7000       -9.3563

The ``composition`` and ``connectivity`` columns are the two blocks whose sum
is the reported energy.

Explaining One Prediction
-------------------------

``synde explain`` prints the full signed decomposition, the applicability
distance, and the model provenance:

.. code-block:: console

   synde explain 'CC(=O)NC' --top 5

.. code-block:: text

   CNC(C)=O   C3H7NO
   ────────────────────────────────────────────────────────────────────
     energy        -462.2355 eV
     composition   -453.1394 eV   (intercept 0.6031)
     connectivity  -9.0961 eV
     status        success

     top connectivity terms  (5 of 34 active, 633 in model)
     term                                     contribution (eV)
     ---------------------------------------  -----------------
     first_order_v1[localized_bond_enthalpy]            -8.3323
     v3_charge_distance_decay_product                   -0.2930
     graph_spectral_energy                              +0.1903
     gasteiger_bond_charge_difference                   -0.1884
     rdkit_chi1n                                        -0.1882

     domain distance  0.451 / q99 3.346

``domain distance`` compares this molecule's standardized descriptor vector to
the training centre.  A value above the reported ``q99`` raises an
applicability warning rather than silently changing the prediction.

Ranking Constitutional Isomers
------------------------------

``synde rank`` requires every candidate to share one molecular formula and
formal charge, which is what makes the ordering a connectivity test:

.. code-block:: console

   synde rank CCCCC 'CC(C)CC' 'CC(C)(C)C'

.. code-block:: text

   C5H12   3 candidates, lowest predicted energy first
   ────────────────────────────────────────────────────────────────────
     #  structure  energy (eV)  Δ vs best  connectivity
     -  ---------  -----------  ---------  ------------
     1  CC(C)(C)C    -458.0003    +0.0000      -11.0020
     2  CC(C)CC      -457.9810    +0.0193      -10.9827
     3  CCCCC        -457.9317    +0.0685      -10.9335

Passing structures with different formulas is rejected with the offending
formulas named, because ranking across formulas is not a supported operation.

Inspecting the Model
--------------------

``synde card`` prints the provenance and the applicability boundary of the
artifact that would be used for scoring:

.. code-block:: console

   synde card
   synde card --json

Batch Files and Pipelines
-------------------------

``--input`` reads one SMILES per line.  Blank lines and ``#`` comments are
ignored, and only the first whitespace-separated field is used, so ordinary
``.smi`` files with trailing names work unchanged:

.. code-block:: console

   synde predict --input molecules.smi --json > energies.json
   cat molecules.smi | synde predict --input -

CSV is the third output format, for spreadsheets and ``pandas.read_csv``:

.. code-block:: console

   synde predict --input molecules.smi --format csv > energies.csv

.. code-block:: text

   input,canonical_smiles,formula,status,predicted_energy,units,...
   CCO,CCO,C2H6O,success,-310.01802879478976,eV,...

By default a structure outside the model domain stops the run.  Pass
``--keep-going`` to score the remainder and report the skipped inputs on
standard error:

.. code-block:: console

   synde predict --input molecules.smi --keep-going

Scoring in Parallel
-------------------

Feature extraction dominates runtime and is independent per structure, so large
files parallelize well.  ``--jobs`` sets the worker process count and ``--jobs 0``
uses every core:

.. code-block:: console

   synde predict --input big.smi --jobs 8 --format csv > energies.csv

Each worker starts an interpreter, imports RDKit, and loads the model once, so
SynDE caps the worker count at one per 100 structures and scores small batches
serially. The effective worker count therefore may be lower than ``--jobs``.

When the run is attached to a terminal and the input is large, a single
updating progress line is written to standard error, so redirected output stays
clean.

Shell Completion
----------------

``synde completion`` prints a completion script for bash, zsh, or fish:

.. code-block:: console

   synde completion bash >> ~/.bashrc
   synde completion zsh >> ~/.zshrc
   synde completion fish > ~/.config/fish/completions/synde.fish

Completion covers the subcommands, the option names, the ``--format`` and
``--color`` values, and file paths for ``--input`` and ``--model``.

Exit Status and Colour
----------------------

The process exits ``0`` on success and ``1`` when any input failed, so
``--keep-going`` runs still signal partial failure to a calling script.
Colour is enabled only for an interactive terminal; it honours the ``NO_COLOR``
and ``FORCE_COLOR`` conventions and the explicit ``--color`` and ``--no-color``
options.

Option Summary
--------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Option
     - Meaning
   * - ``--input FILE``
     - Read SMILES from ``FILE``; ``-`` reads standard input.
   * - ``--format {table,json,csv}``
     - Output format; ``--json`` and ``--csv`` are shorthands.
   * - ``--jobs N``
     - Worker processes for scoring; ``0`` uses every core.
   * - ``--model PATH``
     - Score with a saved artifact instead of the packaged model.
   * - ``--explain``
     - Print the full contribution breakdown (``predict`` only).
   * - ``--top N``
     - Number of connectivity terms shown per structure.
   * - ``--keep-going``
     - Skip out-of-domain inputs instead of stopping.
   * - ``--precision N``
     - Digits after the decimal point (default ``4``).
   * - ``--color {auto,always,never}``
     - Control ANSI colour; ``--no-color`` is a shorthand.
