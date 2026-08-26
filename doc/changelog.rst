Changelog
=========

0.5.0 — 2026-08-27
------------------

Added
^^^^^

- A packaged cross-formula energy predictor with same-formula isomer ranking.
- External weight refinement that returns a new artifact and preserves the
  validated packaged model.
- The ``synde`` command-line interface for prediction, explanation, ranking,
  model-card inspection, structured output, and shell completion.
- Structured input and domain exceptions with machine-readable details.
- Text and notebook reports for predictions, rankings, and model cards.
- Lazy top-level imports and tests that guard the inference dependency boundary.
- Numbered experiment scripts, committed benchmark records, recovered xTB
  labels, and read-only release validation.

Changed
^^^^^^^

- Consolidated the experiment reproducibility contract in
  ``Experiment/README.md``.
- Added an explicit Python 3.11 Black target and moved repository maintenance
  commands under ``scripts/``.
- Expanded the user documentation for installation, command-line use,
  applicability limits, external refinement, and reproducibility.

Validation
^^^^^^^^^^

- The release validator covers 78,513 training molecules in 11,993 formula
  groups and 19,940 external molecules in 3,005 formula- and
  connectivity-disjoint groups.
- Package, lint, test, documentation, and benchmark checks are run from the
  pinned ``synde`` environment.
