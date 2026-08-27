API reference
=============

Graph construction and descriptors
----------------------------------

.. automodule:: synde.graph
   :members:
   :undoc-members:

Energy and ranking predictor
----------------------------

.. autoclass:: synde.energy.SynDEEnergyPredictor
   :members:

.. autoclass:: synde.energy.SynDEEnergyModelCard
   :members:

.. autoclass:: synde.energy.SynDEEnergyPrediction
   :members:

.. autoclass:: synde.energy.SynDEEnergyRanking
   :members:

Errors
------

Every SynDE exception subclasses :class:`ValueError`, so existing handlers keep
working while gaining structured context.

.. automodule:: synde.errors
   :members:
   :undoc-members:

Command line interface
----------------------

The ``synde`` console script is a thin wrapper over the predictor; see
:doc:`cli` for the user-facing documentation.

.. automodule:: synde.cli
   :members: main, build_parser

Presentation helpers
--------------------

Rendering used by the command line interface and by the interactive
``__repr__`` and ``_repr_html_`` hooks.

.. automodule:: synde.report
   :members:

.. automodule:: synde.formatting
   :members:

External refinement
-------------------

.. autoclass:: synde.energy.SynDEEnergyRefiner
   :members:

.. autoclass:: synde.energy.EnergyRefinementRecord
   :members:

.. autoclass:: synde.energy.EnergyRefinementReport
   :members:

Compatibility ranking scorer
----------------------------

.. autoclass:: synde.energy.SynDEScorer
   :members:

.. autoclass:: synde.energy.SynDEModelCard
   :members:

.. autoclass:: synde.energy.SynDEValidationRecord
   :members:

.. autoclass:: synde.energy.MoleculeScoreResult
   :members:

Advanced graph-scoring components
---------------------------------

The symbols below support descriptor inspection, reaction scoring, geometry
comparison, and model-development workflows. ``SynDEEnergyPredictor`` is the
main inference interface; ``SynDEScorer`` preserves compatibility with the
original connectivity-validation artifact.

.. automodule:: synde.energy
   :members:
   :undoc-members:
   :exclude-members: SynDEEnergyPredictor, SynDEEnergyModelCard,
      SynDEEnergyPrediction, SynDEEnergyRanking,
      SynDEEnergyRefiner, EnergyRefinementRecord,
      EnergyRefinementReport, SynDEScorer, SynDEModelCard,
      SynDEValidationRecord, MoleculeScoreResult, OrdCalibratedV4Scorer,
      OrdCalibratedV4ModelCard

Geometry and workflow integrations
----------------------------------

.. automodule:: synde.geometry
   :members:
   :undoc-members:

.. automodule:: synde.integration
   :members:
   :undoc-members:
