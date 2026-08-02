API reference
=============

Graph construction and descriptors
----------------------------------

.. automodule:: synde.graph
   :members:
   :undoc-members:

Canonical SynDE scorer
----------------------

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
comparison, and model-development workflows. ``SynDEScorer`` is the canonical
externally validated isomer-ranking interface.

.. automodule:: synde.energy
   :members:
   :undoc-members:
   :exclude-members: SynDEScorer, SynDEModelCard, SynDEValidationRecord, MoleculeScoreResult, OrdCalibratedV4Scorer, OrdCalibratedV4ModelCard

Geometry and workflow integrations
----------------------------------

.. automodule:: synde.geometry
   :members:
   :undoc-members:

.. automodule:: synde.integration
   :members:
   :undoc-members:
