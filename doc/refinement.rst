External weight refinement
==========================

SynDE can adapt its linear coefficients to reference energies from another
compatible dataset. Refinement is anchored to an existing predictor: it fits
coefficient changes rather than replacing the packaged artifact or mutating it
in memory.

Dataset contract
----------------

Each target must use the units and reference protocol declared by
``predictor.card``. Records must have unique identifiers and contain connected,
closed-shell structures within the base model's supported element and charge
domain. At least two records are required, although a useful fit normally
requires substantially more coverage.

.. code-block:: python

   from synde.energy import EnergyRefinementRecord, SynDEEnergyPredictor

   predictor = SynDEEnergyPredictor.load_default()
   print(predictor.card.target)
   print(predictor.card.units)
   print(predictor.card.reference_protocol)

   development = [
       EnergyRefinementRecord.from_smiles(
           "mol-001",
           "CCO",
           -12.34,
           provenance={"source": "reference-v1"},
       ),
       EnergyRefinementRecord.from_smiles(
           "mol-002",
           "COC",
           -12.10,
           provenance={"source": "reference-v1"},
       ),
   ]

Keep the refinement and evaluation sets separate before inspecting results.
Do not tune ``alpha`` or the enabled weight blocks on the final evaluation set.

Choosing what to refine
-----------------------

Three blocks can be enabled independently:

``intercept``
   Corrects a constant offset. This is often the least flexible first step.

``composition``
   Updates elemental atom-count coefficients and therefore affects
   cross-formula prediction.

``connectivity``
   Updates named graph coefficients and can change constitutional-isomer
   ranking. It needs enough structural diversity to constrain the fit.

For a small external dataset, begin with the intercept and composition block:

.. code-block:: python

   refined, report = predictor.refine(
       development,
       dataset_name="reference-v1-development",
       alpha=10.0,
       refine_intercept=True,
       refine_composition=True,
       refine_connectivity=False,
   )

``alpha`` penalizes standardized coefficient changes away from the base model.
Larger values retain more of the packaged weights; ``alpha=0`` gives an
unanchored least-squares update and should be used only with a well-constrained
design.

Inspecting the fit
------------------

The returned report records the external dataset digest, enabled blocks,
coefficient shift, and in-sample metrics before and after refinement:

.. code-block:: python

   print(report.dataset_sha256)
   print(report.refined_blocks)
   print(report.baseline_metrics)
   print(report.refined_metrics)
   print(report.validation_status)

Improved development error is not evidence of generalization. Evaluate the
returned predictor once on the reserved data and retain the target protocol,
split definition, metrics, and artifact digest with the result.

Saving the result
-----------------

Save the new predictor and, if useful, the standalone report:

.. code-block:: python

   import json
   from pathlib import Path

   model_path = Path("synde-refined.json")
   refined.save(model_path)
   Path("synde-refinement-report.json").write_text(
       json.dumps(report.to_dict(), indent=2) + "\n",
       encoding="utf-8",
   )

   restored = SynDEEnergyPredictor.load(model_path)

The model artifact embeds the same refinement report. Its model card is marked
``external refinement fitted; independent validation required`` and its prior
formula- and connectivity-disjoint evaluation flags are cleared.
