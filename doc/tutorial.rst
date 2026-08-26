Tutorials and examples
======================

Attributing a SynDE ranking
---------------------------

SynDE is a linear equation, so each prediction can be decomposed exactly. The
following example reports the terms that most strongly separate the first two
ranked isomers:

.. code-block:: python

   from synde.energy import SynDEEnergyPredictor
   from synde.graph import GraphBuilder

   predictor = SynDEEnergyPredictor.load_default()
   graphs = [
       GraphBuilder.from_smiles("CCCCC"),
       GraphBuilder.from_smiles("CC(C)CC"),
       GraphBuilder.from_smiles("CC(C)(C)C"),
   ]
   ranked = [
       (graphs[index], output) for index, output in predictor.rank_group(graphs)
   ]

   first, second = ranked[:2]
   differences = {
       name: second[1].connectivity_contributions[name]
       - first[1].connectivity_contributions[name]
       for name in first[1].connectivity_contributions
   }
   for name, delta in sorted(
       differences.items(), key=lambda item: abs(item[1]), reverse=True
   )[:10]:
       print(name, delta)

The displayed values are contributions to a statistical prediction. They can
identify graph motifs associated with a ranking, but should not be interpreted
as an energy decomposition from an electronic-structure Hamiltonian.

Checking applicability warnings
--------------------------------

Predictions are returned rather than silently discarded when a molecule lies
outside a descriptive training boundary. Applications should retain and review
the warning codes:

.. code-block:: python

   molecule = GraphBuilder.from_smiles("CCCCCCCCCCCC")
   output = predictor.predict(molecule)

   if output.warnings:
       print(molecule.canonical_smiles, output.warnings)
       print(output.descriptors["selected_feature_distance"])
       print(output.descriptors["outside_training_composition"])

These warnings are applicability diagnostics, not calibrated uncertainty
intervals or validated abstention decisions.

Serializing auditable output
----------------------------

Result objects are dataclasses with a JSON-compatible ``to_dict()`` method:

.. code-block:: python

   import json

   record = {
       "model_sha256": predictor.model_sha256,
       "evaluation_status": predictor.card.evaluation_status,
       "ranking": [
           {
               "canonical_smiles": graph.canonical_smiles,
               "result": result.to_dict(),
           }
           for graph, result in ranked
       ],
   }
   print(json.dumps(record, indent=2))

Saving and loading a custom artifact
------------------------------------

Refined or otherwise constructed predictors can be persisted as JSON:

.. code-block:: python

   from pathlib import Path

   model_path = Path("synde-refined.json")
   refined.save(model_path)
   restored = SynDEEnergyPredictor.load(model_path)

The artifact includes its model card, coefficient blocks, feature statistics,
composition ranges, and refinement report. Loading computes a SHA-256 digest
that is included in subsequent prediction provenance.

Graph and reaction utilities
----------------------------

The package also exposes lower-level graph, orbital, geometry, and reaction
utilities. These are useful for descriptor inspection and method development,
but they do not inherit the final SynDE model's external-validation claim.

.. code-block:: python

   from synde.graph import GraphBuilder, assign_pi_systems

   graph = GraphBuilder.from_smiles("n1ccccc1")
   pi = assign_pi_systems(graph)
   print(pi.electron_count)

``GraphEnergy.score_reaction`` and ``GraphEnergy.score_its`` provide graph-state
and mapped reaction-centre scores. Their outputs are feasibility-oriented graph
coordinates, not physical transition-state energies.
