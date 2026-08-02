Tutorials and examples
======================

Attributing a SynDE ranking
---------------------------

SynDE is a linear equation, so each prediction can be decomposed exactly. The
following example reports the terms that most strongly separate the first two
ranked isomers:

.. code-block:: python

   from synde.energy import SynDEScorer
   from synde.graph import GraphBuilder

   scorer = SynDEScorer.load_default()
   graphs = [
       GraphBuilder.from_smiles("CCCCC"),
       GraphBuilder.from_smiles("CC(C)CC"),
       GraphBuilder.from_smiles("CC(C)(C)C"),
   ]
   ranked = sorted(
       zip(graphs, scorer.score_group(graphs)), key=lambda row: row[1].score
   )

   first, second = ranked[:2]
   differences = {
       name: second[1].components[name] - first[1].components[name]
       for name in first[1].components
   }
   for name, delta in sorted(
       differences.items(), key=lambda item: abs(item[1]), reverse=True
   )[:10]:
       print(name, delta)

The displayed values are contributions to a statistical prediction. They can
identify graph motifs associated with a ranking, but should not be interpreted
as an energy decomposition from an electronic-structure Hamiltonian.

Serializing auditable output
----------------------------

Result objects are dataclasses with a JSON-compatible ``to_dict()`` method:

.. code-block:: python

   import json

   record = {
       "model_sha256": scorer.model_sha256,
       "externally_validated": scorer.externally_validated,
       "ranking": [
           {
               "canonical_smiles": graph.canonical_smiles,
               "result": result.to_dict(),
           }
           for graph, result in ranked
       ],
   }
   print(json.dumps(record, indent=2))

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
