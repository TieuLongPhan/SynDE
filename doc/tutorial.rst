Tutorials and examples
======================

Graph construction
------------------

``GraphBuilder`` converts SMILES into the normalized graph contract used by
the graph-first model:

.. code-block:: python

   from synde.graph import GraphBuilder, assign_pi_systems

   graph = GraphBuilder.from_smiles("n1ccccc1")
   pi = assign_pi_systems(graph)
   print(pi.electron_count)

The graph layer provides π-system assignment, generalized Hückel descriptors,
frontier-orbital descriptors, HSAB descriptors, and candidate pair ranking.

State score versus ITS score
----------------------------

``score_reaction`` measures the product-minus-reactant graph-state change.
It can be used with or without atom maps, although maps are needed for exact
bond-change diagnostics.

.. code-block:: python

   from synde.energy import GraphEnergy

   energy = GraphEnergy()
   state = energy.score_reaction(
       "[CH2:1]=[CH2:2]>>[CH3:1][CH3:2]"
   )
   print(state.reaction_delta_score)

``score_its`` additionally builds an atom-mapped imaginary transition-state
graph. Use it when the reaction centre is known and a feasibility-oriented
ranking feature is required:

.. code-block:: python

   its = energy.score_its("[CH2:1]=[CH2:2]>>[CH3:1][CH3:2]")
   print(its.components)

Calibration and external comparison
-----------------------------------

The heuristic weights are starting values. Fit a named calibration model only
against a provenance-tracked target and evaluate it on held-out data. Optional
conformer and xTB utilities are comparison/refinement layers; they do not turn
an uncalibrated graph score into a physical energy.
