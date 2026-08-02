import unittest

from synde.energy import TheoryEnergyScorer
from synde.energy.topology_theory_energy import (
    TopologyTheoryEnergyScorer,
    topology_energy_corrections,
)
from synde.graph import GraphBuilder


class TestTopologyTheoryEnergy(unittest.TestCase):
    def corrections(self, smiles: str) -> dict[str, float]:
        return topology_energy_corrections(GraphBuilder.from_smiles(smiles))

    def test_biaryl_bond_loses_full_pi_coupling(self):
        biphenyl = self.corrections("c1ccccc1-c2ccccc2")
        naphthalene = self.corrections("c1ccc2ccccc2c1")

        self.assertGreater(biphenyl["topology_pi_decoupling"], 0.0)
        self.assertEqual(naphthalene["topology_pi_decoupling"], 0.0)

    def test_ortho_substitution_is_more_congested_than_meta_or_para(self):
        ortho = self.corrections("Cc1ccccc1C")["ortho_steric_congestion"]
        meta = self.corrections("Cc1cccc(C)c1")["ortho_steric_congestion"]
        para = self.corrections("Cc1ccc(C)cc1")["ortho_steric_congestion"]

        self.assertGreater(ortho, meta)
        self.assertEqual(meta, para)

    def test_large_ring_refinement_replaces_flat_base_approximation(self):
        nine = self.corrections("C1CCCCCCCC1")["refined_ring_topology"]
        twelve = self.corrections("C1CCCCCCCCCCC1")["refined_ring_topology"]

        self.assertGreater(nine, 0.0)
        self.assertLess(twelve, 0.0)

    def test_experimental_scorer_does_not_change_frozen_result(self):
        graph = GraphBuilder.from_smiles("c1ccccc1-c2ccccc2")
        before = TheoryEnergyScorer().score(graph)
        candidate = TopologyTheoryEnergyScorer().score(graph)
        after = TheoryEnergyScorer().score(graph)

        self.assertEqual(before.score, after.score)
        self.assertNotEqual(candidate.score, before.score)
        self.assertFalse(candidate.provenance["calibrated"])
        self.assertFalse(candidate.provenance["fitted_coefficients"])
        self.assertFalse(candidate.provenance["permanent_holdout_evaluated"])
        self.assertTrue(candidate.provenance["experimental"])


if __name__ == "__main__":
    unittest.main()
