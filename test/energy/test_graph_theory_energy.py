import unittest

from synde.energy import GraphTheoryEnergyScorer, TheoryEnergyScorer
from synde.graph import GraphBuilder


class TestGraphTheoryEnergyScorer(unittest.TestCase):
    def test_candidate_removes_only_geometry_dependent_components(self):
        graph = GraphBuilder.from_smiles("CC1CCCC1")
        frozen = TheoryEnergyScorer().score(graph)
        candidate = GraphTheoryEnergyScorer().score(graph)

        removed = (
            frozen.components["ring_strain"] + frozen.components["steric_congestion"]
        )
        self.assertAlmostEqual(candidate.score, frozen.score - removed)
        self.assertEqual(candidate.components["ring_strain"], 0.0)
        self.assertEqual(candidate.components["steric_congestion"], 0.0)
        self.assertEqual(
            candidate.components["sigma_bond_energy"],
            frozen.components["sigma_bond_energy"],
        )

    def test_candidate_is_uncalibrated_and_does_not_mutate_frozen(self):
        graph = GraphBuilder.from_smiles("C1CCCCC1")
        before = TheoryEnergyScorer().score(graph)
        candidate = GraphTheoryEnergyScorer().score(graph)
        after = TheoryEnergyScorer().score(graph)

        self.assertEqual(before.score, after.score)
        self.assertFalse(candidate.provenance["calibrated"])
        self.assertFalse(candidate.provenance["fitted_coefficients"])
        self.assertFalse(candidate.provenance["permanent_holdout_evaluated"])
        self.assertEqual(
            candidate.provenance["excluded_components"],
            ["ring_strain", "steric_congestion"],
        )


if __name__ == "__main__":
    unittest.main()
