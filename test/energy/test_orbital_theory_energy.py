import unittest

from synde.energy import OrbitalTheoryEnergyScorer, TheoryEnergyScorer
from synde.graph import GraphBuilder


class TestOrbitalTheoryEnergyScorer(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = OrbitalTheoryEnergyScorer()

    def test_amide_uses_orbital_delta_instead_of_manual_resonance(self):
        result = self.scorer.score(GraphBuilder.from_smiles("CC(=O)N"))

        self.assertEqual(result.status, "success")
        self.assertIn("orbital_pi_extension", result.components)
        self.assertNotIn("lone_pair_resonance", result.components)
        self.assertEqual(result.descriptors["n_orbital_pi_electrons"], 4)

    def test_nitrile_is_supported_with_four_pi_electrons(self):
        result = self.scorer.score(GraphBuilder.from_smiles("C#N"))

        self.assertEqual(result.status, "success")
        self.assertEqual(result.descriptors["n_orbital_pi_electrons"], 4)
        self.assertNotIn("PI_ORBITAL_MULTIPLICITY_UNSUPPORTED", result.warnings)
        self.assertLess(result.components["orbital_pi_extension"], 0.0)

    def test_allene_is_treated_as_two_pi_systems(self):
        result = self.scorer.score(GraphBuilder.from_smiles("C=C=C"))

        self.assertEqual(result.status, "success")
        self.assertEqual(result.descriptors["n_orbital_pi_systems"], 2)
        self.assertEqual(result.descriptors["n_orbital_pi_electrons"], 4)

    def test_frozen_theory_scorer_remains_unchanged_and_separate(self):
        graph = GraphBuilder.from_smiles("CC(=O)N")
        frozen = TheoryEnergyScorer().score(graph)
        experimental = self.scorer.score(graph)

        self.assertEqual(frozen.provenance["parameter_set"], "theory-organic-v1-frozen")
        self.assertTrue(experimental.provenance["experimental"])
        self.assertFalse(experimental.provenance["permanent_holdout_evaluated"])
        self.assertEqual(
            experimental.provenance["evaluation_protocol"],
            "synde-ord-orbital-holdout-v1",
        )
        self.assertFalse(experimental.provenance["holdout_tuned"])


if __name__ == "__main__":
    unittest.main()
