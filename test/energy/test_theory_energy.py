import unittest

from synde.energy import TheoryEnergyScorer, theory_energy_corrections
from synde.graph import GraphBuilder


class TestTheoryEnergy(unittest.TestCase):
    def corrections(self, smiles: str) -> dict[str, float]:
        return theory_energy_corrections(GraphBuilder.from_smiles(smiles).graph)

    def test_enhanced_score_is_uncalibrated_and_additive(self):
        result = TheoryEnergyScorer().score(GraphBuilder.from_smiles("CC(=O)N"))

        self.assertFalse(result.provenance["calibrated"])
        self.assertFalse(result.provenance["fitted_coefficients"])
        self.assertTrue(result.provenance["benchmark_informed_development"])
        self.assertEqual(result.provenance["parameter_set"], "theory-organic-v1-frozen")
        self.assertAlmostEqual(result.score, sum(result.components.values()))
        self.assertIn("lone_pair_resonance", result.components)

    def test_amide_has_stronger_lone_pair_reward_than_ester(self):
        amide = self.corrections("CC(=O)N")["lone_pair_resonance"]
        ester = self.corrections("CC(=O)O")["lone_pair_resonance"]

        self.assertLess(amide, ester)
        self.assertLess(ester, 0.0)

    def test_alkene_substitution_is_stabilizing(self):
        ethene = self.corrections("C=C")["alkene_substitution"]
        tetramethyl = self.corrections("CC(C)=C(C)C")["alkene_substitution"]

        self.assertEqual(ethene, 0.0)
        self.assertLess(tetramethyl, ethene)

    def test_small_ring_alkene_gets_extra_strain(self):
        cyclopropene = self.corrections("C1=CC1")["unsaturated_ring_strain"]
        cyclohexene = self.corrections("C1=CCCCC1")["unsaturated_ring_strain"]

        self.assertGreater(cyclopropene, cyclohexene)

    def test_hetero_pi_correction_leaves_carbon_alkene_reference_unchanged(self):
        alkene = self.corrections("C=C")["hetero_pi_bond_correction"]
        carbonyl = self.corrections("C=O")["hetero_pi_bond_correction"]
        imine = self.corrections("C=N")["hetero_pi_bond_correction"]

        self.assertEqual(alkene, 0.0)
        self.assertLess(carbonyl, imine)
        self.assertLess(imine, 0.0)

    def test_extended_resonance_covers_common_delocalized_groups(self):
        amidine = self.corrections("CC(=N)N")["extended_resonance"]
        sulfonamide = self.corrections("CS(=O)(=O)N")["extended_resonance"]
        nitro = self.corrections("C[N+](=O)[O-]")["extended_resonance"]

        self.assertLess(amidine, 0.0)
        self.assertLess(sulfonamide, 0.0)
        self.assertLess(nitro, 0.0)


if __name__ == "__main__":
    unittest.main()
