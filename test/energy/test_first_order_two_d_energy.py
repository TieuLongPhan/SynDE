import unittest

from synde.energy import FirstOrderTwoDEnergyScorer
from synde.graph import GraphBuilder


class TestFirstOrderTwoDEnergyScorer(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = FirstOrderTwoDEnergyScorer()

    def score(self, smiles: str):
        return self.scorer.score(GraphBuilder.from_smiles(smiles))

    def test_localized_bond_inventory_uses_multiple_bond_enthalpies(self):
        ethane = self.score("CC")
        ethene = self.score("C=C")
        ethyne = self.score("C#C")

        self.assertAlmostEqual(
            ethane.components["localized_bond_enthalpy"],
            -(347.0 + 6 * 413.0),
        )
        self.assertAlmostEqual(
            ethene.components["localized_bond_enthalpy"],
            -(614.0 + 4 * 413.0),
        )
        self.assertAlmostEqual(
            ethyne.components["localized_bond_enthalpy"],
            -(839.0 + 2 * 413.0),
        )

    def test_huckel_terms_are_increments_not_local_pi_bonds(self):
        ethene = self.score("C=C")
        butadiene = self.score("C=CC=C")
        benzene = self.score("c1ccccc1")

        self.assertAlmostEqual(ethene.components["acyclic_pi_delocalization"], 0.0)
        self.assertLess(butadiene.components["acyclic_pi_delocalization"], 0.0)
        self.assertLess(benzene.components["aromatic_pi_delocalization"], 0.0)

    def test_resonance_families_remain_separate(self):
        amide = self.score("CC(=O)N")
        ester = self.score("CC(=O)O")

        self.assertLess(amide.components["carbonyl_n_lone_pair_delocalization"], 0.0)
        self.assertLess(ester.components["oxygen_lone_pair_delocalization"], 0.0)

    def test_branching_and_forced_strain_have_expected_signs(self):
        isobutane = self.score("CC(C)C")
        cyclopropane = self.score("C1CC1")

        self.assertLess(isobutane.components["protobranching_13"], 0.0)
        self.assertGreater(cyclopropane.components["forced_small_ring_strain"], 0.0)

    def test_rigid_intramolecular_hbond_is_only_a_raw_candidate(self):
        ortho = self.score("O=Cc1ccccc1O")
        para = self.score("O=Cc1ccc(O)cc1")

        self.assertEqual(
            ortho.descriptors["raw_topology_terms"]["rigid_intramolecular_hbond"],
            -20.0,
        )
        self.assertEqual(
            para.descriptors["raw_topology_terms"]["rigid_intramolecular_hbond"],
            0.0,
        )
        self.assertNotIn("rigid_intramolecular_hbond", ortho.components)

    def test_all_components_sum_and_provenance_is_strictly_two_d(self):
        result = self.score("CC(=O)N")

        self.assertAlmostEqual(result.score, sum(result.components.values()))
        self.assertEqual(result.units, "kJ/mol_score")
        self.assertFalse(result.provenance["calibrated"])
        self.assertFalse(result.provenance["fitted_coefficients"])
        self.assertFalse(result.provenance["uses_coordinates"])
        self.assertFalse(result.provenance["uses_conformers"])
        self.assertFalse(result.provenance["uses_xtb"])
        self.assertTrue(result.provenance["permanent_holdout_evaluated"])
        self.assertEqual(
            result.provenance["permanent_holdout_protocol"],
            "synde-ord-first-order-2d-permanent-holdout-v1",
        )
        self.assertFalse(result.provenance["holdout_tuned"])

    def test_explicit_and_implicit_hydrogen_are_equivalent(self):
        implicit = self.score("CC")
        explicit = self.score("[H]C([H])([H])C([H])([H])[H]")

        self.assertAlmostEqual(implicit.score, explicit.score)
        for name in implicit.components:
            self.assertAlmostEqual(
                implicit.components[name],
                explicit.components[name],
            )


if __name__ == "__main__":
    unittest.main()
