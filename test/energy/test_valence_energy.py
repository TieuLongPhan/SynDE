import unittest

from synde.energy import ValenceEnergyScorer
from synde.graph import GraphBuilder


class TestValenceEnergyScorer(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = ValenceEnergyScorer()

    def score(self, smiles: str):
        return self.scorer.score(GraphBuilder.from_smiles(smiles))

    def test_localized_alkene_and_conjugated_pi_terms_do_not_overlap(self):
        ethene = self.score("C=C")
        butadiene = self.score("C=CC=C")
        benzene = self.score("c1ccccc1")

        ethene_raw = ethene.descriptors["raw_energy_terms"]
        butadiene_raw = butadiene.descriptors["raw_energy_terms"]
        benzene_raw = benzene.descriptors["raw_energy_terms"]

        self.assertLess(ethene.components["lewis_local_pi_reference"], 0.0)
        self.assertAlmostEqual(ethene_raw["acyclic_pi_delocalization"], 0.0)
        self.assertLess(butadiene_raw["acyclic_pi_delocalization"], 0.0)
        self.assertLess(
            benzene_raw["aromatic_pi_delocalization"],
            butadiene_raw["acyclic_pi_delocalization"],
        )
        self.assertEqual(benzene.components["aromatic_pi_delocalization"], 0.0)

    def test_lone_pair_increment_is_separate_from_carbonyl_pi_bond(self):
        ketone = self.score("CC(=O)C")
        amide = self.score("CC(=O)N")

        self.assertAlmostEqual(
            ketone.components["carbonyl_n_lone_pair_delocalization"], 0.0
        )
        self.assertEqual(amide.components["carbonyl_n_lone_pair_delocalization"], 0.0)
        self.assertLess(
            amide.descriptors["raw_energy_terms"][
                "carbonyl_n_lone_pair_delocalization"
            ],
            0.0,
        )
        self.assertAlmostEqual(
            ketone.components["lewis_local_pi_reference"],
            amide.components["lewis_local_pi_reference"],
        )

    def test_lone_pair_donor_elements_are_separate(self):
        amide = self.score("CC(=O)N")
        ester = self.score("CC(=O)O")

        self.assertEqual(amide.components["carbonyl_n_lone_pair_delocalization"], 0.0)
        self.assertEqual(amide.components["oxygen_lone_pair_delocalization"], 0.0)
        self.assertLess(ester.components["oxygen_lone_pair_delocalization"], 0.0)
        self.assertEqual(ester.components["carbonyl_n_lone_pair_delocalization"], 0.0)

    def test_nitrogen_donor_environments_are_separate(self):
        amide = self.score("CC(=O)N")
        aniline = self.score("Nc1ccccc1")
        amidine = self.score("CC(=N)N")

        self.assertLess(
            amide.descriptors["raw_energy_terms"][
                "carbonyl_n_lone_pair_delocalization"
            ],
            0.0,
        )
        self.assertLess(aniline.components["aryl_n_lone_pair_delocalization"], 0.0)
        self.assertLess(
            amidine.descriptors["raw_energy_terms"]["imine_n_lone_pair_delocalization"],
            0.0,
        )

    def test_protobranching_counts_excess_one_three_interactions(self):
        isobutane = self.score("CC(C)C")
        neopentane = self.score("CC(C)(C)C")

        self.assertAlmostEqual(isobutane.components["protobranching"], -0.08)
        self.assertAlmostEqual(neopentane.components["protobranching"], -0.24)

    def test_small_saturated_ring_strain_is_narrow_and_active(self):
        cyclopropane = self.score("C1CC1")
        cyclohexane = self.score("C1CCCCC1")

        self.assertGreater(cyclopropane.components["small_saturated_ring_strain"], 0.0)
        self.assertGreater(
            cyclopropane.descriptors["raw_energy_terms"]["small_saturated_ring_strain"],
            0.0,
        )
        self.assertEqual(
            cyclohexane.descriptors["raw_energy_terms"]["small_saturated_ring_strain"],
            0.0,
        )

    def test_sulfonamide_has_explicit_raw_sulfonyl_donation(self):
        result = self.score("CS(=O)(=O)N")

        self.assertEqual(result.status, "success")
        self.assertEqual(result.components["sulfonyl_n_lone_pair_delocalization"], 0.0)
        self.assertLess(
            result.descriptors["raw_energy_terms"][
                "sulfonyl_n_lone_pair_delocalization"
            ],
            0.0,
        )

    def test_triple_and_cumulene_orbitals_are_supported(self):
        nitrile = self.score("C#N")
        allene = self.score("C=C=C")

        self.assertEqual(nitrile.status, "success")
        self.assertEqual(nitrile.descriptors["n_orbital_pi_electrons"], 4)
        self.assertEqual(allene.status, "success")
        self.assertEqual(allene.descriptors["n_orbital_pi_systems"], 2)

    def test_explicit_and_implicit_hydrogen_scores_match(self):
        implicit = self.score("CC")
        explicit = self.score("[H]C([H])([H])C([H])([H])[H]")

        self.assertAlmostEqual(implicit.score, explicit.score)
        for name in implicit.components:
            self.assertAlmostEqual(implicit.components[name], explicit.components[name])

    def test_disconnected_components_are_additive(self):
        mixture = self.score("CC.O")
        ethane = self.score("CC")
        water = self.score("O")

        self.assertAlmostEqual(mixture.score, ethane.score + water.score)

    def test_provenance_declares_zero_fit_development_model(self):
        result = self.score("CC(=O)N")

        self.assertFalse(result.provenance["calibrated"])
        self.assertFalse(result.provenance["fitted_coefficients"])
        self.assertTrue(result.provenance["benchmark_informed_development"])
        self.assertFalse(result.provenance["permanent_holdout_evaluated"])
        self.assertEqual(result.descriptors["synkit_label_profile"], "full")
        self.assertEqual(result.components["aromatic_pi_delocalization"], 0.0)
        self.assertEqual(result.components["mixed_pi_delocalization"], 0.0)
        self.assertEqual(result.components["acyclic_pi_delocalization"], 0.0)
        self.assertEqual(result.components["graph_polarization"], 0.0)
        self.assertLess(
            result.descriptors["raw_energy_terms"]["graph_polarization"], 0.0
        )
        self.assertAlmostEqual(result.score, sum(result.components.values()))


if __name__ == "__main__":
    unittest.main()
