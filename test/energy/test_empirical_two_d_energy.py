import importlib.util
import unittest
from unittest.mock import patch

from synde.energy import EmpiricalTwoDEnergyScorer
from synde.graph import GraphBuilder


class TestEmpiricalBackendAvailability(unittest.TestCase):
    def test_missing_optional_backend_is_explicit(self) -> None:
        scorer = EmpiricalTwoDEnergyScorer()
        with patch(
            "synde.energy.empirical_two_d_energy._load_joback_backend",
            return_value=None,
        ):
            result = scorer.score(GraphBuilder.from_smiles("CCO"))

        self.assertEqual(result.status, "unsupported")
        self.assertIsNone(result.score)
        self.assertEqual(result.warnings, ("THERMO_NOT_INSTALLED",))


@unittest.skipUnless(
    importlib.util.find_spec("thermo") is not None,
    "optional thermo backend is not installed",
)
class TestEmpiricalTwoDEnergyScorer(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = EmpiricalTwoDEnergyScorer()

    def score(self, smiles: str):
        return self.scorer.score(GraphBuilder.from_smiles(smiles))

    def test_joback_reference_is_exposed_as_named_group_contributions(self):
        result = self.score("CCCC")

        self.assertEqual(result.status, "success")
        self.assertEqual(result.units, "kJ/mol")
        self.assertAlmostEqual(result.components["joback_reference_offset"], 68.29)
        self.assertAlmostEqual(result.components["joback_01_ch3"], 2 * -76.45)
        self.assertAlmostEqual(result.components["joback_02_ch2"], 2 * -20.64)
        self.assertAlmostEqual(result.score, sum(result.components.values()))

    def test_huckel_and_lone_pair_terms_are_separately_ablatable(self):
        benzene = self.score("c1ccccc1")
        amide = self.score("CC(=O)N")

        self.assertLess(benzene.components["huckel_structural_pi_delocalization"], 0.0)
        self.assertLess(amide.components["huckel_lone_pair_delocalization"], 0.0)

    def test_only_connectivity_forced_small_ring_strain_is_active(self):
        cyclopropane = self.score("C1CC1")
        cyclohexane = self.score("C1CCCCC1")

        self.assertAlmostEqual(
            cyclopropane.components["forced_small_ring_strain"], 115.0
        )
        self.assertEqual(cyclohexane.components["forced_small_ring_strain"], 0.0)

    def test_provenance_enforces_graph_only_zero_fit_contract(self):
        result = self.score("CCO")

        self.assertFalse(result.provenance["calibrated"])
        self.assertFalse(result.provenance["fitted_coefficients"])
        self.assertFalse(result.provenance["uses_coordinates"])
        self.assertFalse(result.provenance["uses_conformers"])
        self.assertFalse(result.provenance["uses_xtb"])
        self.assertFalse(result.provenance["uses_ord_labels_at_inference"])
        self.assertFalse(result.provenance["permanent_holdout_evaluated"])

    def test_explicit_and_implicit_hydrogen_are_equivalent(self):
        implicit = self.score("CC")
        explicit = self.score("[H]C([H])([H])C([H])([H])[H]")

        self.assertAlmostEqual(implicit.score, explicit.score)
        self.assertEqual(
            implicit.descriptors["joback_group_counts"],
            explicit.descriptors["joback_group_counts"],
        )


if __name__ == "__main__":
    unittest.main()
