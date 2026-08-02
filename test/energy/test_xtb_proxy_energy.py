import unittest

from synde.energy import (
    GFN2_XTB_VALENCE_REFERENCE_EV,
    GFN2XTBProxyScorer,
)
from synde.graph import GraphBuilder


class TestGFN2XTBProxyScorer(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = GFN2XTBProxyScorer()

    def score(self, smiles: str):
        return self.scorer.score(GraphBuilder.from_smiles(smiles))

    def test_score_is_explicitly_uncalibrated_proxy(self) -> None:
        result = self.score("CCO")

        self.assertEqual(result.status, "success")
        self.assertEqual(result.units, "eV_proxy")
        self.assertFalse(result.provenance["calibrated"])
        self.assertFalse(result.provenance["fitted_coefficients"])
        self.assertTrue(result.provenance["proxy"])
        self.assertFalse(result.provenance["ord_label_provenance_verified"])
        self.assertAlmostEqual(result.score, sum(result.components.values()))

    def test_atomic_reference_uses_neutral_gfn2_valence_levels(self) -> None:
        methane = self.score("C")
        expected = (
            GFN2_XTB_VALENCE_REFERENCE_EV["C"] + 4 * GFN2_XTB_VALENCE_REFERENCE_EV["H"]
        )

        self.assertAlmostEqual(methane.components["gfn2_valence_reference"], expected)
        self.assertEqual(
            methane.descriptors["element_counts_including_implicit_h"],
            {"C": 1, "H": 4},
        )

    def test_larger_hydrocarbon_has_more_negative_total_proxy(self) -> None:
        methane = self.score("C")
        ethane = self.score("CC")

        self.assertLess(ethane.score, methane.score)

    def test_explicit_and_implicit_hydrogens_match(self) -> None:
        implicit = self.score("CC")
        explicit = self.score("[H]C([H])([H])C([H])([H])[H]")

        self.assertAlmostEqual(implicit.score, explicit.score)
        self.assertEqual(implicit.components, explicit.components)

    def test_unsupported_elements_are_not_silently_approximated(self) -> None:
        result = self.score("[Na+]")

        self.assertEqual(result.status, "unsupported")
        self.assertIsNone(result.score)
        self.assertIn("GFN2_PROXY_UNSUPPORTED_ELEMENTS:Na", result.warnings)


if __name__ == "__main__":
    unittest.main()
