import importlib.util
import math
import unittest
from unittest.mock import patch

from synde.energy import GFN2SinglePointScorer
from synde.graph import GraphBuilder


class TestGFN2SinglePointScorer(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = GFN2SinglePointScorer()

    def test_missing_optional_backend_is_explicit(self) -> None:
        with patch(
            "synde.energy.semiempirical_energy._load_tblite_calculator",
            return_value=None,
        ):
            result = self.scorer.score(GraphBuilder.from_smiles("CCO"))

        self.assertEqual(result.status, "unsupported")
        self.assertIsNone(result.score)
        self.assertEqual(result.warnings, ("TBLITE_NOT_INSTALLED",))

    def test_open_shell_requires_an_explicit_spin_policy(self) -> None:
        result = self.scorer.score(GraphBuilder.from_smiles("[CH3]"))

        self.assertEqual(result.status, "unsupported")
        self.assertIsNone(result.score)
        self.assertIn("OPEN_SHELL_REQUIRES_EXPLICIT_UHF", result.warnings)

    def test_even_electron_diradical_is_not_assumed_to_be_singlet(self) -> None:
        result = self.scorer.score(GraphBuilder.from_smiles("[O][O]"))

        self.assertEqual(result.status, "unsupported")
        self.assertEqual(result.descriptors["radical_electrons"], 2)
        self.assertIn("OPEN_SHELL_REQUIRES_EXPLICIT_UHF", result.warnings)

    @unittest.skipUnless(
        importlib.util.find_spec("tblite") is not None,
        "optional tblite backend is not installed",
    )
    def test_actual_gfn2_energy_is_finite_and_uncalibrated(self) -> None:
        result = self.scorer.score(GraphBuilder.from_smiles("CCO"))

        self.assertIn(result.status, {"success", "partial"})
        self.assertTrue(math.isfinite(result.score))
        self.assertEqual(result.units, "eV")
        self.assertFalse(result.provenance["calibrated"])
        self.assertFalse(result.provenance["fitted_coefficients"])
        self.assertTrue(result.provenance["semiempirical"])
        self.assertFalse(result.provenance["proxy"])
        self.assertTrue(result.provenance["permanent_holdout_evaluated"])
        self.assertFalse(result.provenance["holdout_tuned"])
        self.assertAlmostEqual(result.score, sum(result.components.values()))
        self.assertEqual(
            result.descriptors["preoptimization_force_field"],
            "MMFF94",
        )


if __name__ == "__main__":
    unittest.main()
