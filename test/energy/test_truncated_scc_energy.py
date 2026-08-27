import math
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from rdkit import Chem

from synde.energy import GFN2TwoCycleConfig, GFN2TwoCycleScorer
from synde.energy.truncated_scc_energy import (
    gfn2_pair_repulsion,
    parse_gfn2_repulsion_parameters,
    parse_xtb_scc_iterations,
)
from synde.graph import GraphBuilder

PARAMETERS = """
$Z= 1
 REPA= 2.000000
 REPB= 1.000000
$end
$Z= 6
 REPA= 1.250000
 REPB= 4.200000
$end
$Z= 8
 REPA= 1.500000
 REPB= 6.000000
$end
"""

XTB_OUTPUT = """
 iter      E             dE          RMSdq      gap      omega  full diag
   1    -11.5000000 -0.115000E+02  0.374E+00   13.52       0.0  T
   2    -11.5200000 -0.200000E-01  0.212E+00   13.15       1.0  T

   *** convergence criteria cannot be satisfied within 2 iterations ***
scf: Self consistent charge iterator did not converge
"""


class TestGFN2TwoCycleScorer(unittest.TestCase):
    def test_parameter_and_iteration_parsers_are_explicit(self) -> None:
        parameters = parse_gfn2_repulsion_parameters(PARAMETERS)
        rows = parse_xtb_scc_iterations(XTB_OUTPUT)

        self.assertEqual(parameters[6], (1.25, 4.2))
        self.assertEqual(set(rows), {1, 2})
        self.assertAlmostEqual(rows[2]["energy_hartree"], -11.52)
        self.assertAlmostEqual(rows[2]["charge_rms"], 0.212)
        self.assertAlmostEqual(rows[2]["gap_ev"], 13.15)

    def test_pair_repulsion_matches_the_stated_equation(self) -> None:
        molecule = Chem.AddHs(Chem.MolFromSmiles("[H][H]"))
        conformer = Chem.Conformer(molecule.GetNumAtoms())
        conformer.SetAtomPosition(0, (0.0, 0.0, 0.0))
        conformer.SetAtomPosition(1, (0.74, 0.0, 0.0))
        molecule.AddConformer(conformer)
        parameters = {1: (2.0, 1.0)}
        distance_bohr = 0.74 / 0.529177210903
        expected = math.exp(-2.0 * distance_bohr) / distance_bohr

        self.assertAlmostEqual(
            gfn2_pair_repulsion(molecule, parameters),
            expected,
            places=12,
        )

    def test_mocked_score_is_two_unscaled_additive_terms(self) -> None:
        parameter_file = Path("/tmp/synde-test-param-gfn2.txt")
        parameter_file.write_text(PARAMETERS, encoding="utf-8")
        scorer = GFN2TwoCycleScorer(
            GFN2TwoCycleConfig(
                executable="/mock/xtb",
                parameter_file=parameter_file,
            )
        )
        with patch(
            "synde.energy.truncated_scc_energy._run_two_cycle_xtb",
            return_value=(XTB_OUTPUT, 1),
        ):
            result = scorer.score(GraphBuilder.from_smiles("CCO"))

        self.assertEqual(result.status, "partial")
        self.assertTrue(math.isfinite(result.score))
        self.assertAlmostEqual(result.score, sum(result.components.values()))
        self.assertEqual(
            set(result.components),
            {"two_cycle_scc_energy", "explicit_pair_repulsion"},
        )
        self.assertFalse(result.provenance["calibrated"])
        self.assertFalse(result.provenance["fitted_coefficients"])
        self.assertFalse(result.provenance["target_rescaling"])
        self.assertFalse(result.provenance["actual_converged_gfn2"])
        self.assertEqual(result.provenance["scc_charge_updates"], 2)
        self.assertTrue(result.provenance["permanent_holdout_evaluated"])
        self.assertFalse(result.provenance["holdout_tuned"])
        self.assertIn("INTENTIONAL_SCC_NONCONVERGENCE", result.warnings)
        parameter_file.unlink()

    def test_open_shell_is_explicitly_unsupported(self) -> None:
        scorer = GFN2TwoCycleScorer(
            GFN2TwoCycleConfig(
                executable="/mock/xtb",
                parameter_file="/missing",
            )
        )
        result = scorer.score(GraphBuilder.from_smiles("[CH3]"))

        # Parameter availability is checked before molecular applicability.
        self.assertEqual(result.status, "unsupported")

    @unittest.skipUnless(shutil.which("xtb"), "xTB executable is not installed")
    def test_actual_two_cycle_score_is_finite(self) -> None:
        result = GFN2TwoCycleScorer().score(GraphBuilder.from_smiles("CCO"))

        self.assertEqual(result.status, "partial")
        self.assertTrue(math.isfinite(result.score))
        self.assertTrue(result.descriptors["expected_nonconvergence_abort"])
        self.assertAlmostEqual(result.score, sum(result.components.values()))


if __name__ == "__main__":
    unittest.main()
