from __future__ import annotations
import unittest
from synde import GraphEnergy
from synde.integration import SFEnergy


class TestGraphEnergy(unittest.TestCase):
    def setUp(self) -> None:
        self.energy = GraphEnergy()

    def test_molecule_score_is_typed_and_uses_score_units(self) -> None:
        result = self.energy.score_molecule_from_smiles("c1ccccc1")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.units, "score")
        self.assertIn("pi_stabilization", result.components)
        self.assertIn("graph_identity", result.descriptors)

    def test_equivalent_smiles_reuse_cached_result(self) -> None:
        first = self.energy.score_molecule_from_smiles("C(C)O")
        second = self.energy.score_molecule_from_smiles("CCO")
        self.assertIs(first, second)

    def test_enhanced_theory_score_remains_explicitly_uncalibrated(self) -> None:
        result = self.energy.score_molecule_theory_from_smiles("CC(=O)N")

        self.assertFalse(result.provenance["calibrated"])
        self.assertFalse(result.provenance["fitted_coefficients"])
        self.assertIn("lone_pair_resonance", result.components)

    def test_experimental_orbital_theory_supports_nitrile(self) -> None:
        result = self.energy.score_molecule_orbital_theory_from_smiles("C#N")

        self.assertEqual(result.status, "success")
        self.assertTrue(result.provenance["experimental"])
        self.assertEqual(result.descriptors["n_orbital_pi_electrons"], 4)

    def test_graph_theory_candidate_excludes_geometry_dependent_terms(self) -> None:
        result = self.energy.score_molecule_graph_theory_from_smiles("C1CCCCC1")

        self.assertTrue(result.provenance["experimental"])
        self.assertEqual(result.components["ring_strain"], 0.0)
        self.assertEqual(result.components["steric_congestion"], 0.0)

    def test_valence_scorer_exposes_partitioned_pi_terms(self) -> None:
        result = self.energy.score_molecule_valence_from_smiles("CC(=O)N")

        self.assertIn("lewis_local_pi_reference", result.components)
        self.assertIn("aromatic_pi_delocalization", result.components)
        self.assertIn("carbonyl_n_lone_pair_delocalization", result.components)

    def test_first_order_two_d_scorer_is_frozen_graph_only_and_cached(self) -> None:
        first = self.energy.score_molecule_first_order_two_d_from_smiles("C(C)O")
        second = self.energy.score_molecule_first_order_two_d_from_smiles("CCO")

        self.assertIs(first, second)
        self.assertEqual(first.units, "kJ/mol_score")
        self.assertFalse(first.provenance["uses_coordinates"])
        self.assertFalse(first.provenance["uses_conformers"])
        self.assertFalse(first.provenance["uses_xtb"])
        self.assertTrue(first.provenance["permanent_holdout_evaluated"])

    def test_xtb_proxy_is_uncalibrated_and_cached(self) -> None:
        first = self.energy.score_molecule_xtb_proxy_from_smiles("C(C)O")
        second = self.energy.score_molecule_xtb_proxy_from_smiles("CCO")

        self.assertIs(first, second)
        self.assertEqual(first.units, "eV_proxy")
        self.assertFalse(first.provenance["calibrated"])

    def test_gfn2_singlepoint_reuses_cached_result(self) -> None:
        first = self.energy.score_molecule_gfn2_from_smiles("C(C)O")
        second = self.energy.score_molecule_gfn2_from_smiles("CCO")

        self.assertIs(first, second)
        self.assertEqual(first.units, "eV")
        self.assertFalse(first.provenance["calibrated"])

    def test_gfn2_two_cycle_reuses_cached_result(self) -> None:
        first = self.energy.score_molecule_gfn2_two_cycle_from_smiles("C(C)O")
        second = self.energy.score_molecule_gfn2_two_cycle_from_smiles("CCO")

        self.assertIs(first, second)
        self.assertEqual(first.units, "eV")
        self.assertIn(first.status, {"partial", "unsupported"})
        if first.status == "unsupported":
            self.assertIn("XTB_EXECUTABLE_NOT_FOUND", first.warnings)
        else:
            self.assertFalse(first.provenance["calibrated"])
            self.assertFalse(first.provenance["actual_converged_gfn2"])

    def test_mapped_reaction_extracts_bond_changes(self) -> None:
        result = self.energy.score_reaction("[CH2:1]=[CH2:2]>>[CH3:1][CH3:2]")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.units, "score")
        self.assertEqual(len(result.bond_changes), 1)
        self.assertEqual(result.bond_changes[0]["kind"], "order_changed")

    def test_missing_atom_mapping_is_explicit(self) -> None:
        result = self.energy.score_reaction("C=C>>CC")
        self.assertEqual(result.status, "partial")
        self.assertIn("REACTION_MAPPING_MISSING", result.warnings)

    def test_reordering_and_unchanged_component_do_not_change_delta(self) -> None:
        first = self.energy.score_reaction(
            "[CH2:1]=[CH2:2].[OH2:3]>>[CH3:1][CH3:2].[OH2:3]"
        )
        second = self.energy.score_reaction(
            "[OH2:3].[CH2:1]=[CH2:2]>>[OH2:3].[CH3:1][CH3:2]"
        )

        self.assertEqual(first.status, "success")
        self.assertEqual(second.status, "success")
        self.assertAlmostEqual(first.reaction_delta_score, second.reaction_delta_score)
        self.assertEqual(first.bond_changes, second.bond_changes)

    def test_unchanged_component_cancels_out_perfectly(self) -> None:
        # A + C >> B + C
        with_spectator = self.energy.score_reaction(
            "[CH2:1]=[CH2:2].[OH2:3]>>[CH3:1][CH3:2].[OH2:3]"
        )
        # A >> B
        without_spectator = self.energy.score_reaction(
            "[CH2:1]=[CH2:2]>>[CH3:1][CH3:2]"
        )
        self.assertAlmostEqual(
            with_spectator.reaction_delta_score, without_spectator.reaction_delta_score
        )

    def test_sf_energy_sorting_using_graph_mode(self) -> None:
        sf = SFEnergy(energy_type="GRAPH")
        reactions = [
            "[CH2:1]=[CH2:2].[OH2:3]>>[CH3:1][CH3:2].[OH2:3]",
            "[CH2:1]=[CH2:2]>>[CH3:1][CH3:2]",
        ]
        sorted_rsmi, energies = sf.sort_reactions(reactions, sort=True)
        self.assertEqual(len(sorted_rsmi), 2)
        self.assertEqual(len(energies), 2)
        self.assertAlmostEqual(energies[0], energies[1])


if __name__ == "__main__":
    unittest.main()
