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
