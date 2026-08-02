from __future__ import annotations

import unittest

from synde.energy import GraphEnergy
from synde.graph import GraphBuilder, ITSGraphBuilder


class TestITSGraphAndScoring(unittest.TestCase):
    def test_builder_marks_mapped_bond_order_change(self) -> None:
        reactant = GraphBuilder.from_smiles("[CH2:1]=[CH2:2]")
        product = GraphBuilder.from_smiles("[CH3:1][CH3:2]")

        its = ITSGraphBuilder().build([reactant], [product])

        self.assertEqual(its.reacting_atom_maps, (1, 2))
        self.assertEqual(its.bond_changes[0]["kind"], "order_changed")
        self.assertEqual(its.graph.edges[1, 2]["reactant_order"], 2.0)
        self.assertEqual(its.graph.edges[1, 2]["product_order"], 1.0)

    def test_score_its_keeps_state_delta_and_edit_terms_separate(self) -> None:
        result = GraphEnergy().score_its("[CH2:1]=[CH2:2]>>[CH3:1][CH3:2]")

        self.assertEqual(result.status, "success")
        self.assertEqual(result.units, "score")
        self.assertEqual(result.reacting_atom_maps, (1, 2))
        self.assertIn("state_delta", result.components)
        self.assertIn("order_change_penalty", result.components)
        self.assertEqual(result.state_delta_score, result.components["state_delta"])
        self.assertIsNotNone(result.its_graph)

    def test_score_its_requires_complete_atom_mapping(self) -> None:
        result = GraphEnergy().score_its("C=C>>CC")

        self.assertEqual(result.status, "unsupported")
        self.assertIn("REACTION_MAPPING_MISSING", result.warnings)
        self.assertIsNone(result.its_score)

    def test_score_its_accepts_explicit_mapped_hydrogenation(self) -> None:
        result = GraphEnergy().score_its(
            "[CH2:1]=[CH2:2].[H:3][H:4]>>[CH2:1]([H:3])[CH2:2][H:4]"
        )

        self.assertIsNotNone(result.its_score)
        self.assertIn("formed_bond_penalty", result.components)
        self.assertEqual(
            {change["kind"] for change in result.bond_changes},
            {"formed", "broken", "order_changed"},
        )


if __name__ == "__main__":
    unittest.main()
