import unittest

from synde.graph import GraphBuilder


class TestGraphBuilder(unittest.TestCase):
    def test_smiles_builds_normalized_graph_with_atom_maps(self) -> None:
        result = GraphBuilder.from_smiles("[CH3:1][OH:2]")

        self.assertEqual(result.status, "success")
        self.assertEqual(result.graph.number_of_nodes(), 2)
        self.assertEqual(
            {attrs["atom_map"] for _, attrs in result.graph.nodes(data=True)},
            {1, 2},
        )

    def test_invalid_smiles_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GraphBuilder.from_smiles("not a smiles")

    def test_synkit_full_label_graph_attributes_are_preserved(self) -> None:
        result = GraphBuilder.from_smiles("CC(=O)N")
        nitrogen = next(
            attrs
            for _, attrs in result.graph.nodes(data=True)
            if attrs["element"] == "N"
        )
        carbonyl = next(
            attrs
            for _, _, attrs in result.graph.edges(data=True)
            if attrs["order"] == 2.0
        )

        self.assertGreaterEqual(nitrogen["estimated_lone_pairs"], 1)
        self.assertGreaterEqual(nitrogen["available_lone_pairs"], 1)
        self.assertGreater(nitrogen["valence_electrons"], 0)
        self.assertIsNotNone(nitrogen["oxidation_state"])
        self.assertEqual(carbonyl["sigma_order"], 1.0)
        self.assertEqual(carbonyl["pi_order"], 1.0)
        self.assertEqual(carbonyl["kekule_order"], 2.0)

    def test_explicit_mapped_hydrogens_are_preserved(self) -> None:
        result = GraphBuilder.from_smiles("[CH2:1]([H:3])[CH2:2][H:4]")

        self.assertEqual(
            {attrs["atom_map"] for _, attrs in result.graph.nodes(data=True)},
            {1, 2, 3, 4},
        )

    def test_reaction_conversion_uses_synkit_state_and_its_graphs(self) -> None:
        reactant, product, its = GraphBuilder.reaction_states_from_smiles(
            "[CH2:1]=[CH2:2]>>[CH3:1][CH3:2]"
        )

        self.assertEqual(reactant.graph.edges[1, 2]["order"], 2.0)
        self.assertEqual(product.graph.edges[1, 2]["order"], 1.0)
        self.assertEqual(its.edges[1, 2]["order"], (2.0, 1.0))
