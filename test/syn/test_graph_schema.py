from __future__ import annotations

import math
import unittest

import networkx as nx

from synde import GraphBuilder, GraphValidationError, normalize_graph


class TestGraphSchema(unittest.TestCase):
    def test_v2_smiles_builder_preserves_chemical_attributes(self) -> None:
        normalized = GraphBuilder.from_smiles("[CH3:7][OH:8].[Cl-:9]")
        graph = normalized.graph

        self.assertEqual(normalized.status, "success")
        self.assertEqual(graph.number_of_nodes(), 3)
        self.assertEqual(graph.number_of_edges(), 1)
        self.assertEqual(
            {data["atom_map"] for _, data in graph.nodes(data=True)}, {7, 8, 9}
        )
        self.assertEqual(
            {data["component_id"] for _, data in graph.nodes(data=True)}, {0, 1}
        )
        for _, data in graph.nodes(data=True):
            for field in (
                "element",
                "atomic_number",
                "formal_charge",
                "aromatic",
                "hybridization",
                "total_hcount",
                "radical_electrons",
                "component_id",
            ):
                self.assertIn(field, data)

    def test_node_relabeling_keeps_identity_and_does_not_mutate_input(self) -> None:
        graph = nx.Graph()
        graph.add_node("a", element="C", aromatic=False, hcount=2)
        graph.add_node("b", element="C", aromatic=False, hcount=2)
        graph.add_edge("a", "b", order=2.0)
        relabeled = nx.relabel_nodes(graph, {"a": 100, "b": 200}, copy=True)

        first = normalize_graph(graph)
        second = normalize_graph(relabeled)

        self.assertEqual(first.identity, second.identity)
        self.assertNotIn("atomic_number", graph.nodes["a"])
        self.assertNotIn("component_id", graph.nodes["a"])

    def test_nonfinite_partial_charge_is_cleared_with_warning(self) -> None:
        graph = nx.Graph()
        graph.add_node(0, element="C", partial_charge=math.nan)
        normalized = normalize_graph(graph)

        self.assertEqual(normalized.status, "partial")
        self.assertIn("NONFINITE_PARTIAL_CHARGE", normalized.warning_codes())
        self.assertIsNone(normalized.graph.nodes[0]["partial_charge"])

    def test_synkit_smiles_adapter_produces_finite_or_missing_partial_charges(
        self,
    ) -> None:
        normalized = GraphBuilder.from_smiles("CCO")

        charges = [
            data["partial_charge"] for _, data in normalized.graph.nodes(data=True)
        ]
        self.assertTrue(
            all(charge is None or math.isfinite(charge) for charge in charges)
        )
        self.assertNotIn("NONFINITE_PARTIAL_CHARGE", normalized.warning_codes())

    def test_unsupported_element_is_structured_partial_result(self) -> None:
        normalized = GraphBuilder.from_smiles("[Fe]")

        self.assertEqual(normalized.status, "partial")
        self.assertIn("UNSUPPORTED_ELEMENT", normalized.warning_codes())

    def test_strict_graph_rejects_missing_element(self) -> None:
        graph = nx.Graph()
        graph.add_node(0)

        with self.assertRaises(GraphValidationError):
            normalize_graph(graph, strict=True)

    def test_non_strict_graph_records_missing_element(self) -> None:
        graph = nx.Graph()
        graph.add_node(0)
        normalized = normalize_graph(graph, strict=False)

        self.assertEqual(normalized.status, "partial")
        self.assertIn("MISSING_ELEMENT", normalized.warning_codes())
        self.assertEqual(normalized.graph.nodes[0]["element"], "?")


if __name__ == "__main__":
    unittest.main()
