import unittest
import networkx as nx
from synde import GeometryScorer


class TestGeometryScoring(unittest.TestCase):
    def test_geometry_returns_named_force_field_value(self):
        result = GeometryScorer().score_smiles("CCO")
        self.assertIn(result.status, {"success", "partial"})
        self.assertIsNotNone(result.force_field_energy_kcal_mol)
        self.assertEqual(result.provenance["force_field"], "MMFF94")

    def test_distance_terms_respond_to_controlled_separation(self):
        graph = nx.Graph()
        graph.add_node(0, partial_charge=1.0)
        graph.add_node(1, partial_charge=-1.0)
        near = GeometryScorer.nonbonded_terms_for_positions(
            graph, {0: (0, 0, 0), 1: (1, 0, 0)}
        )
        far = GeometryScorer.nonbonded_terms_for_positions(
            graph, {0: (0, 0, 0), 1: (4, 0, 0)}
        )
        self.assertGreater(abs(near["charge_distance"]), abs(far["charge_distance"]))
        self.assertGreater(near["repulsion"], far["repulsion"])
