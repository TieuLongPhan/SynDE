from importlib.util import find_spec
import unittest

HAS_RDKIT = find_spec("rdkit") is not None

import networkx as nx
from synde.energy.syn.rdkit_graph_builder import RDKitGraphBuilder


@unittest.skipUnless(HAS_RDKIT, "RDKit not available")
class TestRDKitGraphBuilder(unittest.TestCase):
    def test_from_smiles_benzene(self) -> None:
        G = RDKitGraphBuilder.from_smiles("c1ccccc1", compute_gasteiger=False)
        self.assertIsInstance(G, nx.Graph)
        self.assertEqual(G.number_of_nodes(), 6)
        self.assertEqual(G.number_of_edges(), 6)
        # attributes present
        any_node = next(iter(G.nodes()))
        self.assertIn("element", G.nodes[any_node])
        self.assertIn("aromatic", G.nodes[any_node])
        self.assertIn("hcount", G.nodes[any_node])
        self.assertIn("partial_charge", G.nodes[any_node])

    def test_invalid_smiles_raises(self) -> None:
        with self.assertRaises(ValueError):
            RDKitGraphBuilder.from_smiles("not_a_smiles")


if __name__ == "__main__":
    unittest.main()
