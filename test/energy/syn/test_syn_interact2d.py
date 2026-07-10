import unittest
import networkx as nx
from synde.energy.syn.params import SynParams
from synde.energy.syn.syn_interact2d import SynInteract2D
from synde.energy.syn.huckel2d import Huckel2D


def make_two_ethenes_with_charges() -> nx.Graph:
    # Two disconnected C=C fragments; add partial charges for Coul_pc path
    G = nx.Graph()
    # fragment A
    G.add_node(0, element="C", aromatic=False, hcount=2, partial_charge=-0.1)
    G.add_node(1, element="C", aromatic=False, hcount=2, partial_charge=+0.1)
    G.add_edge(0, 1, order=2.0)
    # fragment B
    G.add_node(2, element="C", aromatic=False, hcount=2, partial_charge=+0.05)
    G.add_node(3, element="C", aromatic=False, hcount=2, partial_charge=-0.05)
    G.add_edge(2, 3, order=2.0)
    return G


class TestSynInteract2D(unittest.TestCase):
    def setUp(self) -> None:
        self.params = SynParams()
        self.si = SynInteract2D(self.params)
        self.hm = Huckel2D(self.params)

    def test_rank_pairs_basic(self) -> None:
        G = make_two_ethenes_with_charges()
        rows = self.si.rank_pairs_from_graph(G, top_k=10, export="basic")
        # 2 x 2 → 4 pairs
        self.assertEqual(len(rows), 4)
        # Must contain required fields
        for r in rows:
            self.assertIn("DE", r)
            self.assertIn("S_front", r)
            self.assertIn("Coul_pi", r)
            self.assertIn("Coul_pc", r)
            self.assertIn("Steric", r)

    def test_rank_pairs_export_all(self) -> None:
        G = make_two_ethenes_with_charges()
        rows = self.si.rank_pairs_from_graph(G, top_k=10, export="all")
        self.assertEqual(len(rows), 4)
        self.assertTrue(all("alphaA" in r and "alphaB" in r for r in rows))
        self.assertTrue(all("E_A" in r and "E_B" in r for r in rows))

    def test_no_pi_component_returns_empty(self) -> None:
        # Single methane-like node only; no pi edges
        G = nx.Graph()
        G.add_node(0, element="C", aromatic=False, hcount=3)
        rows = self.si.rank_pairs_from_graph(G, top_k=10, export="basic")
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
