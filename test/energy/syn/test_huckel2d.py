import unittest
import networkx as nx
import numpy as np

from synde.energy.syn.params import SynParams
from synde.energy.syn.huckel2d import Huckel2D


def make_ethene_graph() -> nx.Graph:
    G = nx.Graph()
    # Two carbons double-bonded
    G.add_node(0, element="C", aromatic=False, hcount=2, partial_charge=0.0)
    G.add_node(1, element="C", aromatic=False, hcount=2, partial_charge=0.0)
    G.add_edge(0, 1, order=2.0)
    return G


def make_benzene_graph() -> nx.Graph:
    G = nx.Graph()
    for i in range(6):
        G.add_node(i, element="C", aromatic=True, hcount=1, partial_charge=0.0)
    for i in range(6):
        G.add_edge(i, (i + 1) % 6, order=1.0)  # aromatic flags drive pi selection
    return G


class TestHuckel2D(unittest.TestCase):
    def setUp(self) -> None:
        self.params = SynParams()
        self.h = Huckel2D(self.params)

    def test_pi_layer_ethene(self) -> None:
        G = make_ethene_graph()
        PG = self.h.pi_layer(G)
        self.assertEqual(PG.number_of_nodes(), 2)
        self.assertEqual(PG.number_of_edges(), 1)

    def test_initial_effects_halogen_and_plusM(self) -> None:
        # C=C with one side attached to O (single -> +M) and a halogen (−I)
        G = make_ethene_graph()
        # attach an O to node 0 (single bond)
        G.add_node(2, element="O", aromatic=False, hcount=1)
        G.add_edge(0, 2, order=1.0)
        # attach Cl to node 1
        G.add_node(3, element="Cl", aromatic=False, hcount=0)
        G.add_edge(1, 3, order=1.0)

        PG = self.h.pi_layer(G)
        initials = self.h.initial_effects(G, PG)
        kinds = [k for _, k, _ in initials]
        self.assertIn("+M", kinds)
        self.assertIn("-I", kinds)

    def test_propagate_distance_attenuation(self) -> None:
        # line of 3 pi atoms to test attenuation over distance
        G = nx.Graph()
        for i in range(3):
            G.add_node(i, element="C", aromatic=False)
        G.add_edge(0, 1, order=2.0)
        G.add_edge(1, 2, order=2.0)
        PG = self.h.pi_layer(G)

        # place a +M effect at atom 0
        initials = [(0, "+M", 0.3)]
        alpha = self.h.propagate(PG, initials)
        # alpha should diminish as distance increases (att_M < 1)
        self.assertGreater(alpha[0], alpha[1])
        self.assertGreater(alpha[1], alpha[2])

    def test_build_and_solve_ethene(self) -> None:
        G = make_ethene_graph()
        PG = self.h.pi_layer(G)
        alpha = self.h.propagate(PG, [])
        H, nodes = self.h.build_huckel(PG, alpha)
        E, C = self.h.solve(H)
        # For 2-site H with beta=-1, alpha=0 → eigenvalues [-1, 1]
        self.assertAlmostEqual(E[0], -1.0, places=6)
        self.assertAlmostEqual(E[1], 1.0, places=6)
        # eigenvectors are orthonormal
        self.assertAlmostEqual(float(np.dot(C[:, 0], C[:, 1])), 0.0, places=6)

    def test_descriptors_and_pi_mulliken(self) -> None:
        G = make_benzene_graph()
        PG = self.h.pi_layer(G)
        alpha = self.h.propagate(PG, [])
        H, nodes = self.h.build_huckel(PG, alpha)
        E, C = self.h.solve(H)
        d = self.h.descriptors(E, C)
        self.assertIn("h", d)
        self.assertIn("l", d)
        q = self.h.pi_mulliken(E, C)
        self.assertEqual(len(q), PG.number_of_nodes())
        # sanity: not NaN/Inf
        self.assertTrue(np.all(np.isfinite(q)))

    def test_total_pi_energy_benzene(self) -> None:
        # Classic Hückel toy energy (beta=-1): -8 for benzene
        G = make_benzene_graph()
        PG = self.h.pi_layer(G)
        alpha = self.h.propagate(PG, [])
        H, _ = self.h.build_huckel(PG, alpha)
        E, _ = self.h.solve(H)
        Epi = self.h.total_pi_energy(E)
        self.assertAlmostEqual(Epi, -8.0, places=6)


if __name__ == "__main__":
    unittest.main()
