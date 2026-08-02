import unittest

from synde.graph import GraphBuilder, ITSGraphBuilder


class TestITSGraphBuilder(unittest.TestCase):
    def test_order_change_is_retained_on_its_edge(self) -> None:
        reactant = GraphBuilder.from_smiles("[CH2:1]=[CH2:2]")
        product = GraphBuilder.from_smiles("[CH3:1][CH3:2]")

        its = ITSGraphBuilder().build([reactant], [product])

        self.assertEqual(its.graph.edges[1, 2]["edit_type"], "order_changed")
        self.assertEqual(its.reacting_atom_maps, (1, 2))
