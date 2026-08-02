import unittest

from synde.energy import MoleculeScorer
from synde.graph import GraphBuilder


class TestMoleculeScorer(unittest.TestCase):
    def test_score_reports_graph_components(self) -> None:
        result = MoleculeScorer().score(GraphBuilder.from_smiles("c1ccccc1"))

        self.assertEqual(result.units, "score")
        self.assertIn("pi_stabilization", result.components)
