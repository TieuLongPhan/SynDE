import unittest

from synde.energy import ReactionScorer
from synde.graph import GraphBuilder


class TestReactionScorer(unittest.TestCase):
    def test_mapped_order_change_is_reported(self) -> None:
        result = ReactionScorer().score(
            [GraphBuilder.from_smiles("[CH2:1]=[CH2:2]")],
            [GraphBuilder.from_smiles("[CH3:1][CH3:2]")],
            "[CH2:1]=[CH2:2]>>[CH3:1][CH3:2]",
        )

        self.assertEqual(result.bond_changes[0]["kind"], "order_changed")
