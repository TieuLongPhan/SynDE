import unittest
from synde import GraphXTBCascade


class TestGraphXTBCascade(unittest.TestCase):
    def test_shortlist_and_cache_use_injected_evaluator(self):
        reactions = [
            "[CH2:1]=[CH2:2]>>[CH3:1][CH3:2]",
            "[CH3:1][CH3:2]>>[CH2:1]=[CH2:2]",
        ]
        calls = []

        def evaluator(reaction, level):
            calls.append((reaction, level))
            return -1.0

        cascade = GraphXTBCascade()
        first = cascade.screen(reactions, top_k=1, xtb_evaluator=evaluator)
        second = cascade.screen(reactions, top_k=1, xtb_evaluator=evaluator)
        self.assertEqual(first.shortlist_size, 1)
        self.assertEqual(sum(row.selected for row in first.rows), 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(second.rows[0].xtb_delta_e, first.rows[0].xtb_delta_e)
