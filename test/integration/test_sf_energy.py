import unittest

from synde.integration import SFEnergy


class TestSFEnergy(unittest.TestCase):
    def test_graph_mode_sorts_without_external_backend(self) -> None:
        reactions, scores = SFEnergy(energy_type="GRAPH").sort_reactions(
            ["[CH2:1]=[CH2:2]>>[CH3:1][CH3:2]"], sort=True
        )

        self.assertEqual(len(reactions), 1)
        self.assertEqual(len(scores), 1)

    def test_graph_mode_ranks_failed_inputs_last(self) -> None:
        reactions, scores = SFEnergy(energy_type="GRAPH").sort_reactions(
            ["invalid", "[CH2:1]=[CH2:2]>>[CH3:1][CH3:2]"], sort=True
        )

        self.assertEqual(reactions[-1], "invalid")
        self.assertEqual(scores[-1], float("inf"))

    def test_its_mode_uses_mapped_its_score(self) -> None:
        reactions, scores = SFEnergy(energy_type="ITS").sort_reactions(
            ["[CH2:1]=[CH2:2]>>[CH3:1][CH3:2]"], sort=True
        )

        self.assertEqual(len(reactions), 1)
        self.assertIsInstance(scores[0], float)
