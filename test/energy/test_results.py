import unittest

from synde.energy.results import MoleculeScoreResult


class TestResults(unittest.TestCase):
    def test_result_serializes_to_dictionary(self) -> None:
        result = MoleculeScoreResult("success", 1.0, "score", {}, {}, (), {})

        self.assertEqual(result.to_dict()["score"], 1.0)
