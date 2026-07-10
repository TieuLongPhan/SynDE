"""Regression evidence for representative legacy SYN v1 behavior.

This test intentionally records legacy limitations that v2 will address.  It must
not be reused as a scientific validation of the legacy model.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import unittest

from synde.energy.syn import SynEnergy

FIXTURE_PATH = Path(__file__).with_name("fixtures") / "syn_v1_baseline.json"


class TestSynV1Baseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        cls.energy = SynEnergy()

    def test_energy_cases_match_captured_legacy_behavior(self) -> None:
        for case in self.fixture["energy_cases"].values():
            with self.subTest(smiles=case["smiles"]):
                result = self.energy.energy_from_smiles(
                    case["smiles"], compute_gasteiger=False
                )
                self.assertEqual(result["status"], case["status"])
                self.assertEqual(result["n_pi"], case["n_pi"])
                self.assertAlmostEqual(
                    result["energy_beta"], case["energy_beta"], places=12
                )
                self.assertEqual(len(result["E"]), len(case["orbital_energies"]))
                for actual, expected in zip(result["E"], case["orbital_energies"]):
                    self.assertAlmostEqual(actual, expected, places=12)

    def test_pair_score_histograms_match_captured_legacy_behavior(self) -> None:
        for case in self.fixture["pair_cases"].values():
            with self.subTest(smiles=case["smiles"]):
                rows = self.energy.rank_pairs_from_smiles(case["smiles"], top_k=100)
                histogram = Counter(f"{row['DE']:.12g}" for row in rows)
                self.assertEqual(len(rows), case["pair_count"])
                self.assertEqual(dict(histogram), case["score_histogram"])


if __name__ == "__main__":
    unittest.main()
