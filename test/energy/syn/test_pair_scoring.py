from __future__ import annotations

import unittest

from synde.energy.syn import (
    ComponentFrontier,
    GraphPairScorer,
    RDKitGraphBuilder,
    directional_fmo,
)


class TestGraphPairScorer(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = GraphPairScorer()

    def test_symmetric_glyoxals_receive_one_graph_only_score(self) -> None:
        # Glyoxal (O=CC=O) contains 2 equivalent reactable oxygen atoms (valence 2, max valence 3)
        graph = RDKitGraphBuilder.from_smiles_v2("O=CC=O.O=CC=O")
        rows = self.scorer.rank(graph, top_k=100)

        # 2 oxygens on left * 2 oxygens on right = 4 pairs
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            {round(row.pair_compatibility_score, 12) for row in rows},
            {round(rows[0].pair_compatibility_score, 12)},
        )
        self.assertTrue(all("fmo_A_to_B" in row.components for row in rows))
        self.assertTrue(all("fmo_B_to_A" in row.components for row in rows))

    def test_rank_grouped_equivalence(self) -> None:
        graph = RDKitGraphBuilder.from_smiles_v2("O=CC=O.O=CC=O")
        rows = self.scorer.rank(graph, top_k=100, group_equivalence=True)

        # Grouped equivalence should collapse the 4 equivalent pairs into 1 row
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0].pairs), 4)

    def test_valence_impossible_proposed_bonds_are_filtered(self) -> None:
        # Benzene carbons are fully saturated (valence 4, max valence 4)
        graph = RDKitGraphBuilder.from_smiles_v2("c1ccccc1.c1ccccc1")
        rows = self.scorer.rank(graph, top_k=100)
        self.assertEqual(len(rows), 0)

        # Ethene carbons are fully saturated (valence 4, max valence 4)
        graph2 = RDKitGraphBuilder.from_smiles_v2("C=C.C=C")
        rows2 = self.scorer.rank(graph2, top_k=100)
        self.assertEqual(len(rows2), 0)

    def test_directional_fmo_rejects_inverted_gap(self) -> None:
        donor = ComponentFrontier(0, (0,), 1.0, 2.0, {0: 1.0}, {0: 1.0})
        acceptor = ComponentFrontier(1, (1,), 0.0, 0.5, {1: 1.0}, {1: 1.0})

        result = directional_fmo(donor, acceptor, 0, 1)

        self.assertFalse(result.valid)
        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.warning, "INVALID_DIRECTIONAL_GAP")

    def test_directional_fmo_regularizes_small_positive_gap(self) -> None:
        donor = ComponentFrontier(0, (0,), 0.0, 1.0, {0: 1.0}, {0: 1.0})
        acceptor = ComponentFrontier(1, (1,), 0.0, 0.01, {1: 1.0}, {1: 1.0})

        result = directional_fmo(donor, acceptor, 0, 1, gap_floor=0.05)

        self.assertTrue(result.valid)
        self.assertEqual(result.regularized_gap, 0.05)
        self.assertEqual(result.warning, "FRONTIER_GAP_REGULARIZED")


if __name__ == "__main__":
    unittest.main()
