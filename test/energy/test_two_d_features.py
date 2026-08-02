import unittest

from synde.energy import (
    MoleculeScorer,
    extract_heuristic_features,
    extract_two_d_features,
)
from synde.graph import GraphBuilder


class TestTwoDFeatures(unittest.TestCase):
    def test_features_are_deterministic_and_geometry_free(self) -> None:
        graph = GraphBuilder.from_smiles("CC(=O)N")
        result = MoleculeScorer().score(graph)

        first = extract_two_d_features(graph, result)
        second = extract_two_d_features(graph, result)

        self.assertEqual(first, second)
        self.assertIn("heuristic_sigma_bond_energy", first)
        self.assertTrue(any(name.startswith("atom_env_") for name in first))
        self.assertTrue(any(name.startswith("atom_env_r2_") for name in first))
        self.assertTrue(any(name.startswith("bond_env_") for name in first))
        self.assertTrue(any(name.startswith("atom_pair_") for name in first))

    def test_constitutional_isomers_have_different_environment_features(self) -> None:
        scorer = MoleculeScorer()
        straight = GraphBuilder.from_smiles("CCCC")
        branched = GraphBuilder.from_smiles("CC(C)C")

        straight_features = extract_two_d_features(straight, scorer.score(straight))
        branched_features = extract_two_d_features(branched, scorer.score(branched))

        self.assertNotEqual(straight_features, branched_features)

    def test_heuristic_features_contain_only_reweightable_components(self) -> None:
        result = MoleculeScorer().score(GraphBuilder.from_smiles("CCO"))
        features = extract_heuristic_features(result)

        self.assertEqual(len(features), len(result.components))
        self.assertTrue(all(name.startswith("heuristic_") for name in features))
        self.assertFalse(any(name.startswith("atom_env_") for name in features))

    def test_distant_positional_environments_are_represented(self) -> None:
        scorer = MoleculeScorer()
        ortho = GraphBuilder.from_smiles("Cc1ccccc1C")
        para = GraphBuilder.from_smiles("Cc1ccc(C)cc1")
        ortho_features = extract_two_d_features(ortho, scorer.score(ortho))
        para_features = extract_two_d_features(para, scorer.score(para))

        ortho_pairs = {
            key: value
            for key, value in ortho_features.items()
            if key.startswith("atom_pair_")
        }
        para_pairs = {
            key: value
            for key, value in para_features.items()
            if key.startswith("atom_pair_")
        }
        self.assertNotEqual(ortho_pairs, para_pairs)
