from __future__ import annotations

import pytest

from synde.energy import (
    InterpretableTwoDV2Scorer,
    extract_named_empirical_two_d_features,
)
from synde.graph import GraphBuilder


def test_named_v2_features_are_deterministic_and_unhashed() -> None:
    graph = GraphBuilder.from_smiles("CC(=O)N")
    score = InterpretableTwoDV2Scorer().score(graph)

    first = extract_named_empirical_two_d_features(graph, score)
    second = extract_named_empirical_two_d_features(graph, score)

    assert first == second
    assert any(name.startswith("benson_nn[") for name in first)
    assert any(name.startswith("bond_environment[") for name in first)
    assert any(name.startswith("path_1_3[") for name in first)
    assert "huckel_pi_stabilization" in first
    assert not any("hash" in name or "_bin_" in name for name in first)


def test_uncalibrated_v2_uses_only_named_kj_mol_additions() -> None:
    graph = GraphBuilder.from_smiles("C1CCCC1")
    result = InterpretableTwoDV2Scorer().score(graph)

    assert result.score == pytest.approx(sum(result.components.values()))
    assert result.components["conventional_medium_ring_strain"] == 26.0
    assert result.provenance["calibrated"] is False
    assert result.provenance["fitted_coefficients"] is False
    assert result.provenance["uses_coordinates"] is False
    assert result.provenance["uses_conformers"] is False
    assert result.provenance["uses_xtb"] is False


def test_constitutional_isomers_have_different_named_feature_ledgers() -> None:
    straight = GraphBuilder.from_smiles("CCCCC")
    branched = GraphBuilder.from_smiles("CC(C)CC")

    straight_features = extract_named_empirical_two_d_features(straight)
    branched_features = extract_named_empirical_two_d_features(branched)

    assert straight_features != branched_features
    assert straight_features["graph_zagreb_m1"] != branched_features["graph_zagreb_m1"]
