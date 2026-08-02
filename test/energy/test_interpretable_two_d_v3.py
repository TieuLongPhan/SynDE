from __future__ import annotations

from collections import Counter
import math

import networkx as nx
import numpy as np
import pytest

from synde.energy import (
    CHARGE_TOPOLOGY_FAMILY,
    CYCLE_JUNCTION_FAMILY,
    GRAPH_STERIC_FAMILY,
    HUCKEL_DENSITY_FAMILY,
    QUANTUM_GRAPH_FEATURE_SCHEMA_V3,
    RESONANCE_TOPOLOGY_FAMILY,
    V3_FEATURE_DEFINITIONS,
    V3_FEATURE_FAMILIES,
    V3_FEATURE_NAMES,
    extract_quantum_graph_v3_features,
    v3_feature_family,
)
from synde.graph import GraphBuilder, GraphNormalizer


def test_v3_manifest_is_fixed_unique_and_fully_grouped() -> None:
    assert QUANTUM_GRAPH_FEATURE_SCHEMA_V3.endswith("fixed-52-v1")
    assert len(V3_FEATURE_NAMES) == 52
    assert len(set(V3_FEATURE_NAMES)) == 52
    assert tuple(definition.name for definition in V3_FEATURE_DEFINITIONS) == (
        V3_FEATURE_NAMES
    )
    assert set(V3_FEATURE_FAMILIES) == set(V3_FEATURE_NAMES)
    assert Counter(V3_FEATURE_FAMILIES.values()) == {
        HUCKEL_DENSITY_FAMILY: 16,
        CHARGE_TOPOLOGY_FAMILY: 14,
        RESONANCE_TOPOLOGY_FAMILY: 8,
        CYCLE_JUNCTION_FAMILY: 8,
        GRAPH_STERIC_FAMILY: 6,
    }


def test_v3_features_are_deterministic_complete_and_finite() -> None:
    molecule = GraphBuilder.from_smiles("CC(=O)Nc1ccccc1O")

    first = extract_quantum_graph_v3_features(molecule)
    second = extract_quantum_graph_v3_features(molecule)

    assert tuple(first) == V3_FEATURE_NAMES
    assert first == second
    assert all(math.isfinite(value) for value in first.values())
    assert sum(value != 0 for value in first.values()) > 30


def test_v3_features_are_invariant_to_atom_relabeling() -> None:
    molecule = GraphBuilder.from_smiles("CC1=CC=CC=C1C(=O)N")
    mapping = {
        node: f"atom-{index * 17 + 3}"
        for index, node in enumerate(reversed(list(molecule.graph.nodes)))
    }
    relabeled_graph = nx.relabel_nodes(molecule.graph, mapping, copy=True)
    relabeled = GraphNormalizer().normalize(
        relabeled_graph,
        canonical_smiles=molecule.canonical_smiles,
        source="relabel-invariance-test",
    )

    original_features = extract_quantum_graph_v3_features(molecule)
    relabeled_features = extract_quantum_graph_v3_features(relabeled)

    assert original_features.keys() == relabeled_features.keys()
    np.testing.assert_allclose(
        list(original_features.values()),
        list(relabeled_features.values()),
        rtol=1e-12,
        atol=1e-12,
    )


def test_v3_benzene_density_terms_respect_degenerate_symmetry() -> None:
    features = extract_quantum_graph_v3_features(
        GraphBuilder.from_smiles("c1ccccc1")
    )

    assert features["v3_huckel_component_count"] == 1
    assert features["v3_huckel_component_size_max"] == 6
    assert features["v3_huckel_density_electron_trace"] == pytest.approx(6.0)
    assert features["v3_huckel_bond_order_alternation_mean"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert features["v3_huckel_homo_projector_ipr_mean"] == pytest.approx(1 / 6)
    assert features["v3_huckel_lumo_projector_ipr_mean"] == pytest.approx(1 / 6)


def test_v3_disconnected_and_empty_pi_cases_are_zero_safe() -> None:
    disconnected = extract_quantum_graph_v3_features(
        GraphBuilder.from_smiles("C=C.C=C")
    )
    saturated = extract_quantum_graph_v3_features(GraphBuilder.from_smiles("CC"))

    assert disconnected["v3_huckel_component_count"] == 2
    assert disconnected["v3_huckel_density_electron_trace"] == pytest.approx(4.0)
    assert saturated["v3_huckel_component_count"] == 0
    assert saturated["v3_huckel_bond_order_abs_sum"] == 0
    assert all(math.isfinite(value) for value in disconnected.values())
    assert all(math.isfinite(value) for value in saturated.values())


def test_v3_ring_and_graph_steric_terms_have_expected_directions() -> None:
    benzene = extract_quantum_graph_v3_features(
        GraphBuilder.from_smiles("c1ccccc1")
    )
    ortho_xylene = extract_quantum_graph_v3_features(
        GraphBuilder.from_smiles("Cc1ccccc1C")
    )
    meta_xylene = extract_quantum_graph_v3_features(
        GraphBuilder.from_smiles("Cc1cccc(C)c1")
    )
    bicyclic = extract_quantum_graph_v3_features(
        GraphBuilder.from_smiles("C1CC2CCC1C2")
    )

    assert benzene["v3_cycle_cycle_rank"] == 1
    assert bicyclic["v3_cycle_cycle_rank"] == 2
    assert bicyclic["v3_cycle_junction_atom_count"] > 0
    assert bicyclic["v3_cycle_bridgehead_atom_count"] == 2
    assert ortho_xylene["v3_steric_aromatic_ortho_substituent_pairs"] == 1
    assert meta_xylene["v3_steric_aromatic_meta_substituent_pairs"] == 1


def test_v3_family_lookup_rejects_unregistered_terms() -> None:
    assert (
        v3_feature_family("v3_huckel_density_electron_trace")
        == HUCKEL_DENSITY_FAMILY
    )
    with pytest.raises(ValueError, match="Unknown v3 feature"):
        v3_feature_family("v3_unregistered_target_informed_term")
