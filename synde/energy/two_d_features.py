"""Deterministic 2D atom-, bond-, and topology-environment features."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import networkx as nx

from synde.graph.graph_schema import NormalizedMolecularGraph

from .results import MoleculeScoreResult

TWO_D_FEATURE_SCHEMA = "synde-2d-features-v2"


@dataclass(frozen=True)
class TwoDFeatureConfig:
    """Fixed dimensions for hashed local environment counts."""

    atom_environment_bins: int = 128
    bond_environment_bins: int = 128
    extended_atom_environment_bins: int = 192
    atom_environment_radius: int = 3
    pair_environment_bins: int = 128
    maximum_pair_distance: int = 4
    ring_environment_bins: int = 64


def extract_two_d_features(
    normalized: NormalizedMolecularGraph,
    result: MoleculeScoreResult,
    config: TwoDFeatureConfig | None = None,
) -> dict[str, float]:
    """Return geometry-free features suitable for an interpretable linear model."""
    config = config or TwoDFeatureConfig()
    graph = normalized.graph
    features = extract_heuristic_features(result)
    _add_global_features(features, graph)
    for node in graph.nodes:
        signature = _atom_signature(graph, node)
        _increment_hashed(features, "atom_env", signature, config.atom_environment_bins)
    _add_extended_atom_environments(features, graph, config)
    for left, right, attrs in graph.edges(data=True):
        signature = _bond_signature(graph, left, right, attrs)
        _increment_hashed(features, "bond_env", signature, config.bond_environment_bins)
    _add_pair_environments(features, graph, config)
    _add_ring_environments(features, graph, config)
    return features


def extract_heuristic_features(result: MoleculeScoreResult) -> dict[str, float]:
    """Return only named heuristic components for pure weight recalibration."""
    return {
        f"heuristic_{key}": float(value) for key, value in result.components.items()
    }


def _increment_hashed(
    features: dict[str, float], prefix: str, signature: str, bins: int
) -> None:
    digest = hashlib.sha256(signature.encode()).digest()
    index = int.from_bytes(digest[:8], "big") % bins
    sign = 1.0 if digest[8] & 1 else -1.0
    name = f"{prefix}_{index:03d}"
    features[name] = features.get(name, 0.0) + sign


def _atom_signature(graph: nx.Graph, node: Any) -> str:
    attrs = graph.nodes[node]
    neighbors = []
    for neighbor in graph.neighbors(node):
        edge = graph.edges[node, neighbor]
        neighbors.append(
            ":".join(
                (
                    graph.nodes[neighbor]["element"],
                    _bond_class(edge),
                    str(int(graph.nodes[neighbor].get("formal_charge", 0))),
                    str(int(bool(graph.nodes[neighbor].get("aromatic", False)))),
                )
            )
        )
    neighbors.sort()
    return "|".join(
        (
            attrs["element"],
            str(int(bool(attrs.get("aromatic", False)))),
            str(attrs.get("hybridization", "UNSPECIFIED")),
            str(int(attrs.get("formal_charge", 0))),
            str(int(attrs.get("total_hcount", 0))),
            str(len(neighbors)),
            ",".join(neighbors),
        )
    )


def _atom_seed_signature(graph: nx.Graph, node: Any) -> str:
    attrs = graph.nodes[node]
    return ":".join(
        (
            attrs["element"],
            str(int(bool(attrs.get("aromatic", False)))),
            str(attrs.get("hybridization", "UNSPECIFIED")),
            str(int(attrs.get("formal_charge", 0))),
            str(int(attrs.get("total_hcount", 0))),
            str(graph.degree(node)),
        )
    )


def _add_extended_atom_environments(
    features: dict[str, float], graph: nx.Graph, config: TwoDFeatureConfig
) -> None:
    labels = {node: _atom_seed_signature(graph, node) for node in graph.nodes}
    for radius in range(1, config.atom_environment_radius + 1):
        next_labels = {}
        for node in graph.nodes:
            neighbors = sorted(
                f"{_bond_class(graph.edges[node, neighbor])}:{labels[neighbor]}"
                for neighbor in graph.neighbors(node)
            )
            payload = f"{labels[node]}|{','.join(neighbors)}"
            next_labels[node] = hashlib.sha256(payload.encode()).hexdigest()
        labels = next_labels
        if radius < 2:
            continue
        for signature in labels.values():
            _increment_hashed(
                features,
                f"atom_env_r{radius}",
                signature,
                config.extended_atom_environment_bins,
            )


def _add_pair_environments(
    features: dict[str, float], graph: nx.Graph, config: TwoDFeatureConfig
) -> None:
    nodes = list(graph.nodes)
    paths = dict(
        nx.all_pairs_shortest_path_length(graph, cutoff=config.maximum_pair_distance)
    )
    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            distance = paths.get(left, {}).get(right)
            if distance is None or distance < 2:
                continue
            endpoints = sorted(
                (_atom_seed_signature(graph, left), _atom_seed_signature(graph, right))
            )
            signature = f"{distance}|{endpoints[0]}|{endpoints[1]}"
            _increment_hashed(
                features, "atom_pair", signature, config.pair_environment_bins
            )


def _add_ring_environments(
    features: dict[str, float], graph: nx.Graph, config: TwoDFeatureConfig
) -> None:
    for cycle in nx.cycle_basis(graph):
        elements = sorted(graph.nodes[node]["element"] for node in cycle)
        aromatic_atoms = sum(
            bool(graph.nodes[node].get("aromatic", False)) for node in cycle
        )
        signature = f"{len(cycle)}|{aromatic_atoms}|{','.join(elements)}"
        _increment_hashed(features, "ring_env", signature, config.ring_environment_bins)


def _bond_signature(
    graph: nx.Graph, left: Any, right: Any, attrs: dict[str, Any]
) -> str:
    endpoints = sorted(
        (
            _endpoint_signature(graph, left),
            _endpoint_signature(graph, right),
        )
    )
    return "|".join(
        (
            endpoints[0],
            endpoints[1],
            _bond_class(attrs),
            str(int(bool(attrs.get("conjugated", False)))),
            str(int(bool(attrs.get("is_in_ring", attrs.get("in_ring", False))))),
        )
    )


def _endpoint_signature(graph: nx.Graph, node: Any) -> str:
    attrs = graph.nodes[node]
    return ":".join(
        (
            attrs["element"],
            str(int(bool(attrs.get("aromatic", False)))),
            str(attrs.get("hybridization", "UNSPECIFIED")),
            str(int(attrs.get("formal_charge", 0))),
            str(graph.degree(node)),
        )
    )


def _bond_class(attrs: dict[str, Any]) -> str:
    if attrs.get("aromatic", False):
        return "aromatic"
    order = float(attrs.get("order", 1.0))
    if order >= 2.5:
        return "triple"
    if order >= 1.5:
        return "double"
    return "single"


def _add_global_features(features: dict[str, float], graph: nx.Graph) -> None:
    features["n_nodes"] = float(graph.number_of_nodes())
    features["n_edges"] = float(graph.number_of_edges())
    features["n_components"] = float(nx.number_connected_components(graph))
    features["formal_charge_total"] = float(
        sum(int(attrs.get("formal_charge", 0)) for _, attrs in graph.nodes(data=True))
    )
    features["formal_charge_absolute"] = float(
        sum(
            abs(int(attrs.get("formal_charge", 0)))
            for _, attrs in graph.nodes(data=True)
        )
    )
    features["aromatic_atoms"] = float(
        sum(bool(attrs.get("aromatic", False)) for _, attrs in graph.nodes(data=True))
    )
    features["aromatic_bonds"] = float(
        sum(
            bool(attrs.get("aromatic", False)) for _, _, attrs in graph.edges(data=True)
        )
    )
    features["conjugated_bonds"] = float(
        sum(
            bool(attrs.get("conjugated", False))
            for _, _, attrs in graph.edges(data=True)
        )
    )
    cycles = nx.cycle_basis(graph)
    for size in (3, 4, 5, 6, 7):
        features[f"rings_{size}"] = float(sum(len(cycle) == size for cycle in cycles))
    features["rings_8_plus"] = float(sum(len(cycle) >= 8 for cycle in cycles))
    for degree in range(1, 5):
        features[f"heavy_degree_{degree}"] = float(
            sum(
                sum(graph.nodes[n]["element"] != "H" for n in graph.neighbors(node))
                == degree
                for node in graph.nodes
                if graph.nodes[node]["element"] != "H"
            )
        )
    features["randic_index"] = float(
        sum(
            1.0 / max(1.0, (graph.degree(left) * graph.degree(right)) ** 0.5)
            for left, right in graph.edges
        )
    )
