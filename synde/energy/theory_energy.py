"""Additional fixed 2D chemistry corrections for uncalibrated scoring.

The constants encode broad textbook-scale trends and are deliberately not fit
to ORD or another molecular-energy dataset.  This scorer wraps the stable base
heuristic so calibrated feature schemas remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from synde.graph.graph_schema import NormalizedMolecularGraph

from .molecule_scoring import MoleculeScorer
from .results import MoleculeScoreResult


@dataclass(frozen=True)
class TheoryEnergyConfig:
    """Fixed theory-derived corrections in existing heuristic score units."""

    parameter_set: str = "theory-organic-v1-frozen"
    amide_resonance_reward: float = 0.70
    ester_resonance_reward: float = 0.45
    thioester_resonance_reward: float = 0.35
    aryl_amine_resonance_reward: float = 0.25
    aryl_oxygen_resonance_reward: float = 0.12
    alkene_substituent_reward: float = 0.06
    three_ring_alkene_penalty: float = 0.55
    four_ring_alkene_penalty: float = 0.25
    bridgehead_alkene_penalty: float = 0.50
    carbonyl_pi_correction: float = 1.25
    imine_pi_correction: float = 0.75
    nitrogen_pi_correction: float = 0.30
    nitrogen_oxygen_pi_correction: float = 1.50
    oxygen_pi_correction: float = 1.25
    thiocarbonyl_pi_correction: float = 0.40
    amidine_resonance_reward: float = 0.55
    sulfonamide_resonance_reward: float = 0.50
    sulfonate_resonance_reward: float = 0.30
    nitro_resonance_reward: float = 0.45


class TheoryEnergyScorer:
    """Base uncalibrated heuristic plus fixed missing-chemistry corrections."""

    def __init__(
        self,
        config: TheoryEnergyConfig | None = None,
        base_scorer: MoleculeScorer | None = None,
    ) -> None:
        self.config = config or TheoryEnergyConfig()
        self.base_scorer = base_scorer or MoleculeScorer()

    def score(self, graph: NormalizedMolecularGraph) -> MoleculeScoreResult:
        """Return the enhanced fixed score without fitted coefficients."""
        base = self.base_scorer.score(graph)
        if base.score is None:
            return base
        corrections = theory_energy_corrections(graph.graph, self.config)
        components = {**base.components, **corrections}
        descriptors = {
            **base.descriptors,
            "base_model_name": base.provenance.get("model_name"),
        }
        model = {
            "model_name": "synde-theory-2d-v1",
            "mode": "graph",
            "calibrated": False,
            "fitted_coefficients": False,
            "benchmark_informed_development": True,
            "parameter_set": self.config.parameter_set,
        }
        return MoleculeScoreResult(
            status=base.status,
            score=float(sum(components.values())),
            units=base.units,
            components=components,
            descriptors=descriptors,
            warnings=base.warnings,
            provenance=model,
        )


def theory_energy_corrections(
    graph: nx.Graph, config: TheoryEnergyConfig | None = None
) -> dict[str, float]:
    """Return separately inspectable, fixed higher-level chemistry terms."""
    config = config or TheoryEnergyConfig()
    return {
        "lone_pair_resonance": _lone_pair_resonance(graph, config),
        "alkene_substitution": _alkene_substitution(graph, config),
        "unsaturated_ring_strain": _unsaturated_ring_strain(graph, config),
        "hetero_pi_bond_correction": _hetero_pi_bond_correction(graph, config),
        "extended_resonance": _extended_resonance(graph, config),
    }


def _lone_pair_resonance(graph: nx.Graph, config: TheoryEnergyConfig) -> float:
    reward = 0.0
    carbonyl_carbons: set[Any] = set()
    for carbon, attrs in graph.nodes(data=True):
        if attrs["element"] != "C":
            continue
        if any(
            graph.nodes[neighbor]["element"] in {"O", "S"}
            and _bond_order(graph, carbon, neighbor) >= 1.5
            for neighbor in graph.neighbors(carbon)
        ):
            carbonyl_carbons.add(carbon)
            for donor in graph.neighbors(carbon):
                donor_element = graph.nodes[donor]["element"]
                if _bond_order(graph, carbon, donor) >= 1.5:
                    continue
                if int(graph.nodes[donor].get("formal_charge", 0)) > 0:
                    continue
                if donor_element == "N":
                    reward += config.amide_resonance_reward
                elif donor_element == "O":
                    reward += config.ester_resonance_reward
                elif donor_element == "S":
                    reward += config.thioester_resonance_reward

    for carbon, attrs in graph.nodes(data=True):
        if attrs["element"] != "C" or carbon in carbonyl_carbons:
            continue
        conjugated_carbon = bool(attrs.get("aromatic", False)) or any(
            _bond_order(graph, carbon, neighbor) >= 1.5
            for neighbor in graph.neighbors(carbon)
        )
        if not conjugated_carbon:
            continue
        for donor in graph.neighbors(carbon):
            if _bond_order(graph, carbon, donor) >= 1.5:
                continue
            donor_attrs = graph.nodes[donor]
            if int(donor_attrs.get("formal_charge", 0)) > 0:
                continue
            if donor_attrs["element"] == "N":
                reward += config.aryl_amine_resonance_reward
            elif donor_attrs["element"] == "O":
                reward += config.aryl_oxygen_resonance_reward
    return float(-reward)


def _alkene_substitution(graph: nx.Graph, config: TheoryEnergyConfig) -> float:
    substituents = 0
    for left, right, attrs in graph.edges(data=True):
        if bool(attrs.get("aromatic", False)) or float(attrs.get("order", 1.0)) < 1.5:
            continue
        if graph.nodes[left]["element"] != "C" or graph.nodes[right]["element"] != "C":
            continue
        substituents += sum(
            graph.nodes[neighbor]["element"] == "C"
            for neighbor in graph.neighbors(left)
            if neighbor != right
        )
        substituents += sum(
            graph.nodes[neighbor]["element"] == "C"
            for neighbor in graph.neighbors(right)
            if neighbor != left
        )
    return float(-config.alkene_substituent_reward * substituents)


def _unsaturated_ring_strain(graph: nx.Graph, config: TheoryEnergyConfig) -> float:
    cycles = nx.minimum_cycle_basis(graph)
    memberships: dict[Any, list[int]] = {}
    for cycle in cycles:
        for node in cycle:
            memberships.setdefault(node, []).append(len(cycle))
    penalty = 0.0
    for left, right, attrs in graph.edges(data=True):
        if bool(attrs.get("aromatic", False)) or float(attrs.get("order", 1.0)) < 1.5:
            continue
        shared_sizes = [
            len(cycle) for cycle in cycles if left in cycle and right in cycle
        ]
        if 3 in shared_sizes:
            penalty += config.three_ring_alkene_penalty
        elif 4 in shared_sizes:
            penalty += config.four_ring_alkene_penalty
        for node in (left, right):
            small_memberships = [
                size for size in memberships.get(node, []) if size <= 7
            ]
            if len(small_memberships) >= 2 and graph.degree(node) >= 3:
                penalty += config.bridgehead_alkene_penalty
    return float(penalty)


def _hetero_pi_bond_correction(graph: nx.Graph, config: TheoryEnergyConfig) -> float:
    """Correct generic Hückel π scales using fixed average bond-energy gaps."""
    correction = 0.0
    for left, right, attrs in graph.edges(data=True):
        if bool(attrs.get("aromatic", False)) or float(attrs.get("order", 1.0)) < 1.5:
            continue
        elements = frozenset(
            (graph.nodes[left]["element"], graph.nodes[right]["element"])
        )
        if elements == {"C", "O"}:
            correction += config.carbonyl_pi_correction
        elif elements == {"C", "N"}:
            correction += config.imine_pi_correction
        elif elements == {"N"}:
            correction += config.nitrogen_pi_correction
        elif elements == {"N", "O"}:
            correction += config.nitrogen_oxygen_pi_correction
        elif elements == {"O"}:
            correction += config.oxygen_pi_correction
        elif elements == {"C", "S"}:
            correction += config.thiocarbonyl_pi_correction
    return float(-correction)


def _extended_resonance(graph: nx.Graph, config: TheoryEnergyConfig) -> float:
    """Cover common delocalized groups omitted by the one-p-orbital assignment."""
    reward = 0.0
    for center, attrs in graph.nodes(data=True):
        element = attrs["element"]
        neighbors = list(graph.neighbors(center))
        if element == "C" and any(
            graph.nodes[node]["element"] == "N"
            and _bond_order(graph, center, node) >= 1.5
            for node in neighbors
        ):
            reward += config.amidine_resonance_reward * sum(
                graph.nodes[node]["element"] == "N"
                and _bond_order(graph, center, node) < 1.5
                and int(graph.nodes[node].get("formal_charge", 0)) <= 0
                for node in neighbors
            )
        elif element == "S" and any(
            graph.nodes[node]["element"] == "O"
            and _bond_order(graph, center, node) >= 1.5
            for node in neighbors
        ):
            for donor in neighbors:
                if _bond_order(graph, center, donor) >= 1.5:
                    continue
                donor_element = graph.nodes[donor]["element"]
                if donor_element == "N":
                    reward += config.sulfonamide_resonance_reward
                elif donor_element == "O":
                    reward += config.sulfonate_resonance_reward
        elif element == "N" and any(
            graph.nodes[node]["element"] == "O"
            and _bond_order(graph, center, node) >= 1.5
            for node in neighbors
        ):
            if any(
                graph.nodes[node]["element"] == "O"
                and _bond_order(graph, center, node) < 1.5
                for node in neighbors
            ):
                reward += config.nitro_resonance_reward
    return float(-reward)


def hypervalent_resonance_correction(
    graph: nx.Graph, config: TheoryEnergyConfig | None = None
) -> float:
    """Return sulfonyl resonance not represented by the orbital-level model."""
    config = config or TheoryEnergyConfig()
    reward = 0.0
    for sulfur, attrs in graph.nodes(data=True):
        if attrs["element"] != "S":
            continue
        neighbors = list(graph.neighbors(sulfur))
        if not any(
            graph.nodes[node]["element"] == "O"
            and _bond_order(graph, sulfur, node) >= 1.5
            for node in neighbors
        ):
            continue
        for donor in neighbors:
            if _bond_order(graph, sulfur, donor) >= 1.5:
                continue
            if graph.nodes[donor]["element"] == "N":
                reward += config.sulfonamide_resonance_reward
            elif graph.nodes[donor]["element"] == "O":
                reward += config.sulfonate_resonance_reward
    return float(-reward)


def _bond_order(graph: nx.Graph, left: Any, right: Any) -> float:
    return float(graph.edges[left, right].get("order", 1.0))
