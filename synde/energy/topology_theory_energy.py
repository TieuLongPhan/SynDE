"""Experimental graph-topology corrections for the fixed theory heuristic.

The corrections target effects that are physically meaningful in two
dimensions: reduced pi overlap across rotatable biaryl bonds, ortho
substituent congestion, and ring-junction/medium-ring strain.  They are kept
separate from :mod:`synde.energy.theory_energy` so its frozen parameter set is
never changed by development experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import networkx as nx

from synde.graph.generalized_huckel import GeneralizedHuckel
from synde.graph.graph_schema import NormalizedMolecularGraph
from synde.graph.pi_system import PiAssignmentResult, assign_pi_systems

from .molecule_scoring import MoleculeScorer
from .results import MoleculeScoreResult
from .theory_energy import TheoryEnergyScorer

# Approximate relative van-der-Waals bulk, normalized near carbon.  These are
# deliberately coarse chemical reference values, not fitted descriptors.
_ATOM_BULK = {
    "B": 1.05,
    "C": 1.00,
    "N": 0.90,
    "O": 0.82,
    "F": 0.78,
    "Si": 1.35,
    "P": 1.25,
    "S": 1.18,
    "Cl": 1.18,
    "Br": 1.30,
    "I": 1.48,
}


@dataclass(frozen=True)
class TopologyTheoryEnergyConfig:
    """Fixed coefficients for the experimental topology layer."""

    parameter_set: str = "theory-organic-v2-topology-development"
    biaryl_coupling_scale: float = 0.75
    ortho_congestion_weight: float = 0.08
    bridgehead_penalty: float = 0.12
    spiro_penalty: float = 0.04


class TopologyTheoryEnergyScorer:
    """Add narrow topology corrections to the frozen theory scorer."""

    def __init__(
        self,
        config: TopologyTheoryEnergyConfig | None = None,
        base_scorer: MoleculeScorer | None = None,
    ) -> None:
        self.config = config or TopologyTheoryEnergyConfig()
        self.base_scorer = base_scorer or MoleculeScorer()
        self.frozen_scorer = TheoryEnergyScorer(base_scorer=self.base_scorer)
        self.huckel = GeneralizedHuckel(self.base_scorer.huckel.parameters)

    def score(self, normalized: NormalizedMolecularGraph) -> MoleculeScoreResult:
        """Return the experimental score without mutating the frozen scorer."""
        frozen = self.frozen_scorer.score(normalized)
        if frozen.score is None:
            return frozen
        corrections = topology_energy_corrections(
            normalized,
            self.config,
            huckel=self.huckel,
            pi_weight=self.base_scorer.config.two_d.pi_weight,
        )
        components = {**frozen.components, **corrections}
        provenance = {
            "model_name": "synde-theory-2d-v2-topology-experimental",
            "mode": "graph",
            "calibrated": False,
            "fitted_coefficients": False,
            "experimental": True,
            "benchmark_informed_development": True,
            "parameter_set": self.config.parameter_set,
            "permanent_holdout_evaluated": False,
            "holdout_tuned": False,
        }
        return MoleculeScoreResult(
            status=frozen.status,
            score=float(sum(components.values())),
            units=frozen.units,
            components=components,
            descriptors=frozen.descriptors,
            warnings=frozen.warnings,
            provenance=provenance,
        )


def topology_energy_corrections(
    normalized: NormalizedMolecularGraph,
    config: TopologyTheoryEnergyConfig | None = None,
    *,
    huckel: GeneralizedHuckel | None = None,
    pi_weight: float = 1.33,
) -> dict[str, float]:
    """Return separately inspectable topology corrections."""
    config = config or TopologyTheoryEnergyConfig()
    graph = normalized.graph
    return {
        "topology_pi_decoupling": _topology_pi_decoupling(
            normalized, config, huckel or GeneralizedHuckel(), pi_weight
        ),
        "ortho_steric_congestion": _ortho_steric_congestion(graph, config),
        "refined_ring_topology": _refined_ring_topology(graph, config),
    }


def _topology_pi_decoupling(
    normalized: NormalizedMolecularGraph,
    config: TopologyTheoryEnergyConfig,
    huckel: GeneralizedHuckel,
    pi_weight: float,
) -> float:
    assignment = assign_pi_systems(normalized)
    scaled_graph = assignment.pi_graph.copy()
    changed = False
    for left, right, attrs in scaled_graph.edges(data=True):
        if not _is_rotatable_biaryl_bond(scaled_graph, left, right, attrs):
            continue
        attrs["coupling_scale"] = config.biaryl_coupling_scale
        changed = True
    if not changed:
        return 0.0
    scaled_assignment: PiAssignmentResult = replace(assignment, pi_graph=scaled_graph)
    full = huckel.solve(assignment).pi_stabilization
    attenuated = huckel.solve(scaled_assignment).pi_stabilization
    return float(pi_weight * (attenuated - full))


def _is_rotatable_biaryl_bond(
    graph: nx.Graph, left: Any, right: Any, attrs: dict[str, Any]
) -> bool:
    return bool(
        graph.nodes[left].get("aromatic", False)
        and graph.nodes[right].get("aromatic", False)
        and not attrs.get("aromatic", False)
        and not attrs.get("is_in_ring", attrs.get("in_ring", False))
        and float(attrs.get("order", 1.0)) < 1.5
    )


def _ortho_steric_congestion(
    graph: nx.Graph, config: TopologyTheoryEnergyConfig
) -> float:
    """Penalize adjacent aromatic sites bearing non-ring heavy substituents."""
    congestion = 0.0
    for left, right, attrs in graph.edges(data=True):
        if not attrs.get("aromatic", False):
            continue
        left_substituents = _external_substituents(graph, left)
        right_substituents = _external_substituents(graph, right)
        if not left_substituents or not right_substituents:
            continue
        left_bulk = sum(
            _substituent_bulk(graph, node, left) for node in left_substituents
        )
        right_bulk = sum(
            _substituent_bulk(graph, node, right) for node in right_substituents
        )
        congestion += left_bulk * right_bulk
    return float(config.ortho_congestion_weight * congestion)


def _external_substituents(graph: nx.Graph, aromatic_node: Any) -> list[Any]:
    return [
        neighbor
        for neighbor in graph.neighbors(aromatic_node)
        if graph.nodes[neighbor]["element"] != "H"
        and not graph.nodes[neighbor].get("aromatic", False)
    ]


def _substituent_bulk(graph: nx.Graph, root: Any, anchor: Any) -> float:
    """Estimate local substituent bulk through two graph shells."""
    bulk = _ATOM_BULK.get(graph.nodes[root]["element"], 1.0)
    for neighbor in graph.neighbors(root):
        if neighbor == anchor or graph.nodes[neighbor]["element"] == "H":
            continue
        bulk += 0.30 * _ATOM_BULK.get(graph.nodes[neighbor]["element"], 1.0)
        for outer in graph.neighbors(neighbor):
            if outer in {root, anchor} or graph.nodes[outer]["element"] == "H":
                continue
            bulk += 0.10 * _ATOM_BULK.get(graph.nodes[outer]["element"], 1.0)
    return float(bulk)


def _refined_ring_topology(
    graph: nx.Graph, config: TopologyTheoryEnergyConfig
) -> float:
    """Refine flat large-ring strain and identify ring junction classes."""
    cycles = nx.cycle_basis(graph)
    # The base model assigns 0.40 to every >=8 ring.  These increments replace
    # that flat approximation with broad cycloalkane strain trends.
    large_ring_increment = {8: 0.02, 9: 0.14, 10: 0.12, 11: -0.19}
    correction = sum(
        large_ring_increment.get(len(cycle), -0.40 if len(cycle) >= 12 else 0.0)
        for cycle in cycles
    )
    cycle_nodes = [set(cycle) for cycle in cycles]
    bridgeheads: set[Any] = set()
    spiro_centers: set[Any] = set()
    for index, first in enumerate(cycle_nodes):
        for second in cycle_nodes[index + 1 :]:
            shared = first & second
            if len(shared) == 1:
                spiro_centers.update(shared)
            elif len(shared) >= 2:
                bridgeheads.update(node for node in shared if graph.degree(node) >= 3)
    correction += config.bridgehead_penalty * len(bridgeheads)
    correction += config.spiro_penalty * len(spiro_centers - bridgeheads)
    return float(correction)


__all__ = [
    "TopologyTheoryEnergyConfig",
    "TopologyTheoryEnergyScorer",
    "topology_energy_corrections",
]
