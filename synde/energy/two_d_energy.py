"""Fixed, theory-guided 2D molecular-energy heuristic components.

The constants are approximate chemistry reference values and hand-defined
dimensionless corrections. They are not fitted to the ORD benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from synde.graph.generalized_huckel import GeneralizedHuckelResult
from synde.graph.pi_system import PiAssignmentResult

from .local_energy import ATOM_BASELINE

# Approximate σ-bond strengths in kJ/mol, used only as fixed ratios. Multiple
# bonds still contain one σ bond; their π contribution is handled by Hückel.
SIGMA_BOND_STRENGTH_KJ_MOL: dict[tuple[str, str], float] = {
    ("C", "C"): 347.0,
    ("C", "H"): 413.0,
    ("C", "N"): 305.0,
    ("C", "O"): 358.0,
    ("C", "S"): 272.0,
    ("H", "N"): 391.0,
    ("H", "O"): 463.0,
    ("H", "S"): 347.0,
    ("N", "N"): 163.0,
    ("N", "O"): 201.0,
    ("O", "O"): 146.0,
    ("O", "P"): 335.0,
    ("S", "S"): 266.0,
    ("O", "S"): 364.0,
    ("C", "F"): 485.0,
    ("C", "Cl"): 327.0,
    ("Br", "C"): 285.0,
    ("C", "I"): 213.0,
}

IMPLICIT_H_BOND_KJ_MOL = {
    "C": 413.0,
    "N": 391.0,
    "O": 463.0,
    "S": 347.0,
    "P": 322.0,
    "B": 389.0,
    "Si": 318.0,
}

# Approximate cycloalkane strain energies divided by 100 kJ/mol.  These fixed
# values encode only broad trends and do not depend on the ORD labels.
RING_STRAIN_SCORE = {3: 1.15, 4: 1.10, 5: 0.26, 6: 0.0, 7: 0.26}


@dataclass(frozen=True)
class TwoDEnergyConfig:
    """Explicit fixed constants for the uncalibrated 2D heuristic."""

    sigma_scale: float = 0.01
    pi_weight: float = 1.33
    aromatic_reward_per_atom: float = 0.0
    antiaromatic_penalty_per_atom: float = 0.25
    charge_magnitude_penalty: float = 0.50
    charge_pair_weight: float = 0.20
    fused_ring_penalty: float = 0.08
    large_ring_penalty: float = 0.40
    tertiary_branch_reward: float = 0.08
    quaternary_branch_reward: float = 0.12
    steric_branch_weight: float = 0.04
    steric_bond_weight: float = 0.02


def two_d_energy_components(
    graph: nx.Graph,
    assignment: PiAssignmentResult,
    huckel: GeneralizedHuckelResult,
    config: TwoDEnergyConfig | None = None,
) -> dict[str, float]:
    """Return separately inspectable fixed 2D molecular-energy terms."""
    config = config or TwoDEnergyConfig()
    return {
        "atom_reference": _atom_reference(graph),
        "sigma_bond_energy": _sigma_bond_energy(graph, config),
        "pi_stabilization": config.pi_weight * huckel.pi_stabilization,
        "cyclic_pi_correction": _cyclic_pi_correction(graph, assignment, config),
        "ring_strain": _ring_strain(graph, config),
        "charge_electrostatic": _charge_electrostatic(graph, config),
        "branching_stabilization": _branching_stabilization(graph, config),
        "steric_congestion": _steric_congestion(graph, config),
    }


def _atom_reference(graph: nx.Graph) -> float:
    return float(
        sum(
            ATOM_BASELINE.get(attrs["element"], 0.0)
            for _, attrs in graph.nodes(data=True)
        )
    )


def _sigma_bond_strength(left: str, right: str) -> float:
    first, second = sorted((left, right))
    return SIGMA_BOND_STRENGTH_KJ_MOL.get((first, second), 330.0)


def _sigma_bond_energy(graph: nx.Graph, config: TwoDEnergyConfig) -> float:
    total = 0.0
    for left, right in graph.edges:
        total += _sigma_bond_strength(
            graph.nodes[left]["element"], graph.nodes[right]["element"]
        )
    for _, attrs in graph.nodes(data=True):
        if attrs["element"] != "H":
            total += float(attrs.get("total_hcount", 0)) * IMPLICIT_H_BOND_KJ_MOL.get(
                attrs["element"], 350.0
            )
    return float(-config.sigma_scale * total)


def _cyclic_pi_correction(
    graph: nx.Graph, assignment: PiAssignmentResult, config: TwoDEnergyConfig
) -> float:
    included = {atom.node: atom for atom in assignment.atoms if atom.included}
    correction = 0.0
    for cycle in nx.cycle_basis(graph):
        if not cycle or any(node not in included for node in cycle):
            continue
        cycle_edges = [
            (cycle[index], cycle[(index + 1) % len(cycle)])
            for index in range(len(cycle))
        ]
        if any(not assignment.pi_graph.has_edge(*edge) for edge in cycle_edges):
            continue
        electrons = sum(int(included[node].electrons or 0) for node in cycle)
        if electrons % 4 == 2:
            correction -= config.aromatic_reward_per_atom * len(cycle)
        elif electrons > 0 and electrons % 4 == 0:
            correction += config.antiaromatic_penalty_per_atom * len(cycle)
    return float(correction)


def _ring_strain(graph: nx.Graph, config: TwoDEnergyConfig) -> float:
    cycles = nx.cycle_basis(graph)
    strain = sum(
        RING_STRAIN_SCORE.get(len(cycle), config.large_ring_penalty) for cycle in cycles
    )
    memberships: dict[Any, int] = {}
    for cycle in cycles:
        for node in cycle:
            memberships[node] = memberships.get(node, 0) + 1
    strain += config.fused_ring_penalty * sum(
        max(0, count - 1) for count in memberships.values()
    )
    return float(strain)


def _charge_electrostatic(graph: nx.Graph, config: TwoDEnergyConfig) -> float:
    charged = [
        (node, int(attrs.get("formal_charge", 0)))
        for node, attrs in graph.nodes(data=True)
        if int(attrs.get("formal_charge", 0)) != 0
    ]
    score = config.charge_magnitude_penalty * sum(abs(charge) for _, charge in charged)
    for index, (left, left_charge) in enumerate(charged):
        for right, right_charge in charged[index + 1 :]:
            try:
                distance = nx.shortest_path_length(graph, left, right)
            except nx.NetworkXNoPath:
                continue
            score += (
                config.charge_pair_weight
                * left_charge
                * right_charge
                / max(distance, 1)
            )
    return float(score)


def _steric_congestion(graph: nx.Graph, config: TwoDEnergyConfig) -> float:
    heavy_degree = {
        node: sum(
            graph.nodes[neighbor]["element"] != "H"
            for neighbor in graph.neighbors(node)
        )
        for node in graph.nodes
    }
    branch = sum(max(0, degree - 3) ** 2 for degree in heavy_degree.values())
    bond_crowding = sum(
        max(0, (heavy_degree[left] - 1) * (heavy_degree[right] - 1) - 1)
        for left, right in graph.edges
        if graph.nodes[left]["element"] != "H" and graph.nodes[right]["element"] != "H"
    )
    return float(
        config.steric_branch_weight * branch + config.steric_bond_weight * bond_crowding
    )


def _branching_stabilization(graph: nx.Graph, config: TwoDEnergyConfig) -> float:
    """Apply a small fixed reward for moderate branching at saturated carbon."""
    tertiary = 0
    quaternary = 0
    for node, attrs in graph.nodes(data=True):
        if attrs["element"] != "C" or attrs.get("hybridization") != "SP3":
            continue
        carbon_neighbors = sum(
            graph.nodes[neighbor]["element"] == "C"
            for neighbor in graph.neighbors(node)
        )
        tertiary += carbon_neighbors == 3
        quaternary += carbon_neighbors == 4
    return float(
        -config.tertiary_branch_reward * tertiary
        - config.quaternary_branch_reward * quaternary
    )
