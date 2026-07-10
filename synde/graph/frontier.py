"""Degeneracy-safe component frontier descriptors and graph FMO features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .generalized_huckel import GeneralizedHuckelResult


@dataclass(frozen=True)
class ComponentFrontier:
    component_id: int
    nodes: tuple[Any, ...]
    homo_energy: float | None
    lumo_energy: float | None
    homo_density: dict[Any, float]
    lumo_density: dict[Any, float]


@dataclass(frozen=True)
class DirectionalFMO:
    donor_component: int
    acceptor_component: int
    donor_atom: Any
    acceptor_atom: Any
    gap: float | None
    regularized_gap: float | None
    score: float
    valid: bool
    warning: str | None = None


def frontiers_from_graph_result(
    graph, result: GeneralizedHuckelResult
) -> tuple[ComponentFrontier, ...]:
    """Combine frontiers using component IDs from a normalized pi graph."""
    grouped: dict[int, list] = {}
    for system in result.systems:
        component_id = int(graph.nodes[system.nodes[0]]["component_id"])
        grouped.setdefault(component_id, []).append(system)
    return tuple(
        _combine_frontier(component_id, systems)
        for component_id, systems in grouped.items()
    )


def _combine_frontier(component_id: int, systems: list) -> ComponentFrontier:
    nodes = tuple(node for system in systems for node in system.nodes)
    homo_energy = max(
        (system.homo_energy for system in systems if system.homo_energy is not None),
        default=None,
    )
    lumo_energy = min(
        (system.lumo_energy for system in systems if system.lumo_energy is not None),
        default=None,
    )
    homo_density = {node: 0.0 for node in nodes}
    lumo_density = {node: 0.0 for node in nodes}
    for system in systems:
        if system.homo_energy == homo_energy:
            homo_density.update(
                zip(system.nodes, (float(x) for x in system.homo_density))
            )
        if system.lumo_energy == lumo_energy and system.lumo_density is not None:
            lumo_density.update(
                zip(system.nodes, (float(x) for x in system.lumo_density))
            )
    return ComponentFrontier(
        component_id, nodes, homo_energy, lumo_energy, homo_density, lumo_density
    )


def directional_fmo(
    donor: ComponentFrontier,
    acceptor: ComponentFrontier,
    donor_atom: Any,
    acceptor_atom: Any,
    *,
    gap_floor: float = 0.05,
) -> DirectionalFMO:
    """Return a graph-only donor-to-acceptor compatibility descriptor.

    Non-positive gaps are invalid rather than replaced by an artificial tiny
    denominator.  Small positive gaps are regularized and flagged.
    """
    if donor.homo_energy is None or acceptor.lumo_energy is None:
        return DirectionalFMO(
            donor.component_id,
            acceptor.component_id,
            donor_atom,
            acceptor_atom,
            None,
            None,
            0.0,
            False,
            "FRONTIER_UNAVAILABLE",
        )
    gap = float(acceptor.lumo_energy - donor.homo_energy)
    if gap <= 0:
        return DirectionalFMO(
            donor.component_id,
            acceptor.component_id,
            donor_atom,
            acceptor_atom,
            gap,
            None,
            0.0,
            False,
            "INVALID_DIRECTIONAL_GAP",
        )
    regularized = max(gap, gap_floor)
    warning = "FRONTIER_GAP_REGULARIZED" if gap < gap_floor else None
    score = (
        donor.homo_density.get(donor_atom, 0.0)
        * acceptor.lumo_density.get(acceptor_atom, 0.0)
        / regularized
    )
    return DirectionalFMO(
        donor.component_id,
        acceptor.component_id,
        donor_atom,
        acceptor_atom,
        gap,
        regularized,
        float(score),
        True,
        warning,
    )
