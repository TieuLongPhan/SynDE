"""Experimental orbital-level pi graphs for donors, triples, and cumulenes."""

from __future__ import annotations

import math
from typing import Any

import networkx as nx

from .graph_schema import NormalizedMolecularGraph
from .pi_system import PiAssignmentResult, PiAtom, PiSystem, assign_pi_systems

OrbitalNode = tuple[Any, int]


def assign_orbital_pi(normalized: NormalizedMolecularGraph) -> PiAssignmentResult:
    """Expand stable pi assignments with explicit orthogonal orbitals."""
    graph = normalized.graph
    base = assign_pi_systems(normalized)
    orbital_graph = nx.Graph()
    reasons: dict[OrbitalNode, str] = {}
    for node, attrs in base.pi_graph.nodes(data=True):
        orbital = (node, 0)
        orbital_graph.add_node(orbital, **attrs)
        reasons[orbital] = "stable one-orbital assignment"
    for left, right, attrs in base.pi_graph.edges(data=True):
        orbital_graph.add_edge((left, 0), (right, 0), **attrs)

    handled_triple_atoms = _add_triple_orbitals(graph, orbital_graph, reasons)
    _orthogonalize_cumulenes(graph, orbital_graph, reasons)
    _orthogonalize_hypervalent_sulfur(graph, orbital_graph, reasons)
    _add_lone_pair_donors(graph, orbital_graph, reasons, excluded=handled_triple_atoms)
    warnings = tuple(
        warning
        for warning in base.warnings
        if not (
            warning.code == "PI_ORBITAL_MULTIPLICITY_UNSUPPORTED"
            and any(node in handled_triple_atoms for node in warning.nodes)
        )
    )
    atoms = tuple(
        PiAtom(
            node=orbital,
            included=True,
            electrons=int(attrs["pi_electrons"]),
            reason=reasons[orbital],
            confidence="medium" if orbital[1] else "high",
        )
        for orbital, attrs in orbital_graph.nodes(data=True)
    )
    systems = _orbital_systems(orbital_graph)
    return PiAssignmentResult(
        pi_graph=orbital_graph,
        atoms=atoms,
        systems=systems,
        status="partial" if warnings else "success",
        warnings=warnings,
    )


def _add_triple_orbitals(
    graph: nx.Graph,
    orbital_graph: nx.Graph,
    reasons: dict[OrbitalNode, str],
) -> set[Any]:
    handled: set[Any] = set()
    triple_edges = [
        (left, right, attrs)
        for left, right, attrs in graph.edges(data=True)
        if float(attrs.get("order", 1.0)) >= 2.5
    ]
    for left, right, attrs in triple_edges:
        handled.update((left, right))
        for atom in (left, right):
            for index in (0, 1):
                orbital = (atom, index)
                _add_or_update_orbital(
                    orbital_graph, orbital, graph.nodes[atom], electrons=1
                )
                reasons[orbital] = "orthogonal triple-bond pi orbital"
        for index in (0, 1):
            orbital_graph.add_edge((left, index), (right, index), **attrs)
    for atom in handled:
        for neighbor in graph.neighbors(atom):
            if neighbor in handled or (neighbor, 0) not in orbital_graph:
                continue
            attrs = graph.edges[atom, neighbor]
            if bool(attrs.get("conjugated", False)):
                orbital_graph.add_edge((atom, 0), (neighbor, 0), **attrs)
    return handled


def _orthogonalize_cumulenes(
    graph: nx.Graph,
    orbital_graph: nx.Graph,
    reasons: dict[OrbitalNode, str],
) -> None:
    for center, attrs in graph.nodes(data=True):
        double_neighbors = [
            neighbor
            for neighbor in graph.neighbors(center)
            if 1.5 <= float(graph.edges[center, neighbor].get("order", 1.0)) < 2.5
            and not bool(graph.edges[center, neighbor].get("aromatic", False))
        ]
        if len(double_neighbors) != 2 or attrs.get("hybridization") != "SP":
            continue
        first, second = double_neighbors
        if (center, 0) not in orbital_graph:
            continue
        _add_or_update_orbital(orbital_graph, (center, 0), attrs, electrons=1)
        _add_or_update_orbital(orbital_graph, (center, 1), attrs, electrons=1)
        reasons[(center, 0)] = "first orthogonal cumulene pi orbital"
        reasons[(center, 1)] = "second orthogonal cumulene pi orbital"
        for neighbor in double_neighbors:
            if orbital_graph.has_edge((center, 0), (neighbor, 0)):
                orbital_graph.remove_edge((center, 0), (neighbor, 0))
        orbital_graph.add_edge((center, 0), (first, 0), **graph.edges[center, first])
        orbital_graph.add_edge((center, 1), (second, 0), **graph.edges[center, second])


def _orthogonalize_hypervalent_sulfur(
    graph: nx.Graph,
    orbital_graph: nx.Graph,
    reasons: dict[OrbitalNode, str],
) -> None:
    """Give each S=O bond an independent sulfur p-like orbital."""
    for sulfur, attrs in graph.nodes(data=True):
        if attrs["element"] != "S":
            continue
        oxygen_neighbors = [
            neighbor
            for neighbor in graph.neighbors(sulfur)
            if graph.nodes[neighbor]["element"] == "O"
            and float(graph.edges[sulfur, neighbor].get("order", 1.0)) >= 1.5
        ]
        if len(oxygen_neighbors) < 2:
            continue
        for neighbor in oxygen_neighbors:
            if orbital_graph.has_edge((sulfur, 0), (neighbor, 0)):
                orbital_graph.remove_edge((sulfur, 0), (neighbor, 0))
        for index, oxygen in enumerate(sorted(oxygen_neighbors, key=repr)):
            sulfur_orbital = (sulfur, index)
            _add_or_update_orbital(orbital_graph, sulfur_orbital, attrs, electrons=1)
            reasons[sulfur_orbital] = "orthogonal hypervalent sulfur pi orbital"
            orbital_graph.add_edge(
                sulfur_orbital, (oxygen, 0), **graph.edges[sulfur, oxygen]
            )


def _add_lone_pair_donors(
    graph: nx.Graph,
    orbital_graph: nx.Graph,
    reasons: dict[OrbitalNode, str],
    *,
    excluded: set[Any],
) -> None:
    for donor, attrs in graph.nodes(data=True):
        if donor in excluded or (donor, 0) in orbital_graph:
            continue
        if attrs["element"] not in {"N", "O", "S"}:
            continue
        if int(attrs.get("formal_charge", 0)) > 0:
            continue
        if not (attrs.get("available_lp", False) or attrs.get("lone_pairs", 0)):
            continue
        targets = [
            target_orbital
            for neighbor in graph.neighbors(donor)
            if _is_donor_coupling(graph, donor, neighbor)
            for target_orbital in _donor_target_orbitals(graph, orbital_graph, neighbor)
        ]
        if not targets:
            continue
        orbital = (donor, 0)
        _add_or_update_orbital(orbital_graph, orbital, attrs, electrons=2)
        reasons[orbital] = "conjugated lone-pair donor orbital"
        target_counts: dict[Any, int] = {}
        for target, _ in targets:
            target_counts[target] = target_counts.get(target, 0) + 1
        for target_orbital in targets:
            target = target_orbital[0]
            edge_attrs = dict(graph.edges[donor, target])
            multiplicity = target_counts[target]
            if multiplicity > 1:
                edge_attrs["coupling_scale"] = 1.0 / math.sqrt(multiplicity)
            orbital_graph.add_edge(orbital, target_orbital, **edge_attrs)


def _donor_target_orbitals(
    graph: nx.Graph, orbital_graph: nx.Graph, target: Any
) -> list[OrbitalNode]:
    orbitals = [orbital for orbital in orbital_graph.nodes if orbital[0] == target]
    if graph.nodes[target]["element"] == "S" and len(orbitals) > 1:
        return sorted(orbitals, key=repr)
    return [(target, 0)] if (target, 0) in orbital_graph else []


def _is_donor_coupling(graph: nx.Graph, donor: Any, target: Any) -> bool:
    edge = graph.edges[donor, target]
    if float(edge.get("order", 1.0)) >= 1.5:
        return False
    if (
        graph.nodes[target]["element"] == "S"
        and sum(
            graph.nodes[neighbor]["element"] == "O"
            and float(graph.edges[target, neighbor].get("order", 1.0)) >= 1.5
            for neighbor in graph.neighbors(target)
        )
        >= 2
    ):
        return graph.nodes[donor]["element"] in {"N", "O", "S"}
    return bool(edge.get("conjugated", False))


def _add_or_update_orbital(
    orbital_graph: nx.Graph,
    orbital: OrbitalNode,
    attrs: dict[str, Any],
    *,
    electrons: int,
) -> None:
    data = {**attrs, "pi_electrons": electrons}
    if orbital in orbital_graph:
        orbital_graph.nodes[orbital].update(data)
    else:
        orbital_graph.add_node(orbital, **data)


def _orbital_systems(orbital_graph: nx.Graph) -> tuple[PiSystem, ...]:
    systems = []
    for nodes in nx.connected_components(orbital_graph):
        ordered = tuple(sorted(nodes, key=repr))
        edges = tuple(
            sorted(
                (
                    tuple(sorted((left, right), key=repr))
                    for left, right in orbital_graph.subgraph(nodes).edges
                ),
                key=repr,
            )
        )
        systems.append(
            PiSystem(
                nodes=ordered,
                edges=edges,
                electron_count=sum(
                    int(orbital_graph.nodes[node]["pi_electrons"]) for node in nodes
                ),
            )
        )
    return tuple(sorted(systems, key=lambda system: repr(system.nodes)))


__all__ = ["OrbitalNode", "assign_orbital_pi"]
