"""Explicit pi-orbital eligibility and electron assignment for SYN v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from .graph_schema import GraphWarning, NormalizedMolecularGraph

SUPPORTED_PI_ELEMENTS = frozenset({"C", "N", "O", "S"})


@dataclass(frozen=True)
class PiAtom:
    """Per-atom pi-orbital assignment diagnostic."""

    node: Any
    included: bool
    electrons: int | None
    reason: str
    confidence: str


@dataclass(frozen=True)
class PiSystem:
    """One connected system of coupled p orbitals."""

    nodes: tuple[Any, ...]
    edges: tuple[tuple[Any, Any], ...]
    electron_count: int


@dataclass(frozen=True)
class PiAssignmentResult:
    """Pi-system assignment for a normalized molecular graph."""

    pi_graph: nx.Graph
    atoms: tuple[PiAtom, ...]
    systems: tuple[PiSystem, ...]
    status: str
    warnings: tuple[GraphWarning, ...]

    @property
    def electron_count(self) -> int:
        """Return the total number of assigned pi electrons."""

        return sum(system.electron_count for system in self.systems)

    def atom_for(self, node: Any) -> PiAtom:
        """Return the diagnostic record for ``node``."""

        for atom in self.atoms:
            if atom.node == node:
                return atom
        raise KeyError(node)

    def warning_codes(self) -> tuple[str, ...]:
        return tuple(warning.code for warning in self.warnings)


class PiSystemAssigner:
    """Assign one-p-orbital Hückel systems for supported organic graphs.

    The first v2 model intentionally supports one p orbital per atom.  Triple
    bonds, radicals, and unsupported heteroatom environments are reported as
    diagnostics instead of being forced into an incorrect one-orbital model.
    """

    def assign(self, normalized: NormalizedMolecularGraph) -> PiAssignmentResult:
        """Build coupled pi systems and explicit electron counts."""

        graph = normalized.graph
        warnings: list[GraphWarning] = []
        atoms: list[PiAtom] = []
        included: dict[Any, PiAtom] = {}
        for node, attrs in graph.nodes(data=True):
            diagnostic, warning = self._assign_atom(graph, node, attrs)
            atoms.append(diagnostic)
            if warning is not None:
                warnings.append(warning)
            if diagnostic.included:
                included[node] = diagnostic

        pi_graph = nx.Graph()
        for node, diagnostic in included.items():
            pi_graph.add_node(
                node, **graph.nodes[node], pi_electrons=diagnostic.electrons
            )
        for left, right, attrs in graph.edges(data=True):
            if left in included and right in included and self._is_pi_coupling(attrs):
                pi_graph.add_edge(left, right, **attrs)

        systems = self._systems(pi_graph)
        if warnings and not systems:
            status = "unsupported"
        elif warnings:
            status = "partial"
        else:
            status = "success"
        return PiAssignmentResult(
            pi_graph=pi_graph,
            atoms=tuple(atoms),
            systems=systems,
            status=status,
            warnings=tuple(warnings),
        )

    def _assign_atom(
        self,
        graph: nx.Graph,
        node: Any,
        attrs: dict[str, Any],
    ) -> tuple[PiAtom, GraphWarning | None]:
        element = attrs["element"]
        if attrs.get("radical_electrons", 0):
            return (
                PiAtom(node, False, None, "open-shell atom", "unsupported"),
                GraphWarning(
                    "OPEN_SHELL_NOT_SUPPORTED",
                    "Pi assignment excludes atoms with radical electrons.",
                    (node,),
                ),
            )
        if self._has_triple_bond(graph, node):
            return (
                PiAtom(
                    node,
                    False,
                    None,
                    "triple bond needs multiple p orbitals",
                    "unsupported",
                ),
                GraphWarning(
                    "PI_ORBITAL_MULTIPLICITY_UNSUPPORTED",
                    "The one-orbital pi model does not yet support triple-bond atoms.",
                    (node,),
                ),
            )
        if element not in SUPPORTED_PI_ELEMENTS:
            return self._unsupported_element(node, element)
        if attrs.get("aromatic", False):
            return self._aromatic_assignment(node, attrs)
        if self._has_double_bond(graph, node):
            return self._double_bond_assignment(node, attrs)
        return (
            PiAtom(node, False, None, "no supported p-orbital environment", "high"),
            None,
        )

    @staticmethod
    def _unsupported_element(
        node: Any, element: str
    ) -> tuple[PiAtom, GraphWarning | None]:
        return (
            PiAtom(
                node, False, None, f"unsupported pi element {element}", "unsupported"
            ),
            GraphWarning(
                "UNSUPPORTED_PI_ELEMENT",
                f"Pi assignment does not yet support element {element!r}.",
                (node,),
            ),
        )

    @staticmethod
    def _aromatic_assignment(
        node: Any, attrs: dict[str, Any]
    ) -> tuple[PiAtom, GraphWarning | None]:
        element = attrs["element"]
        charge = int(attrs.get("formal_charge", 0))
        hcount = int(attrs.get("total_hcount", 0))
        if element == "C":
            electrons = {1: 0, -1: 2}.get(charge, 1)
            if charge not in {-1, 0, 1}:
                return PiSystemAssigner._ambiguous_charge(node, attrs)
            return PiAtom(node, True, electrons, "aromatic carbon", "high"), None
        if element == "N":
            if charge not in {-1, 0, 1}:
                return PiSystemAssigner._ambiguous_charge(node, attrs)
            if hcount > 0:
                return (
                    PiAtom(node, True, 2, "pyrrole-like aromatic nitrogen", "high"),
                    None,
                )
            if charge == -1:
                return (
                    PiAtom(node, True, 2, "anionic aromatic nitrogen", "medium"),
                    None,
                )
            return (
                PiAtom(node, True, 1, "pyridine-like aromatic nitrogen", "high"),
                None,
            )
        if element in {"O", "S"}:
            if charge not in {-1, 0, 1}:
                return PiSystemAssigner._ambiguous_charge(node, attrs)
            return (
                PiAtom(node, True, 2, f"aromatic {element} lone-pair donor", "medium"),
                None,
            )
        return PiSystemAssigner._unsupported_element(node, element)

    @staticmethod
    def _double_bond_assignment(
        node: Any, attrs: dict[str, Any]
    ) -> tuple[PiAtom, GraphWarning | None]:
        element = attrs["element"]
        charge = int(attrs.get("formal_charge", 0))
        if charge not in {-1, 0, 1}:
            return PiSystemAssigner._ambiguous_charge(node, attrs)
        if element == "C":
            electrons = {1: 0, 0: 1, -1: 2}[charge]
            return (
                PiAtom(node, True, electrons, "non-aromatic sp2 carbon", "high"),
                None,
            )
        if element == "N":
            electrons = 2 if charge == -1 else 1
            return (
                PiAtom(node, True, electrons, "non-aromatic pi nitrogen", "medium"),
                None,
            )
        if element in {"O", "S"}:
            electrons = 2 if charge == -1 else 1
            return (
                PiAtom(node, True, electrons, f"non-aromatic pi {element}", "medium"),
                None,
            )
        return PiSystemAssigner._unsupported_element(node, element)

    @staticmethod
    def _ambiguous_charge(
        node: Any, attrs: dict[str, Any]
    ) -> tuple[PiAtom, GraphWarning]:
        return (
            PiAtom(
                node,
                False,
                None,
                "formal charge is outside the initial rule set",
                "unsupported",
            ),
            GraphWarning(
                "PI_ELECTRON_ASSIGNMENT_AMBIGUOUS",
                "Formal charge is outside the initial pi-electron rule set.",
                (node,),
            ),
        )

    @staticmethod
    def _has_double_bond(graph: nx.Graph, node: Any) -> bool:
        return any(
            float(data.get("order", 1.0)) >= 2.0
            for _, _, data in graph.edges(node, data=True)
        )

    @staticmethod
    def _has_triple_bond(graph: nx.Graph, node: Any) -> bool:
        return any(
            float(data.get("order", 1.0)) >= 3.0
            for _, _, data in graph.edges(node, data=True)
        )

    @staticmethod
    def _is_pi_coupling(attrs: dict[str, Any]) -> bool:
        return (
            bool(attrs.get("aromatic", False))
            or bool(attrs.get("conjugated", False))
            or float(attrs.get("order", 1.0)) >= 2.0
        )

    @staticmethod
    def _systems(pi_graph: nx.Graph) -> tuple[PiSystem, ...]:
        systems: list[PiSystem] = []
        for nodes in nx.connected_components(pi_graph):
            ordered_nodes = tuple(sorted(nodes, key=repr))
            edges = tuple(
                sorted(
                    (
                        tuple(sorted((left, right), key=repr))
                        for left, right in pi_graph.subgraph(nodes).edges()
                    ),
                    key=repr,
                )
            )
            systems.append(
                PiSystem(
                    nodes=ordered_nodes,
                    edges=edges,
                    electron_count=sum(
                        int(pi_graph.nodes[node]["pi_electrons"]) for node in nodes
                    ),
                )
            )
        return tuple(
            sorted(
                systems, key=lambda system: (repr(system.nodes), system.electron_count)
            )
        )


def assign_pi_systems(normalized: NormalizedMolecularGraph) -> PiAssignmentResult:
    """Convenience wrapper around :class:`PiSystemAssigner`."""

    return PiSystemAssigner().assign(normalized)
