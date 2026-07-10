"""Normalized molecular graph contract for the graph-first SYN v2 API.

The legacy SYN modules accept a minimal NetworkX graph.  This module provides
an immutable envelope around a copied, normalized graph with enough chemical
metadata for later pi-system, local-energy, and reaction modules.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable, Mapping

import networkx as nx

try:
    from rdkit import Chem  # type: ignore
except Exception:  # pragma: no cover - optional at import time
    Chem = None  # type: ignore


SUPPORTED_ELEMENTS = frozenset(
    {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}
)


@dataclass(frozen=True)
class GraphWarning:
    """A structured warning emitted while normalizing a molecular graph."""

    code: str
    message: str
    nodes: tuple[Any, ...] = ()


class GraphValidationError(ValueError):
    """Raised for malformed graph input when strict validation is enabled."""


@dataclass(frozen=True)
class NormalizedMolecularGraph:
    """Validated molecular graph plus identity, provenance, and warnings.

    ``graph`` is a private copy of the input graph.  Node identifiers are kept
    unchanged so atom-map and caller-level identities remain usable by later
    reaction modules.
    """

    graph: nx.Graph
    identity: str
    canonical_smiles: str | None
    status: str
    warnings: tuple[GraphWarning, ...]
    source: str

    def warning_codes(self) -> tuple[str, ...]:
        """Return warning codes in encounter order."""

        return tuple(warning.code for warning in self.warnings)

    def to_dict(self) -> dict[str, Any]:
        """Return serializable graph metadata; the NetworkX graph is omitted."""

        return {
            "identity": self.identity,
            "canonical_smiles": self.canonical_smiles,
            "status": self.status,
            "warnings": [
                {"code": w.code, "message": w.message, "nodes": list(w.nodes)}
                for w in self.warnings
            ],
            "source": self.source,
            "n_nodes": self.graph.number_of_nodes(),
            "n_edges": self.graph.number_of_edges(),
        }


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 1.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _canonical_component_key(graph: nx.Graph, nodes: Iterable[Any]) -> str:
    component = graph.subgraph(nodes).copy()
    for _, data in component.nodes(data=True):
        data["_identity_label"] = "|".join(
            (
                str(data.get("element", "?")),
                str(data.get("formal_charge", 0)),
                str(data.get("aromatic", False)),
                str(data.get("hybridization", "UNSPECIFIED")),
            )
        )
    for _, _, data in component.edges(data=True):
        data["_identity_order"] = f"{_as_float(data.get('order')):.3f}"
    return nx.weisfeiler_lehman_graph_hash(
        component,
        node_attr="_identity_label",
        edge_attr="_identity_order",
    )


def _graph_identity(graph: nx.Graph, canonical_smiles: str | None) -> str:
    if canonical_smiles is not None:
        payload = f"syn-v2-smiles:{canonical_smiles}"
    else:
        component_keys = sorted(
            _canonical_component_key(graph, nodes)
            for nodes in nx.connected_components(graph)
        )
        payload = "syn-v2-graph:" + ".".join(component_keys)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_number(element: str) -> int:
    if Chem is None:
        return 0
    try:
        return int(Chem.GetPeriodicTable().GetAtomicNumber(element))
    except Exception:
        return 0


class GraphNormalizer:
    """Create a normalized copy of a NetworkX molecular graph.

    In strict mode, missing element labels and malformed edge orders are input
    errors.  Unsupported chemical environments remain structured ``partial``
    results so callers can inspect the graph and decide whether to continue.
    """

    def __init__(self, *, strict: bool = True) -> None:
        self.strict = strict

    def normalize(
        self,
        graph: nx.Graph,
        *,
        canonical_smiles: str | None = None,
        source: str = "networkx",
    ) -> NormalizedMolecularGraph:
        """Return a normalized copy of ``graph``.

        The input graph is never mutated.  MultiGraphs and directed graphs are
        rejected because SYN's current molecular representation is an undirected
        simple graph.
        """

        if (
            not isinstance(graph, nx.Graph)
            or graph.is_directed()
            or graph.is_multigraph()
        ):
            raise GraphValidationError(
                "A molecular input must be an undirected NetworkX Graph, not a "
                "directed graph or MultiGraph."
            )

        normalized = nx.Graph()
        warnings: list[GraphWarning] = []
        for node, data in graph.nodes(data=True):
            normalized.add_node(node, **self._normalize_node(node, data, warnings))
        for left, right, data in graph.edges(data=True):
            normalized.add_edge(
                left, right, **self._normalize_edge(left, right, data, warnings)
            )

        self._assign_component_ids(normalized)
        status = "partial" if warnings else "success"
        return NormalizedMolecularGraph(
            graph=normalized,
            identity=_graph_identity(normalized, canonical_smiles),
            canonical_smiles=canonical_smiles,
            status=status,
            warnings=tuple(warnings),
            source=source,
        )

    def _normalize_node(
        self,
        node: Any,
        data: Mapping[str, Any],
        warnings: list[GraphWarning],
    ) -> dict[str, Any]:
        attrs = dict(data)
        element = attrs.get("element")
        if not isinstance(element, str) or not element:
            if self.strict:
                raise GraphValidationError(
                    f"Node {node!r} has no valid 'element' attribute."
                )
            element = "?"
            warnings.append(
                GraphWarning(
                    "MISSING_ELEMENT",
                    "Node has no valid element label; using '?'.",
                    (node,),
                )
            )
        if element not in SUPPORTED_ELEMENTS:
            warnings.append(
                GraphWarning(
                    "UNSUPPORTED_ELEMENT",
                    f"Element {element!r} is outside the initial v2 support boundary.",
                    (node,),
                )
            )

        raw_charge = attrs.get("partial_charge")
        partial_charge: float | None
        if raw_charge is None:
            partial_charge = None
        else:
            try:
                partial_charge = float(raw_charge)
            except (TypeError, ValueError):
                partial_charge = None
            if partial_charge is None or not math.isfinite(partial_charge):
                warnings.append(
                    GraphWarning(
                        "NONFINITE_PARTIAL_CHARGE",
                        "Partial charge is missing, invalid, or non-finite; it was cleared.",
                        (node,),
                    )
                )
                partial_charge = None

        atom_map = attrs.get("atom_map")
        atom_map = _as_int(atom_map) if atom_map not in (None, 0) else None
        formal_charge = _as_int(attrs.get("formal_charge", attrs.get("charge", 0)))
        radical_electrons = _as_int(
            attrs.get("radical_electrons", attrs.get("radical", 0))
        )
        if radical_electrons:
            warnings.append(
                GraphWarning(
                    "OPEN_SHELL_NOT_SUPPORTED",
                    "The initial v2 support boundary excludes open-shell atoms.",
                    (node,),
                )
            )

        attrs.update(
            {
                "element": element,
                "atomic_number": _as_int(
                    attrs.get("atomic_number"), _atomic_number(element)
                ),
                "formal_charge": formal_charge,
                "aromatic": bool(attrs.get("aromatic", False)),
                "hybridization": str(attrs.get("hybridization", "UNSPECIFIED")),
                "total_hcount": _as_int(
                    attrs.get("total_hcount", attrs.get("hcount", 0))
                ),
                "radical_electrons": radical_electrons,
                "atom_map": atom_map,
                "partial_charge": partial_charge,
                "is_in_ring": bool(
                    attrs.get("is_in_ring", attrs.get("in_ring", False))
                ),
                "ring_sizes": tuple(
                    sorted(_as_int(size) for size in attrs.get("ring_sizes", ()))
                ),
            }
        )
        return attrs

    def _normalize_edge(
        self,
        left: Any,
        right: Any,
        data: Mapping[str, Any],
        warnings: list[GraphWarning],
    ) -> dict[str, Any]:
        attrs = dict(data)
        raw_order = attrs.get("order", 1.0)
        try:
            order = float(raw_order)
        except (TypeError, ValueError):
            order = float("nan")
        if not math.isfinite(order) or order <= 0:
            if self.strict:
                raise GraphValidationError(
                    f"Edge ({left!r}, {right!r}) has invalid bond order {raw_order!r}."
                )
            warnings.append(
                GraphWarning(
                    "INVALID_BOND_ORDER",
                    "Bond order is invalid; using a single-bond fallback.",
                    (left, right),
                )
            )
            order = 1.0
        attrs.update(
            {
                "order": order,
                "aromatic": bool(attrs.get("aromatic", False)),
                "conjugated": bool(attrs.get("conjugated", False)),
                "in_ring": bool(attrs.get("in_ring", False)),
                "stereo": str(attrs.get("stereo", "NONE")),
                "bond_type": str(attrs.get("bond_type", "UNSPECIFIED")),
            }
        )
        return attrs

    @staticmethod
    def _assign_component_ids(graph: nx.Graph) -> None:
        components = list(nx.connected_components(graph))
        components.sort(
            key=lambda nodes: (_canonical_component_key(graph, nodes), len(nodes))
        )
        for component_id, nodes in enumerate(components):
            for node in nodes:
                graph.nodes[node]["component_id"] = component_id
                graph.nodes[node]["degree"] = int(graph.degree(node))


def normalize_graph(
    graph: nx.Graph,
    *,
    strict: bool = True,
    canonical_smiles: str | None = None,
    source: str = "networkx",
) -> NormalizedMolecularGraph:
    """Convenience wrapper around :class:`GraphNormalizer`."""

    return GraphNormalizer(strict=strict).normalize(
        graph, canonical_smiles=canonical_smiles, source=source
    )
