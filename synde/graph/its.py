"""Atom-mapped imaginary transition-state graph construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from .graph_schema import NormalizedMolecularGraph


@dataclass(frozen=True)
class ITSGraph:
    """Mapped union graph whose edges retain reactant/product bond orders."""

    graph: nx.Graph
    reacting_atom_maps: tuple[int, ...]
    bond_changes: tuple[dict[str, Any], ...]
    reactant_graph: nx.Graph
    product_graph: nx.Graph


class ITSGraphBuilder:
    """Build an ITS graph only when atom mapping gives an unambiguous correspondence."""

    def build(
        self,
        reactants: list[NormalizedMolecularGraph],
        products: list[NormalizedMolecularGraph],
        *,
        native_its: nx.Graph | None = None,
    ) -> ITSGraph:
        left = self._mapped_union(reactants, side="reactant")
        right = self._mapped_union(products, side="product")
        if set(left.nodes) != set(right.nodes):
            raise ValueError("REACTION_NOT_BALANCED")
        for atom_map in left.nodes:
            if left.nodes[atom_map]["element"] != right.nodes[atom_map]["element"]:
                raise ValueError("REACTION_NOT_BALANCED")

        its = nx.Graph()
        for atom_map in sorted(left.nodes):
            reactant = dict(left.nodes[atom_map])
            product = dict(right.nodes[atom_map])
            attrs = dict(reactant)
            attrs.update(
                atom_map=atom_map,
                reactant_formal_charge=reactant["formal_charge"],
                product_formal_charge=product["formal_charge"],
                reactant_aromatic=reactant["aromatic"],
                product_aromatic=product["aromatic"],
            )
            its.add_node(atom_map, **attrs)

        left_bonds = self._bonds(left)
        right_bonds = self._bonds(right)
        changes: list[dict[str, Any]] = []
        for pair in sorted(
            set(left_bonds) | set(right_bonds), key=lambda item: tuple(sorted(item))
        ):
            reactant_order = left_bonds.get(pair)
            product_order = right_bonds.get(pair)
            kind = self._kind(reactant_order, product_order)
            atom_maps = tuple(sorted(pair))
            attrs = {
                "reactant_order": reactant_order,
                "product_order": product_order,
                "edit_type": kind,
                "order": product_order if product_order is not None else reactant_order,
            }
            its.add_edge(*atom_maps, **attrs)
            if kind != "unchanged":
                changes.append(
                    {
                        "atom_maps": list(atom_maps),
                        "reactant_order": reactant_order,
                        "product_order": product_order,
                        "kind": kind,
                    }
                )

        center = tuple(
            sorted({atom_map for change in changes for atom_map in change["atom_maps"]})
        )
        return ITSGraph(native_its or its, center, tuple(changes), left, right)

    @staticmethod
    def _mapped_union(items: list[NormalizedMolecularGraph], *, side: str) -> nx.Graph:
        graph = nx.Graph()
        for item in items:
            for _, attrs in item.graph.nodes(data=True):
                atom_map = attrs.get("atom_map")
                if atom_map is None:
                    raise ValueError("REACTION_MAPPING_MISSING")
                if atom_map in graph:
                    raise ValueError("REACTION_NOT_BALANCED")
                graph.add_node(int(atom_map), **dict(attrs))
            for left, right, attrs in item.graph.edges(data=True):
                left_map = item.graph.nodes[left].get("atom_map")
                right_map = item.graph.nodes[right].get("atom_map")
                if left_map is None or right_map is None:
                    raise ValueError("REACTION_MAPPING_MISSING")
                graph.add_edge(int(left_map), int(right_map), **dict(attrs))
        return graph

    @staticmethod
    def _bonds(graph: nx.Graph) -> dict[frozenset[int], float]:
        return {
            frozenset((int(left), int(right))): float(attrs["order"])
            for left, right, attrs in graph.edges(data=True)
        }

    @staticmethod
    def _kind(reactant_order: float | None, product_order: float | None) -> str:
        if reactant_order is None:
            return "formed"
        if product_order is None:
            return "broken"
        return "unchanged" if reactant_order == product_order else "order_changed"
