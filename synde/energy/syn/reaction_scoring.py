"""Mapped reaction-SMILES validation, bond changes, and graph delta scores."""

from __future__ import annotations
from dataclasses import dataclass
from .graph_schema import NormalizedMolecularGraph
from .local_energy import element_counts
from .molecule_scoring import MoleculeScorer
from .results import ReactionScoreResult


@dataclass(frozen=True)
class ReactionScoringConfig:
    require_atom_mapping: bool = True


class ReactionScorer:
    def __init__(
        self,
        molecule_scorer: MoleculeScorer | None = None,
        config: ReactionScoringConfig | None = None,
    ) -> None:
        self.molecule_scorer = molecule_scorer or MoleculeScorer()
        self.config = config or ReactionScoringConfig()

    def score(
        self,
        reactants: list[NormalizedMolecularGraph],
        products: list[NormalizedMolecularGraph],
        reaction_smiles: str,
    ) -> ReactionScoreResult:
        reactant_results = [self.molecule_scorer.score(item) for item in reactants]
        product_results = [self.molecule_scorer.score(item) for item in products]
        warnings = [
            warning
            for result in (*reactant_results, *product_results)
            for warning in result.warnings
        ]
        mapped, mapping_warnings = self._mapped_graphs(reactants, products)
        warnings.extend(mapping_warnings)
        reactant_score = sum(result.score or 0.0 for result in reactant_results)
        product_score = sum(result.score or 0.0 for result in product_results)
        changes = self._bond_changes(*mapped) if mapped is not None else ()
        status = "partial" if warnings else "success"
        return ReactionScoreResult(
            status,
            float(product_score - reactant_score),
            "score",
            float(reactant_score),
            float(product_score),
            changes,
            {
                "molecule_delta": float(product_score - reactant_score),
                "bond_change_count": float(len(changes)),
            },
            tuple(dict.fromkeys(warnings)),
            {
                "model_name": "graph-energy-v2",
                "mode": "graph",
                "reaction_smiles": reaction_smiles,
                "sign_convention": "products_minus_reactants",
            },
        )

    def _mapped_graphs(self, reactants, products):
        left = self._merge(reactants)
        right = self._merge(products)

        left_map_list = [
            data["atom_map"]
            for _, data in left.nodes(data=True)
            if data.get("atom_map") is not None
        ]
        right_map_list = [
            data["atom_map"]
            for _, data in right.nodes(data=True)
            if data.get("atom_map") is not None
        ]

        left_maps = set(left_map_list)
        right_maps = set(right_map_list)

        left_has_none = any(
            data.get("atom_map") is None for _, data in left.nodes(data=True)
        )
        right_has_none = any(
            data.get("atom_map") is None for _, data in right.nodes(data=True)
        )
        if left_has_none or right_has_none:
            return None, (
                ("REACTION_MAPPING_MISSING",)
                if self.config.require_atom_mapping
                else ()
            )

        # Check duplicate mappings on same side
        if len(left_map_list) != len(left_maps) or len(right_map_list) != len(
            right_maps
        ):
            return None, ("REACTION_NOT_BALANCED",)

        if left_maps != right_maps:
            return None, ("REACTION_NOT_BALANCED",)

        # Check element consistency for each atom map
        left_elements = {
            data["atom_map"]: data["element"] for _, data in left.nodes(data=True)
        }
        right_elements = {
            data["atom_map"]: data["element"] for _, data in right.nodes(data=True)
        }
        for map_id, left_el in left_elements.items():
            if left_el != right_elements.get(map_id):
                return None, ("REACTION_NOT_BALANCED",)

        if element_counts(left) != element_counts(right) or sum(
            data["formal_charge"] for _, data in left.nodes(data=True)
        ) != sum(data["formal_charge"] for _, data in right.nodes(data=True)):
            return None, ("REACTION_NOT_BALANCED",)

        return (left, right), ()

    @staticmethod
    def _merge(items):
        import networkx as nx

        graph = nx.Graph()
        for component_index, item in enumerate(items):
            mapping = {node: (component_index, node) for node in item.graph.nodes()}
            graph = nx.compose(graph, nx.relabel_nodes(item.graph, mapping, copy=True))
        return graph

    @staticmethod
    def _bond_changes(left, right):
        def bonds(graph):
            return {
                frozenset(
                    (graph.nodes[a]["atom_map"], graph.nodes[b]["atom_map"])
                ): float(data["order"])
                for a, b, data in graph.edges(data=True)
            }

        lb, rb = bonds(left), bonds(right)
        changes = []
        for pair in sorted(set(lb) | set(rb), key=lambda item: sorted(item)):
            if lb.get(pair) != rb.get(pair):
                changes.append(
                    {
                        "atom_maps": sorted(pair),
                        "reactant_order": lb.get(pair),
                        "product_order": rb.get(pair),
                        "kind": (
                            "formed"
                            if pair not in lb
                            else "broken" if pair not in rb else "order_changed"
                        ),
                    }
                )
        return tuple(changes)
