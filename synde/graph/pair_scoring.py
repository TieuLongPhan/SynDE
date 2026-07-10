"""Explainable graph-only reactive-pair ranking for SYN v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from .frontier import directional_fmo, frontiers_from_graph_result
from .generalized_huckel import GeneralizedHuckel, HuckelParameters
from .hsab import hsab_descriptor, local_hsab_compatibility
from .pi_system import assign_pi_systems
from .graph_schema import NormalizedMolecularGraph


@dataclass(frozen=True)
class PairScoringConfig:
    gap_floor: float = 0.05
    fmo_weight: float = 1.0
    hsab_weight: float = 0.25
    charge_weight: float = 0.05
    accessibility_weight: float = 0.1
    proposed_bond_order: float = 1.0


@dataclass(frozen=True)
class PairScoreResult:
    atom_a: Any
    atom_b: Any
    component_a: int
    component_b: int
    pair_compatibility_score: float
    direction: str
    components: dict[str, float]
    warnings: tuple[str, ...]
    equivalence_key: tuple[Any, ...]


@dataclass(frozen=True)
class PairScoreGroup:
    """Symmetry-equivalent candidate pairs represented by one score row."""

    representative: PairScoreResult
    pairs: tuple[tuple[Any, Any], ...]
    equivalence_key: tuple[Any, ...]

    @property
    def equivalent_pairs(self) -> tuple[tuple[Any, Any], ...]:
        """Backward-compatible alias for the grouped candidate pairs."""
        return self.pairs


class GraphPairScorer:
    """Rank cross-component pi atom pairs; scores are not interaction energies."""

    def __init__(
        self,
        config: PairScoringConfig | None = None,
        parameters: HuckelParameters | None = None,
    ) -> None:
        self.config = config or PairScoringConfig()
        self.huckel = GeneralizedHuckel(parameters)

    def rank(
        self,
        normalized: NormalizedMolecularGraph,
        *,
        top_k: int = 50,
        group_equivalence: bool = False,
    ) -> list[PairScoreResult] | list[PairScoreGroup]:
        assignment = assign_pi_systems(normalized)
        huckel = self.huckel.solve(assignment)
        frontiers = frontiers_from_graph_result(assignment.pi_graph, huckel)
        hsab = {
            frontier.component_id: hsab_descriptor(frontier) for frontier in frontiers
        }
        results: list[PairScoreResult] = []
        for index, left in enumerate(frontiers):
            for right in frontiers[index + 1 :]:
                for atom_a in left.nodes:
                    for atom_b in right.nodes:
                        if not self._valence_allowed(
                            normalized.graph, atom_a
                        ) or not self._valence_allowed(normalized.graph, atom_b):
                            continue
                        forward = directional_fmo(
                            left, right, atom_a, atom_b, gap_floor=self.config.gap_floor
                        )
                        backward = directional_fmo(
                            right, left, atom_b, atom_a, gap_floor=self.config.gap_floor
                        )
                        fmo = forward.score + backward.score
                        hsab_score = local_hsab_compatibility(
                            hsab[left.component_id],
                            hsab[right.component_id],
                            atom_a,
                            atom_b,
                        ) + local_hsab_compatibility(
                            hsab[right.component_id],
                            hsab[left.component_id],
                            atom_b,
                            atom_a,
                        )
                        charge = self._charge_complementarity(
                            normalized.graph, atom_a, atom_b
                        )
                        accessibility = self._accessibility(
                            normalized.graph, assignment.pi_graph, atom_a
                        ) + self._accessibility(
                            normalized.graph, assignment.pi_graph, atom_b
                        )
                        score = (
                            self.config.fmo_weight * fmo
                            + self.config.hsab_weight * hsab_score
                            + self.config.charge_weight * charge
                            + self.config.accessibility_weight * accessibility
                        )
                        warnings = tuple(
                            w for w in (forward.warning, backward.warning) if w
                        )
                        direction = (
                            "A_to_B"
                            if forward.score > backward.score
                            else "B_to_A" if backward.score > forward.score else "mixed"
                        )
                        components = {
                            "fmo_A_to_B": forward.score,
                            "fmo_B_to_A": backward.score,
                            "hsab": hsab_score,
                            "charge_complementarity": charge,
                            "accessibility": accessibility,
                        }
                        results.append(
                            PairScoreResult(
                                atom_a,
                                atom_b,
                                left.component_id,
                                right.component_id,
                                float(score),
                                direction,
                                components,
                                warnings,
                                self._equivalence_key(
                                    normalized.graph,
                                    assignment.pi_graph,
                                    atom_a,
                                    atom_b,
                                    left,
                                    right,
                                ),
                            )
                        )

        # Sort results by score first
        results.sort(
            key=lambda item: (
                -item.pair_compatibility_score,
                repr(item.atom_a),
                repr(item.atom_b),
            )
        )

        if group_equivalence:
            groups: dict[tuple[Any, ...], list[PairScoreResult]] = {}
            for res in results:
                groups.setdefault(res.equivalence_key, []).append(res)

            grouped_results: list[PairScoreGroup] = []
            for eq_key, res_list in groups.items():
                res_list.sort(key=lambda r: (repr(r.atom_a), repr(r.atom_b)))
                rep = res_list[0]
                pairs = tuple((r.atom_a, r.atom_b) for r in res_list)
                grouped_results.append(
                    PairScoreGroup(
                        representative=rep, pairs=pairs, equivalence_key=eq_key
                    )
                )
            # Sort grouped results by representative's score
            grouped_results.sort(
                key=lambda item: (
                    -item.representative.pair_compatibility_score,
                    repr(item.representative.atom_a),
                    repr(item.representative.atom_b),
                )
            )
            return grouped_results[:top_k]

        return results[:top_k]

    @staticmethod
    def _valence_allowed(graph: nx.Graph, node: Any) -> bool:
        element = graph.nodes[node]["element"]
        if graph.nodes[node].get("radical_electrons", 0):
            return False

        max_valence = {
            "H": 1,
            "B": 4,
            "C": 4,
            "N": 4,
            "O": 3,
            "F": 1,
            "Si": 4,
            "P": 5,
            "S": 6,
            "Cl": 1,
            "Br": 1,
            "I": 1,
        }
        if element not in max_valence:
            return False

        current_valence = sum(
            graph[node][neighbor].get("order", 1.0) for neighbor in graph[node]
        )
        current_valence += graph.nodes[node].get("total_hcount", 0)

        return (current_valence + 1.0) <= max_valence[element]

    @staticmethod
    def _charge_complementarity(graph: nx.Graph, left: Any, right: Any) -> float:
        q_left = graph.nodes[left].get("partial_charge")
        q_right = graph.nodes[right].get("partial_charge")
        if q_left is None:
            q_left = graph.nodes[left]["formal_charge"]
        if q_right is None:
            q_right = graph.nodes[right]["formal_charge"]
        return float(-float(q_left) * float(q_right))

    @staticmethod
    def _accessibility(graph: nx.Graph, pi_graph: nx.Graph, node: Any) -> float:
        substituents = sum(
            1 for neighbor in graph.neighbors(node) if neighbor not in pi_graph
        )
        return float(1.0 / (1.0 + substituents))

    @staticmethod
    def _equivalence_key(
        graph: nx.Graph,
        pi_graph: nx.Graph,
        left: Any,
        right: Any,
        left_frontier,
        right_frontier,
    ) -> tuple[Any, ...]:
        def key(node: Any, frontier) -> tuple[Any, ...]:
            attrs = graph.nodes[node]
            return (
                attrs["element"],
                attrs["formal_charge"],
                round(frontier.homo_density.get(node, 0.0), 10),
                round(frontier.lumo_density.get(node, 0.0), 10),
                GraphPairScorer._accessibility(graph, pi_graph, node),
            )

        return key(left, left_frontier), key(right, right_frontier)
