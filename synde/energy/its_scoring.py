"""Mapped ITS reaction-centre feasibility scoring in graph-model units."""

from __future__ import annotations

from dataclasses import dataclass

from synde.graph.builder import GraphBuilder
from synde.graph.its import ITSGraph, ITSGraphBuilder
from synde.graph.pair_scoring import GraphPairScorer
from synde.graph.graph_schema import NormalizedMolecularGraph

from .molecule_scoring import MoleculeScorer
from .results import ITSScoreResult


@dataclass(frozen=True)
class ITSScoringConfig:
    """Initial transparent weights for the barrier-like ITS heuristic."""

    state_delta_weight: float = 1.0
    formed_bond_weight: float = 1.0
    broken_bond_weight: float = 1.0
    order_change_weight: float = 0.5
    valence_reorganization_weight: float = 0.10
    aromaticity_reorganization_weight: float = 0.20
    fmo_weight: float = 1.0
    hsab_weight: float = 0.25
    charge_weight: float = 0.05


class ITSScorer:
    """Score mapped graph edits and actual formed-bond interaction descriptors.

    Lower values indicate a more favourable initial heuristic.  The score is a
    reaction-ranking feature, never an asserted transition-state energy.
    """

    def __init__(
        self,
        molecule_scorer: MoleculeScorer | None = None,
        pair_scorer: GraphPairScorer | None = None,
        config: ITSScoringConfig | None = None,
    ) -> None:
        self.molecule_scorer = molecule_scorer or MoleculeScorer()
        self.pair_scorer = pair_scorer or GraphPairScorer()
        self.config = config or ITSScoringConfig()
        self.builder = ITSGraphBuilder()

    def score(
        self,
        reactants: list[NormalizedMolecularGraph],
        products: list[NormalizedMolecularGraph],
        reaction_smiles: str,
    ) -> ITSScoreResult:
        state_delta = self._state_delta(reactants, products)
        try:
            its = self.builder.build(reactants, products)
        except ValueError as error:
            return ITSScoreResult(
                "unsupported",
                None,
                state_delta,
                "score",
                {},
                (),
                (),
                (str(error),),
                {
                    "model_name": "synde-its-v1",
                    "mode": "its",
                    "reaction_smiles": reaction_smiles,
                },
            )

        components = self._edit_and_reorganization(its)
        interaction, warnings = self._formed_bond_interactions(its)
        components.update(interaction)
        components["state_delta"] = state_delta
        total = self.config.state_delta_weight * state_delta
        total += components["formed_bond_penalty"] + components["broken_bond_penalty"]
        total += (
            components["order_change_penalty"]
            + components["valence_reorganization_penalty"]
        )
        total += components["aromaticity_reorganization_penalty"]
        total -= components["interaction_stabilization"]
        return ITSScoreResult(
            "partial" if warnings else "success",
            float(total),
            float(state_delta),
            "score",
            components,
            its.bond_changes,
            its.reacting_atom_maps,
            tuple(warnings),
            {
                "model_name": "synde-its-v1",
                "mode": "its",
                "reaction_smiles": reaction_smiles,
                "sign_convention": "lower_is_more_favourable",
                "weight_config": self.config.__dict__,
            },
            its,
        )

    def _state_delta(
        self,
        reactants: list[NormalizedMolecularGraph],
        products: list[NormalizedMolecularGraph],
    ) -> float:
        return float(
            sum(self.molecule_scorer.score(item).score or 0.0 for item in products)
            - sum(self.molecule_scorer.score(item).score or 0.0 for item in reactants)
        )

    def _edit_and_reorganization(self, its: ITSGraph) -> dict[str, float]:
        formed = broken = order_changed = 0.0
        for change in its.bond_changes:
            delta = abs(
                (change["product_order"] or 0.0) - (change["reactant_order"] or 0.0)
            )
            if change["kind"] == "formed":
                formed += delta
            elif change["kind"] == "broken":
                broken += delta
            else:
                order_changed += delta
        valence = 0.0
        aromaticity = 0.0
        for atom_map in its.reacting_atom_maps:
            reactant_valence = sum(
                float(data["order"])
                for _, _, data in its.reactant_graph.edges(atom_map, data=True)
            )
            product_valence = sum(
                float(data["order"])
                for _, _, data in its.product_graph.edges(atom_map, data=True)
            )
            valence += abs(product_valence - reactant_valence)
            node = its.graph.nodes[atom_map]
            aromaticity += float(node["reactant_aromatic"] != node["product_aromatic"])
        return {
            "formed_bond_penalty": self.config.formed_bond_weight * formed,
            "broken_bond_penalty": self.config.broken_bond_weight * broken,
            "order_change_penalty": self.config.order_change_weight * order_changed,
            "valence_reorganization_penalty": self.config.valence_reorganization_weight
            * valence,
            "aromaticity_reorganization_penalty": self.config.aromaticity_reorganization_weight
            * aromaticity,
        }

    def _formed_bond_interactions(
        self, its: ITSGraph
    ) -> tuple[dict[str, float], list[str]]:
        formed = {
            frozenset(change["atom_maps"])
            for change in its.bond_changes
            if change["kind"] == "formed"
        }
        if not formed:
            return {
                "fmo_interaction": 0.0,
                "hsab_interaction": 0.0,
                "charge_interaction": 0.0,
                "interaction_stabilization": 0.0,
            }, []
        normalized = GraphBuilder.from_graph(its.reactant_graph)
        candidates = self.pair_scorer.rank(normalized, top_k=10000)
        matched = [
            row
            for row in candidates
            if frozenset(
                (
                    normalized.graph.nodes[row.atom_a]["atom_map"],
                    normalized.graph.nodes[row.atom_b]["atom_map"],
                )
            )
            in formed
        ]
        warnings: list[str] = []
        if len(
            {
                frozenset(
                    (
                        normalized.graph.nodes[row.atom_a]["atom_map"],
                        normalized.graph.nodes[row.atom_b]["atom_map"],
                    )
                )
                for row in matched
            }
        ) != len(formed):
            warnings.append("FORMED_BOND_NOT_PAIR_SCORABLE")
        fmo = sum(
            row.components["fmo_A_to_B"] + row.components["fmo_B_to_A"]
            for row in matched
        )
        hsab = sum(row.components["hsab"] for row in matched)
        charge = sum(row.components["charge_complementarity"] for row in matched)
        stabilization = (
            self.config.fmo_weight * fmo
            + self.config.hsab_weight * hsab
            + self.config.charge_weight * charge
        )
        return {
            "fmo_interaction": float(fmo),
            "hsab_interaction": float(hsab),
            "charge_interaction": float(charge),
            "interaction_stabilization": float(stabilization),
        }, warnings
