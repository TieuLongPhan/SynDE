"""Runtime scorer for a development-only v3 quantum-graph candidate."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from synde.graph.graph_schema import NormalizedMolecularGraph

from .first_order_two_d_energy import FirstOrderTwoDEnergyScorer
from .interpretable_two_d_v2 import extract_named_empirical_two_d_features
from .interpretable_two_d_v3 import (
    QUANTUM_GRAPH_FEATURE_SCHEMA_V3,
    extract_quantum_graph_v3_features,
)
from .results import MoleculeScoreResult


@dataclass(frozen=True)
class QuantumGraphV3ModelCard:
    """Metadata that prevents a development candidate being mistaken for a freeze."""

    model_name: str
    status: str
    target: str
    feature_definition_sha256: str
    training_groups: int
    training_molecules: int
    admitted_v3_families: tuple[str, ...]
    selection_method: str
    selection_alpha: float
    refit_alpha: float
    formula_centered: bool
    uses_coordinates: bool
    spent_v2_holdout_loaded: bool
    external_validation_complete: bool


class QuantumGraphV3Scorer:
    """Apply the v2+v3 linear graph score.

    The returned coordinate has no absolute thermochemical meaning. Only
    score differences inside one formula/formal-charge group are in scope.
    """

    def __init__(
        self,
        *,
        card: QuantumGraphV3ModelCard,
        weights: dict[str, float],
    ) -> None:
        if not weights:
            raise ValueError("A v3 candidate must contain at least one weight.")
        if card.uses_coordinates:
            raise ValueError("The quantum-graph v3 scorer must remain coordinate-free.")
        if card.spent_v2_holdout_loaded:
            raise ValueError("A v3 development card cannot load the spent holdout.")
        self.card = card
        self.weights = dict(weights)
        self.base_scorer = FirstOrderTwoDEnergyScorer()

    def features(self, normalized: NormalizedMolecularGraph) -> dict[str, float]:
        """Return the combined named v2 and fixed v3 graph coordinates."""
        base = self.base_scorer.score(normalized)
        features = extract_named_empirical_two_d_features(normalized, base)
        features.update(extract_quantum_graph_v3_features(normalized))
        return features

    def score(self, normalized: NormalizedMolecularGraph) -> MoleculeScoreResult:
        features = self.features(normalized)
        contributions = {
            name: float(weight * features.get(name, 0.0))
            for name, weight in self.weights.items()
        }
        return MoleculeScoreResult(
            status="success",
            score=float(sum(contributions.values())),
            units="reference_label_relative_coordinate",
            components=contributions,
            descriptors={
                "graph_identity": normalized.identity,
                "canonical_smiles": normalized.canonical_smiles,
                "comparable_within_formula_only": True,
                "selected_terms_present": sum(
                    name in features for name in self.weights
                ),
                "selected_term_count": len(self.weights),
                "development_only_nonconfirmatory": True,
            },
            warnings=("V3_DEVELOPMENT_ONLY_NONCONFIRMATORY",),
            provenance={
                "model_name": self.card.model_name,
                "mode": "quantum-chemistry-informed-labeled-graph-surrogate",
                "feature_schema": QUANTUM_GRAPH_FEATURE_SCHEMA_V3,
                "uses_coordinates": False,
                "uses_conformers": False,
                "uses_reference_labels_at_inference": False,
                "spent_v2_holdout_loaded": False,
                "external_validation_complete": (
                    self.card.external_validation_complete
                ),
            },
        )

    def score_group(
        self, candidates: list[NormalizedMolecularGraph]
    ) -> list[MoleculeScoreResult]:
        """Score one formula/formal-charge group without implying absolutes."""
        if not candidates:
            return []
        from .two_d_calibration import molecular_formula_charge_key

        keys = {molecular_formula_charge_key(candidate) for candidate in candidates}
        if len(keys) != 1:
            raise ValueError(
                "Quantum-graph v3 group scoring requires one "
                "formula/formal-charge group."
            )
        return [self.score(candidate) for candidate in candidates]

    @classmethod
    def load(cls, path: Path) -> "QuantumGraphV3Scorer":
        payload = json.loads(path.read_text(encoding="utf-8"))
        card_payload: dict[str, Any] = dict(payload["card"])
        card_payload["admitted_v3_families"] = tuple(
            card_payload["admitted_v3_families"]
        )
        return cls(
            card=QuantumGraphV3ModelCard(**card_payload),
            weights={
                str(name): float(value) for name, value in payload["weights"].items()
            },
        )

    def to_dict(self) -> dict[str, object]:
        from dataclasses import asdict

        card = asdict(self.card)
        card["admitted_v3_families"] = list(self.card.admitted_v3_families)
        return {"card": card, "weights": dict(self.weights)}


__all__ = ["QuantumGraphV3ModelCard", "QuantumGraphV3Scorer"]
