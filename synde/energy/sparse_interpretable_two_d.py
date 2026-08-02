"""Runtime model for sparse, named, formula-relative 2D calibration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from synde.graph.graph_schema import NormalizedMolecularGraph

from .first_order_two_d_energy import FirstOrderTwoDEnergyScorer
from .interpretable_two_d_v2 import (
    NAMED_FEATURE_SCHEMA,
    extract_named_empirical_two_d_features,
    uncalibrated_v2_additions,
)
from .results import MoleculeScoreResult


@dataclass(frozen=True)
class SparseInterpretableTwoDModelCard:
    """Metadata needed to interpret a fitted sparse 2D equation."""

    model_name: str
    target: str
    units: str
    feature_schema: str
    feature_profile: str
    training_groups: int
    training_molecules: int
    selection_method: str
    selection_alpha: float
    selection_l1_ratio: float
    selection_folds: int
    minimum_selection_frequency: int
    refit_method: str
    refit_alpha: float
    formula_centered_training: bool
    uses_coordinates: bool
    calibrated: bool
    fitted_coefficients: bool
    version: str = "development-v1"


class SparseInterpretableTwoDScorer:
    """Apply a saved linear equation over human-readable graph terms.

    The raw coordinate is meaningful only for differences among molecules with
    the same formula and formal charge.  Group centering is unnecessary for
    ranking because it changes every member by the same constant.
    """

    def __init__(
        self,
        *,
        card: SparseInterpretableTwoDModelCard,
        weights: dict[str, float],
        training_scale: dict[str, float],
        residual_std: float | None = None,
        development_metrics: dict[str, float] | None = None,
        applicability: dict[str, Any] | None = None,
    ) -> None:
        if card.feature_schema != NAMED_FEATURE_SCHEMA:
            raise ValueError(
                f"Unsupported feature schema {card.feature_schema!r}; "
                f"expected {NAMED_FEATURE_SCHEMA!r}."
            )
        if not weights:
            raise ValueError("Sparse model must contain at least one coefficient.")
        if set(weights) != set(training_scale):
            raise ValueError("Weights and training scales must name the same terms.")
        self.card = card
        self.weights = dict(weights)
        self.training_scale = dict(training_scale)
        self.residual_std = residual_std
        self.development_metrics = dict(development_metrics or {})
        self.applicability = dict(applicability or {})
        self.base_scorer = FirstOrderTwoDEnergyScorer()

    def score(self, normalized: NormalizedMolecularGraph) -> MoleculeScoreResult:
        base = self.base_scorer.score(normalized)
        features = extract_named_empirical_two_d_features(normalized, base)
        return self._result(normalized, base, features)

    def score_group(
        self, candidates: list[NormalizedMolecularGraph]
    ) -> list[MoleculeScoreResult]:
        """Score one formula/charge group and attach centered OOD distances."""
        if not candidates:
            return []
        from .two_d_calibration import molecular_formula_charge_key

        keys = {molecular_formula_charge_key(candidate) for candidate in candidates}
        if len(keys) != 1:
            raise ValueError(
                "Sparse v2 group scoring requires one formula/formal-charge group."
            )
        bases = [self.base_scorer.score(candidate) for candidate in candidates]
        features = [
            extract_named_empirical_two_d_features(candidate, base)
            for candidate, base in zip(candidates, bases)
        ]
        matrix = [[row.get(name, 0.0) for name in self.weights] for row in features]
        means = [
            sum(row[index] for row in matrix) / len(matrix)
            for index in range(len(self.weights))
        ]
        distances = []
        for row in matrix:
            squared = [
                ((value - means[index]) / max(self.training_scale[name], 1e-12)) ** 2
                for index, (name, value) in enumerate(zip(self.weights, row))
            ]
            distances.append((sum(squared) / len(squared)) ** 0.5)
        return [
            self._result(candidate, base, row, centered_distance=distance)
            for candidate, base, row, distance in zip(
                candidates, bases, features, distances
            )
        ]

    def _result(
        self,
        normalized: NormalizedMolecularGraph,
        base: MoleculeScoreResult,
        features: dict[str, float],
        *,
        centered_distance: float | None = None,
    ) -> MoleculeScoreResult:
        contributions = {
            name: float(weight * features.get(name, 0.0))
            for name, weight in self.weights.items()
        }
        active = sum(name in features for name in self.weights)
        elements = {
            str(attrs["element"])
            for _, attrs in normalized.graph.nodes(data=True)
            if attrs["element"] != "H"
        }
        charge = sum(
            int(attrs.get("formal_charge", 0))
            for _, attrs in normalized.graph.nodes(data=True)
        )
        supported_elements = set(self.applicability.get("supported_elements", ()))
        supported_charges = set(self.applicability.get("supported_formal_charges", ()))
        warnings = []
        if supported_elements and not elements <= supported_elements:
            warnings.append("V2_UNSUPPORTED_ELEMENT")
        if supported_charges and charge not in supported_charges:
            warnings.append("V2_UNSUPPORTED_FORMAL_CHARGE")
        threshold = self.applicability.get("centered_feature_distance_q99")
        if (
            centered_distance is not None
            and threshold is not None
            and centered_distance > float(threshold)
        ):
            warnings.append("V2_OUTSIDE_CENTERED_FEATURE_DISTANCE")
        uncalibrated_additions = uncalibrated_v2_additions(normalized, base)
        return MoleculeScoreResult(
            status="partial" if warnings else "success",
            score=float(sum(contributions.values())),
            units=self.card.units,
            components=contributions,
            descriptors={
                "graph_identity": normalized.identity,
                "canonical_smiles": normalized.canonical_smiles,
                "comparable_within_formula_only": True,
                "underlying_first_order_status": base.status,
                "underlying_first_order_warnings": base.warnings,
                "underlying_first_order_score": base.score,
                "prespecified_uncalibrated_v2_score": (
                    float(base.score + sum(uncalibrated_additions.values()))
                    if base.score is not None
                    else None
                ),
                "selected_terms_present": active,
                "selected_term_count": len(self.weights),
                "centered_standardized_feature_distance": centered_distance,
                "applicability_threshold_q99": threshold,
                "training_residual_std": self.residual_std,
                "development_metrics": self.development_metrics,
            },
            warnings=tuple(warnings),
            provenance={
                "model_name": self.card.model_name,
                "mode": "labeled-graph-only-formula-relative",
                "calibrated": True,
                "fitted_coefficients": True,
                "uses_coordinates": False,
                "uses_conformers": False,
                "uses_xtb": False,
                "uses_ord_labels_at_inference": False,
                "feature_schema": self.card.feature_schema,
                "selection_method": self.card.selection_method,
                "permanent_holdout_evaluated": False,
                "holdout_tuned": False,
            },
        )

    @classmethod
    def load(cls, path: Path) -> SparseInterpretableTwoDScorer:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            card=SparseInterpretableTwoDModelCard(**payload["card"]),
            weights={
                str(name): float(value) for name, value in payload["weights"].items()
            },
            training_scale={
                str(name): float(value)
                for name, value in payload["training_scale"].items()
            },
            residual_std=payload.get("training_residual_std"),
            development_metrics=payload.get("development_metrics"),
            applicability=payload.get("applicability"),
        )

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return {
            "card": asdict(self.card),
            "weights": self.weights,
            "training_scale": self.training_scale,
            "training_residual_std": self.residual_std,
            "development_metrics": self.development_metrics,
            "applicability": self.applicability,
        }


__all__ = [
    "SparseInterpretableTwoDModelCard",
    "SparseInterpretableTwoDScorer",
]
