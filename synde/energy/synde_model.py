"""Canonical runtime for the frozen SynDE formula-relative ranking model.

The coefficient artifact was frozen before external evaluation.  Its original
model card is intentionally preserved; a separate validation record upgrades
the runtime status only after the two hashes have been checked together.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
from importlib.resources import as_file, files
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from synde.graph.graph_schema import NormalizedMolecularGraph

from .first_order_two_d_energy import FirstOrderTwoDEnergyScorer
from .interpretable_two_d_v2 import extract_named_empirical_two_d_features
from .interpretable_two_d_v3 import extract_quantum_graph_v3_features
from .results import MoleculeScoreResult


FROZEN_MODEL_SHA256 = (
    "6b04dd12dd22643662f0ea894266bde07d8750b7506447da7545bc675ed0c166"
)


@dataclass(frozen=True)
class SynDEModelCard:
    """Immutable calibration-time provenance stored with the equation."""

    model_name: str
    status_at_freeze: str
    target: str
    rdkit_generation_version: str
    xtb_generation_version: str
    feature_definition_sha256: str
    training_source: str
    training_source_sha256: str
    training_groups: int
    training_molecules: int
    selected_profile: str
    selection_method: str
    selection_alpha: float | None
    refit_alpha: float
    formula_centered: bool
    uses_coordinates: bool
    training_labels_previously_used_for_prior_model_evaluation: bool
    external_validation_complete_at_freeze: bool

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SynDEModelCard":
        """Translate the immutable artifact's historical field names."""
        translated = dict(payload)
        if "status" in translated:
            translated["status_at_freeze"] = translated.pop("status")
        if "v3_external_labels_used_for_v4_development" in translated:
            translated[
                "training_labels_previously_used_for_prior_model_evaluation"
            ] = translated.pop("v3_external_labels_used_for_v4_development")
        if "untouched_v4_test_complete" in translated:
            translated["external_validation_complete_at_freeze"] = translated.pop(
                "untouched_v4_test_complete"
            )
        return cls(**translated)


@dataclass(frozen=True)
class SynDEValidationRecord:
    """Post-freeze external validation attached to the immutable equation."""

    protocol: str
    executed_on: str
    model_sha256: str
    external_formula_and_graph_disjoint: bool
    external_groups: int
    external_molecules: int
    group_scoreability: float
    mean_group_pearson: float
    mean_group_spearman: float
    mean_pairwise_concordance: float
    top1_accuracy: float
    external_labels_used_for_fit_or_tuning: bool
    post_external_model_revision: bool
    all_prespecified_external_performance_criteria_met: bool
    claim_boundary: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SynDEValidationRecord":
        return cls(**{name: payload[name] for name in cls.__dataclass_fields__})

    @property
    def confirms_model(self) -> bool:
        """Return whether the record satisfies the frozen validation boundary."""
        return (
            self.model_sha256 == FROZEN_MODEL_SHA256
            and self.external_formula_and_graph_disjoint
            and not self.external_labels_used_for_fit_or_tuning
            and not self.post_external_model_revision
            and self.all_prespecified_external_performance_criteria_met
        )


class SynDEScorer:
    """Apply the externally validated SynDE score within one formula group."""

    CALIBRATION_ONLY_WARNING = "SYNDE_CALIBRATION_ONLY_NO_VALIDATION_RECORD"
    DISTANCE_WARNING = "SYNDE_OUTSIDE_TRAINING_Q99_FEATURE_DISTANCE"

    def __init__(
        self,
        *,
        card: SynDEModelCard,
        weights: dict[str, float],
        feature_scales: dict[str, float],
        training_distance_q99: float,
        validation: SynDEValidationRecord | None = None,
        model_sha256: str | None = None,
    ) -> None:
        if not weights:
            raise ValueError("A SynDE model must contain at least one weight.")
        if set(weights) != set(feature_scales):
            raise ValueError("SynDE weights and feature scales must have identical keys.")
        if any(not math.isfinite(value) or value <= 0 for value in feature_scales.values()):
            raise ValueError("Every SynDE feature scale must be finite and positive.")
        if not math.isfinite(training_distance_q99) or training_distance_q99 <= 0:
            raise ValueError("The SynDE q99 feature distance must be finite and positive.")
        if card.uses_coordinates:
            raise ValueError("The SynDE scorer must remain coordinate-free.")
        if card.external_validation_complete_at_freeze:
            raise ValueError(
                "The immutable calibration card cannot claim post-freeze validation."
            )
        if card.status_at_freeze != "development_only_no_untouched_test":
            raise ValueError("The immutable calibration card has an unexpected status.")
        if validation is not None:
            if model_sha256 != validation.model_sha256:
                raise ValueError("The validation record does not match the model file.")
            if not validation.confirms_model:
                raise ValueError("The supplied record does not confirm the frozen model.")
        self.card = card
        self.weights = dict(weights)
        self.feature_scales = dict(feature_scales)
        self.training_distance_q99 = float(training_distance_q99)
        self.validation = validation
        self.model_sha256 = model_sha256
        self.base_scorer = FirstOrderTwoDEnergyScorer()

    @property
    def externally_validated(self) -> bool:
        """Whether a matching successful external-validation record was loaded."""
        return self.validation is not None and self.validation.confirms_model

    def features(self, normalized: NormalizedMolecularGraph) -> dict[str, float]:
        """Return the fixed named two-dimensional feature library."""
        first_order = self.base_scorer.score(normalized)
        features = extract_named_empirical_two_d_features(normalized, first_order)
        features.update(extract_quantum_graph_v3_features(normalized))
        return features

    def _score_from_features(
        self,
        normalized: NormalizedMolecularGraph,
        features: dict[str, float],
    ) -> MoleculeScoreResult:
        contributions = {
            name: float(weight * features.get(name, 0.0))
            for name, weight in self.weights.items()
        }
        warnings = () if self.externally_validated else (self.CALIBRATION_ONLY_WARNING,)
        return MoleculeScoreResult(
            status="success",
            score=float(sum(contributions.values())),
            units="GFN2-xTB_formula_relative_model_coordinate",
            components=contributions,
            descriptors={
                "graph_identity": normalized.identity,
                "canonical_smiles": normalized.canonical_smiles,
                "comparable_within_formula_only": True,
                "selected_terms_present": sum(name in features for name in self.weights),
                "selected_term_count": len(self.weights),
                "externally_validated": self.externally_validated,
            },
            warnings=warnings,
            provenance={
                "model_name": "SynDE",
                "frozen_model_sha256": self.model_sha256,
                "target": self.card.target,
                "mode": "named-linear-formula-relative-GFN2-xTB-surrogate",
                "uses_coordinates": False,
                "uses_conformers": False,
                "uses_reference_labels_at_inference": False,
                "calibration_card_status_at_freeze": self.card.status_at_freeze,
                "external_validation_loaded": self.externally_validated,
                "external_validation_protocol": (
                    self.validation.protocol if self.validation is not None else None
                ),
            },
        )

    def score(self, normalized: NormalizedMolecularGraph) -> MoleculeScoreResult:
        """Score one graph without claiming cross-formula comparability."""
        features = self.features(normalized)
        return self._score_from_features(normalized, features)

    def score_group(
        self, candidates: list[NormalizedMolecularGraph]
    ) -> list[MoleculeScoreResult]:
        """Score one formula group and attach descriptive distance warnings."""
        if not candidates:
            return []
        from .two_d_calibration import molecular_formula_charge_key

        keys = {molecular_formula_charge_key(candidate) for candidate in candidates}
        if len(keys) != 1:
            raise ValueError(
                "SynDE group scoring requires one formula/formal-charge group."
            )
        feature_rows = [self.features(candidate) for candidate in candidates]
        matrix = np.asarray(
            [
                [float(features.get(name, 0.0)) for name in self.weights]
                for features in feature_rows
            ],
            dtype=float,
        )
        matrix -= matrix.mean(axis=0, keepdims=True)
        scales = np.asarray(
            [self.feature_scales[name] for name in self.weights], dtype=float
        )
        distances = np.sqrt(np.mean((matrix / scales) ** 2, axis=1))
        output = []
        for candidate, features, distance in zip(candidates, feature_rows, distances):
            result = self._score_from_features(candidate, features)
            warnings = list(result.warnings)
            if distance > self.training_distance_q99:
                warnings.append(self.DISTANCE_WARNING)
            output.append(
                replace(
                    result,
                    descriptors=result.descriptors
                    | {
                        "centered_selected_feature_distance": float(distance),
                        "training_distance_q99": self.training_distance_q99,
                        "distance_warning_is_not_validated_abstention": True,
                    },
                    warnings=tuple(warnings),
                )
            )
        return output

    @classmethod
    def load(
        cls,
        model_path: Path,
        validation_path: Path | None = None,
    ) -> "SynDEScorer":
        """Load a model and, optionally, its matching validation record."""
        model_bytes = model_path.read_bytes()
        model_sha256 = hashlib.sha256(model_bytes).hexdigest()
        payload = json.loads(model_bytes)
        validation = None
        if validation_path is not None:
            validation_payload = json.loads(validation_path.read_text(encoding="utf-8"))
            validation = SynDEValidationRecord.from_dict(validation_payload)
        distance: dict[str, Any] = payload["selected_feature_distance"]
        return cls(
            card=SynDEModelCard.from_dict(payload["card"]),
            weights={str(name): float(value) for name, value in payload["weights"].items()},
            feature_scales={
                str(name): float(value)
                for name, value in payload["feature_scales"].items()
            },
            training_distance_q99=float(distance["training_q99"]),
            validation=validation,
            model_sha256=model_sha256,
        )

    @classmethod
    def load_default(cls) -> "SynDEScorer":
        """Load the packaged frozen model with its successful validation record."""
        package_root = files("synde")
        model_resource = package_root.joinpath("models/synde_frozen_model.json")
        validation_resource = package_root.joinpath(
            "models/synde_external_validation.json"
        )
        if not model_resource.is_file() or not validation_resource.is_file():
            raise FileNotFoundError(
                "The installed SynDE package does not contain its frozen model "
                "and external-validation record. Reinstall the package from a "
                "complete distribution."
            )
        with as_file(model_resource) as model_path, as_file(
            validation_resource
        ) as validation_path:
            return cls.load(model_path, validation_path)

    def to_dict(self) -> dict[str, object]:
        """Serialize the calibration artifact without rewriting its frozen card."""
        return {
            "card": asdict(self.card),
            "weights": dict(self.weights),
            "feature_scales": dict(self.feature_scales),
            "selected_terms": list(self.weights),
            "selected_feature_distance": {
                "definition": (
                    "RMS standardized feature distance after centering within "
                    "candidate formula group"
                ),
                "training_q99": self.training_distance_q99,
                "warning_only_not_validated_abstention": True,
            },
        }


__all__ = [
    "FROZEN_MODEL_SHA256",
    "SynDEModelCard",
    "SynDEScorer",
    "SynDEValidationRecord",
]
