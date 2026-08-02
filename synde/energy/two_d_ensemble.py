"""Explicit ensembles of compatible formula-relative 2D models."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from synde.graph.graph_schema import NormalizedMolecularGraph

from .molecule_scoring import MoleculeScorer
from .two_d_calibration import (
    FormulaRelativeModelCard,
    FormulaRelativePrediction,
    FormulaRelativeRidgeCalibrator,
)


@dataclass(frozen=True)
class FormulaRelativeEnsembleCard:
    """Interpretation metadata for a fixed weighted ensemble."""

    model_name: str
    target: str
    units: str
    weights: tuple[float, ...]
    member_profiles: tuple[str, ...]
    feature_schema: str
    fitted_ensemble_weights: bool = False


@dataclass(frozen=True)
class FormulaRelativeEnsemblePrediction:
    """One conservative aggregate plus its inspectable member predictions."""

    value: float
    units: str
    target: str
    uncertainty: None
    applicability: float
    feature_distance: float
    comparable_within_formula_only: bool
    member_predictions: tuple[FormulaRelativePrediction, ...]
    model_card: FormulaRelativeEnsembleCard


class FormulaRelativeEnsemble:
    """Average compatible formula-relative models using fixed explicit weights."""

    def __init__(
        self,
        models: Iterable[FormulaRelativeRidgeCalibrator],
        weights: Iterable[float] | None = None,
        *,
        model_name: str = "synde-formula-relative-2d-ensemble",
    ) -> None:
        self.models = tuple(models)
        if not self.models:
            raise ValueError("An ensemble requires at least one model.")
        raw_weights = (
            np.ones(len(self.models), dtype=float)
            if weights is None
            else np.asarray(tuple(weights), dtype=float)
        )
        if len(raw_weights) != len(self.models):
            raise ValueError("Ensemble weights must match the number of models.")
        if not np.all(np.isfinite(raw_weights)) or np.any(raw_weights < 0):
            raise ValueError("Ensemble weights must be finite and non-negative.")
        if float(raw_weights.sum()) <= 0:
            raise ValueError("At least one ensemble weight must be positive.")
        self.weights = tuple((raw_weights / raw_weights.sum()).tolist())
        cards = tuple(self._card(model) for model in self.models)
        targets = {card.target for card in cards}
        units = {card.units for card in cards}
        schemas = {card.feature_schema for card in cards}
        if len(targets) != 1 or len(units) != 1 or len(schemas) != 1:
            raise ValueError(
                "Ensemble members must share target, units, and feature schema."
            )
        self.card = FormulaRelativeEnsembleCard(
            model_name=model_name,
            target=next(iter(targets)),
            units=next(iter(units)),
            weights=self.weights,
            member_profiles=tuple(card.feature_profile for card in cards),
            feature_schema=next(iter(schemas)),
        )

    @classmethod
    def load(cls, path: Path) -> FormulaRelativeEnsemble:
        """Load a manifest whose member paths are relative to the manifest."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        models = [
            FormulaRelativeRidgeCalibrator.load(path.parent / member)
            for member in payload["members"]
        ]
        return cls(
            models,
            payload.get("weights"),
            model_name=payload.get("model_name", "synde-formula-relative-2d-ensemble"),
        )

    def predict_group(
        self, features: Iterable[dict[str, float]]
    ) -> list[FormulaRelativeEnsemblePrediction]:
        """Predict an already validated same-formula feature group."""
        rows = list(features)
        member_groups = [model.predict_group(rows) for model in self.models]
        return self._combine(member_groups)

    def predict_graph_group(
        self,
        graphs: Iterable[NormalizedMolecularGraph],
        scorer: MoleculeScorer | None = None,
    ) -> list[FormulaRelativeEnsemblePrediction]:
        """Validate and predict one graph group with every member."""
        candidates = list(graphs)
        member_groups = [
            model.predict_graph_group(candidates, scorer) for model in self.models
        ]
        return self._combine(member_groups)

    def predict_smiles_group(
        self,
        smiles: Iterable[str],
        scorer: MoleculeScorer | None = None,
    ) -> list[FormulaRelativeEnsemblePrediction]:
        """Build and predict a same-formula group directly from SMILES."""
        from synde.graph.builder import GraphBuilder

        graphs = [GraphBuilder.from_smiles(value) for value in smiles]
        return self.predict_graph_group(graphs, scorer)

    def _combine(
        self, member_groups: list[list[FormulaRelativePrediction]]
    ) -> list[FormulaRelativeEnsemblePrediction]:
        lengths = {len(group) for group in member_groups}
        if len(lengths) != 1:
            raise RuntimeError("Ensemble members returned different group sizes.")
        size = next(iter(lengths))
        combined = []
        for index in range(size):
            members = tuple(group[index] for group in member_groups)
            combined.append(
                FormulaRelativeEnsemblePrediction(
                    value=float(
                        sum(
                            weight * prediction.value
                            for weight, prediction in zip(self.weights, members)
                        )
                    ),
                    units=self.card.units,
                    target=self.card.target,
                    uncertainty=None,
                    applicability=min(row.applicability for row in members),
                    feature_distance=max(row.feature_distance for row in members),
                    comparable_within_formula_only=True,
                    member_predictions=members,
                    model_card=self.card,
                )
            )
        return combined

    @staticmethod
    def _card(model: FormulaRelativeRidgeCalibrator) -> FormulaRelativeModelCard:
        if model.card is None:
            raise RuntimeError("Every ensemble member must be fitted or loaded.")
        return model.card
