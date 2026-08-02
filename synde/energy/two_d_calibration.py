"""Calibrated, geometry-free ranking within molecular-formula groups.

The model is trained on features and targets centered independently within
each formula group.  Its output therefore has a deliberately narrow meaning:
differences between molecules with the same formula are meaningful, while the
absolute value and comparisons across formulas are not.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import networkx as nx

from synde.graph.graph_schema import NormalizedMolecularGraph

from .molecule_scoring import MoleculeScorer
from .two_d_features import TWO_D_FEATURE_SCHEMA, extract_two_d_features


@dataclass(frozen=True)
class FormulaCalibrationRecord:
    """One labeled 2D structure used for formula-relative calibration."""

    identifier: str
    formula_group: str
    features: dict[str, float]
    target: float
    provenance: dict[str, Any]


@dataclass(frozen=True)
class FormulaRelativeModelCard:
    """Metadata needed to interpret a formula-relative model."""

    model_name: str
    target: str
    units: str
    feature_names: tuple[str, ...]
    ridge_alpha: float
    training_count: int
    training_groups: int
    centered_within_formula: bool = True
    geometry_features: bool = False
    feature_schema: str = TWO_D_FEATURE_SCHEMA
    feature_profile: str = "custom"
    selection_objective: str = "unspecified"
    uncertainty_kind: str = "training_residual_rmse"
    version: str = "1"


@dataclass(frozen=True)
class FormulaRelativePrediction:
    """A calibrated value that is comparable only within one formula group."""

    value: float
    units: str
    target: str
    uncertainty: float | None
    applicability: float
    feature_distance: float
    comparable_within_formula_only: bool
    model_card: FormulaRelativeModelCard


class FormulaRelativeRidgeCalibrator:
    """Transparent ridge model fitted to within-formula energy differences."""

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        target: str = "formula-relative energy",
        units: str = "target units",
        feature_profile: str = "custom",
        selection_objective: str = "unspecified",
    ) -> None:
        if alpha < 0:
            raise ValueError("Ridge alpha must be non-negative.")
        self.alpha = float(alpha)
        self.target = target
        self.units = units
        self.feature_profile = feature_profile
        self.selection_objective = selection_objective
        self.feature_names: tuple[str, ...] = ()
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.active: np.ndarray | None = None
        self.residual_std: float | None = None
        self.card: FormulaRelativeModelCard | None = None

    def fit(
        self, records: Iterable[FormulaCalibrationRecord]
    ) -> FormulaRelativeModelCard:
        """Fit after centering features and labels within every formula group."""
        rows = list(records)
        grouped: dict[str, list[FormulaCalibrationRecord]] = defaultdict(list)
        for row in rows:
            grouped[row.formula_group].append(row)
        usable = [group for group in grouped.values() if len(group) >= 2]
        if not usable:
            raise ValueError(
                "At least one formula group containing two records is required."
            )

        self.feature_names = tuple(
            sorted({name for group in usable for row in group for name in row.features})
        )
        if not self.feature_names:
            raise ValueError("Calibration records contain no features.")

        centered_x: list[np.ndarray] = []
        centered_y: list[np.ndarray] = []
        for group in usable:
            x = self._matrix(row.features for row in group)
            y = np.asarray([row.target for row in group], dtype=float)
            relative_y = y - y.mean()
            centered_x.append(x - x.mean(axis=0))
            centered_y.append(relative_y)
        if not centered_y:
            raise ValueError("No formula group has varying calibration targets.")
        x = np.vstack(centered_x)
        y = np.concatenate(centered_y)

        self.mean = x.mean(axis=0)
        raw_scale = x.std(axis=0)
        self.active = raw_scale > 1e-12
        if not np.any(self.active):
            raise ValueError("No features vary within the supplied formula groups.")
        self.scale = np.where(self.active, raw_scale, 1.0)
        z = (x[:, self.active] - self.mean[self.active]) / self.scale[self.active]
        penalty = np.eye(z.shape[1], dtype=float) * self.alpha
        active_weights = np.linalg.solve(z.T @ z + penalty, z.T @ y)
        self.weights = np.zeros(len(self.feature_names), dtype=float)
        self.weights[self.active] = active_weights
        residual = y - z @ active_weights
        self.residual_std = float(np.sqrt(np.mean(residual**2)))
        training_count = sum(len(values) for values in centered_y)
        self.card = FormulaRelativeModelCard(
            model_name="synde-formula-relative-2d-ridge",
            target=self.target,
            units=self.units,
            feature_names=self.feature_names,
            ridge_alpha=self.alpha,
            training_count=training_count,
            training_groups=len(centered_y),
            feature_profile=self.feature_profile,
            selection_objective=self.selection_objective,
        )
        return self.card

    def predict(self, features: dict[str, float]) -> FormulaRelativePrediction:
        """Predict a ranking coordinate; compare only within the same formula."""
        return self._prediction_from_vector(self._vector(features))

    def _prediction_from_vector(self, x: np.ndarray) -> FormulaRelativePrediction:
        """Predict from a vector already expressed in the intended reference frame."""
        self._require_fitted()
        assert self.active is not None
        assert self.card is not None
        assert self.mean is not None
        assert self.scale is not None
        assert self.weights is not None
        z = (x[self.active] - self.mean[self.active]) / self.scale[self.active]
        value = float(z @ self.weights[self.active])
        feature_distance = float(np.linalg.norm(z) / np.sqrt(len(z)))
        applicability = 1.0 / (1.0 + feature_distance)
        return FormulaRelativePrediction(
            value=value,
            units=self.units,
            target=self.target,
            uncertainty=self.residual_std,
            applicability=applicability,
            feature_distance=feature_distance,
            comparable_within_formula_only=True,
            model_card=self.card,
        )

    def predict_group(
        self, features: Iterable[dict[str, float]]
    ) -> list[FormulaRelativePrediction]:
        """Center one formula group's features before prediction and distance."""
        rows = list(features)
        if not rows:
            return []
        matrix = self._matrix(rows)
        centered = matrix - matrix.mean(axis=0)
        return [self._prediction_from_vector(row) for row in centered]

    def predict_graph(
        self,
        graph: NormalizedMolecularGraph,
        scorer: MoleculeScorer | None = None,
    ) -> FormulaRelativePrediction:
        """Build 2D features and predict directly from a normalized graph."""
        self._require_current_feature_schema()
        result = (scorer or MoleculeScorer()).score(graph)
        if result.score is None or result.status != "success":
            raise ValueError(
                "The molecular graph is outside the calibrated success set."
            )
        return self.predict(extract_two_d_features(graph, result))

    def predict_graph_group(
        self,
        graphs: Iterable[NormalizedMolecularGraph],
        scorer: MoleculeScorer | None = None,
    ) -> list[FormulaRelativePrediction]:
        """Predict and zero-center graphs after enforcing formula/charge equality."""
        self._require_current_feature_schema()
        candidates = list(graphs)
        if not candidates:
            return []
        if any(
            nx.number_connected_components(graph.graph) != 1 for graph in candidates
        ):
            raise ValueError(
                "Formula-relative calibration supports single-component molecules only."
            )
        keys = {molecular_formula_charge_key(graph) for graph in candidates}
        if len(keys) != 1:
            raise ValueError(
                "Formula-relative predictions require one molecular formula and "
                f"formal charge; received {sorted(keys)}."
            )
        molecule_scorer = scorer or MoleculeScorer()
        features = []
        for graph in candidates:
            result = molecule_scorer.score(graph)
            if result.score is None or result.status != "success":
                raise ValueError(
                    "A molecular graph is outside the calibrated success set."
                )
            features.append(extract_two_d_features(graph, result))
        return self.predict_group(features)

    def predict_smiles_group(
        self,
        smiles: Iterable[str],
        scorer: MoleculeScorer | None = None,
    ) -> list[FormulaRelativePrediction]:
        """Build, validate, and predict a same-formula group directly from SMILES."""
        from synde.graph.builder import GraphBuilder

        return self.predict_graph_group(
            [GraphBuilder.from_smiles(value) for value in smiles], scorer
        )

    def save(self, path: Path) -> None:
        """Save all coefficients and interpretation metadata as JSON."""
        self._require_fitted()
        assert self.active is not None
        assert self.card is not None
        assert self.mean is not None
        assert self.scale is not None
        assert self.weights is not None
        path.write_text(
            json.dumps(
                {
                    "card": asdict(self.card),
                    "mean": self.mean.tolist(),
                    "scale": self.scale.tolist(),
                    "weights": self.weights.tolist(),
                    "active": self.active.tolist(),
                    "residual_std": self.residual_std,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> FormulaRelativeRidgeCalibrator:
        """Load a model written by :meth:`save`."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_card = payload["card"]
        raw_card["feature_names"] = tuple(raw_card["feature_names"])
        card = FormulaRelativeModelCard(**raw_card)
        model = cls(
            alpha=card.ridge_alpha,
            target=card.target,
            units=card.units,
            feature_profile=card.feature_profile,
            selection_objective=card.selection_objective,
        )
        model.feature_names = card.feature_names
        model.mean = np.asarray(payload["mean"], dtype=float)
        model.scale = np.asarray(payload["scale"], dtype=float)
        model.weights = np.asarray(payload["weights"], dtype=float)
        model.active = np.asarray(payload["active"], dtype=bool)
        model.residual_std = payload.get("residual_std")
        model.card = card
        model._validate_loaded_dimensions()
        return model

    def _matrix(self, rows: Iterable[dict[str, float]]) -> np.ndarray:
        return np.asarray([self._vector(row) for row in rows], dtype=float)

    def _vector(self, features: dict[str, float]) -> np.ndarray:
        return np.asarray(
            [features.get(name, 0.0) for name in self.feature_names], dtype=float
        )

    def _require_fitted(self) -> None:
        if any(
            value is None
            for value in (self.mean, self.scale, self.weights, self.active, self.card)
        ):
            raise RuntimeError("Fit or load a formula-relative model first.")

    def _validate_loaded_dimensions(self) -> None:
        expected = len(self.feature_names)
        arrays = (self.mean, self.scale, self.weights, self.active)
        if any(array is None or len(array) != expected for array in arrays):
            raise ValueError("Saved model arrays do not match its feature names.")

    def _require_current_feature_schema(self) -> None:
        self._require_fitted()
        assert self.card is not None
        if self.card.feature_schema != TWO_D_FEATURE_SCHEMA:
            raise RuntimeError(
                f"Model feature schema {self.card.feature_schema!r} is incompatible "
                f"with runtime schema {TWO_D_FEATURE_SCHEMA!r}."
            )


def molecular_formula_charge_key(graph: NormalizedMolecularGraph) -> str:
    """Return a deterministic Hill-formula and formal-charge comparison key."""
    counts: Counter[str] = Counter()
    charge = 0
    for _, attrs in graph.graph.nodes(data=True):
        element = str(attrs["element"])
        counts[element] += 1
        charge += int(attrs.get("formal_charge", 0))
        if element != "H":
            counts["H"] += int(attrs.get("total_hcount", 0))
    if counts.get("C", 0):
        elements = ["C"]
        if counts.get("H", 0):
            elements.append("H")
        elements.extend(
            sorted(element for element in counts if element not in {"C", "H"})
        )
    else:
        elements = sorted(counts)
    formula = "".join(
        element + (str(counts[element]) if counts[element] != 1 else "")
        for element in elements
        if counts[element]
    )
    return f"{formula}|charge={charge}"
