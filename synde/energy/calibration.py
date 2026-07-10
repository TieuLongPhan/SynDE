"""Small, reproducible calibration utilities for named graph-energy targets."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
import numpy as np


@dataclass(frozen=True)
class CalibrationRecord:
    identifier: str
    features: dict[str, float]
    target: float
    provenance: dict[str, Any]


@dataclass(frozen=True)
class ModelCard:
    model_name: str
    target: str
    units: str
    feature_names: tuple[str, ...]
    ridge_alpha: float
    metrics: dict[str, float]
    training_count: int
    version: str = "1"


@dataclass(frozen=True)
class CalibratedPrediction:
    value: float
    units: str
    target: str
    uncertainty: float | None
    applicability: float
    model_card: ModelCard


class RidgeCalibrator:
    """NumPy-only ridge baseline; suitable for transparent first calibration."""

    def __init__(
        self, *, alpha: float = 1e-6, target: str, units: str = "kcal/mol"
    ) -> None:
        self.alpha = alpha
        self.target = target
        self.units = units
        self.feature_names: tuple[str, ...] = ()
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.residual_std: float | None = None
        self.card: ModelCard | None = None

    def fit(self, records: Iterable[CalibrationRecord]) -> ModelCard:
        rows = list(records)
        if len(rows) < 2:
            raise ValueError("At least two calibration records are required.")
        self.feature_names = tuple(
            sorted({key for row in rows for key in row.features})
        )
        x = np.array(
            [
                [row.features.get(key, 0.0) for key in self.feature_names]
                for row in rows
            ],
            dtype=float,
        )
        y = np.array([row.target for row in rows], dtype=float)
        self.mean = x.mean(0)
        self.scale = np.where(x.std(0) == 0, 1.0, x.std(0))
        z = (x - self.mean) / self.scale
        design = np.c_[np.ones(len(z)), z]
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0
        self.weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        predicted = design @ self.weights
        residual = y - predicted
        self.residual_std = float(np.sqrt(np.mean(residual**2)))
        metrics = regression_metrics(y, predicted)
        self.card = ModelCard(
            "graph-energy-ridge",
            self.target,
            self.units,
            self.feature_names,
            self.alpha,
            metrics,
            len(rows),
        )
        return self.card

    def predict(self, features: dict[str, float]) -> CalibratedPrediction:
        if (
            self.weights is None
            or self.mean is None
            or self.scale is None
            or self.card is None
        ):
            raise RuntimeError("Fit or load a calibration model first.")
        x = np.array([features.get(key, 0.0) for key in self.feature_names])
        z = (x - self.mean) / self.scale
        value = float(np.r_[1.0, z] @ self.weights)
        applicability = float(np.linalg.norm(z) / max(1, len(z)) ** 0.5)
        return CalibratedPrediction(
            value, self.units, self.target, self.residual_std, applicability, self.card
        )

    def save(self, path: Path) -> None:
        if (
            self.weights is None
            or self.mean is None
            or self.scale is None
            or self.card is None
        ):
            raise RuntimeError("Fit a model before saving.")
        path.write_text(
            json.dumps(
                {
                    "card": asdict(self.card),
                    "mean": self.mean.tolist(),
                    "scale": self.scale.tolist(),
                    "weights": self.weights.tolist(),
                    "residual_std": self.residual_std,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def deterministic_split(
    records: Iterable[CalibrationRecord], *, train_fraction: float = 0.8
) -> tuple[list[CalibrationRecord], list[CalibrationRecord]]:
    train = []
    test = []
    for record in records:
        bucket = (
            int(hashlib.sha256(record.identifier.encode()).hexdigest()[:8], 16)
            / 0xFFFFFFFF
        )
        (train if bucket < train_fraction else test).append(record)
    return train, test


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = actual - predicted
    return {
        "mae": float(np.mean(abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "median_absolute_error": float(np.median(abs(error))),
    }
