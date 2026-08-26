"""User-facing refinement of SynDE energy weights on external labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from typing import Any, Iterable

import numpy as np

from synde.graph.graph_schema import NormalizedMolecularGraph

from .energy_predictor import SynDEEnergyPredictor, molecular_composition
from .two_d_calibration import molecular_formula_charge_key


@dataclass(frozen=True)
class EnergyRefinementRecord:
    """One externally supplied structure and reference energy."""

    identifier: str
    graph: NormalizedMolecularGraph
    target: float
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_smiles(
        cls,
        identifier: str,
        smiles: str,
        target: float,
        provenance: dict[str, Any] | None = None,
    ) -> "EnergyRefinementRecord":
        """Build a refinement record directly from SMILES.

        :param identifier: Stable identifier for the external observation.
        :param smiles: Molecular structure encoded as SMILES.
        :param target: Reference energy in the base predictor's units.
        :param provenance: Optional source metadata retained by the caller.
        :return: Normalized refinement record.
        """
        from synde.graph.builder import GraphBuilder

        return cls(
            identifier=identifier,
            graph=GraphBuilder.from_smiles(smiles),
            target=float(target),
            provenance=dict(provenance or {}),
        )


@dataclass(frozen=True)
class EnergyRefinementReport:
    """Diagnostics and provenance for one external refinement."""

    dataset_name: str
    dataset_sha256: str
    record_count: int
    formula_groups: int
    alpha: float
    refined_blocks: tuple[str, ...]
    baseline_metrics: dict[str, float]
    refined_metrics: dict[str, float]
    coefficient_l2_shift: float
    validation_status: str = (
        "external refinement fitted; independent validation required"
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return asdict(self)


class SynDEEnergyRefiner:
    """Refine selected weights while shrinking changes toward a base artifact.

    The fit minimizes squared residual error plus ``alpha`` times the squared
    standardized change from the supplied predictor. It returns a new
    predictor and never mutates the validated base artifact.
    """

    def __init__(
        self,
        base: SynDEEnergyPredictor,
        *,
        alpha: float = 10.0,
        refine_intercept: bool = True,
        refine_composition: bool = True,
        refine_connectivity: bool = True,
    ) -> None:
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("Refinement alpha must be finite and non-negative.")
        if not any((refine_intercept, refine_composition, refine_connectivity)):
            raise ValueError("At least one weight block must be refined.")
        self.base = base
        self.alpha = float(alpha)
        self.refine_intercept = refine_intercept
        self.refine_composition = refine_composition
        self.refine_connectivity = refine_connectivity

    def fit(
        self,
        records: Iterable[EnergyRefinementRecord],
        *,
        dataset_name: str,
    ) -> tuple[SynDEEnergyPredictor, EnergyRefinementReport]:
        """Fit coefficient adjustments to an external labeled dataset.

        :param records: Structures and reference energies in base-model units.
        :param dataset_name: Human-readable external dataset identifier.
        :return: New unvalidated predictor and its refinement report.
        """
        rows = list(records)
        self._validate_records(rows, dataset_name)
        baseline = np.asarray(
            [self.base.predict(row.graph).predicted_energy for row in rows],
            dtype=float,
        )
        targets = np.asarray([float(row.target) for row in rows], dtype=float)
        design, columns, scales = self._design(rows)
        standardized = design / scales
        residual = targets - baseline
        adjustment = self._solve(standardized, residual)
        raw_adjustment = adjustment / scales
        refined = self._updated_predictor(rows, columns, raw_adjustment, dataset_name)
        fitted = np.asarray(
            [refined.predict(row.graph).predicted_energy for row in rows], dtype=float
        )
        dataset_sha256 = _dataset_sha256(rows)
        report = EnergyRefinementReport(
            dataset_name=dataset_name,
            dataset_sha256=dataset_sha256,
            record_count=len(rows),
            formula_groups=len(
                {molecular_formula_charge_key(row.graph) for row in rows}
            ),
            alpha=self.alpha,
            refined_blocks=self._refined_blocks(),
            baseline_metrics=_regression_metrics(targets, baseline),
            refined_metrics=_regression_metrics(targets, fitted),
            coefficient_l2_shift=float(np.linalg.norm(adjustment)),
        )
        refined.refinement_report = report.to_dict()
        return refined, report

    def _validate_records(
        self, rows: list[EnergyRefinementRecord], dataset_name: str
    ) -> None:
        if not dataset_name.strip():
            raise ValueError("A nonempty external dataset name is required.")
        if len(rows) < 2:
            raise ValueError("At least two external refinement records are required.")
        identifiers = [row.identifier for row in rows]
        if any(not identifier for identifier in identifiers):
            raise ValueError("Every refinement record requires an identifier.")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Refinement record identifiers must be unique.")
        if any(not math.isfinite(float(row.target)) for row in rows):
            raise ValueError("Every external refinement target must be finite.")

    def _design(
        self, rows: list[EnergyRefinementRecord]
    ) -> tuple[np.ndarray, list[tuple[str, str]], np.ndarray]:
        columns: list[tuple[str, str]] = []
        vectors: list[np.ndarray] = []
        scales: list[float] = []
        if self.refine_intercept:
            columns.append(("intercept", "intercept"))
            vectors.append(np.ones(len(rows), dtype=float))
            scales.append(1.0)
        if self.refine_composition:
            compositions = [molecular_composition(row.graph) for row in rows]
            for name in self.base.composition_weights:
                vector = np.asarray(
                    [float(values.get(name, 0)) for values in compositions],
                    dtype=float,
                )
                columns.append(("composition", name))
                vectors.append(vector)
                scales.append(max(float(np.std(vector)), 1.0))
        if self.refine_connectivity:
            feature_rows = [self.base.features(row.graph) for row in rows]
            for name in self.base.connectivity_weights:
                columns.append(("connectivity", name))
                vectors.append(
                    np.asarray(
                        [float(values.get(name, 0.0)) for values in feature_rows],
                        dtype=float,
                    )
                )
                scales.append(self.base.feature_scales[name])
        return np.column_stack(vectors), columns, np.asarray(scales, dtype=float)

    def _solve(self, design: np.ndarray, residual: np.ndarray) -> np.ndarray:
        if self.alpha == 0:
            return np.linalg.lstsq(design, residual, rcond=None)[0]
        gram = design @ design.T
        dual = np.linalg.solve(
            gram + self.alpha * np.eye(len(residual), dtype=float), residual
        )
        return design.T @ dual

    def _updated_predictor(
        self,
        rows: list[EnergyRefinementRecord],
        columns: list[tuple[str, str]],
        adjustment: np.ndarray,
        dataset_name: str,
    ) -> SynDEEnergyPredictor:
        intercept = self.base.intercept
        composition = dict(self.base.composition_weights)
        connectivity = dict(self.base.connectivity_weights)
        for (block, name), value in zip(columns, adjustment):
            if block == "intercept":
                intercept += float(value)
            elif block == "composition":
                composition[name] += float(value)
            else:
                connectivity[name] += float(value)
        ranges = dict(self.base.composition_ranges)
        observed = [molecular_composition(row.graph) for row in rows]
        for element, (lower, upper) in ranges.items():
            values = [entry.get(element, 0) for entry in observed]
            ranges[element] = (min(lower, *values), max(upper, *values))
        card = replace(
            self.base.card,
            model_name=f"{self.base.card.model_name}-refined",
            training_source=f"{self.base.card.training_source} + {dataset_name}",
            training_source_sha256=hashlib.sha256(
                (
                    f"{self.base.card.training_source_sha256}:"
                    f"{_dataset_sha256(rows)}"
                ).encode("utf-8")
            ).hexdigest(),
            training_groups=len(
                {molecular_formula_charge_key(row.graph) for row in rows}
            ),
            training_molecules=len(rows),
            evaluation_status=(
                "external refinement fitted; independent validation required"
            ),
            formula_disjoint_evaluation=False,
            connectivity_disjoint_evaluation=False,
            composition_model=(
                f"{self.base.card.composition_model}; externally refined"
                if self.refine_composition
                else self.base.card.composition_model
            ),
            connectivity_model=(
                f"{self.base.card.connectivity_model}; externally refined"
                if self.refine_connectivity
                else self.base.card.connectivity_model
            ),
            connectivity_equation_unchanged=not self.refine_connectivity,
        )
        return SynDEEnergyPredictor(
            card=card,
            intercept=intercept,
            composition_weights=composition,
            connectivity_weights=connectivity,
            feature_means=self.base.feature_means,
            feature_scales=self.base.feature_scales,
            training_distance_q99=self.base.training_distance_q99,
            composition_ranges=ranges,
        )

    def _refined_blocks(self) -> tuple[str, ...]:
        flags = (
            ("intercept", self.refine_intercept),
            ("composition", self.refine_composition),
            ("connectivity", self.refine_connectivity),
        )
        return tuple(name for name, enabled in flags if enabled)


def _dataset_sha256(rows: list[EnergyRefinementRecord]) -> str:
    payload = [
        {
            "identifier": row.identifier,
            "graph_identity": row.graph.identity,
            "target": float(row.target),
            "provenance": row.provenance,
        }
        for row in rows
    ]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = actual - predicted
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
    }


__all__ = [
    "EnergyRefinementRecord",
    "EnergyRefinementReport",
    "SynDEEnergyRefiner",
]
