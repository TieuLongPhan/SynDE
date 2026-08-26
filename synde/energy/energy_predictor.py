"""Total-energy prediction and same-formula ranking.

One artifact combines an extensive elemental-composition baseline with the
frozen connectivity-sensitive SynDE equation.  The full prediction is used
across formulas.  Within one formula the composition contribution is constant,
so the same artifact ranks isomers through its connectivity contribution.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
from importlib.resources import as_file, files
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

from synde.errors import SynDEDomainError, describe_domain
from synde.formatting import Palette
from synde.graph.graph_schema import NormalizedMolecularGraph
from synde.report import (
    model_card_summary,
    prediction_headline,
    prediction_html,
    prediction_summary,
    ranking_html,
    ranking_summary,
    top_contributions,
)

from .first_order_two_d_energy import FirstOrderTwoDEnergyScorer
from .interpretable_two_d_v2 import extract_named_empirical_two_d_features
from .interpretable_two_d_v3 import extract_quantum_graph_v3_features

if TYPE_CHECKING:
    from .refinement import EnergyRefinementRecord, EnergyRefinementReport


ENERGY_MODEL_RESOURCE = "models/synde_energy_model.json"


@dataclass(frozen=True)
class SynDEEnergyModelCard:
    """Provenance and applicability boundary of a cross-formula artifact."""

    model_name: str
    schema_version: int
    target: str
    units: str
    reference_protocol: str
    training_source: str
    training_source_sha256: str
    training_groups: int
    training_molecules: int
    evaluation_status: str
    formula_disjoint_evaluation: bool
    connectivity_disjoint_evaluation: bool
    composition_model: str
    connectivity_model: str
    supported_elements: tuple[str, ...]
    supported_formal_charges: tuple[int, ...]
    uses_coordinates_at_inference: bool
    uses_conformers_at_inference: bool
    connectivity_equation_unchanged: bool
    connectivity_refit_on_amended_development_cohort: bool = False
    ordinary_explicit_hydrogen_policy: str = "not_recorded"
    isotope_policy: str = "not_recorded"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SynDEEnergyModelCard":
        translated = dict(payload)
        translated["supported_elements"] = tuple(translated["supported_elements"])
        translated["supported_formal_charges"] = tuple(
            int(value) for value in translated["supported_formal_charges"]
        )
        return cls(**translated)

    def summary(self, *, color: bool | None = None) -> str:
        """Render this model card as an aligned text report.

        :param color: Force ANSI colour on or off; ``None`` auto-detects.
        :type color: bool | None
        :return: Multi-line model-card report.
        :rtype: str
        """
        palette = Palette.automatic() if color is None else Palette(color)
        return model_card_summary(self, palette=palette)

    def __repr__(self) -> str:
        """Return a compact single-line description of this model card.

        :return: Class name with the model name, target, and units.
        :rtype: str
        """
        return (
            f"SynDEEnergyModelCard(model={self.model_name!r}, "
            f"target={self.target!r}, units={self.units!r})"
        )


@dataclass(frozen=True)
class SynDEEnergyPrediction:
    """Auditable prediction of one protocol-defined total energy."""

    status: str
    predicted_energy: float
    units: str
    intercept_contribution: float
    composition_contributions: dict[str, float]
    connectivity_contributions: dict[str, float]
    composition_total: float
    connectivity_total: float
    descriptors: dict[str, Any]
    warnings: tuple[str, ...]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the whole prediction as a JSON-ready mapping.

        :return: Nested mapping of every field on this prediction.
        :rtype: dict[str, Any]
        """
        return asdict(self)

    @property
    def canonical_smiles(self) -> str | None:
        """Return the canonical SMILES recorded for the scored graph.

        :return: Canonical SMILES, or ``None`` when the graph carried none.
        :rtype: str | None
        """
        value = self.descriptors.get("canonical_smiles")
        return None if value is None else str(value)

    @property
    def composition(self) -> dict[str, int]:
        """Return the element counts used by the composition block.

        :return: Element symbol to count mapping including hydrogens.
        :rtype: dict[str, int]
        """
        return dict(self.descriptors.get("composition", {}))

    def top_contributions(self, limit: int = 8) -> list[tuple[str, float]]:
        """Return the largest-magnitude active connectivity terms.

        :param limit: Maximum number of terms returned.
        :type limit: int
        :return: Term name and signed contribution pairs, largest first.
        :rtype: list[tuple[str, float]]
        """
        return top_contributions(self, limit)

    def summary(
        self, *, color: bool | None = None, precision: int = 4, top: int = 8
    ) -> str:
        """Render the auditable breakdown of this prediction as text.

        :param color: Force ANSI colour on or off; ``None`` auto-detects.
        :type color: bool | None
        :param precision: Digits kept after the decimal point.
        :type precision: int
        :param top: Number of connectivity terms listed.
        :type top: int
        :return: Multi-line report suitable for printing to a terminal.
        :rtype: str
        """
        palette = Palette.automatic() if color is None else Palette(color)
        return prediction_summary(self, palette=palette, precision=precision, top=top)

    def __repr__(self) -> str:
        """Return a compact single-line description of this prediction.

        :return: Class name with the scored structure and predicted energy.
        :rtype: str
        """
        return f"SynDEEnergyPrediction({prediction_headline(self)})"

    def _repr_html_(self) -> str:
        """Render this prediction as HTML for Jupyter front ends.

        :return: HTML fragment showing the breakdown table.
        :rtype: str
        """
        return prediction_html(self)


class SynDEEnergyRanking(list):
    """Ordered ``(input index, prediction)`` pairs from :meth:`rank_group`.

    The class behaves exactly like the list of tuples returned by earlier
    releases and only adds rendering helpers, so existing unpacking code keeps
    working unchanged.

    :param entries: Ordered input-index and prediction pairs.
    :type entries: Iterable[tuple[int, SynDEEnergyPrediction]]
    :param labels: Display labels indexed by original input position.
    :type labels: Sequence[str] | None
    """

    def __init__(self, entries, labels=None) -> None:
        super().__init__(entries)
        self.labels = list(labels) if labels is not None else None

    def summary(self, *, color: bool | None = None, precision: int = 4) -> str:
        """Render this ranking as an aligned text table.

        :param color: Force ANSI colour on or off; ``None`` auto-detects.
        :type color: bool | None
        :param precision: Digits kept after the decimal point.
        :type precision: int
        :return: Multi-line ranking table.
        :rtype: str
        """
        palette = Palette.automatic() if color is None else Palette(color)
        return ranking_summary(
            self, labels=self.labels, palette=palette, precision=precision
        )

    def __repr__(self) -> str:
        """Return a compact description of this ranking.

        :return: Class name with the ranked structures in order.
        :rtype: str
        """
        order = ", ".join(
            str(prediction.descriptors.get("canonical_smiles", index))
            for index, prediction in self
        )
        return f"SynDEEnergyRanking([{order}])"

    def _repr_html_(self) -> str:
        """Render this ranking as HTML for Jupyter front ends.

        :return: HTML fragment showing the ranking table.
        :rtype: str
        """
        return ranking_html(self, labels=self.labels)


def molecular_composition(
    normalized: NormalizedMolecularGraph,
) -> dict[str, int]:
    """Return element counts including implicit hydrogens.

    Explicit hydrogen nodes are counted directly.  For the usual implicit-H
    graph, normalized ``total_hcount`` attributes supply the hydrogen count.
    """

    counts: Counter[str] = Counter()
    implicit_hydrogens = 0
    for _, attrs in normalized.graph.nodes(data=True):
        element = str(attrs.get("element", "?"))
        counts[element] += 1
        if element != "H":
            implicit_hydrogens += int(attrs.get("total_hcount", 0))
    counts["H"] += implicit_hydrogens
    return dict(sorted((name, int(value)) for name, value in counts.items() if value))


class SynDEEnergyPredictor:
    """Predict GFN2-xTB total energy and rank constitutional isomers.

    The returned energy is the sum of three inspectable blocks: a fitted
    intercept, an extensive elemental-composition baseline, and a sparse
    connectivity correction over the named SynDE descriptor library.
    """

    DISTANCE_WARNING = "SYNDE_ENERGY_OUTSIDE_TRAINING_Q99_FEATURE_DISTANCE"
    COMPOSITION_WARNING = "SYNDE_ENERGY_OUTSIDE_TRAINING_COMPOSITION_RANGE"

    def __init__(
        self,
        *,
        card: SynDEEnergyModelCard,
        intercept: float,
        composition_weights: dict[str, float],
        connectivity_weights: dict[str, float],
        feature_means: dict[str, float],
        feature_scales: dict[str, float],
        training_distance_q99: float,
        composition_ranges: dict[str, tuple[int, int]],
        model_sha256: str | None = None,
        refinement_report: dict[str, Any] | None = None,
    ) -> None:
        if card.schema_version not in {1, 2}:
            raise ValueError("Unsupported SynDE energy model-card schema.")
        if card.uses_coordinates_at_inference or card.uses_conformers_at_inference:
            raise ValueError("A SynDE energy artifact must remain 2D at inference.")
        if not math.isfinite(intercept):
            raise ValueError("The SynDE energy intercept must be finite.")
        if not composition_weights or not connectivity_weights:
            raise ValueError("Composition and connectivity weights must be nonempty.")
        if set(composition_weights) != set(card.supported_elements):
            raise ValueError(
                "Composition weights must cover exactly the supported elements."
            )
        if set(composition_ranges) != set(card.supported_elements):
            raise ValueError(
                "Composition ranges must cover exactly the supported elements."
            )
        if set(connectivity_weights) != set(feature_means):
            raise ValueError("Connectivity weights and feature means must match.")
        if set(connectivity_weights) != set(feature_scales):
            raise ValueError("Connectivity weights and feature scales must match.")
        if any(not math.isfinite(value) for value in composition_weights.values()):
            raise ValueError("Every composition weight must be finite.")
        if any(not math.isfinite(value) for value in connectivity_weights.values()):
            raise ValueError("Every connectivity weight must be finite.")
        if any(not math.isfinite(value) for value in feature_means.values()):
            raise ValueError("Every feature mean must be finite.")
        if any(
            not math.isfinite(value) or value <= 0 for value in feature_scales.values()
        ):
            raise ValueError("Every feature scale must be finite and positive.")
        if not math.isfinite(training_distance_q99) or training_distance_q99 <= 0:
            raise ValueError("The training q99 feature distance must be positive.")
        if any(lower > upper for lower, upper in composition_ranges.values()):
            raise ValueError("Every composition range must be ordered.")
        self.card = card
        self.intercept = float(intercept)
        self.composition_weights = dict(composition_weights)
        self.connectivity_weights = dict(connectivity_weights)
        self.feature_means = dict(feature_means)
        self.feature_scales = dict(feature_scales)
        self.training_distance_q99 = float(training_distance_q99)
        self.composition_ranges = dict(composition_ranges)
        self.model_sha256 = model_sha256
        self.refinement_report = dict(refinement_report or {})
        self._first_order = FirstOrderTwoDEnergyScorer()

    def features(self, normalized: NormalizedMolecularGraph) -> dict[str, float]:
        """Return the same named coordinate-free descriptor library used in fit."""

        first_order = self._first_order.score(normalized)
        features = extract_named_empirical_two_d_features(normalized, first_order)
        features.update(extract_quantum_graph_v3_features(normalized))
        return features

    def predict(self, normalized: NormalizedMolecularGraph) -> SynDEEnergyPrediction:
        """Predict one total energy; unlike ``SynDEScorer``, formulas may differ."""

        self._validate_graph(normalized)
        composition = molecular_composition(normalized)
        features = self.features(normalized)
        composition_contributions = {
            element: float(weight * composition.get(element, 0))
            for element, weight in self.composition_weights.items()
        }
        connectivity_contributions = {
            name: float(weight * features.get(name, 0.0))
            for name, weight in self.connectivity_weights.items()
        }
        composition_total = float(
            self.intercept + sum(composition_contributions.values())
        )
        connectivity_total = float(sum(connectivity_contributions.values()))
        prediction = float(composition_total + connectivity_total)
        distance = self._feature_distance(features)
        warnings: list[str] = []
        if distance > self.training_distance_q99:
            warnings.append(self.DISTANCE_WARNING)
        outside_composition = self._outside_composition_range(composition)
        if outside_composition:
            warnings.append(self.COMPOSITION_WARNING)
        return SynDEEnergyPrediction(
            status="success",
            predicted_energy=prediction,
            units=self.card.units,
            intercept_contribution=self.intercept,
            composition_contributions=composition_contributions,
            connectivity_contributions=connectivity_contributions,
            composition_total=composition_total,
            connectivity_total=connectivity_total,
            descriptors={
                "graph_identity": normalized.identity,
                "canonical_smiles": normalized.canonical_smiles,
                "composition": composition,
                "selected_connectivity_terms": len(self.connectivity_weights),
                "selected_feature_distance": distance,
                "training_distance_q99": self.training_distance_q99,
                "outside_training_composition": outside_composition,
                "cross_formula_comparable_under_same_target_protocol": True,
                "connectivity_subtotal_used_as_absolute_energy": False,
            },
            warnings=tuple(warnings),
            provenance={
                "model_name": self.card.model_name,
                "model_sha256": self.model_sha256,
                "target": self.card.target,
                "reference_protocol": self.card.reference_protocol,
                "composition_model": self.card.composition_model,
                "connectivity_model": self.card.connectivity_model,
                "evaluation_status": self.card.evaluation_status,
                "uses_coordinates_at_inference": False,
                "uses_conformers_at_inference": False,
                "refinement_report": self.refinement_report or None,
            },
        )

    def predict_many(
        self, candidates: list[NormalizedMolecularGraph]
    ) -> list[SynDEEnergyPrediction]:
        """Predict any collection; no shared-formula restriction is applied."""

        return [self.predict(candidate) for candidate in candidates]

    def predict_group(
        self, candidates: list[NormalizedMolecularGraph]
    ) -> list[SynDEEnergyPrediction]:
        """Predict one formula group with the same artifact used globally.

        Requiring a common elemental composition and formal charge makes the
        ranking interpretation explicit: every composition contribution is
        identical, and all predicted differences come from connectivity.
        """

        if not candidates:
            return []
        outputs = self.predict_many(candidates)
        signatures = {
            (
                tuple(sorted(output.descriptors["composition"].items())),
                sum(
                    int(attrs.get("formal_charge", 0))
                    for _, attrs in candidate.graph.nodes(data=True)
                ),
            )
            for candidate, output in zip(candidates, outputs)
        }
        if len(signatures) != 1:
            from synde.report import composition_formula

            observed = sorted(
                f"{composition_formula(dict(counts))} (charge {charge})"
                for counts, charge in signatures
            )
            raise SynDEDomainError(
                "SynDE group prediction requires one formula/formal-charge "
                f"group; received {len(signatures)}: {', '.join(observed)}.",
                hint=(
                    "rank_group() removes atom-count signal by construction, so "
                    "every candidate must be a constitutional isomer of the "
                    "others. Use predict_many() to compare across formulas."
                ),
                details={"observed_groups": observed},
            )
        composition_totals = {round(output.composition_total, 12) for output in outputs}
        if len(composition_totals) != 1:
            raise RuntimeError("Same-formula composition contributions must be equal.")
        return outputs

    def rank_group(
        self, candidates: list[NormalizedMolecularGraph]
    ) -> "SynDEEnergyRanking":
        """Return a same-formula group ordered from lowest predicted energy.

        The result is a list of ``(input index, prediction)`` pairs exactly as
        in earlier releases, wrapped in a subclass that can render itself.

        :param candidates: Constitutional isomers sharing one formula.
        :type candidates: list[NormalizedMolecularGraph]
        :return: Ordered index and prediction pairs, lowest energy first.
        :rtype: SynDEEnergyRanking
        """
        outputs = self.predict_group(candidates)
        ordered = sorted(enumerate(outputs), key=lambda item: item[1].predicted_energy)
        labels = [
            candidate.canonical_smiles or candidate.identity for candidate in candidates
        ]
        return SynDEEnergyRanking(ordered, labels=labels)

    def predict_smiles(self, smiles: str) -> SynDEEnergyPrediction:
        """Parse one SMILES string and predict its total energy.

        :param smiles: SMILES string for a single neutral closed-shell molecule.
        :type smiles: str
        :return: Auditable prediction for the parsed structure.
        :rtype: SynDEEnergyPrediction
        """
        from synde.graph import GraphBuilder

        return self.predict(GraphBuilder.from_smiles(smiles))

    def predict_many_smiles(self, smiles: Iterable[str]) -> list[SynDEEnergyPrediction]:
        """Parse and predict a collection of SMILES strings.

        :param smiles: SMILES strings; formulas may differ freely.
        :type smiles: Iterable[str]
        :return: One prediction per input, in input order.
        :rtype: list[SynDEEnergyPrediction]
        """
        from synde.graph import GraphBuilder

        return self.predict_many([GraphBuilder.from_smiles(item) for item in smiles])

    def rank_smiles(self, smiles: Iterable[str]) -> "SynDEEnergyRanking":
        """Parse constitutional isomers and rank them by predicted energy.

        :param smiles: SMILES strings that share one formula and formal charge.
        :type smiles: Iterable[str]
        :return: Ordered index and prediction pairs, lowest energy first.
        :rtype: SynDEEnergyRanking
        """
        from synde.graph import GraphBuilder

        return self.rank_group([GraphBuilder.from_smiles(item) for item in smiles])

    def summary(self, *, color: bool | None = None) -> str:
        """Render this artifact's provenance and applicability boundary.

        :param color: Force ANSI colour on or off; ``None`` auto-detects.
        :type color: bool | None
        :return: Multi-line model-card report.
        :rtype: str
        """
        palette = Palette.automatic() if color is None else Palette(color)
        return model_card_summary(
            self.card, palette=palette, model_sha256=self.model_sha256
        )

    def __repr__(self) -> str:
        """Return a compact single-line description of this predictor.

        :return: Class name with the model name and fitted term counts.
        :rtype: str
        """
        return (
            f"SynDEEnergyPredictor(model={self.card.model_name!r}, "
            f"units={self.card.units!r}, "
            f"composition_terms={len(self.composition_weights)}, "
            f"connectivity_terms={len(self.connectivity_weights)})"
        )

    def _repr_html_(self) -> str:
        """Render this artifact's model card as HTML for Jupyter front ends.

        :return: HTML fragment describing the loaded artifact.
        :rtype: str
        """
        from html import escape

        rows = "".join(
            f"<tr><td style='padding:2px 10px 2px 0;opacity:.6'>{escape(name)}</td>"
            f"<td style='padding:2px 0;font-family:monospace'>{escape(str(value))}"
            "</td></tr>"
            for name, value in [
                ("model", self.card.model_name),
                ("units", self.card.units),
                ("protocol", self.card.reference_protocol),
                ("elements", " ".join(self.card.supported_elements)),
                ("connectivity terms", len(self.connectivity_weights)),
                ("evaluation", self.card.evaluation_status),
            ]
        )
        return (
            "<div style='font-family:system-ui,sans-serif'>"
            "<strong>SynDEEnergyPredictor</strong>"
            f"<table style='border-collapse:collapse;font-size:.9em'>{rows}</table>"
            "</div>"
        )

    def domain_description(self) -> str:
        """Summarize the chemistry this artifact was fitted to accept.

        :return: Single-line description of elements, charges, and structure.
        :rtype: str
        """
        return describe_domain(
            self.card.supported_elements, self.card.supported_formal_charges
        )

    def _reject(
        self,
        reason: str,
        normalized: NormalizedMolecularGraph,
        hint: str,
        **details: Any,
    ) -> SynDEDomainError:
        """Build a domain error carrying the input, the rule, and a next step.

        :param reason: Short sentence naming the violated domain rule.
        :type reason: str
        :param normalized: Graph that was rejected.
        :type normalized: NormalizedMolecularGraph
        :param hint: Concrete remedial action offered to the caller.
        :type hint: str
        :param details: Extra machine-readable context attached to the failure.
        :type details: Any
        :return: Error ready to be raised by the caller.
        :rtype: SynDEDomainError
        """
        return SynDEDomainError(
            reason,
            subject=normalized.canonical_smiles or normalized.identity,
            hint=f"{hint}\n  Model domain: {self.domain_description()}",
            details={"model_name": self.card.model_name, **details},
        )

    def _validate_graph(self, normalized: NormalizedMolecularGraph) -> None:
        """Reject any graph outside the active artifact's applicability domain.

        :param normalized: Candidate graph to validate.
        :type normalized: NormalizedMolecularGraph
        :raises SynDEDomainError: If the graph violates any domain rule.
        """
        graph = normalized.graph
        if graph.number_of_nodes() == 0:
            raise self._reject(
                "SynDE energy prediction requires a nonempty graph.",
                normalized,
                "Check that the input structure parsed into at least one atom.",
            )
        import networkx as nx

        if not nx.is_connected(graph):
            components = nx.number_connected_components(graph)
            raise self._reject(
                "SynDE energy prediction requires one connected molecule; "
                f"this input has {components} disconnected fragments.",
                normalized,
                "Split the input on '.' and score each neutral component "
                "separately; salts and solvates are not single molecules.",
                components=components,
            )
        isotopes = sorted(
            {
                int(attrs.get("isotope", 0))
                for _, attrs in graph.nodes(data=True)
                if int(attrs.get("isotope", 0)) != 0
            }
        )
        if isotopes:
            raise self._reject(
                "Isotopically labelled molecules are outside the SynDE model "
                f"domain; found mass numbers {isotopes}.",
                normalized,
                "Remove the isotope labels; the 2D descriptors are "
                "mass-independent, so the unlabelled structure gives the "
                "same prediction.",
                isotopes=isotopes,
            )
        elements = {str(attrs.get("element")) for _, attrs in graph.nodes(data=True)}
        unsupported = sorted(elements - set(self.card.supported_elements))
        if unsupported:
            raise self._reject(
                "Unsupported elements for SynDE energy model: " f"{unsupported}.",
                normalized,
                "The composition baseline has no fitted weight for these "
                "elements. Restrict the input set, or refit with refine() on "
                "labels that cover them.",
                unsupported_elements=unsupported,
            )
        charge = sum(
            int(attrs.get("formal_charge", 0)) for _, attrs in graph.nodes(data=True)
        )
        if charge not in self.card.supported_formal_charges:
            accepted = ", ".join(
                str(value) for value in self.card.supported_formal_charges
            )
            raise self._reject(
                f"Unsupported formal charge for SynDE energy model: {charge} "
                f"(accepted: {accepted}).",
                normalized,
                "Score the neutral form instead; the reference protocol "
                "optimized neutral closed-shell species only.",
                formal_charge=charge,
            )
        radicals = sum(
            int(attrs.get("radical_electrons", 0))
            for _, attrs in graph.nodes(data=True)
        )
        if radicals:
            raise self._reject(
                "SynDE energy prediction requires a closed-shell molecule; "
                f"this input carries {radicals} radical electrons.",
                normalized,
                "Pair the unpaired electrons, or add explicit hydrogens to "
                "close the shell.",
                radical_electrons=radicals,
            )

    def _feature_distance(self, features: dict[str, float]) -> float:
        standardized = [
            (float(features.get(name, 0.0)) - self.feature_means[name])
            / self.feature_scales[name]
            for name in self.connectivity_weights
        ]
        return float(np.sqrt(np.mean(np.square(standardized))))

    def _outside_composition_range(self, composition: dict[str, int]) -> bool:
        for element in set(composition) | set(self.composition_ranges):
            value = composition.get(element, 0)
            lower, upper = self.composition_ranges.get(element, (0, 0))
            if value < lower or value > upper:
                return True
        return False

    @classmethod
    def load(cls, model_path: Path) -> "SynDEEnergyPredictor":
        """Load and hash a versioned cross-formula model artifact."""

        model_bytes = model_path.read_bytes()
        payload = json.loads(model_bytes)
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("Unsupported SynDE energy model schema.")
        return cls(
            card=SynDEEnergyModelCard.from_dict(payload["card"]),
            intercept=float(payload["intercept"]),
            composition_weights={
                str(name): float(value)
                for name, value in payload["composition_weights"].items()
            },
            connectivity_weights={
                str(name): float(value)
                for name, value in payload["connectivity_weights"].items()
            },
            feature_means={
                str(name): float(value)
                for name, value in payload["feature_means"].items()
            },
            feature_scales={
                str(name): float(value)
                for name, value in payload["feature_scales"].items()
            },
            training_distance_q99=float(payload["training_distance_q99"]),
            composition_ranges={
                str(name): (int(bounds[0]), int(bounds[1]))
                for name, bounds in payload["composition_ranges"].items()
            },
            model_sha256=hashlib.sha256(model_bytes).hexdigest(),
            refinement_report=payload.get("refinement_report"),
        )

    @classmethod
    def load_default(cls) -> "SynDEEnergyPredictor":
        """Load the packaged energy-and-ranking artifact."""

        resource = files("synde").joinpath(ENERGY_MODEL_RESOURCE)
        if not resource.is_file():
            raise FileNotFoundError(
                "The installed package has no externally validated SynDE energy "
                "artifact. Run Experiment/run_global_comparators.sh on the active "
                "training and external-validation cohorts, review the result, and "
                "publish the generated model."
            )
        with as_file(resource) as model_path:
            return cls.load(model_path)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the model without embedding its self-referential file hash."""

        payload = {
            "schema_version": 1,
            "card": asdict(self.card),
            "intercept": self.intercept,
            "composition_weights": dict(self.composition_weights),
            "connectivity_weights": dict(self.connectivity_weights),
            "feature_means": dict(self.feature_means),
            "feature_scales": dict(self.feature_scales),
            "training_distance_q99": self.training_distance_q99,
            "composition_ranges": {
                name: list(bounds) for name, bounds in self.composition_ranges.items()
            },
        }
        if self.refinement_report:
            payload["refinement_report"] = dict(self.refinement_report)
        return payload

    def save(self, path: Path) -> None:
        """Write this predictor as a loadable, versioned JSON artifact.

        :param path: Destination JSON path.
        """
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def refine(
        self,
        records: "Iterable[EnergyRefinementRecord]",
        *,
        dataset_name: str,
        alpha: float = 10.0,
        refine_intercept: bool = True,
        refine_composition: bool = True,
        refine_connectivity: bool = True,
    ) -> "tuple[SynDEEnergyPredictor, EnergyRefinementReport]":
        """Return a new predictor refined on externally supplied labels.

        The validated packaged artifact is never mutated. The returned model
        is explicitly marked as requiring fresh independent validation.

        :param records: External structures and targets in this model's units.
        :param dataset_name: Human-readable external dataset identifier.
        :param alpha: Ridge strength anchoring adjustments to current weights.
        :param refine_intercept: Permit refinement of the intercept.
        :param refine_composition: Permit refinement of elemental weights.
        :param refine_connectivity: Permit refinement of connectivity weights.
        :return: Refined predictor and fit diagnostics.
        """
        from .refinement import SynDEEnergyRefiner

        refiner = SynDEEnergyRefiner(
            self,
            alpha=alpha,
            refine_intercept=refine_intercept,
            refine_composition=refine_composition,
            refine_connectivity=refine_connectivity,
        )
        return refiner.fit(records, dataset_name=dataset_name)


__all__ = [
    "ENERGY_MODEL_RESOURCE",
    "SynDEEnergyModelCard",
    "SynDEEnergyPrediction",
    "SynDEEnergyPredictor",
    "molecular_composition",
]
