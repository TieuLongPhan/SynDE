"""Development candidate that keeps only defensible graph-only energy terms."""

from __future__ import annotations

from dataclasses import dataclass

from synde.graph.graph_schema import NormalizedMolecularGraph

from .molecule_scoring import MoleculeScorer
from .results import MoleculeScoreResult
from .theory_energy import TheoryEnergyScorer


@dataclass(frozen=True)
class GraphTheoryEnergyConfig:
    """Configuration for the reduced graph-only theory candidate."""

    parameter_set: str = "theory-organic-v2-graph-development"
    excluded_components: tuple[str, ...] = (
        "ring_strain",
        "steric_congestion",
    )


class GraphTheoryEnergyScorer:
    """Exclude terms whose values cannot be resolved reliably without geometry.

    The scorer retains the frozen v1 chemistry decomposition but sets its
    cycle-basis ring-strain and degree-only steric terms to zero.  No
    coefficient is fitted: development data is used only for this discrete
    term-inclusion decision.
    """

    def __init__(
        self,
        config: GraphTheoryEnergyConfig | None = None,
        base_scorer: MoleculeScorer | None = None,
    ) -> None:
        self.config = config or GraphTheoryEnergyConfig()
        self.base_scorer = base_scorer or MoleculeScorer()
        self.frozen_scorer = TheoryEnergyScorer(base_scorer=self.base_scorer)

    def score(self, normalized: NormalizedMolecularGraph) -> MoleculeScoreResult:
        """Return the v2 development score while preserving frozen v1."""
        frozen = self.frozen_scorer.score(normalized)
        if frozen.score is None:
            return frozen
        components = dict(frozen.components)
        excluded_values = {
            name: float(components.get(name, 0.0))
            for name in self.config.excluded_components
        }
        for name in self.config.excluded_components:
            if name in components:
                components[name] = 0.0
        descriptors = {
            **frozen.descriptors,
            "excluded_component_values": excluded_values,
        }
        provenance = {
            "model_name": "synde-theory-2d-v2-graph-development",
            "mode": "graph",
            "calibrated": False,
            "fitted_coefficients": False,
            "experimental": True,
            "benchmark_informed_development": True,
            "parameter_set": self.config.parameter_set,
            "excluded_components": list(self.config.excluded_components),
            "permanent_holdout_evaluated": False,
            "holdout_tuned": False,
        }
        return MoleculeScoreResult(
            status=frozen.status,
            score=float(sum(components.values())),
            units=frozen.units,
            components=components,
            descriptors=descriptors,
            warnings=frozen.warnings,
            provenance=provenance,
        )


__all__ = ["GraphTheoryEnergyConfig", "GraphTheoryEnergyScorer"]
