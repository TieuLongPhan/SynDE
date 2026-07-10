"""Graph-only molecule score orchestration."""

from __future__ import annotations
from dataclasses import dataclass
from synde.graph.generalized_huckel import GeneralizedHuckel, HuckelParameters
from synde.graph.graph_schema import NormalizedMolecularGraph
from .local_energy import local_score_components
from synde.graph.pi_system import assign_pi_systems
from .results import MoleculeScoreResult


@dataclass(frozen=True)
class MoleculeScoringConfig:
    pi_weight: float = 1.0


class MoleculeScorer:
    def __init__(
        self,
        config: MoleculeScoringConfig | None = None,
        parameters: HuckelParameters | None = None,
    ) -> None:
        self.config = config or MoleculeScoringConfig()
        self.huckel = GeneralizedHuckel(parameters)

    def score(self, normalized: NormalizedMolecularGraph) -> MoleculeScoreResult:
        local = local_score_components(normalized.graph)
        assignment = assign_pi_systems(normalized)
        huckel = self.huckel.solve(assignment)
        components = {
            **local,
            "pi_stabilization": self.config.pi_weight * huckel.pi_stabilization,
        }
        warnings = tuple(
            dict.fromkeys(
                (
                    *normalized.warning_codes(),
                    *assignment.warning_codes(),
                    *huckel.warnings,
                )
            )
        )
        status = "partial" if warnings else "success"
        return MoleculeScoreResult(
            status,
            float(sum(components.values())),
            "score",
            components,
            {
                "graph_identity": normalized.identity,
                "canonical_smiles": normalized.canonical_smiles,
                "n_pi_electrons": assignment.electron_count,
                "parameter_set": huckel.parameter_set,
            },
            warnings,
            {"model_name": "graph-energy-v2", "mode": "graph"},
        )
