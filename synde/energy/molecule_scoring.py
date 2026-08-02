"""Graph-only molecule score orchestration."""

from __future__ import annotations
from dataclasses import dataclass, field
from synde.graph.generalized_huckel import GeneralizedHuckel, HuckelParameters
from synde.graph.graph_schema import NormalizedMolecularGraph
from synde.graph.pi_system import assign_pi_systems
from .results import MoleculeScoreResult
from .two_d_energy import TwoDEnergyConfig, two_d_energy_components


@dataclass(frozen=True)
class MoleculeScoringConfig:
    two_d: TwoDEnergyConfig = field(default_factory=TwoDEnergyConfig)


class MoleculeScorer:
    def __init__(
        self,
        config: MoleculeScoringConfig | None = None,
        parameters: HuckelParameters | None = None,
    ) -> None:
        self.config = config or MoleculeScoringConfig()
        self.huckel = GeneralizedHuckel(parameters)

    def score(self, normalized: NormalizedMolecularGraph) -> MoleculeScoreResult:
        assignment = assign_pi_systems(normalized)
        huckel = self.huckel.solve(assignment)
        components = two_d_energy_components(
            normalized.graph, assignment, huckel, self.config.two_d
        )
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
            {"model_name": "synde-2d-v1", "mode": "graph", "calibrated": False},
        )
