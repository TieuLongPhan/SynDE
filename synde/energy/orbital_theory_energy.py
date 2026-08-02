"""Experimental theory score driven by an expanded orbital-level pi graph."""

from __future__ import annotations

from synde.graph.generalized_huckel import GeneralizedHuckel, HuckelParameters
from synde.graph.graph_schema import NormalizedMolecularGraph
from synde.graph.orbital_pi import assign_orbital_pi

from .molecule_scoring import MoleculeScorer
from .results import MoleculeScoreResult
from .theory_energy import (
    TheoryEnergyConfig,
    hypervalent_resonance_correction,
    theory_energy_corrections,
)


class OrbitalTheoryEnergyScorer:
    """Use explicit donor/triple/cumulene orbitals without ORD calibration."""

    def __init__(
        self,
        config: TheoryEnergyConfig | None = None,
        base_scorer: MoleculeScorer | None = None,
        huckel_parameters: HuckelParameters | None = None,
    ) -> None:
        self.config = config or TheoryEnergyConfig()
        self.base_scorer = base_scorer or MoleculeScorer(parameters=huckel_parameters)
        self.huckel = GeneralizedHuckel(huckel_parameters)

    def score(self, graph: NormalizedMolecularGraph) -> MoleculeScoreResult:
        """Return an experimental score while preserving the stable base scorer."""
        base = self.base_scorer.score(graph)
        if base.score is None:
            return base
        assignment = assign_orbital_pi(graph)
        huckel = self.huckel.solve(assignment)
        pi_weight = self.base_scorer.config.two_d.pi_weight
        orbital_pi_energy = pi_weight * huckel.pi_stabilization
        corrections = theory_energy_corrections(graph.graph, self.config)
        corrections.pop("lone_pair_resonance", None)
        corrections.pop("extended_resonance", None)
        corrections["hypervalent_resonance"] = hypervalent_resonance_correction(
            graph.graph, self.config
        )
        corrections["orbital_pi_extension"] = float(
            orbital_pi_energy - base.components.get("pi_stabilization", 0.0)
        )
        components = {**base.components, **corrections}
        obsolete_base_warnings = {
            "PI_ORBITAL_MULTIPLICITY_UNSUPPORTED",
            "ODD_PI_ELECTRON_COUNT",
        }
        warnings = tuple(
            dict.fromkeys(
                (
                    *(
                        warning
                        for warning in base.warnings
                        if warning not in obsolete_base_warnings
                    ),
                    *huckel.warnings,
                )
            )
        )
        status = "partial" if warnings else "success"
        descriptors = {
            **base.descriptors,
            "n_orbital_pi_electrons": assignment.electron_count,
            "n_orbital_pi_systems": len(assignment.systems),
            "orbital_parameter_set": huckel.parameter_set,
        }
        provenance = {
            "model_name": "synde-orbital-theory-2d-v1-experimental",
            "mode": "orbital-graph",
            "calibrated": False,
            "fitted_coefficients": False,
            "experimental": True,
            "permanent_holdout_evaluated": False,
            "evaluation_protocol": "synde-ord-orbital-holdout-v1",
            "holdout_tuned": False,
        }
        return MoleculeScoreResult(
            status=status,
            score=float(sum(components.values())),
            units=base.units,
            components=components,
            descriptors=descriptors,
            warnings=warnings,
            provenance=provenance,
        )


__all__ = ["OrbitalTheoryEnergyScorer"]
