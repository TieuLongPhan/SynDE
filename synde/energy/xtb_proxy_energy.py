"""Uncalibrated graph proxy for raw GFN2-xTB-like total energies.

The dominant contribution to a raw semiempirical total energy is elemental
composition.  This module combines neutral-atom valence-level references from
the public GFN2-xTB parameter set with SynDE's fixed graph corrections.  It
does not run xTB and must not be interpreted as an xTB calculation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from synde.graph.graph_schema import NormalizedMolecularGraph

from .molecule_scoring import MoleculeScorer, MoleculeScoringConfig
from .results import MoleculeScoreResult
from .two_d_energy import TwoDEnergyConfig

# CODATA-compatible conversion used only to express the existing textbook
# sigma-bond references on an eV-like scale.
KJ_MOL_PER_EV = 96.4853321233

# GFN2-xTB ``lev`` entries (eV) and neutral valence occupations.  Values come
# from the official ``param_gfn2-xtb.txt`` parameter file:
# https://github.com/grimme-lab/xtb/blob/main/param_gfn2-xtb.txt
#
# The sums are composition references, not isolated-atom total energies.
GFN2_XTB_VALENCE_REFERENCE_EV: dict[str, float] = {
    "H": -10.707211,
    "B": 2 * -9.224376 + -7.419002,
    "C": 2 * -13.970922 + 2 * -10.063292,
    "N": 2 * -16.686243 + 3 * -12.523956,
    "O": 2 * -20.229985 + 4 * -15.503117,
    "F": 2 * -23.458179 + 5 * -15.746583,
    "Si": 2 * -14.360932 + 2 * -6.915131,
    "P": 2 * -17.518756 + 3 * -9.842286,
    "S": 2 * -20.029654 + 4 * -11.377694,
    "Cl": 2 * -29.278781 + 5 * -12.673758,
    "Br": 2 * -23.583718 + 5 * -12.588824,
    "I": 2 * -20.949407 + 5 * -12.180159,
}


@dataclass(frozen=True)
class GFN2XTBProxyConfig:
    """Fixed settings for the uncalibrated total-energy proxy."""

    parameter_set: str = "gfn2-xtb-valence-reference-plus-synde-2d-v1"


class GFN2XTBProxyScorer:
    """Estimate a raw GFN2-xTB-like molecular energy without fitted weights."""

    def __init__(
        self,
        config: GFN2XTBProxyConfig | None = None,
        base_scorer: MoleculeScorer | None = None,
    ) -> None:
        self.config = config or GFN2XTBProxyConfig()
        self.base_scorer = base_scorer or MoleculeScorer(
            MoleculeScoringConfig(
                two_d=TwoDEnergyConfig(sigma_scale=1.0 / KJ_MOL_PER_EV)
            )
        )

    def score(self, normalized: NormalizedMolecularGraph) -> MoleculeScoreResult:
        """Return an eV-like proxy, or an explicit unsupported-element result."""
        base = self.base_scorer.score(normalized)
        counts, implicit_hydrogens = _element_counts_with_implicit_hydrogen(normalized)
        unsupported = tuple(
            sorted(
                element
                for element in counts
                if element not in GFN2_XTB_VALENCE_REFERENCE_EV
            )
        )
        descriptors = {
            **base.descriptors,
            "element_counts_including_implicit_h": dict(sorted(counts.items())),
            "implicit_hydrogen_count": implicit_hydrogens,
            "base_model_name": base.provenance.get("model_name"),
        }
        provenance = {
            "model_name": "synde-gfn2-xtb-total-energy-proxy-v1",
            "mode": "graph",
            "calibrated": False,
            "fitted_coefficients": False,
            "proxy": True,
            "target_method_family": "GFN2-xTB",
            "parameter_set": self.config.parameter_set,
            "parameter_source": (
                "grimme-lab/xtb param_gfn2-xtb.txt; doi:10.1021/acs.jctc.8b01176"
            ),
            "benchmark_informed_development": True,
            "ord_label_provenance_verified": False,
        }
        if unsupported:
            warning = "GFN2_PROXY_UNSUPPORTED_ELEMENTS:" + ",".join(unsupported)
            return MoleculeScoreResult(
                status="unsupported",
                score=None,
                units="eV_proxy",
                components={},
                descriptors=descriptors,
                warnings=tuple(dict.fromkeys((*base.warnings, warning))),
                provenance=provenance,
            )

        atomic_reference = sum(
            count * GFN2_XTB_VALENCE_REFERENCE_EV[element]
            for element, count in counts.items()
        )
        components = {"gfn2_valence_reference": float(atomic_reference)}
        components.update(
            {
                name: float(value)
                for name, value in base.components.items()
                if name != "atom_reference"
            }
        )
        return MoleculeScoreResult(
            status=base.status,
            score=float(sum(components.values())),
            units="eV_proxy",
            components=components,
            descriptors=descriptors,
            warnings=base.warnings,
            provenance=provenance,
        )


def _element_counts_with_implicit_hydrogen(
    normalized: NormalizedMolecularGraph,
) -> tuple[Counter[str], int]:
    """Count explicit atoms plus hydrogens represented on heavy-atom labels."""
    counts: Counter[str] = Counter()
    implicit_hydrogens = 0
    for _, attrs in normalized.graph.nodes(data=True):
        element = str(attrs["element"])
        counts[element] += 1
        if element != "H":
            hydrogen_count = int(attrs.get("total_hcount", 0))
            counts["H"] += hydrogen_count
            implicit_hydrogens += hydrogen_count
    return counts, implicit_hydrogens


__all__ = [
    "GFN2_XTB_VALENCE_REFERENCE_EV",
    "KJ_MOL_PER_EV",
    "GFN2XTBProxyConfig",
    "GFN2XTBProxyScorer",
]
