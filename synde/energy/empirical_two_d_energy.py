"""Interpretable empirical energy ledger derived only from a molecular graph.

The localized reference is the Joback group-contribution estimate of the
ideal-gas enthalpy of formation at 298.15 K.  It is augmented by corrections
that Joback does not resolve explicitly: Hückel pi delocalization, lone-pair
donation, formal-charge localization, and connectivity-forced small-ring
strain.  No coordinate, conformer, force-field, xTB, fingerprint, or fitted
ORD quantity enters the score.

The score is intended for formula-controlled molecular ranking.  Its kJ/mol
scale is an empirical bookkeeping scale, not a claim that the unprovenanced
``ord.csv`` labels are thermochemical enthalpies.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import re
from typing import Any

import networkx as nx

from synde.graph.graph_schema import NormalizedMolecularGraph

from .results import MoleculeScoreResult
from .valence_energy import ValenceEnergyConfig, valence_energy_terms


JOBACK_REFERENCE_OFFSET_KJ_MOL = 68.29
HUCKEL_RESONANCE_INTEGRAL_KJ_MOL = 75.0
SCORE_PARAMETER_SET = "empirical-2d-joback-huckel-v1-development"


@dataclass(frozen=True)
class EmpiricalTwoDEnergyConfig:
    """Externally fixed scales for the graph-only empirical score.

    ``huckel_resonance_integral_kj_mol`` is the conventional 17--20 kcal/mol
    magnitude inferred from hydrocarbon thermochemistry, represented here by
    the round value 75 kJ/mol.  Joback group increments and its 68.29 kJ/mol
    reference offset are used verbatim by :mod:`thermo`.
    """

    parameter_set: str = SCORE_PARAMETER_SET
    huckel_resonance_integral_kj_mol: float = HUCKEL_RESONANCE_INTEGRAL_KJ_MOL
    cyclopropane_strain_kj_mol: float = 115.0
    cyclobutane_strain_kj_mol: float = 110.0
    cyclopropene_extra_strain_kj_mol: float = 55.0
    cyclobutene_extra_strain_kj_mol: float = 25.0
    bridgehead_alkene_strain_kj_mol: float = 50.0
    formal_charge_hardness_scale_kj_mol: float = 9.6485


def _load_joback_backend() -> tuple[dict[int, Any], type[Any]] | None:
    """Load the optional Joback backend without making it a core dependency."""
    try:
        module = importlib.import_module("thermo.group_contribution.joback")
    except ImportError:
        return None
    return module.JOBACK_GROUPS, module.Joback


def _component_name(index: int, joback_groups: dict[int, Any]) -> str:
    label = joback_groups[index].group
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return f"joback_{index:02d}_{slug}"


def _joback_components(smiles: str) -> tuple[dict[str, float], dict[int, int], str]:
    backend = _load_joback_backend()
    if backend is None:
        return {}, {}, "THERMO_NOT_INSTALLED"
    joback_groups, joback_type = backend
    joback = joback_type(smiles)
    if not joback.success:
        return {}, dict(joback.counts), str(joback.status)
    components = {
        "joback_reference_offset": JOBACK_REFERENCE_OFFSET_KJ_MOL,
    }
    for index in joback_groups:
        count = int(joback.counts.get(index, 0))
        components[_component_name(index, joback_groups)] = float(
            count * joback_groups[index].Hform
        )
    return components, dict(joback.counts), str(joback.status)


class EmpiricalTwoDEnergyScorer:
    """Score a graph with named conventional empirical chemistry terms."""

    def __init__(self, config: EmpiricalTwoDEnergyConfig | None = None) -> None:
        self.config = config or EmpiricalTwoDEnergyConfig()
        self._valence_config = ValenceEnergyConfig()

    def score(self, normalized: NormalizedMolecularGraph) -> MoleculeScoreResult:
        """Return a graph-only empirical score in a documented kJ/mol scale."""
        joback, counts, joback_status = _joback_components(normalized.canonical_smiles)
        if not joback:
            backend_missing = joback_status == "THERMO_NOT_INSTALLED"
            return MoleculeScoreResult(
                status="unsupported" if backend_missing else "error",
                score=None,
                units="kJ/mol",
                components={},
                descriptors={
                    "graph_identity": normalized.identity,
                    "canonical_smiles": normalized.canonical_smiles,
                    "joback_status": joback_status,
                    "joback_group_counts": counts,
                },
                warnings=(
                    "THERMO_NOT_INSTALLED"
                    if backend_missing
                    else "JOBACK_FRAGMENTATION_FAILED",
                ),
                provenance=self._provenance(),
            )

        raw, assignment = valence_energy_terms(
            normalized,
            config=self._valence_config,
        )
        original_pi_scale = self._valence_config.pi_energy_scale
        pi_scale = self.config.huckel_resonance_integral_kj_mol / original_pi_scale
        structural_pi = sum(
            raw[name]
            for name in (
                "aromatic_pi_delocalization",
                "mixed_pi_delocalization",
                "acyclic_pi_delocalization",
            )
        )
        lone_pair_pi = sum(
            raw[name]
            for name in (
                "carbonyl_n_lone_pair_delocalization",
                "imine_n_lone_pair_delocalization",
                "aryl_n_lone_pair_delocalization",
                "other_n_lone_pair_delocalization",
                "sulfonyl_n_lone_pair_delocalization",
                "oxygen_lone_pair_delocalization",
                "sulfur_lone_pair_delocalization",
                "mixed_n_lone_pair_delocalization",
                "mixed_element_lone_pair_delocalization",
            )
        )
        components = {
            **joback,
            "huckel_structural_pi_delocalization": float(structural_pi * pi_scale),
            "huckel_lone_pair_delocalization": float(lone_pair_pi * pi_scale),
            "formal_charge_localization": float(
                raw["charge_localization"]
                * self.config.formal_charge_hardness_scale_kj_mol
                / self._valence_config.charge_localization_scale
            ),
            "forced_small_ring_strain": self._small_ring_strain(normalized),
            "forced_unsaturated_ring_strain": self._unsaturated_ring_strain(normalized),
        }
        warnings = tuple(
            dict.fromkeys(
                (
                    *normalized.warning_codes(),
                    *assignment.warning_codes(),
                )
            )
        )
        return MoleculeScoreResult(
            status="partial" if warnings else "success",
            score=float(sum(components.values())),
            units="kJ/mol",
            components=components,
            descriptors={
                "graph_identity": normalized.identity,
                "canonical_smiles": normalized.canonical_smiles,
                "joback_status": joback_status,
                "joback_group_counts": counts,
                "n_orbital_pi_electrons": assignment.electron_count,
                "term_scales": {
                    "joback_group_increments": "published kJ/mol",
                    "huckel_terms": (
                        f"dimensionless Hückel energy x "
                        f"{self.config.huckel_resonance_integral_kj_mol} kJ/mol"
                    ),
                    "strain_terms": "fixed empirical kJ/mol",
                    "formal_charge_localization": (
                        "fixed atomic-hardness proxy in kJ/mol"
                    ),
                },
            },
            warnings=warnings,
            provenance=self._provenance(),
        )

    def _small_ring_strain(self, normalized: NormalizedMolecularGraph) -> float:
        graph = normalized.graph
        value = 0.0
        for cycle in nx.cycle_basis(graph):
            if len(cycle) not in {3, 4}:
                continue
            edges = [
                (cycle[index], cycle[(index + 1) % len(cycle)])
                for index in range(len(cycle))
            ]
            if any(
                float(graph.edges[left, right].get("order", 1.0)) >= 1.5
                for left, right in edges
            ):
                continue
            value += (
                self.config.cyclopropane_strain_kj_mol
                if len(cycle) == 3
                else self.config.cyclobutane_strain_kj_mol
            )
        return float(value)

    def _unsaturated_ring_strain(self, normalized: NormalizedMolecularGraph) -> float:
        graph = normalized.graph
        cycles = nx.minimum_cycle_basis(graph)
        memberships: dict[object, list[int]] = {}
        for cycle in cycles:
            for node in cycle:
                memberships.setdefault(node, []).append(len(cycle))
        value = 0.0
        for left, right, attrs in graph.edges(data=True):
            if bool(attrs.get("aromatic", False)):
                continue
            if float(attrs.get("order", 1.0)) < 1.5:
                continue
            shared = [
                len(cycle) for cycle in cycles if left in cycle and right in cycle
            ]
            if 3 in shared:
                value += self.config.cyclopropene_extra_strain_kj_mol
            elif 4 in shared:
                value += self.config.cyclobutene_extra_strain_kj_mol
            for node in (left, right):
                small = [size for size in memberships.get(node, []) if size <= 7]
                if len(small) >= 2 and graph.degree(node) >= 3:
                    value += self.config.bridgehead_alkene_strain_kj_mol
        return float(value)

    def _provenance(self) -> dict[str, object]:
        return {
            "model_name": "synde-empirical-2d-v1-development",
            "mode": "labeled-graph-only",
            "calibrated": False,
            "fitted_coefficients": False,
            "uses_coordinates": False,
            "uses_conformers": False,
            "uses_xtb": False,
            "uses_ord_labels_at_inference": False,
            "experimental": True,
            "benchmark_informed_development": True,
            "parameter_set": self.config.parameter_set,
            "permanent_holdout_evaluated": False,
            "holdout_tuned": False,
        }


__all__ = [
    "EmpiricalTwoDEnergyConfig",
    "EmpiricalTwoDEnergyScorer",
    "HUCKEL_RESONANCE_INTEGRAL_KJ_MOL",
    "JOBACK_REFERENCE_OFFSET_KJ_MOL",
]
