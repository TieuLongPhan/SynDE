"""A first-order empirical energy ledger using only labeled 2D connectivity.

The localized reference is a sum of conventional average bond enthalpies.
Hückel delocalization is evaluated relative to those localized bonds, so it is
not a second pi-bond energy.  The remaining terms represent resonance,
hyperconjugation, formal-charge localization, protobranching, and strain that
is forced by connectivity.

All active constants precede the ORD evaluation and are expressed on one
kJ/mol bookkeeping scale.  Average bond enthalpies are deliberately coarse:
the output is an interpretable ranking score, not a thermochemical prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from synde.graph.generalized_huckel import GeneralizedHuckel, HuckelParameters
from synde.graph.graph_schema import NormalizedMolecularGraph

from .results import MoleculeScoreResult
from .valence_energy import ValenceEnergyConfig, valence_energy_terms


# Conventional gas-phase average bond enthalpies, kJ/mol.  Keys are sorted
# element pairs followed by integer bond order.  The table is deliberately
# compact; missing environments fall back to the corresponding single-bond
# value rather than silently introducing a fitted fragment coefficient.
AVERAGE_BOND_ENTHALPY_KJ_MOL: dict[tuple[str, str, int], float] = {
    ("B", "H", 1): 389.0,
    ("Br", "C", 1): 285.0,
    ("C", "C", 1): 347.0,
    ("C", "C", 2): 614.0,
    ("C", "C", 3): 839.0,
    ("C", "Cl", 1): 327.0,
    ("C", "F", 1): 485.0,
    ("C", "H", 1): 413.0,
    ("C", "I", 1): 213.0,
    ("C", "N", 1): 305.0,
    ("C", "N", 2): 615.0,
    ("C", "N", 3): 891.0,
    ("C", "O", 1): 358.0,
    ("C", "O", 2): 745.0,
    ("C", "P", 1): 264.0,
    ("C", "S", 1): 272.0,
    ("C", "S", 2): 573.0,
    ("H", "N", 1): 391.0,
    ("H", "O", 1): 463.0,
    ("H", "P", 1): 322.0,
    ("H", "S", 1): 347.0,
    ("H", "Si", 1): 318.0,
    ("N", "N", 1): 163.0,
    ("N", "N", 2): 418.0,
    ("N", "N", 3): 945.0,
    ("N", "O", 1): 201.0,
    ("N", "O", 2): 607.0,
    ("O", "O", 1): 146.0,
    ("O", "O", 2): 498.0,
    ("O", "P", 1): 335.0,
    ("O", "P", 2): 544.0,
    ("O", "S", 1): 364.0,
    ("O", "S", 2): 523.0,
    ("S", "S", 1): 266.0,
    ("Si", "Si", 1): 226.0,
}

IMPLICIT_H_BOND_ENTHALPY_KJ_MOL = {
    "B": 389.0,
    "C": 413.0,
    "N": 391.0,
    "O": 463.0,
    "P": 322.0,
    "S": 347.0,
    "Si": 318.0,
}

CARBON_H_BOND_ENTHALPY_BY_HYBRIDIZATION_KJ_MOL = {
    "SP3": 413.0,
    "SP2": 464.0,
    "SP": 536.0,
}


@dataclass(frozen=True)
class FirstOrderTwoDEnergyConfig:
    """Fixed conventional constants for the first-order graph score."""

    parameter_set: str = "first-order-2d-empirical-v1-frozen"
    huckel_resonance_integral_kj_mol: float = 75.0
    hyperconjugation_per_alkyl_substituent_kj_mol: float = 6.0
    protobranching_per_excess_13_contact_kj_mol: float = 8.0
    carbon_graph_energy_scale_kj_mol: float = 8.0
    carbon_wiener_scale_kj_mol: float = 8.0
    rigid_intramolecular_hbond_kj_mol: float = 20.0
    cyclopropane_strain_kj_mol: float = 115.0
    cyclobutane_strain_kj_mol: float = 110.0
    cyclopropene_extra_strain_kj_mol: float = 55.0
    cyclobutene_extra_strain_kj_mol: float = 25.0
    bridgehead_alkene_strain_kj_mol: float = 50.0
    formal_charge_hardness_scale_kj_mol: float = 9.6485
    evaluate_traditional_huckel_sensitivity: bool = False


class FirstOrderTwoDEnergyScorer:
    """Evaluate named, non-overlapping first-order empirical graph terms."""

    def __init__(self, config: FirstOrderTwoDEnergyConfig | None = None) -> None:
        self.config = config or FirstOrderTwoDEnergyConfig()
        self._valence_config = ValenceEnergyConfig()
        self._active_huckel = GeneralizedHuckel()
        self._traditional_huckel = (
            GeneralizedHuckel(
                HuckelParameters(
                    parameter_set="traditional-heteroatom-huckel-v1",
                    beta_aromatic=-1.0,
                    beta_double=-1.0,
                    beta_conjugated=-0.8,
                    hetero_beta_scale=0.85,
                    alpha={
                        "C_aromatic": 0.0,
                        "C_sp2": 0.0,
                        "N_aromatic_pyridine": -0.5,
                        "N_aromatic_pyrrole": -1.5,
                        "N_sp2": -0.5,
                        "O_aromatic": -2.0,
                        "O_sp2": -1.0,
                        "S_aromatic": -1.5,
                        "S_sp2": -1.0,
                    },
                )
            )
            if self.config.evaluate_traditional_huckel_sensitivity
            else None
        )

    def score(self, normalized: NormalizedMolecularGraph) -> MoleculeScoreResult:
        active_raw, assignment = valence_energy_terms(
            normalized,
            config=self._valence_config,
            huckel=self._active_huckel,
        )
        pi_conversion = (
            self.config.huckel_resonance_integral_kj_mol
            / self._valence_config.pi_energy_scale
        )
        structural_names = (
            "aromatic_pi_delocalization",
            "mixed_pi_delocalization",
            "acyclic_pi_delocalization",
        )
        donor_names = (
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
        active_structural_pi = {
            name: float(active_raw[name] * pi_conversion)
            for name in structural_names
        }
        active_donor_pi = {
            name: float(active_raw[name] * pi_conversion)
            for name in donor_names
        }
        inactive_pi_names = {
            "imine_n_lone_pair_delocalization",
            "other_n_lone_pair_delocalization",
            "sulfonyl_n_lone_pair_delocalization",
            "sulfur_lone_pair_delocalization",
        }
        selected_structural_pi = {
            name: value
            for name, value in active_structural_pi.items()
            if name not in inactive_pi_names
        }
        selected_donor_pi = {
            name: value
            for name, value in active_donor_pi.items()
            if name not in inactive_pi_names
        }
        raw_topology = self._raw_topology_terms(normalized)
        raw_hyperconjugation = self._hyperconjugation(normalized)
        unsaturated_strain = self._unsaturated_ring_strain_terms(normalized)
        components = {
            "localized_bond_enthalpy": self._localized_bond_energy(normalized),
            **selected_structural_pi,
            **selected_donor_pi,
            "formal_charge_localization": float(
                active_raw["charge_localization"]
                * self.config.formal_charge_hardness_scale_kj_mol
                / self._valence_config.charge_localization_scale
            ),
            "protobranching_13": self._protobranching(normalized),
            "carbon_skeleton_graph_energy": raw_topology[
                "carbon_skeleton_graph_energy"
            ],
            "forced_small_ring_strain": self._small_ring_strain(normalized),
            "small_unsaturated_ring_strain": unsaturated_strain[
                "small_unsaturated_ring_strain"
            ],
        }
        warnings = tuple(
            dict.fromkeys(
                (
                    *normalized.warning_codes(),
                    *assignment.warning_codes(),
                )
            )
        )
        hybridization_bonds = self._localized_bond_energy(
            normalized,
            resolve_carbon_hybridization=True,
        )
        active_score = float(sum(components.values()))
        all_terms_score = float(
            active_score
            + sum(active_structural_pi.get(name, 0.0) for name in inactive_pi_names)
            + sum(active_donor_pi.get(name, 0.0) for name in inactive_pi_names)
            + unsaturated_strain["bridgehead_alkene_strain"]
        )
        alternative_huckel_scores = {
            "organic_v1_for_all_pi": active_score,
        }
        huckel_parameter_sets = {
            "active": self._active_huckel.parameters.parameter_set,
        }
        if self._traditional_huckel is not None:
            traditional_raw, _ = valence_energy_terms(
                normalized,
                config=self._valence_config,
                huckel=self._traditional_huckel,
            )
            traditional_structural_pi = {
                name: float(traditional_raw[name] * pi_conversion)
                for name in structural_names
            }
            traditional_donor_pi = {
                name: float(traditional_raw[name] * pi_conversion)
                for name in donor_names
            }
            alternative_huckel_scores.update(
                {
                    "traditional_for_all_pi": float(
                        active_score
                        - sum(active_structural_pi.values())
                        - sum(active_donor_pi.values())
                        + sum(traditional_structural_pi.values())
                        + sum(traditional_donor_pi.values())
                    ),
                    "traditional_structural_plus_organic_v1_donor": float(
                        active_score
                        - sum(active_structural_pi.values())
                        + sum(traditional_structural_pi.values())
                    ),
                }
            )
            huckel_parameter_sets["traditional_sensitivity"] = (
                self._traditional_huckel.parameters.parameter_set
            )
        return MoleculeScoreResult(
            status="partial" if warnings else "success",
            score=active_score,
            units="kJ/mol_score",
            components=components,
            descriptors={
                "graph_identity": normalized.identity,
                "canonical_smiles": normalized.canonical_smiles,
                "n_orbital_pi_electrons": assignment.electron_count,
                "n_orbital_pi_systems": len(assignment.systems),
                "huckel_parameter_sets": huckel_parameter_sets,
                "term_scales": {
                    "localized_bond_enthalpy": "average kJ/mol bond enthalpies",
                    "pi_and_resonance": (
                        "dimensionless Hückel increment x 75 kJ/mol"
                    ),
                    "remaining_terms": "fixed empirical kJ/mol",
                },
                "raw_topology_terms": raw_topology,
                "inactive_empirical_terms": {
                    "alkene_hyperconjugation": raw_hyperconjugation,
                    **{
                        name: active_structural_pi[name]
                        for name in inactive_pi_names
                        if name in active_structural_pi
                    },
                    **{
                        name: active_donor_pi[name]
                        for name in inactive_pi_names
                        if name in active_donor_pi
                    },
                    "bridgehead_alkene_strain": unsaturated_strain[
                        "bridgehead_alkene_strain"
                    ],
                    "carbon_skeleton_mean_wiener": raw_topology[
                        "carbon_skeleton_mean_wiener"
                    ],
                    "rigid_intramolecular_hbond": raw_topology[
                        "rigid_intramolecular_hbond"
                    ],
                },
                "raw_bond_alternatives": {
                    "hybridization_resolved_carbon_h": hybridization_bonds,
                },
                "alternative_bond_scores": {
                    "hybridization_resolved_carbon_h": float(
                        active_score
                        - components["localized_bond_enthalpy"]
                        + hybridization_bonds
                    ),
                },
                "alternative_term_scores": {
                    "all_empirical_terms": all_terms_score,
                },
                "alternative_huckel_scores": alternative_huckel_scores,
            },
            warnings=warnings,
            provenance={
                "model_name": "synde-first-order-2d-v1-frozen",
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
                "permanent_holdout_evaluated": True,
                "permanent_holdout_protocol": (
                    "synde-ord-first-order-2d-permanent-holdout-v1"
                ),
                "permanent_holdout_mean_pearson": 0.6273585365824424,
                "permanent_holdout_mean_spearman": 0.5588531582778654,
                "holdout_tuned": False,
            },
        )

    def _raw_topology_terms(
        self, normalized: NormalizedMolecularGraph
    ) -> dict[str, float]:
        """Return traditional carbon-skeleton indices before term selection."""
        graph = normalized.graph
        carbon_nodes = [
            node
            for node, attrs in graph.nodes(data=True)
            if attrs["element"] == "C"
        ]
        carbon_graph = graph.subgraph(carbon_nodes)
        spectral = 0.0
        wiener_mean = 0.0
        pair_count = 0
        for nodes in nx.connected_components(carbon_graph):
            subgraph = carbon_graph.subgraph(nodes)
            ordered = tuple(sorted(nodes, key=repr))
            if len(ordered) >= 2:
                adjacency = nx.to_numpy_array(
                    subgraph,
                    nodelist=ordered,
                    weight=None,
                    dtype=float,
                )
                spectral += float(np.sum(np.abs(np.linalg.eigvalsh(adjacency))))
            paths = dict(nx.all_pairs_shortest_path_length(subgraph))
            for index, left in enumerate(ordered):
                for right in ordered[index + 1 :]:
                    wiener_mean += float(paths[left][right])
                    pair_count += 1
        if pair_count:
            wiener_mean /= pair_count
        return {
            "carbon_skeleton_graph_energy": float(
                self.config.carbon_graph_energy_scale_kj_mol * spectral
            ),
            "carbon_skeleton_mean_wiener": float(
                self.config.carbon_wiener_scale_kj_mol * wiener_mean
            ),
            "rigid_intramolecular_hbond": self._rigid_intramolecular_hbond(
                normalized
            ),
        }

    def _rigid_intramolecular_hbond(
        self, normalized: NormalizedMolecularGraph
    ) -> float:
        """Reward only graph-constrained five- or six-member H-bond motifs."""
        graph = normalized.graph
        donors = [
            node
            for node, attrs in graph.nodes(data=True)
            if attrs["element"] in {"N", "O", "S"}
            and int(attrs.get("total_hcount", 0)) > 0
            and int(attrs.get("formal_charge", 0)) <= 0
        ]
        acceptors = [
            node
            for node, attrs in graph.nodes(data=True)
            if attrs["element"] in {"N", "O", "S"}
            and int(attrs.get("formal_charge", 0)) <= 0
            and (
                bool(attrs.get("available_lone_pairs", 0))
                or int(attrs.get("estimated_lone_pairs", 0)) > 0
            )
        ]
        opportunities: set[tuple[object, object]] = set()
        for donor in donors:
            for acceptor in acceptors:
                if donor == acceptor:
                    continue
                try:
                    path = nx.shortest_path(graph, donor, acceptor)
                except nx.NetworkXNoPath:
                    continue
                if len(path) not in {5, 6}:
                    continue
                flexible_bonds = 0
                rigid_bonds = 0
                for left, right in zip(path, path[1:]):
                    edge = graph.edges[left, right]
                    rigid = (
                        bool(edge.get("aromatic", False))
                        or bool(edge.get("conjugated", False))
                        or bool(edge.get("in_ring", False))
                        or float(edge.get("order", 1.0)) >= 1.5
                    )
                    rigid_bonds += int(rigid)
                    flexible_bonds += int(not rigid)
                if flexible_bonds <= 1 and rigid_bonds >= len(path) - 2:
                    opportunities.add((donor, acceptor))
        selected = 0
        used_atoms: set[object] = set()
        for donor, acceptor in sorted(opportunities, key=repr):
            if donor in used_atoms or acceptor in used_atoms:
                continue
            used_atoms.update((donor, acceptor))
            selected += 1
        return float(
            -self.config.rigid_intramolecular_hbond_kj_mol
            * selected
        )

    def _localized_bond_energy(
        self,
        normalized: NormalizedMolecularGraph,
        *,
        resolve_carbon_hybridization: bool = False,
    ) -> float:
        graph = normalized.graph
        total = 0.0
        for left, right, attrs in graph.edges(data=True):
            elements = tuple(
                sorted(
                    (
                        str(graph.nodes[left]["element"]),
                        str(graph.nodes[right]["element"]),
                    )
                )
            )
            order = int(
                round(float(attrs.get("kekule_order", attrs.get("order", 1.0))))
            )
            key = (elements[0], elements[1], max(1, min(3, order)))
            fallback = (elements[0], elements[1], 1)
            total += AVERAGE_BOND_ENTHALPY_KJ_MOL.get(
                key,
                AVERAGE_BOND_ENTHALPY_KJ_MOL.get(fallback, 330.0),
            )
        for _, attrs in graph.nodes(data=True):
            element = str(attrs["element"])
            if element == "H":
                continue
            strength = IMPLICIT_H_BOND_ENTHALPY_KJ_MOL.get(element, 350.0)
            if resolve_carbon_hybridization and element == "C":
                strength = CARBON_H_BOND_ENTHALPY_BY_HYBRIDIZATION_KJ_MOL.get(
                    str(attrs.get("hybridization", "SP3")),
                    strength,
                )
            total += int(attrs.get("total_hcount", 0)) * strength
        return float(-total)

    def _hyperconjugation(self, normalized: NormalizedMolecularGraph) -> float:
        graph = normalized.graph
        substituents = 0
        for left, right, attrs in graph.edges(data=True):
            if bool(attrs.get("aromatic", False)):
                continue
            if float(attrs.get("order", 1.0)) < 1.5:
                continue
            if (
                graph.nodes[left]["element"] != "C"
                or graph.nodes[right]["element"] != "C"
            ):
                continue
            substituents += sum(
                graph.nodes[node]["element"] == "C"
                for node in graph.neighbors(left)
                if node != right
            )
            substituents += sum(
                graph.nodes[node]["element"] == "C"
                for node in graph.neighbors(right)
                if node != left
            )
        return float(
            -self.config.hyperconjugation_per_alkyl_substituent_kj_mol * substituents
        )

    def _protobranching(self, normalized: NormalizedMolecularGraph) -> float:
        graph = normalized.graph
        excess = 0
        for node, attrs in graph.nodes(data=True):
            if attrs["element"] != "C" or attrs.get("hybridization") != "SP3":
                continue
            carbon_degree = sum(
                graph.nodes[neighbor]["element"] == "C"
                for neighbor in graph.neighbors(node)
            )
            pairs = carbon_degree * (carbon_degree - 1) // 2
            excess += max(0, pairs - max(0, carbon_degree - 1))
        return float(-self.config.protobranching_per_excess_13_contact_kj_mol * excess)

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

    def _unsaturated_ring_strain_terms(
        self, normalized: NormalizedMolecularGraph
    ) -> dict[str, float]:
        graph = normalized.graph
        cycles = nx.minimum_cycle_basis(graph)
        memberships: dict[object, list[int]] = {}
        for cycle in cycles:
            for node in cycle:
                memberships.setdefault(node, []).append(len(cycle))
        small_ring = 0.0
        bridgehead = 0.0
        for left, right, attrs in graph.edges(data=True):
            if bool(attrs.get("aromatic", False)):
                continue
            if float(attrs.get("order", 1.0)) < 1.5:
                continue
            shared = [
                len(cycle) for cycle in cycles if left in cycle and right in cycle
            ]
            if 3 in shared:
                small_ring += self.config.cyclopropene_extra_strain_kj_mol
            elif 4 in shared:
                small_ring += self.config.cyclobutene_extra_strain_kj_mol
            for node in (left, right):
                small = [size for size in memberships.get(node, []) if size <= 7]
                if len(small) >= 2 and graph.degree(node) >= 3:
                    bridgehead += self.config.bridgehead_alkene_strain_kj_mol
        return {
            "small_unsaturated_ring_strain": float(small_ring),
            "bridgehead_alkene_strain": float(bridgehead),
        }


__all__ = [
    "AVERAGE_BOND_ENTHALPY_KJ_MOL",
    "CARBON_H_BOND_ENTHALPY_BY_HYBRIDIZATION_KJ_MOL",
    "FirstOrderTwoDEnergyConfig",
    "FirstOrderTwoDEnergyScorer",
]
