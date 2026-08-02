"""Conventional, non-overlapping molecular-energy terms from a labeled graph.

This experimental scorer uses Synkit 1.5's full atom and bond labels.  Its
localized Lewis pi reference, coupled pi delocalization, and lone-pair donor
increment are defined as mutually exclusive contributions.  Geometry-only
steric, saturated-ring, dispersion, and nonbonded Coulomb energies are omitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from synde.graph.generalized_huckel import GeneralizedHuckel, HuckelParameters
from synde.graph.graph_schema import GraphWarning, NormalizedMolecularGraph
from synde.graph.orbital_pi import assign_orbital_pi
from synde.graph.pi_system import PiAssignmentResult, PiAtom, PiSystem

from .molecule_scoring import MoleculeScorer
from .results import MoleculeScoreResult
from .theory_energy import theory_energy_corrections

# Pauling-like electronegativity and coarse atomic hardness scales.  Their
# internal ratios are fixed; later calibration may adjust only the complete
# graph-polarization term's global weight.
ELECTRONEGATIVITY = {
    "B": 2.04,
    "C": 2.55,
    "N": 3.04,
    "O": 3.44,
    "F": 3.98,
    "Si": 1.90,
    "P": 2.19,
    "S": 2.58,
    "Cl": 3.16,
    "Br": 2.96,
    "I": 2.66,
}

CHEMICAL_HARDNESS = {
    "B": 4.0,
    "C": 5.0,
    "N": 6.0,
    "O": 7.0,
    "F": 8.0,
    "Si": 3.8,
    "P": 4.5,
    "S": 5.0,
    "Cl": 6.0,
    "Br": 5.5,
    "I": 4.8,
}

# Fixed localized hetero-pi increments inherited from the frozen theory model.
# They correct the generic two-orbital Hückel scale once per multiple bond.
HETERO_LOCAL_PI_CORRECTION = {
    frozenset(("C", "O")): -1.25,
    frozenset(("C", "N")): -0.75,
    frozenset(("N", "N")): -0.30,
    frozenset(("N", "O")): -1.50,
    frozenset(("O", "O")): -1.25,
    frozenset(("C", "S")): -0.40,
}


@dataclass(frozen=True)
class ValenceEnergyWeights:
    """Global weights; the uncalibrated conventional model uses all ones."""

    atom_reference: float = 1.0
    lewis_sigma_reference: float = 1.0
    lewis_local_pi_reference: float = 1.0
    # Connectivity provides different confidence about planarity in these
    # environments, so they remain separately calibratable.
    aromatic_pi_delocalization: float = 0.0
    mixed_pi_delocalization: float = 0.0
    acyclic_pi_delocalization: float = 0.0
    carbonyl_n_lone_pair_delocalization: float = 0.0
    imine_n_lone_pair_delocalization: float = 0.0
    aryl_n_lone_pair_delocalization: float = 1.0
    other_n_lone_pair_delocalization: float = 0.0
    sulfonyl_n_lone_pair_delocalization: float = 0.0
    oxygen_lone_pair_delocalization: float = 1.0
    sulfur_lone_pair_delocalization: float = 1.0
    mixed_n_lone_pair_delocalization: float = 0.0
    mixed_element_lone_pair_delocalization: float = 1.0
    hyperconjugation: float = 0.0
    graph_polarization: float = 0.0
    charge_localization: float = 1.0
    protobranching: float = 1.0
    hard_topology: float = 1.0
    small_saturated_ring_strain: float = 1.0


@dataclass(frozen=True)
class ValenceEnergyConfig:
    """Fixed internal parameters for the experimental conventional scorer."""

    parameter_set: str = "valence-2d-v1-development"
    pi_energy_scale: float = 1.33
    polarization_bond_coupling: float = 1.0
    charge_localization_scale: float = 0.10
    protobranching_13_reward: float = 0.08
    weights: ValenceEnergyWeights = field(default_factory=ValenceEnergyWeights)


class ValenceEnergyScorer:
    """Score a molecule with explicit, non-overlapping valence-energy terms."""

    def __init__(
        self,
        config: ValenceEnergyConfig | None = None,
        base_scorer: MoleculeScorer | None = None,
        huckel_parameters: HuckelParameters | None = None,
    ) -> None:
        self.config = config or ValenceEnergyConfig()
        self.base_scorer = base_scorer or MoleculeScorer(parameters=huckel_parameters)
        self.huckel = GeneralizedHuckel(
            huckel_parameters or self.base_scorer.huckel.parameters
        )

    def score(self, normalized: NormalizedMolecularGraph) -> MoleculeScoreResult:
        """Return weighted contributions and retain raw terms for calibration."""
        base = self.base_scorer.score(normalized)
        if base.score is None:
            return base
        raw, assignment = valence_energy_terms(
            normalized,
            config=self.config,
            base_result=base,
            huckel=self.huckel,
        )
        weights = self.config.weights.__dict__
        components = {name: float(value * weights[name]) for name, value in raw.items()}
        obsolete_warnings = {
            "PI_ORBITAL_MULTIPLICITY_UNSUPPORTED",
            "ODD_PI_ELECTRON_COUNT",
        }
        warnings = tuple(
            dict.fromkeys(
                (
                    *(w for w in base.warnings if w not in obsolete_warnings),
                    *assignment.warning_codes(),
                )
            )
        )
        descriptors = {
            **base.descriptors,
            "raw_energy_terms": raw,
            "n_orbital_pi_electrons": assignment.electron_count,
            "n_orbital_pi_systems": len(assignment.systems),
            "synkit_label_profile": "full",
        }
        provenance = {
            "model_name": "synde-valence-2d-v1-experimental",
            "mode": "labeled-graph",
            "calibrated": self.config.weights != ValenceEnergyWeights(),
            "fitted_coefficients": False,
            "experimental": True,
            "benchmark_informed_development": True,
            "parameter_set": self.config.parameter_set,
            "permanent_holdout_evaluated": False,
            "holdout_tuned": False,
        }
        return MoleculeScoreResult(
            status="partial" if warnings else "success",
            score=float(sum(components.values())),
            units="score",
            components=components,
            descriptors=descriptors,
            warnings=warnings,
            provenance=provenance,
        )


def valence_energy_terms(
    normalized: NormalizedMolecularGraph,
    config: ValenceEnergyConfig | None = None,
    *,
    base_result: MoleculeScoreResult | None = None,
    huckel: GeneralizedHuckel | None = None,
) -> tuple[dict[str, float], PiAssignmentResult]:
    """Return raw conventional terms before any global weighting."""
    config = config or ValenceEnergyConfig()
    huckel = huckel or GeneralizedHuckel()
    base_result = base_result or MoleculeScorer(parameters=huckel.parameters).score(
        normalized
    )
    expanded = assign_orbital_pi(normalized)
    core = _without_lone_pair_donors(normalized.graph, expanded)
    delocalization = _pi_delocalization_by_class(normalized.graph, core, huckel)
    lone_pair = _lone_pair_delocalization_by_class(normalized.graph, expanded, huckel)
    localized_pi = _localized_pi_reference(expanded, huckel)
    localized_pi_correction = _localized_pi_correction(expanded)
    hard_topology = theory_energy_corrections(normalized.graph)[
        "unsaturated_ring_strain"
    ]
    hyperconjugation = theory_energy_corrections(normalized.graph)[
        "alkene_substitution"
    ]
    raw = {
        "atom_reference": float(base_result.components.get("atom_reference", 0.0)),
        "lewis_sigma_reference": float(
            base_result.components.get("sigma_bond_energy", 0.0)
        ),
        "lewis_local_pi_reference": float(
            config.pi_energy_scale * localized_pi + localized_pi_correction
        ),
        "aromatic_pi_delocalization": float(
            config.pi_energy_scale * delocalization["aromatic"]
        ),
        "mixed_pi_delocalization": float(
            config.pi_energy_scale * delocalization["mixed"]
        ),
        "acyclic_pi_delocalization": float(
            config.pi_energy_scale * delocalization["acyclic"]
        ),
        "carbonyl_n_lone_pair_delocalization": float(
            config.pi_energy_scale * lone_pair["carbonyl_N"]
        ),
        "imine_n_lone_pair_delocalization": float(
            config.pi_energy_scale * lone_pair["imine_N"]
        ),
        "aryl_n_lone_pair_delocalization": float(
            config.pi_energy_scale * lone_pair["aryl_N"]
        ),
        "other_n_lone_pair_delocalization": float(
            config.pi_energy_scale * lone_pair["other_N"]
        ),
        "sulfonyl_n_lone_pair_delocalization": float(
            config.pi_energy_scale * lone_pair["sulfonyl_N"]
        ),
        "oxygen_lone_pair_delocalization": float(
            config.pi_energy_scale * lone_pair["O"]
        ),
        "sulfur_lone_pair_delocalization": float(
            config.pi_energy_scale * lone_pair["S"]
        ),
        "mixed_n_lone_pair_delocalization": float(
            config.pi_energy_scale * lone_pair["mixed_N"]
        ),
        "mixed_element_lone_pair_delocalization": float(
            config.pi_energy_scale * lone_pair["mixed_element"]
        ),
        "hyperconjugation": float(hyperconjugation),
        "graph_polarization": _graph_polarization(normalized.graph, config),
        "charge_localization": _charge_localization(normalized.graph, config),
        "protobranching": _protobranching(normalized.graph, config),
        "hard_topology": float(hard_topology),
        "small_saturated_ring_strain": _small_saturated_ring_strain(normalized.graph),
    }
    return raw, expanded


def _pi_delocalization_by_class(
    graph: nx.Graph,
    assignment: PiAssignmentResult,
    huckel: GeneralizedHuckel,
) -> dict[str, float]:
    """Partition coupled-pi stabilization by graph-identifiable environment."""
    values = {"aromatic": 0.0, "mixed": 0.0, "acyclic": 0.0}
    for system in assignment.systems:
        system_graph = assignment.pi_graph.subgraph(system.nodes).copy()
        local_assignment = _assignment_from_pi_graph(system_graph)
        coupled = huckel.solve(local_assignment).pi_stabilization
        localized = _localized_pi_reference(local_assignment, huckel)
        atoms = {orbital[0] for orbital in system.nodes}
        aromatic_count = sum(
            bool(graph.nodes[atom].get("aromatic", False)) for atom in atoms
        )
        if aromatic_count == len(atoms):
            category = "aromatic"
        elif aromatic_count:
            category = "mixed"
        else:
            category = "acyclic"
        values[category] += float(coupled - localized)
    return values


def _lone_pair_delocalization_by_class(
    graph: nx.Graph,
    assignment: PiAssignmentResult,
    huckel: GeneralizedHuckel,
) -> dict[str, float]:
    """Partition the exact donor increment by donor-element system class."""
    values = {
        "carbonyl_N": 0.0,
        "imine_N": 0.0,
        "aryl_N": 0.0,
        "other_N": 0.0,
        "sulfonyl_N": 0.0,
        "O": 0.0,
        "S": 0.0,
        "mixed_N": 0.0,
        "mixed_element": 0.0,
    }
    for system in assignment.systems:
        system_graph = assignment.pi_graph.subgraph(system.nodes).copy()
        donors = [
            orbital
            for orbital in system.nodes
            if not _is_structural_pi_orbital(graph, orbital)
        ]
        if not donors:
            continue
        full = huckel.solve(_assignment_from_pi_graph(system_graph)).pi_stabilization
        system_graph.remove_nodes_from(donors)
        core = huckel.solve(_assignment_from_pi_graph(system_graph)).pi_stabilization
        elements = {graph.nodes[orbital[0]]["element"] for orbital in donors}
        if elements == {"N"}:
            nitrogen_classes = {
                _nitrogen_donor_class(graph, assignment.pi_graph, orbital)
                for orbital in donors
            }
            category = (
                next(iter(nitrogen_classes))
                if len(nitrogen_classes) == 1
                else "mixed_N"
            )
        else:
            category = next(iter(elements)) if len(elements) == 1 else "mixed_element"
        values[category] += float(full - core)
    return values


def _nitrogen_donor_class(
    graph: nx.Graph, orbital_graph: nx.Graph, donor_orbital: Any
) -> str:
    """Classify an N donor by the acceptor environment fixed by connectivity."""
    targets = [orbital[0] for orbital in orbital_graph.neighbors(donor_orbital)]
    for target in targets:
        if graph.nodes[target].get("aromatic", False):
            return "aryl_N"
        if (
            graph.nodes[target]["element"] == "S"
            and sum(
                graph.nodes[neighbor]["element"] == "O"
                and float(graph.edges[target, neighbor].get("order", 1.0)) >= 1.5
                for neighbor in graph.neighbors(target)
            )
            >= 2
        ):
            return "sulfonyl_N"
        if graph.nodes[target]["element"] != "C":
            continue
        neighbors = [
            node for node in graph.neighbors(target) if node != donor_orbital[0]
        ]
        if any(
            graph.nodes[node]["element"] in {"O", "S"}
            and float(graph.edges[target, node].get("order", 1.0)) >= 1.5
            for node in neighbors
        ):
            return "carbonyl_N"
        if any(
            graph.nodes[node]["element"] == "N"
            and float(graph.edges[target, node].get("order", 1.0)) >= 1.5
            for node in neighbors
        ):
            return "imine_N"
    return "other_N"


def _protobranching(graph: nx.Graph, config: ValenceEnergyConfig) -> float:
    """Reward excess 1,3 C-C interactions relative to an unbranched center."""
    excess_interactions = 0
    for node, attrs in graph.nodes(data=True):
        if attrs["element"] != "C" or attrs.get("hybridization") != "SP3":
            continue
        carbon_degree = sum(
            graph.nodes[neighbor]["element"] == "C"
            for neighbor in graph.neighbors(node)
        )
        all_pairs = carbon_degree * (carbon_degree - 1) // 2
        linear_reference = max(0, carbon_degree - 1)
        excess_interactions += max(0, all_pairs - linear_reference)
    return float(-config.protobranching_13_reward * excess_interactions)


def _small_saturated_ring_strain(graph: nx.Graph) -> float:
    """Return only unavoidable saturated three/four-membered ring strain."""
    penalty = 0.0
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
        penalty += 1.15 if len(cycle) == 3 else 1.10
    return float(penalty)


def _without_lone_pair_donors(
    graph: nx.Graph, assignment: PiAssignmentResult
) -> PiAssignmentResult:
    pi_graph = assignment.pi_graph.copy()
    donor_orbitals = [
        orbital
        for orbital in pi_graph.nodes
        if not _is_structural_pi_orbital(graph, orbital)
    ]
    pi_graph.remove_nodes_from(donor_orbitals)
    return _assignment_from_pi_graph(
        pi_graph,
        status=assignment.status,
        warnings=assignment.warnings,
    )


def _is_structural_pi_orbital(graph: nx.Graph, orbital: Any) -> bool:
    atom, orbital_index = orbital
    if orbital_index > 0:
        return any(
            float(attrs.get("order", 1.0)) >= 1.5
            for _, _, attrs in graph.edges(atom, data=True)
        )
    attrs = graph.nodes[atom]
    return bool(attrs.get("aromatic", False)) or any(
        float(edge.get("order", 1.0)) >= 1.5
        for _, _, edge in graph.edges(atom, data=True)
    )


def _localized_pi_reference(
    assignment: PiAssignmentResult, huckel: GeneralizedHuckel
) -> float:
    """Sum isolated two-electron pi bonds selected by Synkit Kekule labels."""
    energy = 0.0
    for left, right, attrs in assignment.pi_graph.edges(data=True):
        pi_order = float(
            attrs.get(
                "pi_order",
                max(0.0, float(attrs.get("order", 1.0)) - 1.0),
            )
        )
        if pi_order < 0.5:
            continue
        local_graph = nx.Graph()
        for orbital in (left, right):
            node_attrs = dict(assignment.pi_graph.nodes[orbital])
            node_attrs["pi_electrons"] = 1
            local_graph.add_node(orbital, **node_attrs)
        local_graph.add_edge(left, right, **attrs)
        local = _assignment_from_pi_graph(local_graph)
        energy += huckel.solve(local).pi_stabilization
    return float(energy)


def _localized_pi_correction(assignment: PiAssignmentResult) -> float:
    """Return one fixed hetero-pi increment per localized multiple bond."""
    corrected_bonds: set[frozenset[Any]] = set()
    correction = 0.0
    for left, right, attrs in assignment.pi_graph.edges(data=True):
        if attrs.get("aromatic", False):
            continue
        pi_order = float(
            attrs.get(
                "pi_order",
                max(0.0, float(attrs.get("order", 1.0)) - 1.0),
            )
        )
        if pi_order < 0.5:
            continue
        atom_bond = frozenset((left[0], right[0]))
        if atom_bond in corrected_bonds:
            continue
        corrected_bonds.add(atom_bond)
        elements = frozenset(
            (
                assignment.pi_graph.nodes[left]["element"],
                assignment.pi_graph.nodes[right]["element"],
            )
        )
        correction += HETERO_LOCAL_PI_CORRECTION.get(elements, 0.0)
    return float(correction)


def _assignment_from_pi_graph(
    pi_graph: nx.Graph,
    *,
    status: str = "success",
    warnings: tuple[GraphWarning, ...] = (),
) -> PiAssignmentResult:
    atoms = tuple(
        PiAtom(
            node=node,
            included=True,
            electrons=int(attrs.get("pi_electrons", 0)),
            reason="conventional valence partition",
            confidence="medium",
        )
        for node, attrs in pi_graph.nodes(data=True)
    )
    systems = []
    for nodes in nx.connected_components(pi_graph):
        ordered = tuple(sorted(nodes, key=repr))
        edges = tuple(
            sorted(
                (
                    tuple(sorted((left, right), key=repr))
                    for left, right in pi_graph.subgraph(nodes).edges
                ),
                key=repr,
            )
        )
        systems.append(
            PiSystem(
                nodes=ordered,
                edges=edges,
                electron_count=sum(
                    int(pi_graph.nodes[node].get("pi_electrons", 0)) for node in nodes
                ),
            )
        )
    return PiAssignmentResult(
        pi_graph=pi_graph,
        atoms=atoms,
        systems=tuple(sorted(systems, key=lambda item: repr(item.nodes))),
        status=status,
        warnings=warnings,
    )


def _graph_polarization(graph: nx.Graph, config: ValenceEnergyConfig) -> float:
    """Return a bonded electronegativity-equalization stabilization."""
    total = 0.0
    heavy = [node for node, attrs in graph.nodes(data=True) if attrs["element"] != "H"]
    heavy_graph = graph.subgraph(heavy)
    for nodes in nx.connected_components(heavy_graph):
        ordered = tuple(sorted(nodes, key=repr))
        if len(ordered) < 2:
            continue
        index = {node: position for position, node in enumerate(ordered)}
        size = len(ordered)
        matrix = np.zeros((size, size), dtype=float)
        chi = np.zeros(size, dtype=float)
        for node, position in index.items():
            element = graph.nodes[node]["element"]
            matrix[position, position] = CHEMICAL_HARDNESS.get(element, 5.0)
            chi[position] = ELECTRONEGATIVITY.get(element, 2.5)
        for left, right, attrs in heavy_graph.subgraph(nodes).edges(data=True):
            coupling = config.polarization_bond_coupling * min(
                2.0, float(attrs.get("order", 1.0))
            )
            i, j = index[left], index[right]
            matrix[i, i] += coupling
            matrix[j, j] += coupling
            matrix[i, j] -= coupling
            matrix[j, i] -= coupling
        kkt = np.zeros((size + 1, size + 1), dtype=float)
        kkt[:size, :size] = matrix
        kkt[:size, size] = 1.0
        kkt[size, :size] = 1.0
        rhs = np.concatenate((-chi, np.asarray([0.0])))
        solution = np.linalg.solve(kkt, rhs)[:size]
        total += float(chi @ solution + 0.5 * solution @ matrix @ solution)
    return float(total)


def _charge_localization(graph: nx.Graph, config: ValenceEnergyConfig) -> float:
    score = 0.0
    for _, attrs in graph.nodes(data=True):
        charge = int(attrs.get("formal_charge", 0))
        if not charge:
            continue
        hardness = CHEMICAL_HARDNESS.get(attrs["element"], 5.0)
        score += hardness * charge * charge
    return float(config.charge_localization_scale * score)


__all__ = [
    "ValenceEnergyConfig",
    "ValenceEnergyScorer",
    "ValenceEnergyWeights",
    "valence_energy_terms",
]
