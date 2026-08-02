"""Named empirical 2D terms for an interpretable second-generation score.

The feature vocabulary is deliberately unhashed.  Every value is an exact
function of the labeled molecular graph and has a human-readable chemical or
chemical-graph meaning.  The uncalibrated scorer adds only conventional
kJ/mol-scale increments to the frozen first-order v1 ledger; dimensionless
descriptors are exposed for a separately identified sparse calibration model.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, GraphDescriptors, rdMolDescriptors
from rdkit.Chem.EState import EState
from rdkit.Chem import rdPartialCharges

from synde.graph.generalized_huckel import GeneralizedHuckel
from synde.graph.graph_schema import NormalizedMolecularGraph
from synde.graph.orbital_pi import assign_orbital_pi

from .first_order_two_d_energy import FirstOrderTwoDEnergyScorer
from .results import MoleculeScoreResult

NAMED_FEATURE_SCHEMA = "synde-interpretable-2d-v2-named-features-v1"


@dataclass(frozen=True)
class InterpretableTwoDV2Config:
    """Chemistry-fixed increments for the uncalibrated v2 candidate."""

    parameter_set: str = "interpretable-2d-v2-uncalibrated-development"
    cyclopentane_strain_kj_mol: float = 26.0
    cycloheptane_strain_kj_mol: float = 26.0
    cyclooctane_strain_kj_mol: float = 40.0
    cyclononane_strain_kj_mol: float = 54.0
    cyclodecane_strain_kj_mol: float = 52.0


def _add(features: dict[str, float], name: str, value: float = 1.0) -> None:
    if math.isfinite(value) and value != 0:
        features[name] = features.get(name, 0.0) + float(value)


def _bond_class(attrs: dict[str, Any]) -> str:
    if bool(attrs.get("aromatic", False)):
        return "aromatic"
    order = float(attrs.get("kekule_order", attrs.get("order", 1.0)))
    if order >= 2.5:
        return "triple"
    if order >= 1.5:
        return "double"
    return "single"


def _atom_class(graph: nx.Graph, node: Any) -> str:
    attrs = graph.nodes[node]
    return ";".join(
        (
            str(attrs["element"]),
            f"hyb={attrs.get('hybridization', 'UNSPECIFIED')}",
            f"arom={int(bool(attrs.get('aromatic', False)))}",
            f"q={int(attrs.get('formal_charge', 0)):+d}",
            f"H={int(attrs.get('total_hcount', 0))}",
            f"deg={sum(graph.nodes[n]['element'] != 'H' for n in graph.neighbors(node))}",
            f"ring={int(bool(attrs.get('is_in_ring', attrs.get('in_ring', False))))}",
        )
    )


def _element_state(graph: nx.Graph, node: Any) -> str:
    attrs = graph.nodes[node]
    return ";".join(
        (
            str(attrs["element"]),
            f"hyb={attrs.get('hybridization', 'UNSPECIFIED')}",
            f"arom={int(bool(attrs.get('aromatic', False)))}",
            f"q={int(attrs.get('formal_charge', 0)):+d}",
        )
    )


def _canonical_sequence(values: list[str]) -> str:
    forward = "|".join(values)
    reverse = "|".join(reversed(values))
    return min(forward, reverse)


def _add_local_environment_features(
    features: dict[str, float], graph: nx.Graph
) -> None:
    heavy_nodes = [node for node in graph.nodes if graph.nodes[node]["element"] != "H"]
    for node in heavy_nodes:
        atom = _atom_class(graph, node)
        _add(features, f"atom_state[{atom}]")
        neighbors = sorted(
            f"{_bond_class(graph.edges[node, neighbor])}:{_element_state(graph, neighbor)}"
            for neighbor in graph.neighbors(node)
            if graph.nodes[neighbor]["element"] != "H"
        )
        _add(features, f"benson_nn[{atom}|{','.join(neighbors)}]")

    for left, right, attrs in graph.edges(data=True):
        if "H" in {graph.nodes[left]["element"], graph.nodes[right]["element"]}:
            continue
        endpoints = sorted((_atom_class(graph, left), _atom_class(graph, right)))
        bond = _bond_class(attrs)
        flags = (
            f"conj={int(bool(attrs.get('conjugated', False)))};"
            f"ring={int(bool(attrs.get('is_in_ring', attrs.get('in_ring', False))))}"
        )
        _add(
            features,
            f"bond_environment[{endpoints[0]}|{bond}|{endpoints[1]}|{flags}]",
        )

    # Named 1,3 paths: endpoint--center--endpoint.
    for center in heavy_nodes:
        neighbors = [
            node
            for node in graph.neighbors(center)
            if graph.nodes[node]["element"] != "H"
        ]
        for index, left in enumerate(neighbors):
            for right in neighbors[index + 1 :]:
                sequence = [
                    _element_state(graph, left),
                    _bond_class(graph.edges[left, center]),
                    _atom_class(graph, center),
                    _bond_class(graph.edges[center, right]),
                    _element_state(graph, right),
                ]
                _add(features, f"path_1_3[{_canonical_sequence(sequence)}]")

    # Named 1,4 paths, generated once around their central bond.
    seen: set[tuple[Any, ...]] = set()
    for center_left, center_right in graph.edges:
        if "H" in {
            graph.nodes[center_left]["element"],
            graph.nodes[center_right]["element"],
        }:
            continue
        for left in graph.neighbors(center_left):
            if left == center_right or graph.nodes[left]["element"] == "H":
                continue
            for right in graph.neighbors(center_right):
                if right == center_left or graph.nodes[right]["element"] == "H":
                    continue
                path = (left, center_left, center_right, right)
                reverse = tuple(reversed(path))
                identity = min(path, reverse, key=repr)
                if len(set(path)) != 4 or identity in seen:
                    continue
                seen.add(identity)
                sequence = [
                    _element_state(graph, left),
                    _bond_class(graph.edges[left, center_left]),
                    _atom_class(graph, center_left),
                    _bond_class(graph.edges[center_left, center_right]),
                    _atom_class(graph, center_right),
                    _bond_class(graph.edges[center_right, right]),
                    _element_state(graph, right),
                ]
                _add(features, f"path_1_4[{_canonical_sequence(sequence)}]")


def _add_ring_features(features: dict[str, float], graph: nx.Graph) -> None:
    heavy = graph.subgraph(
        [node for node in graph if graph.nodes[node]["element"] != "H"]
    ).copy()
    cycles = [set(cycle) for cycle in nx.minimum_cycle_basis(heavy)]
    for cycle in cycles:
        elements = ",".join(sorted(graph.nodes[node]["element"] for node in cycle))
        aromatic = sum(bool(graph.nodes[node].get("aromatic", False)) for node in cycle)
        unsaturated_edges = sum(
            left in cycle
            and right in cycle
            and _bond_class(attrs) in {"double", "triple", "aromatic"}
            for left, right, attrs in graph.edges(data=True)
        )
        _add(
            features,
            (
                f"ring[size={len(cycle)};arom_atoms={aromatic};"
                f"unsat_edges={unsaturated_edges};elements={elements}]"
            ),
        )
    fused = spiro = bridged = 0
    for index, left in enumerate(cycles):
        for right in cycles[index + 1 :]:
            overlap = left & right
            if len(overlap) == 1:
                spiro += 1
            elif len(overlap) == 2 and graph.has_edge(*tuple(overlap)):
                fused += 1
            elif len(overlap) >= 2:
                bridged += 1
    _add(features, "ring_junction_spiro_pairs", float(spiro))
    _add(features, "ring_junction_fused_pairs", float(fused))
    _add(features, "ring_junction_bridged_pairs", float(bridged))
    _add(features, "cycle_rank", float(len(cycles)))

    exocyclic = 0
    for left, right, attrs in graph.edges(data=True):
        if _bond_class(attrs) not in {"double", "triple"}:
            continue
        left_ring = any(left in cycle for cycle in cycles)
        right_ring = any(right in cycle for cycle in cycles)
        exocyclic += int(left_ring != right_ring)
    _add(features, "exocyclic_multiple_bonds", float(exocyclic))


def _add_graph_index_features(features: dict[str, float], graph: nx.Graph) -> None:
    heavy = graph.subgraph(
        [node for node in graph if graph.nodes[node]["element"] != "H"]
    ).copy()
    if not heavy:
        return
    degrees = dict(heavy.degree())
    _add(
        features, "graph_zagreb_m1", float(sum(value**2 for value in degrees.values()))
    )
    _add(
        features,
        "graph_zagreb_m2",
        float(sum(degrees[left] * degrees[right] for left, right in heavy.edges)),
    )
    _add(
        features,
        "graph_randic",
        float(
            sum(
                1.0 / math.sqrt(max(1, degrees[left] * degrees[right]))
                for left, right in heavy.edges
            )
        ),
    )
    distances = dict(nx.all_pairs_shortest_path_length(heavy))
    ordered = sorted(heavy.nodes, key=repr)
    pair_distances = [
        distances[left][right]
        for index, left in enumerate(ordered)
        for right in ordered[index + 1 :]
        if right in distances[left]
    ]
    _add(features, "graph_wiener", float(sum(pair_distances)))
    _add(
        features,
        "graph_harary",
        float(sum(1.0 / value for value in pair_distances if value)),
    )
    adjacency = nx.to_numpy_array(heavy, nodelist=ordered, weight=None, dtype=float)
    _add(
        features,
        "graph_spectral_energy",
        float(np.sum(np.abs(np.linalg.eigvalsh(adjacency)))),
    )


def _add_charge_topology_features(features: dict[str, float], graph: nx.Graph) -> None:
    charged = [
        (node, int(attrs.get("formal_charge", 0)))
        for node, attrs in graph.nodes(data=True)
        if int(attrs.get("formal_charge", 0))
    ]
    _add(features, "formal_charge_centers", float(len(charged)))
    _add(features, "formal_charge_absolute", float(sum(abs(q) for _, q in charged)))
    opposite = same = 0.0
    for index, (left, left_q) in enumerate(charged):
        for right, right_q in charged[index + 1 :]:
            try:
                distance = nx.shortest_path_length(graph, left, right)
            except nx.NetworkXNoPath:
                continue
            value = abs(left_q * right_q) / max(1, distance)
            if left_q * right_q < 0:
                opposite += value
            else:
                same += value
    _add(features, "opposite_charge_inverse_graph_distance", opposite)
    _add(features, "same_charge_inverse_graph_distance", same)


def _add_rdkit_empirical_features(
    features: dict[str, float], canonical_smiles: str
) -> None:
    molecule = Chem.MolFromSmiles(canonical_smiles)
    if molecule is None:
        return
    molecule = Chem.RemoveHs(molecule)
    descriptor_functions = {
        "rdkit_bertz_ct": GraphDescriptors.BertzCT,
        "rdkit_balaban_j": GraphDescriptors.BalabanJ,
        "rdkit_hall_kier_alpha": GraphDescriptors.HallKierAlpha,
        "rdkit_kappa1": GraphDescriptors.Kappa1,
        "rdkit_kappa2": GraphDescriptors.Kappa2,
        "rdkit_kappa3": GraphDescriptors.Kappa3,
        "rdkit_chi0n": GraphDescriptors.Chi0n,
        "rdkit_chi1n": GraphDescriptors.Chi1n,
        "rdkit_chi2n": GraphDescriptors.Chi2n,
        "rdkit_chi3n": GraphDescriptors.Chi3n,
        "rdkit_chi4n": GraphDescriptors.Chi4n,
        "rdkit_fraction_csp3": rdMolDescriptors.CalcFractionCSP3,
        "rdkit_mol_logp": Crippen.MolLogP,
        "rdkit_molar_refractivity": Crippen.MolMR,
        "rdkit_labute_asa": rdMolDescriptors.CalcLabuteASA,
        "rdkit_tpsa": rdMolDescriptors.CalcTPSA,
        "rdkit_num_rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds,
        "rdkit_num_bridgeheads": rdMolDescriptors.CalcNumBridgeheadAtoms,
        "rdkit_num_spiro": rdMolDescriptors.CalcNumSpiroAtoms,
    }
    for name, function in descriptor_functions.items():
        try:
            _add(features, name, float(function(molecule)))
        except (RuntimeError, ValueError):
            continue

    ranks = list(Chem.CanonicalRankAtoms(molecule, breakTies=False))
    class_sizes = Counter(ranks).values()
    _add(features, "symmetry_classes", float(len(set(ranks))))
    _add(
        features,
        "symmetry_equivalent_pairs",
        float(sum(size * (size - 1) / 2 for size in class_sizes)),
    )
    if ranks:
        probabilities = np.asarray(list(Counter(ranks).values()), dtype=float) / len(
            ranks
        )
        _add(
            features,
            "symmetry_class_entropy",
            float(-np.sum(probabilities * np.log(probabilities))),
        )

    try:
        estate = np.asarray(EState.EStateIndices(molecule), dtype=float)
        _add(features, "estate_sum", float(np.sum(estate)))
        _add(features, "estate_absolute_sum", float(np.sum(np.abs(estate))))
        _add(features, "estate_min", float(np.min(estate)))
        _add(features, "estate_max", float(np.max(estate)))
        for atom, value in zip(molecule.GetAtoms(), estate):
            _add(features, f"estate_by_element[{atom.GetSymbol()}]", float(value))
    except (RuntimeError, ValueError):
        pass

    try:
        charged_molecule = Chem.Mol(molecule)
        rdPartialCharges.ComputeGasteigerCharges(charged_molecule)
        charges = np.asarray(
            [
                float(atom.GetProp("_GasteigerCharge"))
                for atom in charged_molecule.GetAtoms()
            ]
        )
        if np.all(np.isfinite(charges)):
            _add(features, "gasteiger_absolute_sum", float(np.sum(np.abs(charges))))
            _add(features, "gasteiger_squared_sum", float(np.sum(charges**2)))
            for atom, value in zip(charged_molecule.GetAtoms(), charges):
                _add(
                    features,
                    f"gasteiger_by_element[{atom.GetSymbol()}]",
                    float(value),
                )
            _add(
                features,
                "gasteiger_bond_charge_difference",
                float(
                    sum(
                        abs(
                            charges[bond.GetBeginAtomIdx()]
                            - charges[bond.GetEndAtomIdx()]
                        )
                        for bond in charged_molecule.GetBonds()
                    )
                ),
            )
    except (RuntimeError, ValueError):
        pass


def _add_huckel_fukui_features(
    features: dict[str, float], normalized: NormalizedMolecularGraph
) -> None:
    try:
        assignment = assign_orbital_pi(normalized)
        result = GeneralizedHuckel().solve(assignment)
    except (KeyError, RuntimeError, ValueError):
        return
    _add(features, "huckel_pi_systems", float(len(result.systems)))
    _add(features, "huckel_pi_stabilization", result.pi_stabilization)
    _add(features, "huckel_raw_pi_energy", result.raw_pi_energy)
    gaps = []
    for system in result.systems:
        if system.homo_energy is not None:
            _add(features, "huckel_homo_energy_sum", system.homo_energy)
        if system.lumo_energy is not None:
            _add(features, "huckel_lumo_energy_sum", system.lumo_energy)
        if system.homo_energy is not None and system.lumo_energy is not None:
            gaps.append(system.lumo_energy - system.homo_energy)
        for position, orbital in enumerate(system.nodes):
            atom = orbital[0] if isinstance(orbital, tuple) else orbital
            if atom not in normalized.graph:
                continue
            atom_state = _element_state(normalized.graph, atom)
            _add(
                features,
                f"fukui_homo_density[{atom_state}]",
                float(system.homo_density[position]),
            )
            if system.lumo_density is not None:
                _add(
                    features,
                    f"fukui_lumo_density[{atom_state}]",
                    float(system.lumo_density[position]),
                )
    if gaps:
        _add(features, "huckel_min_frontier_gap", float(min(gaps)))
        _add(features, "huckel_mean_frontier_gap", float(np.mean(gaps)))


def extract_named_empirical_two_d_features(
    normalized: NormalizedMolecularGraph,
    first_order_result: MoleculeScoreResult | None = None,
) -> dict[str, float]:
    """Return an unhashed, auditable empirical and chemical-graph vocabulary."""
    graph = normalized.graph
    features: dict[str, float] = {}
    _add_local_environment_features(features, graph)
    _add_ring_features(features, graph)
    _add_graph_index_features(features, graph)
    _add_charge_topology_features(features, graph)
    _add_rdkit_empirical_features(features, normalized.canonical_smiles)
    _add_huckel_fukui_features(features, normalized)
    if first_order_result is not None:
        for name, value in first_order_result.components.items():
            _add(features, f"first_order_v1[{name}]", float(value))
        inactive = first_order_result.descriptors.get("inactive_empirical_terms", {})
        if isinstance(inactive, dict):
            for name, value in inactive.items():
                if isinstance(value, (int, float)):
                    _add(features, f"inactive_empirical[{name}]", float(value))
    return features


def uncalibrated_v2_additions(
    normalized: NormalizedMolecularGraph,
    base: MoleculeScoreResult,
    config: InterpretableTwoDV2Config | None = None,
) -> dict[str, float]:
    """Return the prespecified kJ/mol additions without rescoring v1."""
    config = config or InterpretableTwoDV2Config()
    inactive = base.descriptors.get("inactive_empirical_terms", {})
    rigid_hbond = float(
        inactive.get("rigid_intramolecular_hbond", 0.0)
        if isinstance(inactive, dict)
        else 0.0
    )
    bridgehead = float(
        inactive.get("bridgehead_alkene_strain", 0.0)
        if isinstance(inactive, dict)
        else 0.0
    )
    table = {
        5: config.cyclopentane_strain_kj_mol,
        7: config.cycloheptane_strain_kj_mol,
        8: config.cyclooctane_strain_kj_mol,
        9: config.cyclononane_strain_kj_mol,
        10: config.cyclodecane_strain_kj_mol,
    }
    graph = normalized.graph
    medium_ring = 0.0
    for cycle in nx.minimum_cycle_basis(graph):
        size = len(cycle)
        if size not in table:
            continue
        edges = [(cycle[index], cycle[(index + 1) % size]) for index in range(size)]
        if any(
            bool(graph.edges[left, right].get("aromatic", False))
            or float(graph.edges[left, right].get("order", 1.0)) >= 1.5
            for left, right in edges
        ):
            continue
        medium_ring += table[size]
    return {
        "conventional_medium_ring_strain": float(medium_ring),
        "rigid_intramolecular_hbond": rigid_hbond,
        "bridgehead_alkene_strain": bridgehead,
    }


class InterpretableTwoDV2Scorer:
    """Uncalibrated v2 score with chemistry-fixed additions to frozen v1."""

    def __init__(self, config: InterpretableTwoDV2Config | None = None) -> None:
        self.config = config or InterpretableTwoDV2Config()
        self.base_scorer = FirstOrderTwoDEnergyScorer()

    def score(self, normalized: NormalizedMolecularGraph) -> MoleculeScoreResult:
        base = self.base_scorer.score(normalized)
        if base.score is None:
            return base
        components = {
            **base.components,
            **uncalibrated_v2_additions(normalized, base, self.config),
        }
        return MoleculeScoreResult(
            status=base.status,
            score=float(sum(components.values())),
            units="kJ/mol_score",
            components=components,
            descriptors={
                **base.descriptors,
                "named_feature_schema": NAMED_FEATURE_SCHEMA,
                "uncalibrated_v2_additions": (
                    "conventional_medium_ring_strain",
                    "rigid_intramolecular_hbond",
                    "bridgehead_alkene_strain",
                ),
            },
            warnings=base.warnings,
            provenance={
                "model_name": "synde-interpretable-2d-v2-uncalibrated-development",
                "mode": "labeled-graph-only",
                "calibrated": False,
                "fitted_coefficients": False,
                "uses_coordinates": False,
                "uses_conformers": False,
                "uses_xtb": False,
                "uses_ord_labels_at_inference": False,
                "benchmark_informed_development": False,
                "parameter_set": self.config.parameter_set,
                "permanent_holdout_evaluated": False,
                "holdout_tuned": False,
            },
        )


__all__ = [
    "InterpretableTwoDV2Config",
    "InterpretableTwoDV2Scorer",
    "NAMED_FEATURE_SCHEMA",
    "extract_named_empirical_two_d_features",
    "uncalibrated_v2_additions",
]
