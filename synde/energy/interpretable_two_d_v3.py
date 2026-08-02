"""Quantum-chemistry-inspired, fixed-vocabulary 2D graph terms for SynDE v3.

The descriptors in this module are deterministic functions of a sanitized
labeled molecular graph. They use no coordinates, conformers, force fields,
reference calculations, or fitted parameters. Hückel density matrices,
Gasteiger charges, and graph-steric quantities are empirical descriptors, not
identified physical observables or energy components.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdPartialCharges

from synde.graph.generalized_huckel import GeneralizedHuckel
from synde.graph.graph_schema import NormalizedMolecularGraph
from synde.graph.orbital_pi import assign_orbital_pi


QUANTUM_GRAPH_FEATURE_SCHEMA_V3 = "synde-quantum-graph-2d-v3-fixed-52-v1"

HUCKEL_DENSITY_FAMILY = "v3_huckel_density_spectral"
CHARGE_TOPOLOGY_FAMILY = "v3_charge_topology"
RESONANCE_TOPOLOGY_FAMILY = "v3_resonance_topology"
CYCLE_JUNCTION_FAMILY = "v3_cycle_ring_junction"
GRAPH_STERIC_FAMILY = "v3_graph_steric"


@dataclass(frozen=True)
class V3FeatureDefinition:
    """One preregistered v3 descriptor definition."""

    name: str
    family: str
    description: str
    units: str = "dimensionless_graph_descriptor"


def _definitions() -> tuple[V3FeatureDefinition, ...]:
    definitions: list[V3FeatureDefinition] = []

    def add(family: str, name: str, description: str) -> None:
        definitions.append(V3FeatureDefinition(name, family, description))

    for name, description in (
        ("component_count", "Number of assigned connected pi-orbital components."),
        ("component_size_max", "Largest assigned pi-orbital component size."),
        ("density_electron_trace", "Sum of traces of occupied Hückel density matrices."),
        ("population_l2", "Sum of squared diagonal occupied populations."),
        (
            "population_variance",
            "Sum of within-component variances of diagonal occupied populations.",
        ),
        ("bond_order_abs_sum", "Sum of absolute occupied-density edge elements."),
        ("bond_order_squared_sum", "Sum of squared occupied-density edge elements."),
        ("bond_order_abs_min", "Minimum absolute occupied-density edge element."),
        ("bond_order_abs_max", "Maximum absolute occupied-density edge element."),
        (
            "bond_order_alternation_mean",
            "Mean absolute bond-order difference for incident pi edges.",
        ),
        ("spectral_moment_2", "Sum of normalized second Hückel spectral moments."),
        ("spectral_moment_3", "Sum of normalized third Hückel spectral moments."),
        ("spectral_moment_4", "Sum of normalized fourth Hückel spectral moments."),
        (
            "occupied_energy_variance",
            "Sum of occupation-weighted within-component orbital-energy variances.",
        ),
        (
            "homo_projector_ipr_mean",
            "Mean inverse-participation ratio of complete HOMO projectors.",
        ),
        (
            "lumo_projector_ipr_mean",
            "Mean inverse-participation ratio of complete LUMO projectors.",
        ),
    ):
        add(HUCKEL_DENSITY_FAMILY, f"v3_huckel_{name}", description)

    for distance in range(1, 5):
        add(
            CHARGE_TOPOLOGY_FAMILY,
            f"v3_charge_abs_difference_r{distance}",
            f"Sum of absolute Gasteiger-charge contrasts at graph distance {distance}.",
        )
        add(
            CHARGE_TOPOLOGY_FAMILY,
            f"v3_charge_product_r{distance}",
            f"Sum of Gasteiger-charge products at graph distance {distance}.",
        )
    for distance in range(2, 5):
        add(
            CHARGE_TOPOLOGY_FAMILY,
            f"v3_charge_donor_acceptor_abs_difference_r{distance}",
            (
                "Sum of absolute charge contrasts for fixed-rule donor--acceptor "
                f"pairs at graph distance {distance}."
            ),
        )
    add(
        CHARGE_TOPOLOGY_FAMILY,
        "v3_charge_distance_decay_abs_difference",
        "All-pair absolute charge contrast weighted by 2^(-graph distance).",
    )
    add(
        CHARGE_TOPOLOGY_FAMILY,
        "v3_charge_distance_decay_product",
        "All-pair signed charge product weighted by 2^(-graph distance).",
    )
    add(
        CHARGE_TOPOLOGY_FAMILY,
        "v3_charge_bond_sign_changes",
        "Number of heavy-atom bonds whose nonzero endpoint charges have opposite signs.",
    )

    for name, description in (
        ("component_count", "Number of connected components in the atom-level pi graph."),
        ("site_count", "Number of atoms in the atom-level pi graph."),
        ("component_size_max", "Largest atom-level pi-component size."),
        ("branch_center_count", "Pi-graph vertices with degree at least three."),
        (
            "cross_conjugated_center_count",
            "Nonaromatic pi-graph vertices with degree at least three.",
        ),
        ("path_count_length2", "Number of undirected simple pi paths with two edges."),
        ("path_count_length3", "Number of undirected simple pi paths with three edges."),
        ("path_count_length4", "Number of undirected simple pi paths with four edges."),
    ):
        add(RESONANCE_TOPOLOGY_FAMILY, f"v3_resonance_{name}", description)

    for name, description in (
        ("cycle_rank", "Basis-independent heavy-atom cyclomatic number."),
        (
            "ring_size_quadratic_deviation",
            "Sum over symmetrized rings of (ring size - 6)^2.",
        ),
        ("ring_size_variance", "Population variance of symmetrized ring sizes."),
        ("junction_atom_count", "Atoms shared by at least two symmetrized rings."),
        ("fused_shared_edge_count", "Edges shared by at least two symmetrized rings."),
        ("bridgehead_atom_count", "Invariant RDKit bridgehead-atom count."),
        (
            "exocyclic_multiple_bond_count",
            "Multiple bonds with exactly one endpoint in a symmetrized ring.",
        ),
        (
            "adjacent_small_ring_pair_count",
            "Pairs of rings of size at most four that touch or are directly linked.",
        ),
    ):
        add(CYCLE_JUNCTION_FAMILY, f"v3_cycle_{name}", description)

    for name, description in (
        (
            "weighted_1_3_crowding",
            "Covalent-radius-product sum over heavy 1,3 endpoint pairs.",
        ),
        (
            "heavy_neighbor_pair_count",
            "Sum of heavy-neighbor pair counts around heavy atoms.",
        ),
        (
            "adjacent_branching_bond_count",
            "Heavy bonds joining two atoms of heavy degree at least three.",
        ),
        (
            "aromatic_ortho_substituent_pairs",
            "Aromatic ring edges whose endpoints both have external heavy substituents.",
        ),
        (
            "aromatic_meta_substituent_pairs",
            "Substituted aromatic atom pairs separated by two aromatic edges.",
        ),
        (
            "double_bond_substitution_product",
            "Sum of endpoint external-heavy-degree products over nonaromatic double bonds.",
        ),
    ):
        add(GRAPH_STERIC_FAMILY, f"v3_steric_{name}", description)
    return tuple(definitions)


V3_FEATURE_DEFINITIONS = _definitions()
V3_FEATURE_NAMES = tuple(definition.name for definition in V3_FEATURE_DEFINITIONS)
V3_FEATURE_FAMILIES = {
    definition.name: definition.family for definition in V3_FEATURE_DEFINITIONS
}

if len(V3_FEATURE_NAMES) != 52 or len(set(V3_FEATURE_NAMES)) != 52:
    raise RuntimeError("The preregistered v3 feature manifest must contain 52 names.")


def _finite(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else 0.0


def _projector_ipr(density: np.ndarray | None) -> float | None:
    if density is None:
        return None
    values = np.asarray(density, dtype=float)
    total = float(np.sum(values))
    if not math.isfinite(total) or total <= 0:
        return None
    normalized = values / total
    return _finite(float(np.dot(normalized, normalized)))


def _add_huckel_density_features(
    output: dict[str, float], normalized: NormalizedMolecularGraph
) -> None:
    try:
        assignment = assign_orbital_pi(normalized)
        result = GeneralizedHuckel().solve(assignment)
    except (KeyError, RuntimeError, ValueError, np.linalg.LinAlgError):
        return

    output["v3_huckel_component_count"] = float(len(result.systems))
    if not result.systems:
        return
    output["v3_huckel_component_size_max"] = float(
        max(len(system.nodes) for system in result.systems)
    )

    all_edge_orders: list[float] = []
    alternation: list[float] = []
    homo_ipr: list[float] = []
    lumo_ipr: list[float] = []
    for system in result.systems:
        size = len(system.nodes)
        if size == 0:
            continue
        coefficients = np.asarray(system.coefficients, dtype=float)
        occupations = np.asarray(system.occupations, dtype=float)
        density = (coefficients * occupations[np.newaxis, :]) @ coefficients.T
        populations = np.diag(density)
        output["v3_huckel_density_electron_trace"] += _finite(np.trace(density))
        output["v3_huckel_population_l2"] += _finite(np.dot(populations, populations))
        output["v3_huckel_population_variance"] += _finite(
            np.var(populations, ddof=0)
        )
        hamiltonian = np.asarray(system.hamiltonian, dtype=float)
        for power in (2, 3, 4):
            output[f"v3_huckel_spectral_moment_{power}"] += _finite(
                np.trace(np.linalg.matrix_power(hamiltonian, power)) / size
            )
        occupied_total = float(np.sum(occupations))
        if occupied_total > 0:
            weights = occupations / occupied_total
            mean_energy = float(np.dot(weights, system.orbital_energies))
            output["v3_huckel_occupied_energy_variance"] += _finite(
                np.dot(weights, (system.orbital_energies - mean_energy) ** 2)
            )

        index = {node: position for position, node in enumerate(system.nodes)}
        incident: dict[Any, list[float]] = defaultdict(list)
        for left, right in assignment.pi_graph.subgraph(system.nodes).edges:
            value = abs(float(density[index[left], index[right]]))
            all_edge_orders.append(value)
            incident[left].append(value)
            incident[right].append(value)
        for values in incident.values():
            for first in range(len(values)):
                for second in range(first + 1, len(values)):
                    alternation.append(abs(values[first] - values[second]))

        value = _projector_ipr(system.homo_density)
        if value is not None:
            homo_ipr.append(value)
        value = _projector_ipr(system.lumo_density)
        if value is not None:
            lumo_ipr.append(value)

    if all_edge_orders:
        edge_array = np.asarray(all_edge_orders, dtype=float)
        output["v3_huckel_bond_order_abs_sum"] = _finite(np.sum(edge_array))
        output["v3_huckel_bond_order_squared_sum"] = _finite(
            np.dot(edge_array, edge_array)
        )
        output["v3_huckel_bond_order_abs_min"] = _finite(np.min(edge_array))
        output["v3_huckel_bond_order_abs_max"] = _finite(np.max(edge_array))
    if alternation:
        output["v3_huckel_bond_order_alternation_mean"] = _finite(
            np.mean(alternation)
        )
    if homo_ipr:
        output["v3_huckel_homo_projector_ipr_mean"] = _finite(np.mean(homo_ipr))
    if lumo_ipr:
        output["v3_huckel_lumo_projector_ipr_mean"] = _finite(np.mean(lumo_ipr))


def _is_donor(atom: Chem.Atom) -> bool:
    return (
        atom.GetSymbol() in {"N", "O", "S"}
        and atom.GetFormalCharge() <= 0
        and atom.GetTotalNumHs(includeNeighbors=True) > 0
    )


def _is_acceptor(atom: Chem.Atom) -> bool:
    element = atom.GetSymbol()
    if element not in {"N", "O", "S"} or atom.GetFormalCharge() > 0:
        return False
    if element == "N" and atom.GetIsAromatic() and atom.GetTotalNumHs() > 0:
        return False
    return element in {"O", "S"} or atom.GetTotalValence() <= 3


def _add_charge_topology_features(
    output: dict[str, float], canonical_smiles: str | None
) -> None:
    if not canonical_smiles:
        return
    molecule = Chem.MolFromSmiles(canonical_smiles)
    if molecule is None:
        return
    molecule = Chem.RemoveHs(molecule)
    try:
        charged = Chem.Mol(molecule)
        rdPartialCharges.ComputeGasteigerCharges(charged)
        charges = np.asarray(
            [float(atom.GetProp("_GasteigerCharge")) for atom in charged.GetAtoms()],
            dtype=float,
        )
    except (KeyError, RuntimeError, ValueError):
        return
    if not np.all(np.isfinite(charges)):
        return

    distances = np.asarray(Chem.GetDistanceMatrix(charged), dtype=float)
    donors = {
        index
        for index in range(charged.GetNumAtoms())
        if _is_donor(charged.GetAtomWithIdx(index))
    }
    acceptors = {
        index
        for index in range(charged.GetNumAtoms())
        if _is_acceptor(charged.GetAtomWithIdx(index))
    }
    for left in range(charged.GetNumAtoms()):
        for right in range(left + 1, charged.GetNumAtoms()):
            distance = int(round(float(distances[left, right])))
            if distance <= 0:
                continue
            difference = abs(float(charges[left] - charges[right]))
            product = float(charges[left] * charges[right])
            if distance <= 4:
                output[f"v3_charge_abs_difference_r{distance}"] += difference
                output[f"v3_charge_product_r{distance}"] += product
            if (
                2 <= distance <= 4
                and (
                    (left in donors and right in acceptors)
                    or (right in donors and left in acceptors)
                )
            ):
                output[
                    f"v3_charge_donor_acceptor_abs_difference_r{distance}"
                ] += difference
            weight = 2.0 ** (-distance)
            output["v3_charge_distance_decay_abs_difference"] += weight * difference
            output["v3_charge_distance_decay_product"] += weight * product
    output["v3_charge_bond_sign_changes"] = float(
        sum(
            charges[bond.GetBeginAtomIdx()] * charges[bond.GetEndAtomIdx()] < 0
            and abs(charges[bond.GetBeginAtomIdx()]) > 1e-12
            and abs(charges[bond.GetEndAtomIdx()]) > 1e-12
            for bond in charged.GetBonds()
        )
    )


def _pi_capable(attrs: dict[str, Any]) -> bool:
    if bool(attrs.get("aromatic", False)):
        return True
    if str(attrs.get("hybridization", "")) in {"SP", "SP2"}:
        return True
    return (
        attrs.get("element") in {"N", "O", "S"}
        and int(attrs.get("formal_charge", 0)) <= 0
        and bool(attrs.get("available_lp", False) or attrs.get("lone_pairs", 0))
    )


def _simple_path_count(graph: nx.Graph, length: int) -> int:
    if length < 1:
        return 0
    paths: set[tuple[Any, ...]] = set()

    def visit(path: tuple[Any, ...]) -> None:
        if len(path) == length + 1:
            reverse = tuple(reversed(path))
            paths.add(min(path, reverse, key=repr))
            return
        for neighbor in graph.neighbors(path[-1]):
            if neighbor not in path:
                visit((*path, neighbor))

    for node in graph:
        visit((node,))
    return len(paths)


def _resonance_graph(graph: nx.Graph) -> nx.Graph:
    resonance = nx.Graph()
    for node, attrs in graph.nodes(data=True):
        if attrs.get("element") != "H" and _pi_capable(attrs):
            resonance.add_node(node)
    for left, right, attrs in graph.edges(data=True):
        if left not in resonance or right not in resonance:
            continue
        if (
            bool(attrs.get("aromatic", False))
            or bool(attrs.get("conjugated", False))
            or float(attrs.get("order", 1.0)) >= 1.5
        ):
            resonance.add_edge(left, right)
    return resonance


def _add_resonance_features(
    output: dict[str, float], graph: nx.Graph
) -> None:
    resonance = _resonance_graph(graph)
    components = list(nx.connected_components(resonance))
    output["v3_resonance_component_count"] = float(len(components))
    output["v3_resonance_site_count"] = float(resonance.number_of_nodes())
    output["v3_resonance_component_size_max"] = float(
        max((len(component) for component in components), default=0)
    )
    output["v3_resonance_branch_center_count"] = float(
        sum(degree >= 3 for _, degree in resonance.degree())
    )
    output["v3_resonance_cross_conjugated_center_count"] = float(
        sum(
            degree >= 3 and not bool(graph.nodes[node].get("aromatic", False))
            for node, degree in resonance.degree()
        )
    )
    for length in (2, 3, 4):
        output[f"v3_resonance_path_count_length{length}"] = float(
            _simple_path_count(resonance, length)
        )


def _ring_edges(ring: tuple[int, ...]) -> set[frozenset[int]]:
    return {
        frozenset((ring[index], ring[(index + 1) % len(ring)]))
        for index in range(len(ring))
    }


def _add_cycle_features(
    output: dict[str, float], normalized: NormalizedMolecularGraph
) -> None:
    graph = normalized.graph
    heavy_nodes = [
        node for node, attrs in graph.nodes(data=True) if attrs.get("element") != "H"
    ]
    heavy = graph.subgraph(heavy_nodes)
    output["v3_cycle_cycle_rank"] = float(
        heavy.number_of_edges()
        - heavy.number_of_nodes()
        + nx.number_connected_components(heavy)
        if heavy.number_of_nodes()
        else 0
    )
    if not normalized.canonical_smiles:
        return
    molecule = Chem.MolFromSmiles(normalized.canonical_smiles)
    if molecule is None:
        return
    molecule = Chem.RemoveHs(molecule)
    rings = [tuple(int(atom) for atom in ring) for ring in Chem.GetSymmSSSR(molecule)]
    sizes = np.asarray([len(ring) for ring in rings], dtype=float)
    if len(sizes):
        output["v3_cycle_ring_size_quadratic_deviation"] = _finite(
            np.sum((sizes - 6.0) ** 2)
        )
        output["v3_cycle_ring_size_variance"] = _finite(np.var(sizes, ddof=0))
    memberships: defaultdict[int, int] = defaultdict(int)
    edge_memberships: defaultdict[frozenset[int], int] = defaultdict(int)
    for ring in rings:
        for atom in ring:
            memberships[atom] += 1
        edges = _ring_edges(ring)
        for edge in edges:
            edge_memberships[edge] += 1
    output["v3_cycle_junction_atom_count"] = float(
        sum(count >= 2 for count in memberships.values())
    )
    output["v3_cycle_fused_shared_edge_count"] = float(
        sum(count >= 2 for count in edge_memberships.values())
    )
    adjacent_small = 0
    for first in range(len(rings)):
        left_atoms = set(rings[first])
        for second in range(first + 1, len(rings)):
            right_atoms = set(rings[second])
            overlap = left_atoms & right_atoms
            if len(rings[first]) <= 4 and len(rings[second]) <= 4:
                directly_linked = any(
                    molecule.GetBondBetweenAtoms(left, right) is not None
                    for left in left_atoms
                    for right in right_atoms
                    if left != right
                )
                adjacent_small += int(bool(overlap) or directly_linked)
    output["v3_cycle_bridgehead_atom_count"] = float(
        rdMolDescriptors.CalcNumBridgeheadAtoms(molecule)
    )
    output["v3_cycle_adjacent_small_ring_pair_count"] = float(adjacent_small)
    ring_atoms = set(memberships)
    output["v3_cycle_exocyclic_multiple_bond_count"] = float(
        sum(
            bond.GetBondTypeAsDouble() >= 2.0
            and (
                (bond.GetBeginAtomIdx() in ring_atoms)
                != (bond.GetEndAtomIdx() in ring_atoms)
            )
            for bond in molecule.GetBonds()
        )
    )


_COVALENT_RADIUS = {
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}


def _add_steric_features(output: dict[str, float], graph: nx.Graph) -> None:
    heavy_nodes = [
        node for node, attrs in graph.nodes(data=True) if attrs.get("element") != "H"
    ]
    heavy = graph.subgraph(heavy_nodes)
    degrees = dict(heavy.degree())
    for center in heavy:
        neighbors = list(heavy.neighbors(center))
        output["v3_steric_heavy_neighbor_pair_count"] += len(neighbors) * (
            len(neighbors) - 1
        ) / 2
        for first in range(len(neighbors)):
            for second in range(first + 1, len(neighbors)):
                left = graph.nodes[neighbors[first]].get("element")
                right = graph.nodes[neighbors[second]].get("element")
                output["v3_steric_weighted_1_3_crowding"] += (
                    _COVALENT_RADIUS.get(str(left), 0.75)
                    * _COVALENT_RADIUS.get(str(right), 0.75)
                )
    output["v3_steric_adjacent_branching_bond_count"] = float(
        sum(degrees[left] >= 3 and degrees[right] >= 3 for left, right in heavy.edges)
    )

    aromatic = nx.Graph()
    substituted: set[Any] = set()
    for node in heavy:
        if bool(graph.nodes[node].get("aromatic", False)):
            aromatic.add_node(node)
    for left, right, attrs in heavy.edges(data=True):
        if (
            left in aromatic
            and right in aromatic
            and bool(attrs.get("aromatic", False))
        ):
            aromatic.add_edge(left, right)
    for node in aromatic:
        if any(neighbor not in aromatic for neighbor in heavy.neighbors(node)):
            substituted.add(node)
    output["v3_steric_aromatic_ortho_substituent_pairs"] = float(
        sum(
            left in substituted and right in substituted
            for left, right in aromatic.edges
        )
    )
    meta_pairs = 0
    ordered = sorted(substituted, key=repr)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            try:
                meta_pairs += int(nx.shortest_path_length(aromatic, left, right) == 2)
            except nx.NetworkXNoPath:
                continue
    output["v3_steric_aromatic_meta_substituent_pairs"] = float(meta_pairs)
    output["v3_steric_double_bond_substitution_product"] = float(
        sum(
            max(0, degrees[left] - 1) * max(0, degrees[right] - 1)
            for left, right, attrs in heavy.edges(data=True)
            if not bool(attrs.get("aromatic", False))
            and 1.5 <= float(attrs.get("order", 1.0)) < 2.5
        )
    )


def extract_quantum_graph_v3_features(
    normalized: NormalizedMolecularGraph,
) -> dict[str, float]:
    """Return the complete fixed 52-coordinate v3 descriptor vector.

    Unsupported or absent substructures produce zeros. Any nonfinite
    intermediate is replaced by zero, so callers always receive the exact
    preregistered manifest in deterministic order.
    """

    output = {name: 0.0 for name in V3_FEATURE_NAMES}
    _add_huckel_density_features(output, normalized)
    _add_charge_topology_features(output, normalized.canonical_smiles)
    _add_resonance_features(output, normalized.graph)
    _add_cycle_features(output, normalized)
    _add_steric_features(output, normalized.graph)
    return {name: _finite(output[name]) for name in V3_FEATURE_NAMES}


def v3_feature_family(name: str) -> str:
    """Return the fixed family of a preregistered v3 term."""

    try:
        return V3_FEATURE_FAMILIES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown v3 feature {name!r}.") from exc


__all__ = [
    "CHARGE_TOPOLOGY_FAMILY",
    "CYCLE_JUNCTION_FAMILY",
    "GRAPH_STERIC_FAMILY",
    "HUCKEL_DENSITY_FAMILY",
    "QUANTUM_GRAPH_FEATURE_SCHEMA_V3",
    "RESONANCE_TOPOLOGY_FAMILY",
    "V3FeatureDefinition",
    "V3_FEATURE_DEFINITIONS",
    "V3_FEATURE_FAMILIES",
    "V3_FEATURE_NAMES",
    "extract_quantum_graph_v3_features",
    "v3_feature_family",
]
