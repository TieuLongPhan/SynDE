"""Interpretable, uncalibrated local graph score features."""

from __future__ import annotations
from collections import Counter
import networkx as nx

ATOM_BASELINE = {
    "B": 0.10,
    "C": 0.0,
    "N": 0.10,
    "O": 0.20,
    "P": 0.15,
    "S": 0.15,
    "F": 0.05,
    "Cl": 0.05,
    "Br": 0.05,
    "I": 0.05,
}


def local_score_components(graph: nx.Graph) -> dict[str, float]:
    """Return local atom/bond/charge/ring topology features in score units."""
    atom = sum(
        ATOM_BASELINE.get(data["element"], 0.0) for _, data in graph.nodes(data=True)
    )
    bond = -sum(float(data.get("order", 1.0)) for _, _, data in graph.edges(data=True))
    charge = 0.25 * sum(
        abs(int(data.get("formal_charge", 0))) for _, data in graph.nodes(data=True)
    )
    cycles = nx.cycle_basis(graph)
    ring = sum(
        0.50 if len(cycle) == 3 else 0.20 if len(cycle) == 4 else 0.0
        for cycle in cycles
    )
    return {
        "local_atom": float(atom),
        "local_bond": float(bond),
        "formal_charge": float(charge),
        "ring_topology": float(ring),
    }


def element_counts(graph: nx.Graph) -> Counter[str]:
    return Counter(data["element"] for _, data in graph.nodes(data=True))
