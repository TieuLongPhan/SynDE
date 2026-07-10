"""Global and local HSAB descriptors derived from generalized frontier data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .frontier import ComponentFrontier


@dataclass(frozen=True)
class HSABDescriptor:
    component_id: int
    electronegativity: float | None
    hardness: float | None
    softness: float | None
    local_nucleophilic_softness: dict[Any, float]
    local_electrophilic_softness: dict[Any, float]


def hsab_descriptor(
    frontier: ComponentFrontier, *, hardness_floor: float = 0.05
) -> HSABDescriptor:
    """Build global and local HSAB features without claiming an energy value."""
    if frontier.homo_energy is None or frontier.lumo_energy is None:
        return HSABDescriptor(frontier.component_id, None, None, None, {}, {})
    electronegativity = -0.5 * (frontier.homo_energy + frontier.lumo_energy)
    hardness = 0.5 * (frontier.lumo_energy - frontier.homo_energy)
    softness = 1.0 / max(hardness, hardness_floor)
    return HSABDescriptor(
        frontier.component_id,
        float(electronegativity),
        float(hardness),
        float(softness),
        {
            node: float(softness * value)
            for node, value in frontier.homo_density.items()
        },
        {
            node: float(softness * value)
            for node, value in frontier.lumo_density.items()
        },
    )


def local_hsab_compatibility(
    donor: HSABDescriptor, acceptor: HSABDescriptor, donor_atom: Any, acceptor_atom: Any
) -> float:
    """Return a dimensionless local soft donor/acceptor compatibility feature."""
    return float(
        donor.local_nucleophilic_softness.get(donor_atom, 0.0)
        * acceptor.local_electrophilic_softness.get(acceptor_atom, 0.0)
    )
