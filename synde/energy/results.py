"""Typed, serializable v2 graph-energy results."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MoleculeScoreResult:
    status: str
    score: float | None
    units: str
    components: dict[str, float]
    descriptors: dict[str, Any]
    warnings: tuple[str, ...]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReactionScoreResult:
    status: str
    reaction_delta_score: float | None
    units: str
    reactant_score: float | None
    product_score: float | None
    bond_changes: tuple[dict[str, Any], ...]
    components: dict[str, float]
    warnings: tuple[str, ...]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ITSScoreResult:
    """Mapped reaction-centre feasibility result; not a physical TS energy."""

    status: str
    its_score: float | None
    state_delta_score: float | None
    units: str
    components: dict[str, float]
    bond_changes: tuple[dict[str, Any], ...]
    reacting_atom_maps: tuple[int, ...]
    warnings: tuple[str, ...]
    provenance: dict[str, Any]
    its_graph: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("its_graph", None)
        return data
