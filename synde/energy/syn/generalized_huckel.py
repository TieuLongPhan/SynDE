"""Generalized one-p-orbital Hückel descriptors for SYN v2.

The values in :class:`HuckelParameters` are an explicitly versioned initial
organic parameter set.  They produce model descriptors, not physical energies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .pi_system import PiAssignmentResult, PiSystem


@dataclass(frozen=True)
class HuckelParameters:
    """Initial versioned alpha/beta configuration for common organic pi systems."""

    parameter_set: str = "organic-v1"
    degeneracy_tolerance: float = 1e-8
    beta_aromatic: float = -1.0
    beta_double: float = -1.0
    beta_conjugated: float = -0.8
    hetero_beta_scale: float = 0.85
    alpha: dict[str, float] = field(
        default_factory=lambda: {
            "C_aromatic": 0.0,
            "C_sp2": 0.0,
            "N_aromatic_pyridine": -0.5,
            "N_aromatic_pyrrole": -0.3,
            "N_sp2": -0.5,
            "O_aromatic": -1.0,
            "O_sp2": -1.0,
            "S_aromatic": -0.5,
            "S_sp2": -0.5,
        }
    )


@dataclass(frozen=True)
class HuckelSystemResult:
    """Orbital and invariant descriptors for one connected pi system."""

    nodes: tuple[Any, ...]
    electron_count: int
    hamiltonian: np.ndarray
    orbital_energies: np.ndarray
    coefficients: np.ndarray
    occupations: np.ndarray
    alpha: dict[Any, float]
    raw_pi_energy: float
    reference_pi_energy: float
    pi_stabilization: float
    pi_populations: np.ndarray
    homo_indices: tuple[int, ...]
    lumo_indices: tuple[int, ...]
    homo_density: np.ndarray
    lumo_density: np.ndarray | None

    @property
    def homo_energy(self) -> float | None:
        return (
            float(self.orbital_energies[self.homo_indices[0]])
            if self.homo_indices
            else None
        )

    @property
    def lumo_energy(self) -> float | None:
        return (
            float(self.orbital_energies[self.lumo_indices[0]])
            if self.lumo_indices
            else None
        )


@dataclass(frozen=True)
class GeneralizedHuckelResult:
    """Descriptor result for all pi systems in one normalized molecule graph."""

    systems: tuple[HuckelSystemResult, ...]
    parameter_set: str
    status: str
    warnings: tuple[str, ...]

    @property
    def raw_pi_energy(self) -> float:
        return float(sum(system.raw_pi_energy for system in self.systems))

    @property
    def reference_pi_energy(self) -> float:
        return float(sum(system.reference_pi_energy for system in self.systems))

    @property
    def pi_stabilization(self) -> float:
        return float(sum(system.pi_stabilization for system in self.systems))


class GeneralizedHuckel:
    """Solve generalized Hückel systems built by :mod:`synde.energy.syn.pi_system`."""

    def __init__(self, parameters: HuckelParameters | None = None) -> None:
        self.parameters = parameters or HuckelParameters()

    def solve(self, assignment: PiAssignmentResult) -> GeneralizedHuckelResult:
        """Solve every assigned pi system using its explicit electron count."""

        results = tuple(
            self._solve_system(assignment, system) for system in assignment.systems
        )
        warnings = list(assignment.warning_codes())
        if any(system.electron_count % 2 for system in results):
            warnings.append("ODD_PI_ELECTRON_COUNT")
        status = assignment.status if assignment.status != "success" else "success"
        return GeneralizedHuckelResult(
            systems=results,
            parameter_set=self.parameters.parameter_set,
            status=status,
            warnings=tuple(warnings),
        )

    def _solve_system(
        self, assignment: PiAssignmentResult, system: PiSystem
    ) -> HuckelSystemResult:
        nodes = tuple(sorted(system.nodes, key=repr))
        index = {node: position for position, node in enumerate(nodes)}
        hamiltonian = np.zeros((len(nodes), len(nodes)), dtype=float)
        alpha = {node: self._alpha(assignment.pi_graph.nodes[node]) for node in nodes}
        for node, value in alpha.items():
            hamiltonian[index[node], index[node]] = value
        for left, right, attrs in assignment.pi_graph.subgraph(nodes).edges(data=True):
            beta = self._beta(
                assignment.pi_graph.nodes[left], assignment.pi_graph.nodes[right], attrs
            )
            hamiltonian[index[left], index[right]] = beta
            hamiltonian[index[right], index[left]] = beta

        orbital_energies, coefficients = np.linalg.eigh(hamiltonian)
        order = np.argsort(orbital_energies)
        orbital_energies = orbital_energies[order]
        coefficients = coefficients[:, order]
        occupations = self.occupations(len(nodes), system.electron_count)
        raw_pi_energy = float(np.dot(occupations, orbital_energies))
        reference_pi_energy = float(
            sum(
                int(assignment.pi_graph.nodes[node]["pi_electrons"]) * alpha[node]
                for node in nodes
            )
        )
        homo_indices, lumo_indices = self.frontier_subspaces(
            occupations, orbital_energies
        )
        pi_populations = np.sum((coefficients**2) * occupations[np.newaxis, :], axis=1)
        return HuckelSystemResult(
            nodes=nodes,
            electron_count=system.electron_count,
            hamiltonian=hamiltonian,
            orbital_energies=orbital_energies,
            coefficients=coefficients,
            occupations=occupations,
            alpha=alpha,
            raw_pi_energy=raw_pi_energy,
            reference_pi_energy=reference_pi_energy,
            pi_stabilization=float(raw_pi_energy - reference_pi_energy),
            pi_populations=pi_populations,
            homo_indices=homo_indices,
            lumo_indices=lumo_indices,
            homo_density=self.subspace_density(coefficients, homo_indices),
            lumo_density=(
                self.subspace_density(coefficients, lumo_indices)
                if lumo_indices
                else None
            ),
        )

    def _alpha(self, attrs: dict[str, Any]) -> float:
        element = attrs["element"]
        if attrs.get("aromatic", False):
            if element == "N":
                key = (
                    "N_aromatic_pyrrole"
                    if attrs.get("total_hcount", 0)
                    else "N_aromatic_pyridine"
                )
            else:
                key = f"{element}_aromatic"
        else:
            key = f"{element}_sp2"
        try:
            return float(self.parameters.alpha[key])
        except KeyError as exc:  # pi assignment should have filtered this already
            raise ValueError(
                f"No alpha parameter for supported pi environment {key!r}."
            ) from exc

    def _beta(
        self, left: dict[str, Any], right: dict[str, Any], attrs: dict[str, Any]
    ) -> float:
        if attrs.get("aromatic", False):
            beta = self.parameters.beta_aromatic
        elif float(attrs.get("order", 1.0)) >= 2.0:
            beta = self.parameters.beta_double
        else:
            beta = self.parameters.beta_conjugated
        if left["element"] != "C" or right["element"] != "C":
            beta *= self.parameters.hetero_beta_scale
        return float(beta)

    @staticmethod
    def occupations(n_orbitals: int, electron_count: int) -> np.ndarray:
        """Return an explicit occupation vector (two electrons then one, if needed)."""

        if electron_count < 0 or electron_count > 2 * n_orbitals:
            raise ValueError(
                f"Electron count {electron_count} is invalid for {n_orbitals} orbitals."
            )
        occupations = np.zeros(n_orbitals, dtype=float)
        remaining = electron_count
        for index in range(n_orbitals):
            occupation = min(2, remaining)
            occupations[index] = occupation
            remaining -= occupation
        return occupations

    def frontier_subspaces(
        self, occupations: np.ndarray, energies: np.ndarray
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return degenerate HOMO and LUMO subspaces from explicit occupations."""

        occupied = np.flatnonzero(occupations > 0)
        unoccupied = np.flatnonzero(occupations < 2)
        if not len(occupied):
            return (), tuple(int(index) for index in unoccupied)
        homo_index = int(occupied[-1])
        homo_energy = energies[homo_index]
        homo = tuple(
            int(index)
            for index in occupied
            if abs(float(energies[index] - homo_energy))
            <= self.parameters.degeneracy_tolerance
        )
        lumo_candidates = [index for index in unoccupied if index > homo_index]
        if not lumo_candidates:
            return homo, ()
        lumo_index = int(lumo_candidates[0])
        lumo_energy = energies[lumo_index]
        lumo = tuple(
            int(index)
            for index in lumo_candidates
            if abs(float(energies[index] - lumo_energy))
            <= self.parameters.degeneracy_tolerance
        )
        return homo, lumo

    @staticmethod
    def subspace_density(
        coefficients: np.ndarray, indices: tuple[int, ...]
    ) -> np.ndarray:
        """Return a basis-invariant local density summed over an orbital subspace."""

        if not indices:
            return np.zeros(coefficients.shape[0], dtype=float)
        return np.sum(coefficients[:, list(indices)] ** 2, axis=1)


def solve_generalized_huckel(
    assignment: PiAssignmentResult,
    parameters: HuckelParameters | None = None,
) -> GeneralizedHuckelResult:
    """Convenience wrapper for :class:`GeneralizedHuckel`."""

    return GeneralizedHuckel(parameters).solve(assignment)
