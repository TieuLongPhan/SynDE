"""Optional zero-fit semiempirical molecular-energy scoring.

This module runs an actual GFN2-xTB single-point calculation through tblite on
a deterministic RDKit conformer.  It is deliberately separate from SynDE's
graph-only scores and from the cheap xTB-like composition proxy.
"""

from __future__ import annotations

import hashlib
import importlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from synde.graph.graph_schema import NormalizedMolecularGraph

from .results import MoleculeScoreResult

ANGSTROM_PER_BOHR = 0.529177210903
HARTREE_TO_EV = 27.211386245988


@dataclass(frozen=True)
class GFN2SinglePointConfig:
    """Frozen, unfitted protocol for deterministic GFN2-xTB scoring."""

    method: str = "GFN2-xTB"
    embedding_method: str = "ETKDGv3"
    seed_namespace: str = "synde-gfn2-singlepoint-mmff-v1"
    preopt_max_iterations: int = 500
    scf_max_iterations: int = 250
    accuracy: float = 1.0


class GFN2SinglePointScorer:
    """Run gas-phase closed-shell GFN2-xTB on a deterministic 3D conformer."""

    def __init__(self, config: GFN2SinglePointConfig | None = None) -> None:
        self.config = config or GFN2SinglePointConfig()

    def score(self, normalized: NormalizedMolecularGraph) -> MoleculeScoreResult:
        """Return an actual semiempirical energy in eV without target fitting."""
        provenance = {
            "model_name": "synde-gfn2-singlepoint-mmff-v1",
            "mode": "semiempirical-3d-singlepoint",
            "method": self.config.method,
            "calibrated": False,
            "fitted_coefficients": False,
            "semiempirical": True,
            "proxy": False,
            "geometry_optimized_at_gfn2": False,
            "benchmark_informed_protocol_selection": True,
            "permanent_holdout_evaluated": True,
            "holdout_tuned": False,
            "holdout_protocol": "synde-ord-gfn2-singlepoint-holdout-v1",
            "holdout_group_key_sha256": (
                "4c4469be1b3a0f454fd05f6df420685b2a94839f4c4925227b028afb923cad55"
            ),
            "parameter_reference": "doi:10.1021/acs.jctc.8b01176",
        }
        molecule = Chem.MolFromSmiles(normalized.canonical_smiles)
        if molecule is None:
            return _failure_result("error", "RDKIT_SMILES_PARSE_FAILED", provenance)
        if len(Chem.GetMolFrags(molecule)) != 1:
            return _failure_result(
                "unsupported",
                "DISCONNECTED_MOLECULE_NOT_SUPPORTED",
                provenance,
            )

        molecule = Chem.AddHs(molecule)
        formal_charge = int(Chem.GetFormalCharge(molecule))
        radical_electrons = sum(
            atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()
        )
        electron_count = sum(atom.GetAtomicNum() for atom in molecule.GetAtoms())
        electron_count -= formal_charge
        if radical_electrons or electron_count % 2:
            return _failure_result(
                "unsupported",
                "OPEN_SHELL_REQUIRES_EXPLICIT_UHF",
                provenance,
                descriptors={
                    "formal_charge": formal_charge,
                    "electron_count": electron_count,
                    "radical_electrons": radical_electrons,
                },
            )

        calculator_type = _load_tblite_calculator()
        if calculator_type is None:
            return _failure_result(
                "unsupported",
                "TBLITE_NOT_INSTALLED",
                provenance,
            )

        seed = _stable_seed(
            normalized.canonical_smiles,
            namespace=self.config.seed_namespace,
        )
        warnings: list[str] = []
        try:
            preoptimizer = _embed_and_preoptimize(
                molecule,
                seed=seed,
                max_iterations=self.config.preopt_max_iterations,
            )
            if preoptimizer["not_converged"]:
                warnings.append("PREOPT_NOT_CONVERGED")
            energy_hartree = _singlepoint(
                calculator_type,
                molecule,
                method=self.config.method,
                formal_charge=formal_charge,
                max_iterations=self.config.scf_max_iterations,
                accuracy=self.config.accuracy,
            )
        except Exception as error:  # noqa: BLE001 - backend exception API varies.
            return _failure_result(
                "error",
                f"GFN2_SINGLEPOINT_FAILED:{type(error).__name__}",
                provenance,
                descriptors={
                    "formal_charge": formal_charge,
                    "electron_count": electron_count,
                    "embedding_seed": seed,
                    "error": str(error),
                },
            )

        energy_ev = float(energy_hartree * HARTREE_TO_EV)
        if not math.isfinite(energy_ev):
            return _failure_result(
                "error",
                "GFN2_NONFINITE_ENERGY",
                provenance,
            )
        descriptors = {
            "graph_identity": normalized.identity,
            "canonical_smiles": normalized.canonical_smiles,
            "formal_charge": formal_charge,
            "unpaired_electrons": 0,
            "radical_electrons": radical_electrons,
            "electron_count": electron_count,
            "atom_count_including_h": molecule.GetNumAtoms(),
            "embedding_seed": seed,
            "embedding_method": self.config.embedding_method,
            "preoptimization_force_field": preoptimizer["force_field"],
            "energy_hartree": float(energy_hartree),
        }
        return MoleculeScoreResult(
            status="partial" if warnings else "success",
            score=energy_ev,
            units="eV",
            components={"gfn2_singlepoint_energy": energy_ev},
            descriptors=descriptors,
            warnings=tuple(warnings),
            provenance=provenance,
        )


def _load_tblite_calculator() -> type[Any] | None:
    """Load the optional tblite backend without making it a core dependency."""
    try:
        module = importlib.import_module("tblite.interface")
    except ImportError:
        return None
    return module.Calculator


def _stable_seed(smiles: str, *, namespace: str) -> int:
    digest = hashlib.sha256(f"{namespace}:{smiles}".encode()).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def _embed_and_preoptimize(
    molecule: Chem.Mol,
    *,
    seed: int,
    max_iterations: int,
) -> dict[str, str | bool]:
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useRandomCoords = False
    params.maxIterations = 1000
    if AllChem.EmbedMolecule(molecule, params) < 0:
        raise ValueError("ETKDGv3 embedding failed")

    if AllChem.MMFFHasAllMoleculeParams(molecule):
        status = AllChem.MMFFOptimizeMolecule(
            molecule,
            mmffVariant="MMFF94",
            maxIters=max_iterations,
        )
        force_field = "MMFF94"
    elif AllChem.UFFHasAllMoleculeParams(molecule):
        status = AllChem.UFFOptimizeMolecule(
            molecule,
            maxIters=max_iterations,
        )
        force_field = "UFF"
    else:
        raise ValueError("No MMFF94 or UFF parameters for molecule")
    if status < 0:
        raise ValueError(f"{force_field} preoptimization failed")
    return {
        "force_field": force_field,
        "not_converged": bool(status),
    }


def _singlepoint(
    calculator_type: type[Any],
    molecule: Chem.Mol,
    *,
    method: str,
    formal_charge: int,
    max_iterations: int,
    accuracy: float,
) -> float:
    conformer = molecule.GetConformer()
    positions_angstrom = np.asarray(
        [
            [
                conformer.GetAtomPosition(index).x,
                conformer.GetAtomPosition(index).y,
                conformer.GetAtomPosition(index).z,
            ]
            for index in range(molecule.GetNumAtoms())
        ],
        dtype=float,
    )
    atomic_numbers = np.asarray(
        [atom.GetAtomicNum() for atom in molecule.GetAtoms()],
        dtype=int,
    )
    calculator = calculator_type(
        method,
        atomic_numbers,
        positions_angstrom / ANGSTROM_PER_BOHR,
        charge=formal_charge,
        uhf=0,
    )
    calculator.set("verbosity", 0)
    calculator.set("max-iter", max_iterations)
    calculator.set("accuracy", accuracy)
    result = calculator.singlepoint()
    return float(result.get("energy"))


def _failure_result(
    status: str,
    warning: str,
    provenance: dict[str, Any],
    *,
    descriptors: dict[str, Any] | None = None,
) -> MoleculeScoreResult:
    return MoleculeScoreResult(
        status=status,
        score=None,
        units="eV",
        components={},
        descriptors=descriptors or {},
        warnings=(warning,),
        provenance=provenance,
    )


__all__ = [
    "GFN2SinglePointConfig",
    "GFN2SinglePointScorer",
]
