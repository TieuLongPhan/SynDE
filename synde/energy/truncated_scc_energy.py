"""Frozen two-cycle GFN2-xTB score with an explicit pair-repulsion term.

The score is deliberately *not* a converged GFN2-xTB single-point energy::

    S_2 = E_SCC^(2) + E_rep

``E_SCC^(2)`` is the electronic/SCC energy printed after exactly two charge
updates. ``E_rep`` is recomputed in Python from the published GFN2 pair
potential and the original GFN2 element parameters.  Both terms enter with
their physical coefficient of one; no ORD labels, regression coefficients, or
target-dependent rescaling are used.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem

from synde.graph.graph_schema import NormalizedMolecularGraph

from .results import MoleculeScoreResult
from .semiempirical_energy import (
    ANGSTROM_PER_BOHR,
    HARTREE_TO_EV,
    _embed_and_preoptimize,
    _stable_seed,
)

_ITERATION_PATTERN = re.compile(
    r"^\s*(?P<iteration>\d+)\s+"
    r"(?P<energy>[-+0-9.Ee]+)\s+"
    r"(?P<delta>[-+0-9.Ee]+)\s+"
    r"(?P<rmsdq>[-+0-9.Ee]+)\s+"
    r"(?P<gap>[-+0-9.Ee]+)\s+",
    flags=re.MULTILINE,
)
_REPA_PATTERN = re.compile(r"\bREPA=\s*([-+0-9.Ee]+)")
_REPB_PATTERN = re.compile(r"\bREPB=\s*([-+0-9.Ee]+)")


@dataclass(frozen=True)
class GFN2TwoCycleConfig:
    """Frozen, zero-fit protocol selected using development groups only."""

    executable: str | None = None
    parameter_file: str | Path | None = None
    embedding_method: str = "ETKDGv3"
    seed_namespace: str = "synde-gfn2-two-cycle-mmff-v1"
    preopt_max_iterations: int = 500
    timeout_seconds: int = 60


class GFN2TwoCycleScorer:
    """Evaluate the additive two-cycle SCC plus pair-repulsion score."""

    def __init__(self, config: GFN2TwoCycleConfig | None = None) -> None:
        self.config = config or GFN2TwoCycleConfig()

    def score(self, normalized: NormalizedMolecularGraph) -> MoleculeScoreResult:
        """Return the uncalibrated two-term score in eV."""
        provenance: dict[str, Any] = {
            "model_name": "synde-gfn2-two-cycle-mmff-v1",
            "mode": "truncated-semiempirical-3d-score",
            "method": "GFN2-xTB Hamiltonian, exactly two SCC charge updates",
            "functional_form": "S_2 = E_SCC^(2) + E_rep",
            "term_coefficients": {
                "two_cycle_scc_energy": 1.0,
                "explicit_pair_repulsion": 1.0,
            },
            "calibrated": False,
            "fitted_coefficients": False,
            "target_rescaling": False,
            "semiempirical": True,
            "proxy": False,
            "actual_converged_gfn2": False,
            "intentional_scc_truncation": True,
            "scc_charge_updates": 2,
            "geometry_optimized_at_gfn2": False,
            "benchmark_informed_protocol_selection": True,
            "permanent_holdout_evaluated": True,
            "holdout_tuned": False,
            "holdout_protocol": ("synde-ord-gfn2-two-cycle-permanent-holdout-v1"),
            "holdout_group_key_sha256": (
                "9faddc3096bfe20f6f6cccfc1eb1e4d8c8502478fbbaba6829fcf9846875e54c"
            ),
            "parameter_reference": "doi:10.1021/acs.jctc.8b01176",
            "repulsion_equation": (
                "sum_A<B Zeff_A*Zeff_B/R_AB * " "exp[-sqrt(alpha_A*alpha_B)*R_AB^k_AB]"
            ),
        }
        executable = self.config.executable or shutil.which("xtb")
        if executable is None:
            return _failure_result(
                "unsupported", "XTB_EXECUTABLE_NOT_FOUND", provenance
            )

        try:
            parameter_file = _locate_parameter_file(
                executable,
                configured=self.config.parameter_file,
            )
            parameter_text = parameter_file.read_text(encoding="utf-8")
            repulsion_parameters = parse_gfn2_repulsion_parameters(parameter_text)
        except (OSError, ValueError) as error:
            return _failure_result(
                "unsupported",
                f"GFN2_PARAMETER_FILE_UNAVAILABLE:{type(error).__name__}",
                provenance,
                descriptors={"error": str(error)},
            )
        provenance["parameter_file_sha256"] = hashlib.sha256(
            parameter_text.encode()
        ).hexdigest()

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

        missing_elements = sorted(
            {
                atom.GetAtomicNum()
                for atom in molecule.GetAtoms()
                if atom.GetAtomicNum() not in repulsion_parameters
            }
        )
        if missing_elements:
            return _failure_result(
                "unsupported",
                "GFN2_REPULSION_PARAMETERS_MISSING",
                provenance,
                descriptors={"missing_atomic_numbers": missing_elements},
            )

        seed = _stable_seed(
            normalized.canonical_smiles,
            namespace=self.config.seed_namespace,
        )
        warnings: list[str] = ["INTENTIONAL_SCC_NONCONVERGENCE"]
        try:
            preoptimizer = _embed_and_preoptimize(
                molecule,
                seed=seed,
                max_iterations=self.config.preopt_max_iterations,
            )
            if preoptimizer["not_converged"]:
                warnings.append("PREOPT_NOT_CONVERGED")
            output, return_code = _run_two_cycle_xtb(
                executable,
                molecule,
                formal_charge=formal_charge,
                parameter_file=parameter_file,
                timeout_seconds=self.config.timeout_seconds,
            )
            iterations = parse_xtb_scc_iterations(output)
            if 2 not in iterations:
                raise ValueError("xTB output does not contain SCC iteration 2")
            expected_abort = (
                "convergence criteria cannot be satisfied within 2 iterations" in output
                and "did not converge" in output
            )
            if return_code != 0 and not expected_abort:
                raise RuntimeError(
                    f"xTB failed for a reason other than the defined truncation "
                    f"(return code {return_code})"
                )
            pair_repulsion_hartree = gfn2_pair_repulsion(
                molecule,
                repulsion_parameters,
            )
        except Exception as error:  # noqa: BLE001 - subprocess/backend errors vary.
            return _failure_result(
                "error",
                f"GFN2_TWO_CYCLE_FAILED:{type(error).__name__}",
                provenance,
                descriptors={
                    "formal_charge": formal_charge,
                    "electron_count": electron_count,
                    "embedding_seed": seed,
                    "error": str(error),
                },
            )

        second = iterations[2]
        scc_ev = second["energy_hartree"] * HARTREE_TO_EV
        pair_repulsion_ev = pair_repulsion_hartree * HARTREE_TO_EV
        score = scc_ev + pair_repulsion_ev
        if not math.isfinite(score):
            return _failure_result(
                "error",
                "GFN2_TWO_CYCLE_NONFINITE_ENERGY",
                provenance,
            )
        components = {
            "two_cycle_scc_energy": float(scc_ev),
            "explicit_pair_repulsion": float(pair_repulsion_ev),
        }
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
            "scc_iteration_1_energy_hartree": iterations[1]["energy_hartree"],
            "scc_iteration_2_energy_hartree": second["energy_hartree"],
            "scc_iteration_2_delta_hartree": second["delta_hartree"],
            "scc_iteration_2_charge_rms": second["charge_rms"],
            "scc_iteration_2_homo_lumo_gap_ev": second["gap_ev"],
            "explicit_pair_repulsion_hartree": pair_repulsion_hartree,
            "xtb_return_code": return_code,
            "expected_nonconvergence_abort": expected_abort,
        }
        return MoleculeScoreResult(
            status="partial",
            score=float(score),
            units="eV",
            components=components,
            descriptors=descriptors,
            warnings=tuple(warnings),
            provenance=provenance,
        )


def parse_gfn2_repulsion_parameters(text: str) -> dict[int, tuple[float, float]]:
    """Parse ``atomic number -> (alpha, effective charge)`` from GFN2 data."""
    parameters: dict[int, tuple[float, float]] = {}
    for block in text.split("$Z=")[1:]:
        fields = block.split()
        if not fields:
            continue
        try:
            atomic_number = int(fields[0])
        except ValueError:
            continue
        repa = _REPA_PATTERN.search(block)
        repb = _REPB_PATTERN.search(block)
        if repa is not None and repb is not None:
            parameters[atomic_number] = (float(repa.group(1)), float(repb.group(1)))
    if not parameters:
        raise ValueError("No GFN2 REPA/REPB blocks found")
    return parameters


def gfn2_pair_repulsion(
    molecule: Chem.Mol,
    parameters: dict[int, tuple[float, float]],
) -> float:
    """Return the explicit GFN2 pair repulsion in hartree."""
    conformer = molecule.GetConformer()
    energy = 0.0
    for left, atom_left in enumerate(molecule.GetAtoms()):
        alpha_left, zeff_left = parameters[atom_left.GetAtomicNum()]
        point_left = conformer.GetAtomPosition(left)
        for right in range(left + 1, molecule.GetNumAtoms()):
            atom_right = molecule.GetAtomWithIdx(right)
            alpha_right, zeff_right = parameters[atom_right.GetAtomicNum()]
            point_right = conformer.GetAtomPosition(right)
            distance_bohr = point_left.Distance(point_right) / ANGSTROM_PER_BOHR
            if distance_bohr <= 0:
                raise ValueError("Coincident atoms in pair-repulsion evaluation")
            exponent = (
                1.0
                if atom_left.GetAtomicNum() <= 2 and atom_right.GetAtomicNum() <= 2
                else 1.5
            )
            energy += (
                zeff_left
                * zeff_right
                / distance_bohr
                * math.exp(
                    -math.sqrt(alpha_left * alpha_right) * distance_bohr**exponent
                )
            )
    return float(energy)


def parse_xtb_scc_iterations(output: str) -> dict[int, dict[str, float]]:
    """Parse the stable numeric SCC iteration table emitted by xTB 6.x."""
    table = re.search(
        r"iter\s+E\s+dE\s+RMSdq\s+gap.*?\n(?P<body>.*?)\n\s*\*\*\*",
        output,
        flags=re.DOTALL,
    )
    if table is None:
        return {}
    rows: dict[int, dict[str, float]] = {}
    for match in _ITERATION_PATTERN.finditer(table.group("body")):
        iteration = int(match.group("iteration"))
        rows[iteration] = {
            "energy_hartree": float(match.group("energy")),
            "delta_hartree": float(match.group("delta")),
            "charge_rms": float(match.group("rmsdq")),
            "gap_ev": float(match.group("gap")),
        }
    return rows


def _locate_parameter_file(
    executable: str,
    *,
    configured: str | Path | None,
) -> Path:
    candidates: list[Path] = []
    if configured is not None:
        candidates.append(Path(configured).expanduser())
    xtbpath = os.environ.get("XTBPATH")
    if xtbpath:
        candidates.append(Path(xtbpath) / "param_gfn2-xtb.txt")
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "share" / "xtb" / "param_gfn2-xtb.txt")
    prefix = Path(executable).expanduser().resolve().parent.parent
    candidates.append(prefix / "share" / "xtb" / "param_gfn2-xtb.txt")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "param_gfn2-xtb.txt not found in configured path, XTBPATH, "
        "CONDA_PREFIX, or the xTB installation prefix"
    )


def _run_two_cycle_xtb(
    executable: str,
    molecule: Chem.Mol,
    *,
    formal_charge: int,
    parameter_file: Path,
    timeout_seconds: int,
) -> tuple[str, int]:
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "XTBPATH": str(parameter_file.parent),
    }
    with tempfile.TemporaryDirectory(prefix="synde-gfn2-two-cycle-") as directory:
        coordinate_file = Path(directory) / "molecule.xyz"
        _write_xyz(coordinate_file, molecule)
        process = subprocess.run(
            [
                executable,
                coordinate_file.name,
                "--gfn",
                "2",
                "--sp",
                "--chrg",
                str(formal_charge),
                "--uhf",
                "0",
                "--iterations",
                "2",
                "--norestart",
            ],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=environment,
        )
    return process.stdout + process.stderr, process.returncode


def _write_xyz(path: Path, molecule: Chem.Mol) -> None:
    conformer = molecule.GetConformer()
    lines = [str(molecule.GetNumAtoms()), "SynDE deterministic RDKit geometry"]
    for index, atom in enumerate(molecule.GetAtoms()):
        point = conformer.GetAtomPosition(index)
        lines.append(
            f"{atom.GetSymbol():<3} {point.x: .10f} " f"{point.y: .10f} {point.z: .10f}"
        )
    path.write_text("\n".join((*lines, "")), encoding="utf-8")


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
    "GFN2TwoCycleConfig",
    "GFN2TwoCycleScorer",
    "gfn2_pair_repulsion",
    "parse_gfn2_repulsion_parameters",
    "parse_xtb_scc_iterations",
]
