from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

from rdkit import Chem
from rdkit.Chem import AllChem

from synkit.IO import setup_logging

logger = setup_logging()


@dataclass
class ForceFieldConfig:
    """
    Simple container for force-field related defaults.

    :param default_method: Default force field method to use (e.g. "MMFF94" or "UFF").
    :param min_iter: Minimum iterations used when max_iter="auto".
    :param max_iter: Hard upper bound for iterations when max_iter="auto".
    :param incr_iter: Per-atom increment used when max_iter="auto".
    :param num_threads: Default number of threads to pass to RDKit minimizers.
    """

    default_method: str = "MMFF94"
    min_iter: int = 20
    max_iter: int = 2000
    incr_iter: int = 10
    num_threads: int = 1
    _available_methods: List[str] = field(
        default_factory=lambda: ["MMFF", "MMFF94", "MMFF94s", "UFF"]
    )


class ForceField:
    """
    Object-oriented interface to RDKit force-field utilities (MMFF / UFF).

    The class wraps common operations:
      - Minimization of all embedded conformers
      - Compute energy of a specified conformer
      - Extract lowest-energy conformer as a new molecule

    Usage example
    -------------
    >>> ff = ForceField()
    >>> mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    >>> AllChem.EmbedMolecule(mol)
    >>> mol_min = ff.minimize(mol)  # returns minimized molecule
    """

    def __init__(self, config: Optional[ForceFieldConfig] = None) -> None:
        """
        Initialize the ForceField helper.

        :param config: Optional ForceFieldConfig to override defaults.
        """
        self._config = config or ForceFieldConfig()

    # ---------- dunder helpers ----------
    def __repr__(self) -> str:
        return (
            f"<ForceField default_method={self.default_method!r}"
            + f" threads={self.num_threads}>"
        )

    # ---------- convenience / introspection ----------
    @property
    def config(self) -> ForceFieldConfig:
        """Return the configuration dataclass used by this instance."""
        return self._config

    @property
    def default_method(self) -> str:
        """Default force field method (e.g. 'MMFF94')."""
        return self._config.default_method

    @property
    def num_threads(self) -> int:
        """Default number of threads used in optimizations."""
        return self._config.num_threads

    @property
    def available_methods(self) -> List[str]:
        """List of allowed / recognised force field method strings."""
        return list(self._config._available_methods)

    @classmethod
    def show_help(cls) -> str:
        """
        Return a short usage summary for quick interactive help.

        :return: Short usage string.
        """
        return (
            cls.__doc__
            or "ForceField helper for RDKit minimization & energy calculation."
        )

    # ---------- public API ----------
    def minimize(
        self,
        molecule: Chem.Mol,
        force_field_method: Optional[str] = None,
        max_iter: Union[int, str] = "auto",
        return_energies: bool = False,
        num_threads: Optional[int] = None,
        **kwargs,
    ) -> Union[Chem.Mol, Tuple[Chem.Mol, List[float]]]:
        """
        Minimize all conformers of ``molecule`` using the requested force field.

        :param molecule: RDKit molecule containing one or more conformers.
        :param force_field_method: "MMFF94", "MMFF94s", "UFF", or "MMFF" shorthand.
                                  If None, uses instance default.
        :param max_iter: Number of iterations or "auto" to compute from molecule size.
        :param return_energies: If True, return (molecule, energies_list).
        :param num_threads: Number of threads to pass to RDKit (overrides config).
        :param kwargs: Extra keyword args forwarded to RDKit optimizer (if supported).
        :returns: Minimized molecule or (molecule, energies).
        :raises ValueError: if molecule has no conformers or an invalid force field
        is given.
        :raises RuntimeError: on RDKit initialization / execution errors.
        """
        force_field_method = force_field_method or self.default_method
        self._validate_force_field(force_field_method)
        self._ensure_has_conformers(molecule)

        # copy molecule so original is left intact
        mol = Chem.Mol(molecule)

        if mol.GetNumConformers() <= 1:
            # nothing to minimize (still return energies if requested: compute energy)
            if return_energies:
                energies = [self.compute_energy(mol, 0, force_field_method)]
                return mol, energies
            return mol

        if max_iter == "auto" or max_iter is None:
            max_iter = self._get_max_iter_from_molecule_size(
                mol,
                min_iter=self._config.min_iter,
                max_iter=self._config.max_iter,
                incr_iter=self._config.incr_iter,
            )

        num_threads = int(num_threads or self.num_threads)

        # Normalize variant names
        ff_for_call = "MMFF94" if force_field_method == "MMFF" else force_field_method

        try:
            if ff_for_call.startswith("MMFF"):
                # sanitize and run MMFF optimizer
                AllChem.MMFFSanitizeMolecule(mol)
                results = AllChem.MMFFOptimizeMoleculeConfs(
                    mol,
                    maxIters=int(max_iter),
                    numThreads=num_threads,
                    mmffVariant=ff_for_call,
                    **kwargs,
                )
            else:
                # UFF
                results = AllChem.UFFOptimizeMoleculeConfs(
                    mol, maxIters=int(max_iter), numThreads=num_threads, **kwargs
                )
        except Exception as exc:  # pragma: no cover - RDKit runtime may raise
            logger.warning(
                "Force field minimization raised an exception (%s)."
                + " Returning molecule unchanged.",
                exc,
            )
            raise RuntimeError(f"Force field minimization failed: {exc}") from exc

        # results is a list of (convergence_flag, energy) tuples
        energies = [float(r[1]) for r in results]
        # The exact meaning of the convergence flag can depend on RDKit; we check if any
        # conformer shows a non-terminal flag value and warn if all are non-converged.
        converged_flags = [r[0] for r in results]
        if not any((f == 0 or f == 1) for f in converged_flags):  # conservative check
            logger.warning(
                "%s minimization produced unexpected convergence flags"
                + " after %s iterations: %s",
                ff_for_call,
                max_iter,
                converged_flags,
            )

        if return_energies:
            return mol, energies
        return mol

    def compute_energy(
        self,
        molecule: Chem.Mol,
        conformer_id: int,
        force_field_method: Optional[str] = None,
    ) -> float:
        """
        Compute the force-field energy for a single conformer.

        :param molecule: RDKit molecule containing the conformer.
        :param conformer_id: ID (index) of the conformer to evaluate.
        :param force_field_method: Force-field (MMFF94, MMFF94s, or UFF).
        If None uses instance default.
        :returns: Energy (float) in the units RDKit reports (kcal / mol).
        :raises ValueError: on invalid conformer id or unsupported force field.
        :raises RuntimeError: on RDKit initialization errors.
        """
        force_field_method = force_field_method or self.default_method
        self._validate_force_field(force_field_method)
        self._ensure_has_conformers(molecule)

        if conformer_id >= molecule.GetNumConformers() or conformer_id < 0:
            raise ValueError(
                f"Conformer id {conformer_id} is out of bounds"
                + f" (0..{molecule.GetNumConformers()-1})."
            )

        ff_for_call = "MMFF94" if force_field_method == "MMFF" else force_field_method
        try:
            if ff_for_call.startswith("MMFF"):
                mmff_props = AllChem.MMFFGetMoleculeProperties(
                    molecule, mmffVariant=ff_for_call
                )
                if not mmff_props:
                    raise RuntimeError(
                        f"MMFF properties initialization failed for variant {ff_for_call}"
                    )
                ff = AllChem.MMFFGetMoleculeForceField(
                    molecule, mmff_props, confId=int(conformer_id)
                )
            else:
                ff = AllChem.UFFGetMoleculeForceField(
                    molecule, confId=int(conformer_id)
                )

            if ff is None:
                raise RuntimeError("RDKit failed to create a force-field object.")
            energy = float(ff.CalcEnergy())
            return energy
        except Exception as exc:  # pragma: no cover - depends on RDKit internals
            logger.error("Error computing force-field energy: %s", exc)
            raise RuntimeError(f"Error computing force-field energy: {exc}") from exc

    def get_lowest_energy_conformer(
        self, molecule: Chem.Mol, force_field_method: Optional[str] = None
    ) -> Chem.Mol:
        """
        Return a NEW RDKit molecule (copy) containing only the lowest-energy conformer.

        :param molecule: RDKit molecule with >= 1 conformers.
        :param force_field_method: Force-field used to score conformers.
        :returns: New RDKit molecule with a single conformer (the lowest-energy one).
        :raises ValueError: if the molecule contains no conformers or
        no finite-energy conformer is found.
        """
        force_field_method = force_field_method or self.default_method
        self._validate_force_field(force_field_method)
        self._ensure_has_conformers(molecule)

        best_energy = float("inf")
        best_id: Optional[int] = None

        for conf in molecule.GetConformers():
            cid = conf.GetId()
            try:
                e = self.compute_energy(molecule, cid, force_field_method)
            except RuntimeError:
                # skip conformers that fail scoring
                continue
            if e < best_energy:
                best_energy = e
                best_id = cid

        if best_id is None:
            raise ValueError("Failed to find any conformer with a finite energy.")

        new_mol = Chem.Mol(molecule)
        new_mol.RemoveAllConformers()
        new_mol.AddConformer(molecule.GetConformer(best_id), assignId=True)
        return new_mol

    # ---------- internal helpers ----------
    def _get_max_iter_from_molecule_size(
        self, molecule: Chem.Mol, min_iter: int, max_iter: int, incr_iter: int
    ) -> int:
        """
        Compute a reasonable maximum iteration count based on number of atoms.

        :returns: integer maximum iterations.
        """
        num_atoms = molecule.GetNumAtoms()
        return min(int(max_iter), int(min_iter + num_atoms * int(incr_iter)))

    def _validate_force_field(self, method: str) -> None:
        """
        Raise ValueError if method is not supported.
        """
        if method not in self.available_methods and method not in [
            "MMFF94",
            "MMFF94s",
            "UFF",
        ]:
            raise ValueError(
                f"Unsupported force field method: {method!r}."
                + f" Supported: {self.available_methods}"
            )

    @staticmethod
    def _ensure_has_conformers(molecule: Chem.Mol) -> None:
        """
        Ensure molecule has at least one conformer; otherwise raise ValueError.
        """
        if molecule is None or molecule.GetNumConformers() == 0:
            raise ValueError("Provided molecule has no conformers.")
