import os
import re
import shutil
import tempfile
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from rdkit import Chem
from rdkit.Chem import AllChem
from joblib import Parallel, delayed

from synkit.IO import setup_logging

logger = setup_logging()


class XTBMinimize:
    """
    SMILES → 3D (RDKit) → XYZ → xTB geometry optimization.

    Supports xTB optimization *level* ladder and auto-sets ``--acc`` per level.
    (Energy/gradient targets shown for reference; xTB enforces them internally.)

    Level table:
        level     Econv [Eh]   Gconv [Eh·α⁻¹]   acc
        ------    ----------   ---------------   ----
        crude     5e-4         1e-2              3.00
        sloppy    1e-4         6e-3              3.00
        loose     5e-5         4e-3              2.00
        lax       2e-5         2e-3              2.00
        normal    5e-6         1e-3              1.00
        tight     1e-6         8e-4              0.20
        vtight    1e-7         2e-4              0.05
        extreme   5e-8         5e-5              0.01

    :param smiles: SMILES representation of the molecule.
    :type smiles: str
    :param xtb_executable: xTB binary name or absolute path, defaults to "xtb".
    :type xtb_executable: str, optional
    :param embed_max_attempts: Max attempts for RDKit ETKDG embedding, defaults to 3.
    :type embed_max_attempts: int, optional
    :param embed_seed: Random seed for embedding (None for random), defaults to 42.
    :type embed_seed: Optional[int], optional
    :param charge: Total molecular charge, forwarded as ``--chrg``, defaults to 0.
    :type charge: int, optional
    :param multiplicity: Spin multiplicity (1=singlet) → ``--uhf=multiplicity-1``.
    Defaults to 1.
    :type multiplicity: int, optional
    :param gfn: GFN level (0/1/2) → ``--gfn``, or None to omit, defaults to 2.
    :type gfn: Optional[int], optional
    :param alpb: ALPB solvent keyword (e.g. "water"), or None to omit, defaults to None.
    :type alpb: Optional[str], optional
    """

    # Unit conversions
    HARTREE_TO_KJMOL: float = 2625.499638
    HARTREE_TO_KCALMOL: float = 627.509474
    HARTREE_TO_EV: float = 27.211386245988

    # Level → recommended --acc (Econv/Gconv listed for reference)
    LEVEL_SPECS: Dict[str, Dict[str, float]] = {
        "crude": {"Econv": 5e-4, "Gconv": 1e-2, "acc": 3.00},
        "sloppy": {"Econv": 1e-4, "Gconv": 6e-3, "acc": 3.00},
        "loose": {"Econv": 5e-5, "Gconv": 4e-3, "acc": 2.00},
        "lax": {"Econv": 2e-5, "Gconv": 2e-3, "acc": 2.00},
        "normal": {"Econv": 5e-6, "Gconv": 1e-3, "acc": 1.00},
        "tight": {"Econv": 1e-6, "Gconv": 8e-4, "acc": 0.20},
        "vtight": {"Econv": 1e-7, "Gconv": 2e-4, "acc": 0.05},
        "extreme": {"Econv": 5e-8, "Gconv": 5e-5, "acc": 0.01},
    }

    def __init__(
        self,
        smiles: str,
        xtb_executable: str = "xtb",
        embed_max_attempts: int = 3,
        embed_seed: Optional[int] = 42,
        *,
        charge: int = 0,
        multiplicity: int = 1,
        gfn: Optional[int] = 2,
        alpb: Optional[str] = None,
    ) -> None:
        self.smiles: str = smiles
        self.xtb_executable: str = xtb_executable
        self.embed_max_attempts: int = max(1, int(embed_max_attempts))
        self.embed_seed: Optional[int] = embed_seed

        self.charge: int = int(charge)
        self.multiplicity: int = max(1, int(multiplicity))
        self.gfn: Optional[int] = gfn if gfn is None else int(gfn)
        self.alpb: Optional[str] = alpb

    # ---------------------------------------------------------------------
    # Compatibility helpers (names kept similar to older code)
    # ---------------------------------------------------------------------
    @staticmethod
    def smiles_to_3D(smiles: str) -> Optional[Chem.Mol]:
        """
        Convert a SMILES string to a 3D RDKit molecule (single quick attempt).

        Prefer using the instance method via :meth:`optimize` which retries and
        pre-relaxes more robustly.

        :param smiles: SMILES to convert.
        :type smiles: str
        :return: RDKit molecule with 3D coordinates (or None on failure).
        :rtype: Optional[Chem.Mol]
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception:
                try:
                    AllChem.UFFOptimizeMolecule(mol)
                except Exception:
                    pass
            logger.info("3D molecule generation and quick pre-optimization completed.")
            return mol
        except Exception as e:
            logger.error(f"Failed in generating 3D structure from SMILES: {e}")
            return None

    @staticmethod
    def save_mol_to_xyz(molecule: Chem.Mol, filename: str = "molecule.xyz") -> str:
        """
        Save an RDKit molecule (with a conformer) to an XYZ file.

        :param molecule: RDKit molecule with a conformer.
        :type molecule: Chem.Mol
        :param filename: Output XYZ filename, defaults to "molecule.xyz".
        :type filename: str, optional
        :return: Path to the saved XYZ file.
        :rtype: str
        :raises RuntimeError: If the molecule has no conformer.
        """
        conf = molecule.GetConformer()
        if conf is None:
            raise RuntimeError("No conformer present on molecule.")
        num_atoms = molecule.GetNumAtoms()
        with open(filename, "w") as fh:
            fh.write(f"{num_atoms}\nXYZ file generated by RDKit\n")
            for i in range(num_atoms):
                sym = molecule.GetAtomWithIdx(i).GetSymbol()
                pos = conf.GetAtomPosition(i)
                fh.write(f"{sym} {pos.x:.10f} {pos.y:.10f} {pos.z:.10f}\n")
        logger.info(f"XYZ file '{filename}' has been generated and saved.")
        return filename

    # ---------------------------------------------------------------------
    # Internal RDKit + xTB utilities
    # ---------------------------------------------------------------------
    def _validate_smiles(self) -> Chem.Mol:
        """
        Parse and hydrogenate SMILES.

        :return: RDKit molecule with explicit Hs.
        :rtype: Chem.Mol
        :raises ValueError: If SMILES is invalid.
        """
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {self.smiles}")
        return Chem.AddHs(mol)

    def _embed_and_relax(self, mol: Chem.Mol) -> Chem.Mol:
        """
        Embed with ETKDGv3 and pre-relax using MMFF or UFF.

        Retries embedding up to ``embed_max_attempts`` with seed tweaks.

        :param mol: RDKit molecule with explicit hydrogens.
        :type mol: Chem.Mol
        :return: Molecule with a 3D conformer.
        :rtype: Chem.Mol
        :raises RuntimeError: If embedding fails after the allowed attempts.
        """
        params = AllChem.ETKDGv3()
        if self.embed_seed is not None:
            params.randomSeed = int(self.embed_seed)

        for attempt in range(1, self.embed_max_attempts + 1):
            code = AllChem.EmbedMolecule(mol, params)
            if code == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                except Exception:
                    try:
                        AllChem.UFFOptimizeMolecule(mol)
                    except Exception:
                        pass
                logger.info(
                    "3D embedding and pre-optimization completed (attempt %d).", attempt
                )
                return mol
            if self.embed_seed is not None:
                params.randomSeed += attempt
        raise RuntimeError(
            f"Failed to generate 3D conformer after {self.embed_max_attempts} attempts."
        )

    def _which_xtb(self) -> str:
        """
        Locate the xTB executable.

        :return: Resolved path to xTB executable.
        :rtype: str
        :raises FileNotFoundError: If xTB is not found or not executable.
        """
        path = shutil.which(self.xtb_executable)
        if path:
            return path
        if os.path.isfile(self.xtb_executable) and os.access(
            self.xtb_executable, os.X_OK
        ):
            return self.xtb_executable
        raise FileNotFoundError(
            f"xTB executable '{self.xtb_executable}' not found on PATH or not executable."
        )

    @classmethod
    def _acc_for_level(cls, level: str) -> float:
        """
        Get recommended ``--acc`` for a given xTB level.

        :param level: One of: crude, sloppy, loose, lax, normal, tight, vtight, extreme.
        :type level: str
        :return: Accuracy parameter value for ``--acc``.
        :rtype: float
        :raises ValueError: If level is unknown.
        """
        spec = cls.LEVEL_SPECS.get(level.lower())
        if not spec:
            raise ValueError(f"Unknown level '{level}'. Valid: {list(cls.LEVEL_SPECS)}")
        return spec["acc"]

    @staticmethod
    def _parse_energy(stdout: str, stderr: str) -> Optional[float]:
        """
        Parse final total energy (Hartree) from xTB output.

        :param stdout: xTB stdout text.
        :type stdout: str
        :param stderr: xTB stderr text.
        :type stderr: str
        :return: Energy in Hartree (Eh) if found, else None.
        :rtype: Optional[float]
        """
        text = (stdout or "") + "\n" + (stderr or "")

        # Prefer the last line that mentions TOTAL and ENERGY
        last_hit: Optional[str] = None
        for line in text.splitlines():
            u = line.upper()
            if "TOTAL" in u and "ENERGY" in u:
                last_hit = line
        if last_hit:
            toks = last_hit.replace("=", " ").split()
            for tok in reversed(toks):
                try:
                    return float(tok)
                except Exception:
                    continue

        # Fallback: last floating number followed by Eh
        matches = re.findall(r"(-?\d+\.\d+(?:[Ee][+-]?\d+)?)\s*E?h\b", text)
        if matches:
            try:
                return float(matches[-1])
            except Exception:
                return None
        return None

    def _build_xtb_cmd(
        self,
        xyz_path: str,
        level: str,
        *,
        xtb_omp_threads: Optional[int],
    ) -> Dict[str, Any]:
        """
        Build xTB command, environment, and derived options.

        :param xyz_path: Path to input XYZ file.
        :type xyz_path: str
        :param level: xTB optimization level token (e.g., "normal").
        :type level: str
        :param xtb_omp_threads: Value for OMP_NUM_THREADS (None to leave as-is).
        :type xtb_omp_threads: Optional[int]
        :return: Dict with ``cmd`` (List[str]) and ``env`` (dict).
        :rtype: Dict[str, Any]
        """
        xtb_bin = self._which_xtb()
        acc = self._acc_for_level(level)

        cmd: List[str] = [xtb_bin, xyz_path, "--opt", level]
        if self.charge:
            cmd += ["--chrg", str(self.charge)]
        uhf = max(0, self.multiplicity - 1)
        if uhf:
            cmd += ["--uhf", str(uhf)]
        if self.gfn is not None:
            cmd += ["--gfn", str(int(self.gfn))]
        if self.alpb:
            cmd += ["--alpb", str(self.alpb)]
        cmd += ["--acc", f"{acc:g}"]

        env = os.environ.copy()
        if xtb_omp_threads is not None:
            env["OMP_NUM_THREADS"] = str(int(xtb_omp_threads))

        return {"cmd": cmd, "env": env}

    # ---------------------------------------------------------------------
    # Public per-molecule API
    # ---------------------------------------------------------------------
    def optimize(
        self,
        save_dir: str = "./",
        *,
        level: str = "normal",
        timeout: Optional[int] = 600,
        clean: bool = True,
        keep_intermediates: bool = False,
        xtb_omp_threads: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run xTB geometry optimization for ``self.smiles`` and collect results.

        :param save_dir: Directory to write the optimized XYZ (created if missing).
        :type save_dir: str
        :param level: One of {"crude","sloppy","loose","lax","normal","tight",
        "vtight","extreme"}.
        :type level: str
        :param timeout: Subprocess timeout in seconds (per job), defaults to 600.
        :type timeout: Optional[int]
        :param clean: If True, delete temporary workdir (default). If False, persist it.
        :type clean: bool
        :param keep_intermediates: If True, copy temp inputs/outputs/stderr/stdout into
                                   ``save_dir/intermediates_<stamp>``.
        :type keep_intermediates: bool
        :param xtb_omp_threads: If provided, set OMP_NUM_THREADS for the xTB subprocess.
        :type xtb_omp_threads: Optional[int]
        :return: Result dict with keys:
                 ``smiles, status, message, energy_Eh, energy_kJmol, energy_kcalmol,
                   energy_eV, optimized_file, stdout, stderr``.
        :rtype: Dict[str, Any]
        """
        result: Dict[str, Any] = {
            "smiles": self.smiles,
            "status": "error",
            "message": "",
            "energy_Eh": None,
            "energy_kJmol": None,
            "energy_kcalmol": None,
            "energy_eV": None,
            "optimized_file": None,
            "stdout": None,
            "stderr": None,
        }

        level = level.lower()
        if level not in self.LEVEL_SPECS:
            result["message"] = (
                f"Invalid level '{level}'. Valid: {list(self.LEVEL_SPECS)}"
            )
            return result

        os.makedirs(save_dir, exist_ok=True)

        # RDKit: parse, embed, pre-opt
        try:
            mol = self._validate_smiles()
            mol = self._embed_and_relax(mol)
        except Exception as e:
            result["message"] = f"Embedding failed: {e}"
            logger.exception(result["message"])
            return result

        stamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        base = f"mol_{stamp}"
        inter_dir = os.path.join(save_dir, f"intermediates_{base}")

        with tempfile.TemporaryDirectory() as tmp:
            try:
                inp_xyz = os.path.join(tmp, base + ".xyz")
                # write input xyz
                self.save_mol_to_xyz(mol, inp_xyz)

                spec = self._build_xtb_cmd(
                    inp_xyz, level, xtb_omp_threads=xtb_omp_threads
                )
                logger.debug("Running xTB: %s", " ".join(spec["cmd"]))
                proc = subprocess.run(
                    spec["cmd"],
                    cwd=tmp,
                    env=spec["env"],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
                if proc.returncode != 0:
                    raise subprocess.CalledProcessError(
                        proc.returncode, spec["cmd"], proc.stdout, proc.stderr
                    )

                # Locate optimized xyz
                opt_path: Optional[str] = None
                for candidate in ("xtbopt.xyz", "geopt.xyz", "xtbopt.xyz"):
                    p = os.path.join(tmp, candidate)
                    if os.path.exists(p):
                        opt_path = p
                        break

                # Save optimized file into save_dir
                if opt_path:
                    out_name = base + "_xtb_optimized.xyz"
                    out_path = os.path.join(save_dir, out_name)
                    shutil.copy(opt_path, out_path)
                    result["optimized_file"] = os.path.abspath(out_path)
                    logger.info("Optimized file saved to %s", result["optimized_file"])
                else:
                    logger.warning(
                        "No optimized file produced by xTB for %s", self.smiles
                    )

                # Energy parsing and unit conversions
                energy_eh = self._parse_energy(proc.stdout or "", proc.stderr or "")
                if energy_eh is not None:
                    result["energy_Eh"] = energy_eh
                    result["energy_kJmol"] = energy_eh * self.HARTREE_TO_KJMOL
                    result["energy_kcalmol"] = energy_eh * self.HARTREE_TO_KCALMOL
                    result["energy_eV"] = energy_eh * self.HARTREE_TO_EV

                result["stdout"] = proc.stdout
                result["stderr"] = proc.stderr
                result["status"] = "success"
                result["message"] = "Optimization completed."

                # Persist intermediates if asked (or if clean=False)
                if keep_intermediates or not clean:
                    os.makedirs(inter_dir, exist_ok=True)
                    try:
                        shutil.copy(
                            inp_xyz, os.path.join(inter_dir, os.path.basename(inp_xyz))
                        )
                        if opt_path:
                            shutil.copy(
                                opt_path,
                                os.path.join(inter_dir, os.path.basename(opt_path)),
                            )
                        with open(os.path.join(inter_dir, "xtb_stdout.txt"), "w") as fh:
                            fh.write(proc.stdout or "")
                        with open(os.path.join(inter_dir, "xtb_stderr.txt"), "w") as fh:
                            fh.write(proc.stderr or "")
                        logger.debug("Saved intermediate files to %s", inter_dir)
                    except Exception:
                        logger.exception("Failed to persist intermediates")

                return result

            except subprocess.CalledProcessError as e:
                result["message"] = f"xTB failed (rc={e.returncode})"
                result["stdout"] = e.output
                result["stderr"] = e.stderr
                logger.exception(result["message"])
                return result
            except FileNotFoundError as e:
                result["message"] = str(e)
                logger.exception(result["message"])
                return result
            except Exception as e:
                result["message"] = f"Unexpected error: {e}"
                logger.exception(result["message"])
                return result

    def fit(
        self,
        save_dir: str = "./",
        clean_xyz: bool = False,
        level: str = "loose",
    ) -> float:
        """
        Compatibility wrapper that executes the workflow and returns only energy.

        This mirrors your original API:
        - writes an optimized XYZ to ``save_dir`` unless ``clean_xyz=True``.
        - removes *all* ``.xyz`` files (including optimized) if ``clean_xyz=True``.

        :param save_dir: Directory to save the optimized file, defaults to "./".
        :type save_dir: str
        :param clean_xyz: If True, delete all .xyz artifacts after run, defaults to False.
        :type clean_xyz: bool
        :param level: xTB level ("crude"…"extreme"). Default "loose" for backward compat.
        :type level: str
        :return: Final energy in Hartree (Eh). Returns 0.0 on failure/unavailable.
        :rtype: float
        """
        try:
            res = self.optimize(save_dir=save_dir, level=level, clean=True)
            energy = res.get("energy_Eh")
            # If the legacy flag asks to clean *everything*, remove optimized XYZ as well.
            if clean_xyz and res.get("optimized_file"):
                try:
                    os.remove(res["optimized_file"])
                except Exception:
                    logger.debug("Could not remove optimized xyz during clean_xyz=True")
            return float(energy) if energy is not None else 0.0
        except Exception as e:
            logger.error(f"An error occurred during fit(): {e}")
            return 0.0

    # ---------------------------------------------------------------------
    # Batch (parallel) API
    # ---------------------------------------------------------------------
    @classmethod
    def _worker(cls, s: str, save_dir: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal worker for joblib.

        :param s: One SMILES string.
        :type s: str
        :param save_dir: Output directory.
        :type save_dir: str
        :param params: Keyword parameter bag.
        :type params: Dict[str, Any]
        :return: Result dict from :meth:`optimize`.
        :rtype: Dict[str, Any]
        """
        inst = cls(
            s,
            xtb_executable=params.get("xtb_executable", "xtb"),
            embed_max_attempts=params.get("embed_max_attempts", 3),
            embed_seed=params.get("embed_seed", 42),
            charge=params.get("charge", 0),
            multiplicity=params.get("multiplicity", 1),
            gfn=params.get("gfn", 2),
            alpb=params.get("alpb"),
        )
        return inst.optimize(
            save_dir=save_dir,
            level=params.get("level", "normal"),
            timeout=params.get("timeout", 600),
            clean=params.get("clean", True),
            keep_intermediates=params.get("keep_intermediates", False),
            xtb_omp_threads=params.get("xtb_omp_threads"),
        )

    @staticmethod
    def _safe_remove_dir(path: str) -> None:
        """
        Remove a directory safely:
         - refuse to remove root or empty paths
         - log exceptions instead of raising
        """
        try:
            if not path:
                logger.warning("Refusing to remove empty path")
                return
            abs_path = os.path.abspath(path)
            # refuse to delete root or home
            if abs_path in (os.path.abspath(os.sep), os.path.expanduser("~")):
                logger.error("Refusing to remove unsafe directory: %s", abs_path)
                return
            if not os.path.exists(abs_path):
                logger.debug("Path does not exist, nothing to remove: %s", abs_path)
                return
            shutil.rmtree(abs_path)
            logger.info("Removed directory: %s", abs_path)
        except Exception:
            logger.exception("Failed to remove directory %s", path)

    @classmethod
    def process_smiles_list(
        cls,
        smiles_list: Sequence[str],
        save_dir: str = "./",
        *,
        n_jobs: int = 1,
        backend: str = "loky",
        level: str = "normal",
        timeout: Optional[int] = 600,
        clean: bool = True,
        keep_intermediates: bool = False,
        xtb_executable: str = "xtb",
        embed_max_attempts: int = 3,
        embed_seed: Optional[int] = 42,
        charge: int = 0,
        multiplicity: int = 1,
        gfn: Optional[int] = 2,
        alpb: Optional[str] = None,
        xtb_omp_threads: Optional[int] = None,
        cleanup_save_dir: bool = False,
        remove_save_dir_after: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Optimize a list of SMILES strings in parallel.

        :param smiles_list: Sequence of SMILES to process.
        :param save_dir: Directory for outputs.
        :param n_jobs: Number of parallel workers (-1 for all cores).
        :param backend: Joblib backend; "loky" recommended with RDKit.
        :param level: xTB optimization level token (default "normal").
        :param timeout: Per-molecule timeout in seconds.
        :param clean: If True, delete temp workdirs; if False, persist them.
        :param keep_intermediates: If True, copy temp inputs/outputs to intermediates dir.
        :param xtb_executable: xTB binary name/path (default "xtb").
        :param embed_max_attempts: ETKDG embedding attempts.
        :param embed_seed: Embedding seed (None for random).
        :param charge: Molecule charge.
        :param multiplicity: Spin multiplicity.
        :param gfn: GFN flavor (0/1/2) or None to omit.
        :param alpb: ALPB solvent keyword, or None.
        :param xtb_omp_threads: Force OMP threads for xTB subprocesses.
        :param cleanup_save_dir: If True, clear ``save_dir`` before running.
        :param remove_save_dir_after: If True, remove ``save_dir``
        after the batch completes.
        :return: List of per-SMILES result dicts from :meth:`optimize`.
        """
        # Optional pre-clean of save_dir
        if cleanup_save_dir and os.path.isdir(save_dir):
            logger.info("cleanup_save_dir=True: clearing save_dir %s", save_dir)
            for fn in os.listdir(save_dir):
                p = os.path.join(save_dir, fn)
                try:
                    if os.path.isdir(p):
                        shutil.rmtree(p)
                    else:
                        os.remove(p)
                except Exception:
                    logger.exception("Failed to remove %s during cleanup_save_dir", p)

        os.makedirs(save_dir, exist_ok=True)

        common: Dict[str, Any] = dict(
            xtb_executable=xtb_executable,
            embed_max_attempts=embed_max_attempts,
            embed_seed=embed_seed,
            level=level,
            timeout=timeout,
            clean=clean,
            keep_intermediates=keep_intermediates,
            charge=charge,
            multiplicity=multiplicity,
            gfn=gfn,
            alpb=alpb,
            xtb_omp_threads=xtb_omp_threads,
        )
        tasks = (delayed(cls._worker)(s, save_dir, common) for s in smiles_list)
        results = Parallel(n_jobs=n_jobs, backend=backend)(tasks)

        # Optionally remove the save_dir after the batch has finished.
        if remove_save_dir_after:
            logger.info(
                "remove_save_dir_after=True: attempting to remove save_dir %s", save_dir
            )
            cls._safe_remove_dir(save_dir)

        return results
