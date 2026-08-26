# synde/energy/xtb/geo_opt.py
from __future__ import annotations

import os
import re
import shutil
import tempfile
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem
from joblib import Parallel, delayed

from synkit.IO import setup_logging

logger = setup_logging()

# Module-level xTB discovery (non-fatal; used by tests to decide skipping)
_XTB_PATH = shutil.which("xtb")
HAS_XTB: bool = _XTB_PATH is not None
if not HAS_XTB:
    logger.warning(
        "[xtb] 'xtb' executable not found on PATH. Geometry optimization tests "
        "will be skipped and runtime calls will fail if xTB is required."
    )


@dataclass
class GeoOptConfig:
    """
    Configuration bundle for xTB geometry optimization and RDKit pre-embedding.

    :param xtb_executable: xTB binary name or absolute path, defaults to ``'xtb'``.
    :type xtb_executable: str
    :param embed_max_attempts: Maximum ETKDG embedding attempts, defaults to ``3``.
    :type embed_max_attempts: int
    :param embed_seed: Random seed for ETKDG (``None`` for random), defaults to ``42``.
    :type embed_seed: Optional[int]
    :param charge: Total molecular charge (forwarded as ``--chrg``), defaults to ``0``.
    :type charge: int
    :param multiplicity: Spin multiplicity (1=singlet) mapped to
                         ``--uhf=multiplicity-1``, defaults to ``1``.
    :type multiplicity: int
    :param gfn: GFN flavor (0/1/2) forwarded as ``--gfn``; ``None`` to omit.
                Defaults to ``2``.
    :type gfn: Optional[int]
    :param alpb: ALPB solvent keyword (e.g. ``'water'``) forwarded as ``--alpb``.
                 ``None`` to omit.
    :type alpb: Optional[str]
    """

    xtb_executable: str = "xtb"
    embed_max_attempts: int = 3
    embed_seed: Optional[int] = 42
    charge: int = 0
    multiplicity: int = 1
    gfn: Optional[int] = 2
    alpb: Optional[str] = None


class GeoOpt:
    """
    SMILES → 3D (RDKit) → XYZ → xTB geometry optimization (geo-opt).

    Also supports single-point (SP) energy evaluations.
    """

    # Unit conversions
    HARTREE_TO_KJMOL: float = 2625.499638
    HARTREE_TO_KCALMOL: float = 627.509474
    HARTREE_TO_EV: float = 27.211386245988

    # Optimization levels → recommend --acc (Econv/Gconv are informative only)
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

    def __init__(self, smiles: str, config: Optional[GeoOptConfig] = None) -> None:
        """
        Construct the geometry optimizer for a single SMILES.

        :param smiles: SMILES representation of the molecule.
        :type smiles: str
        :param config: Optional :class:`GeoOptConfig` to override defaults.
        :type config: Optional[GeoOptConfig]
        """
        self._smiles = smiles
        self._cfg = config or GeoOptConfig()

    # Dunder & properties
    def __repr__(self) -> str:
        return (
            f"<GeoOpt smiles={self._smiles!r} xtb={self._cfg.xtb_executable!r} "
            f"charge={self._cfg.charge} mult={self._cfg.multiplicity} "
            f"gfn={self._cfg.gfn} alpb={self._cfg.alpb!r}>"
        )

    @property
    def smiles(self) -> str:
        """Return the input SMILES string."""
        return self._smiles

    @property
    def config(self) -> GeoOptConfig:
        """Return the instance configuration dataclass."""
        return self._cfg

    @classmethod
    def show_help(cls) -> str:
        """Short usage summary."""
        return cls.__doc__ or "xTB geometry optimizer."

    # Compatibility helpers
    @staticmethod
    def smiles_to_3D(smiles: str) -> Optional[Chem.Mol]:
        """
        Convert SMILES into a 3D RDKit molecule (single attempt).

        :param smiles: SMILES string to embed.
        :type smiles: str
        :return: RDKit molecule with 3D coords or ``None`` on failure.
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
            logger.info("3D generation and quick pre-optimization completed.")
            return mol
        except Exception as e:
            logger.error("Failed to generate 3D structure from SMILES: %s", e)
            return None

    @staticmethod
    def save_mol_to_xyz(molecule: Chem.Mol, filename: str = "molecule.xyz") -> str:
        """
        Write an RDKit molecule (with a conformer) to an XYZ file.

        :param molecule: RDKit molecule that must contain a conformer.
        :type molecule: Chem.Mol
        :param filename: Output XYZ path.
        :type filename: str
        :return: The filename that was written.
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
        logger.info("XYZ file '%s' has been saved.", filename)
        return filename

    # Internal helpers (RDKit + xTB)
    def _validate_smiles(self) -> Chem.Mol:
        """
        Parse and hydrogenate the instance SMILES.

        :return: RDKit molecule with explicit hydrogens.
        :rtype: Chem.Mol
        :raises ValueError: If the SMILES is invalid.
        """
        mol = Chem.MolFromSmiles(self._smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {self._smiles}")
        return Chem.AddHs(mol)

    def _embed_and_relax(self, mol: Chem.Mol) -> Chem.Mol:
        """
        Embed a 3D conformer (ETKDGv3) and pre-relax via MMFF (fallback UFF).

        Retries up to ``embed_max_attempts`` and perturbs the seed.
        """
        params = AllChem.ETKDGv3()
        if self._cfg.embed_seed is not None:
            params.randomSeed = int(self._cfg.embed_seed)

        for attempt in range(1, max(1, int(self._cfg.embed_max_attempts)) + 1):
            code = AllChem.EmbedMolecule(mol, params)
            if code == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                except Exception:
                    try:
                        AllChem.UFFOptimizeMolecule(mol)
                    except Exception:
                        pass
                logger.info("Embedding + pre-opt completed (attempt %d).", attempt)
                return mol
            if self._cfg.embed_seed is not None:
                params.randomSeed += attempt
        raise RuntimeError(
            "Failed to generate 3D conformer after "
            f"{self._cfg.embed_max_attempts} attempts."
        )

    def _which_xtb(self) -> str:
        """
        Resolve the xTB executable path.

        :return: Absolute path to the xTB binary.
        :rtype: str
        :raises FileNotFoundError: If xTB cannot be found or executed.
        """
        path = shutil.which(self._cfg.xtb_executable)
        if path:
            return path
        if os.path.isfile(self._cfg.xtb_executable) and os.access(
            self._cfg.xtb_executable, os.X_OK
        ):
            return self._cfg.xtb_executable
        raise FileNotFoundError(
            f"xTB executable '{self._cfg.xtb_executable}' not found on PATH "
            "or not executable."
        )

    @classmethod
    def _acc_for_level(cls, level: str) -> float:
        """
        Get the recommended ``--acc`` value for a given xTB level.

        :param level: Level name token.
        :type level: str
        :return: Accuracy parameter for ``--acc``.
        :rtype: float
        :raises ValueError: If the level name is unknown.
        """
        spec = cls.LEVEL_SPECS.get(level.lower())
        if not spec:
            raise ValueError(f"Unknown level '{level}'. Valid: {list(cls.LEVEL_SPECS)}")
        return spec["acc"]

    @staticmethod
    def _parse_energy(stdout: str, stderr: str) -> Optional[float]:
        """
        Parse the final total energy (Hartree) from xTB stdout/stderr.

        :param stdout: Captured xTB stdout text.
        :type stdout: str
        :param stderr: Captured xTB stderr text.
        :type stderr: str
        :return: Energy in Hartree if found, else ``None``.
        :rtype: Optional[float]
        """
        text = (stdout or "") + "\n" + (stderr or "")
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
        # fallback: last float followed by Eh
        matches = re.findall(r"(-?\d+\.\d+(?:[Ee][+-]?\d+)?)\s*E?h\b", text)
        if matches:
            try:
                return float(matches[-1])
            except Exception:
                return None
        return None

    def _build_xtb_cmd(
        self, xyz_path: str, level: str, *, xtb_omp_threads: Optional[int]
    ) -> Dict[str, Any]:
        """
        Construct the xTB command and environment for optimization.

        :param xyz_path: Path to the input XYZ file.
        :type xyz_path: str
        :param level: xTB optimization level token.
        :type level: str
        :param xtb_omp_threads: Optional OMP thread cap for the xTB subprocess.
        :type xtb_omp_threads: Optional[int]
        :return: Mapping with ``cmd`` and ``env``.
        :rtype: Dict[str, Any]
        """
        xtb_bin = self._which_xtb()
        acc = self._acc_for_level(level)

        cmd: List[str] = [xtb_bin, xyz_path, "--opt", level]
        if self._cfg.charge:
            cmd += ["--chrg", str(self._cfg.charge)]
        uhf = max(0, self._cfg.multiplicity - 1)
        if uhf:
            cmd += ["--uhf", str(uhf)]
        if self._cfg.gfn is not None:
            cmd += ["--gfn", str(int(self._cfg.gfn))]
        if self._cfg.alpb:
            cmd += ["--alpb", str(self._cfg.alpb)]
        cmd += ["--acc", f"{acc:g}"]

        env = os.environ.copy()
        if xtb_omp_threads is not None:
            env["OMP_NUM_THREADS"] = str(int(xtb_omp_threads))
        return {"cmd": cmd, "env": env}

    def _prepare_temp_xyz(self, tmp_dir: str, base: str, mol: Chem.Mol) -> str:
        """Write the input XYZ into a temporary directory and return its path."""
        inp_xyz = os.path.join(tmp_dir, base + ".xyz")
        self.save_mol_to_xyz(mol, inp_xyz)
        return inp_xyz

    def _run_xtb_process(
        self,
        inp_xyz: str,
        level: str,
        xtb_omp_threads: Optional[int],
        tmp_dir: str,
        timeout: Optional[int],
    ) -> subprocess.CompletedProcess:
        """
        Execute the xTB subprocess for optimization and return the completed proc.

        :raises subprocess.CalledProcessError: If xTB returns non-zero code.
        """
        spec = self._build_xtb_cmd(inp_xyz, level, xtb_omp_threads=xtb_omp_threads)
        logger.debug("Running xTB: %s", " ".join(spec["cmd"]))
        proc = subprocess.run(
            spec["cmd"],
            cwd=tmp_dir,
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
        return proc

    def _locate_optimized_file(self, tmp_dir: str) -> Optional[str]:
        """Locate the optimized XYZ in the temporary working directory."""
        for candidate in ("xtbopt.xyz", "geopt.xyz", "xtbopt.xyz"):
            p = os.path.join(tmp_dir, candidate)
            if os.path.exists(p):
                return p
        return None

    def _save_optimized_file(self, opt_path: str, save_dir: str, base: str) -> str:
        """Copy the optimized XYZ to the output directory and return the path."""
        out_name = base + "_xtb_optimized.xyz"
        out_path = os.path.join(save_dir, out_name)
        shutil.copy(opt_path, out_path)
        return os.path.abspath(out_path)

    def _persist_intermediates(
        self,
        inter_dir: str,
        inp_xyz: str,
        opt_path: Optional[str],
        proc: subprocess.CompletedProcess,
    ) -> None:
        """Persist input/output XYZ and stdout/stderr for debugging."""
        os.makedirs(inter_dir, exist_ok=True)
        try:
            shutil.copy(inp_xyz, os.path.join(inter_dir, os.path.basename(inp_xyz)))
            if opt_path:
                shutil.copy(
                    opt_path, os.path.join(inter_dir, os.path.basename(opt_path))
                )
            with open(os.path.join(inter_dir, "xtb_stdout.txt"), "w") as fh:
                fh.write(proc.stdout or "")
            with open(os.path.join(inter_dir, "xtb_stderr.txt"), "w") as fh:
                fh.write(proc.stderr or "")
            logger.debug("Saved intermediate files to %s", inter_dir)
        except Exception:
            logger.exception("Failed to persist intermediates")

    def _run_and_collect(
        self,
        tmp_dir: str,
        base: str,
        save_dir: str,
        mol: Chem.Mol,
        level: str,
        timeout: Optional[int],
        clean: bool,
        keep_intermediates: bool,
        xtb_omp_threads: Optional[int],
        inter_dir: str,
    ) -> Tuple[subprocess.CompletedProcess, Optional[str], Optional[str]]:
        """
        Prepare input, run xTB, collect outputs, and optionally persist intermediates.
        """
        inp_xyz = self._prepare_temp_xyz(tmp_dir, base, mol)
        proc = self._run_xtb_process(inp_xyz, level, xtb_omp_threads, tmp_dir, timeout)
        opt_path = self._locate_optimized_file(tmp_dir)
        saved_opt_path: Optional[str] = None
        if opt_path:
            saved_opt_path = self._save_optimized_file(opt_path, save_dir, base)
        if keep_intermediates or not clean:
            try:
                self._persist_intermediates(inter_dir, inp_xyz, opt_path, proc)
            except Exception:
                logger.exception("Failed during persistence of intermediate files.")
        return proc, opt_path, saved_opt_path

    # Geometry optimization
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
        Run xTB geometry optimization and collect results.

        :return: Result dict with keys: ``smiles, status, message, energy_Eh,
                  energy_kJmol, energy_kcalmol, energy_eV, optimized_file, stdout,
                  stderr``.
        """
        result: Dict[str, Any] = {
            "smiles": self._smiles,
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

        lvl = level.lower()
        if lvl not in self.LEVEL_SPECS:
            result["message"] = (
                f"Invalid level '{level}'. Valid: {list(self.LEVEL_SPECS)}"
            )
            return result

        os.makedirs(save_dir, exist_ok=True)

        # 1) RDKit embedding + quick pre-relax
        try:
            mol = self._validate_smiles()
            mol = self._embed_and_relax(mol)
        except Exception as e:
            result["message"] = f"Embedding failed: {e}"
            logger.exception(result["message"])
            return result

        # 2) Run xTB within a temporary directory and collect outputs
        stamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        base = f"mol_{stamp}"
        inter_dir = os.path.join(save_dir, f"intermediates_{base}")

        with tempfile.TemporaryDirectory() as tmp:
            try:
                proc, _opt_tmp, saved_opt = self._run_and_collect(
                    tmp_dir=tmp,
                    base=base,
                    save_dir=save_dir,
                    mol=mol,
                    level=lvl,
                    timeout=timeout,
                    clean=clean,
                    keep_intermediates=keep_intermediates,
                    xtb_omp_threads=xtb_omp_threads,
                    inter_dir=inter_dir,
                )
            except subprocess.CalledProcessError as e:
                result["message"] = f"xTB failed (rc={e.returncode})"
                result["stdout"] = getattr(e, "output", None) or getattr(
                    e, "stdout", None
                )
                result["stderr"] = getattr(e, "stderr", None)
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

        # 3) Post-process outputs (linear, no branching)
        energy_eh = self._parse_energy((proc.stdout or ""), (proc.stderr or ""))
        if energy_eh is not None:
            result["energy_Eh"] = energy_eh
            result["energy_kJmol"] = energy_eh * self.HARTREE_TO_KJMOL
            result["energy_kcalmol"] = energy_eh * self.HARTREE_TO_KCALMOL
            result["energy_eV"] = energy_eh * self.HARTREE_TO_EV

        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
        result["optimized_file"] = saved_opt
        result["status"] = "success"
        result["message"] = "Optimization completed."
        return result

    # Single-point energy
    def _build_sp_cmd(
        self,
        xyz: str,
        *,
        xtb_omp_threads: Optional[int],
        gfn: Optional[int],
        alpb: Optional[str],
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Build the xTB command & environment for a single-point (no --opt).
        """
        xtb_bin = self._which_xtb()
        cmd: List[str] = [xtb_bin, xyz]
        if self._cfg.charge:
            cmd += ["--chrg", str(self._cfg.charge)]
        uhf = max(0, self._cfg.multiplicity - 1)
        if uhf:
            cmd += ["--uhf", str(uhf)]
        if gfn is not None:
            cmd += ["--gfn", str(int(gfn))]
        elif self._cfg.gfn is not None:
            cmd += ["--gfn", str(int(self._cfg.gfn))]
        if alpb:
            cmd += ["--alpb", str(alpb)]
        elif self._cfg.alpb:
            cmd += ["--alpb", str(self._cfg.alpb)]
        env = os.environ.copy()
        if xtb_omp_threads is not None:
            env["OMP_NUM_THREADS"] = str(int(xtb_omp_threads))
        return cmd, env

    def _write_xyz_for_sp(self, mol: Chem.Mol, tmp_dir: str) -> str:
        """
        Write molecule to a temp XYZ and return its path.
        """
        xyz = os.path.join(tmp_dir, "sp_input.xyz")
        self.save_mol_to_xyz(mol, xyz)
        return xyz

    def _run_sp_process(
        self,
        cmd: List[str],
        env: Dict[str, str],
        timeout: Optional[int],
    ) -> subprocess.CompletedProcess:
        """
        Run the SP subprocess and return the CompletedProcess.
        """
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
        return proc

    def single_point(
        self,
        *,
        mol: Optional[Chem.Mol] = None,
        xyz_path: Optional[str] = None,
        timeout: Optional[int] = 300,
        xtb_omp_threads: Optional[int] = None,
        gfn: Optional[int] = None,
        alpb: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run an xTB single-point (SP) energy on a molecule or XYZ file.

        If neither ``mol`` nor ``xyz_path`` is provided the instance SMILES is
        embedded and pre-relaxed first.

        :return: Result dict with keys: ``status, message, energy_Eh, energy_kJmol,
                  energy_kcalmol, energy_eV, stdout, stderr, used_xyz``.
        """
        result: Dict[str, Any] = {
            "status": "error",
            "message": "",
            "energy_Eh": None,
            "energy_kJmol": None,
            "energy_kcalmol": None,
            "energy_eV": None,
            "stdout": None,
            "stderr": None,
            "used_xyz": None,
        }

        # Resolve xTB early
        try:
            _ = self._which_xtb()
        except FileNotFoundError as e:
            result["message"] = str(e)
            return result

        tmp_ctx = None
        try:
            if xyz_path:
                used_xyz = xyz_path
            else:
                if mol is None:
                    mol = self._validate_smiles()
                    mol = self._embed_and_relax(mol)
                tmp_ctx = tempfile.TemporaryDirectory()
                used_xyz = self._write_xyz_for_sp(mol, tmp_ctx.name)

            cmd, env = self._build_sp_cmd(
                used_xyz, xtb_omp_threads=xtb_omp_threads, gfn=gfn, alpb=alpb
            )
            proc = self._run_sp_process(cmd, env, timeout)
            result["stdout"] = proc.stdout
            result["stderr"] = proc.stderr
            result["used_xyz"] = used_xyz

            if proc.returncode != 0:
                result["message"] = f"xTB single-point failed (rc={proc.returncode})"
                return result

            energy_eh = self._parse_energy(proc.stdout or "", proc.stderr or "")
            if energy_eh is not None:
                result["energy_Eh"] = energy_eh
                result["energy_kJmol"] = energy_eh * self.HARTREE_TO_KJMOL
                result["energy_kcalmol"] = energy_eh * self.HARTREE_TO_KCALMOL
                result["energy_eV"] = energy_eh * self.HARTREE_TO_EV
                result["status"] = "success"
                result["message"] = "Single-point completed."
            else:
                result["message"] = "Single-point completed but energy not found."
                result["status"] = "success"
            return result

        except Exception as e:
            result["message"] = f"SP subprocess failed: {e}"
            return result

        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()

    # Backward-compatible wrapper returning only energy
    def fit(
        self, save_dir: str = "./", clean_xyz: bool = False, level: str = "loose"
    ) -> float:
        """
        Run optimization and return the final energy (legacy API).
        """
        try:
            res = self.optimize(save_dir=save_dir, level=level, clean=True)
            energy = res.get("energy_Eh")
            if clean_xyz and res.get("optimized_file"):
                try:
                    os.remove(res["optimized_file"])
                except Exception:
                    logger.debug("Could not remove optimized xyz during clean_xyz=True")
            return float(energy) if energy is not None else 0.0
        except Exception as e:
            logger.error("An error occurred during fit(): %s", e)
            return 0.0

    # Batch (parallel) API
    @classmethod
    def _worker(cls, s: str, save_dir: str, params: Dict[str, Any]) -> Dict[str, Any]:
        inst = cls(
            s,
            GeoOptConfig(
                xtb_executable=params.get("xtb_executable", "xtb"),
                embed_max_attempts=params.get("embed_max_attempts", 3),
                embed_seed=params.get("embed_seed", 42),
                charge=params.get("charge", 0),
                multiplicity=params.get("multiplicity", 1),
                gfn=params.get("gfn", 2),
                alpb=params.get("alpb"),
            ),
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
        Remove a directory safely (refuse dangerous targets; log errors).
        """
        try:
            if not path:
                logger.warning("Refusing to remove empty path")
                return
            abs_path = os.path.abspath(path)
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
        """
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
                    logger.exception("Failed to remove %s during cleanup", p)

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

        if remove_save_dir_after:
            logger.info(
                "remove_save_dir_after=True: attempting to remove save_dir %s", save_dir
            )
            cls._safe_remove_dir(save_dir)

        return results
