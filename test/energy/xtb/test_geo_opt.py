# tests/test_geo_opt_runtime_skip.py
import os
import shutil
import tempfile
import unittest
from shutil import which

from synde.energy.xtb.geo_opt import GeoOpt  # import directly as requested


class TestGeoOpt(unittest.TestCase):
    def setUp(self) -> None:
        # Runtime detection of xTB presence (decide skip at runtime)
        self.has_xtb = which("xtb") is not None

        self.smiles = "CCC"  # propane
        self.opt = GeoOpt(self.smiles)
        self.test_dir = tempfile.mkdtemp(prefix="xtb_min_test_")

    def tearDown(self) -> None:
        try:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir)
        except Exception:
            # best-effort cleanup
            pass

    # ---------- basic utilities (always run) ----------
    def test_show_help_returns_str(self) -> None:
        txt = GeoOpt.show_help()
        self.assertIsInstance(txt, str)
        self.assertTrue(len(txt) > 0)

    def test_smiles_to_3D(self) -> None:
        mol = self.opt.smiles_to_3D(self.smiles)
        self.assertIsNotNone(mol, "smiles_to_3D returned None for a valid SMILES")
        self.assertTrue(mol.GetNumConformers() > 0, "No conformers after embedding")

    def test_save_mol_to_xyz(self) -> None:
        mol = self.opt.smiles_to_3D(self.smiles)
        self.assertIsNotNone(mol)
        out_path = os.path.join(self.test_dir, "test_molecule.xyz")
        fname = self.opt.save_mol_to_xyz(mol, out_path)
        self.assertTrue(os.path.exists(fname), "XYZ file was not created")
        self.assertGreater(os.stat(fname).st_size, 0, "XYZ file is empty")

    # ---------- optimization (runtime-skip if xtb missing) ----------
    def test_optimize_method(self) -> None:
        if not self.has_xtb:
            self.skipTest("Skipping xTB integration test (xtb not found on PATH)")

        result = self.opt.optimize(
            save_dir=self.test_dir,
            level="crude",
            timeout=300,
            clean=True,
            keep_intermediates=False,
            xtb_omp_threads=1,
        )
        self.assertIsInstance(result, dict)
        self.assertIn("status", result)
        self.assertEqual(
            result["status"], "success", f"xTB optimize failed: {result.get('message')}"
        )
        energy = result.get("energy_Eh")
        if energy is not None:
            self.assertIsInstance(energy, float)
        # optimized file optional, but if present it must exist
        if result.get("optimized_file"):
            self.assertTrue(os.path.exists(result["optimized_file"]))

    def test_optimize_invalid_level(self) -> None:
        # No xTB required here; we fail early on level validation.
        result = self.opt.optimize(save_dir=self.test_dir, level="invalid_level")
        self.assertEqual(result["status"], "error")
        self.assertIn("Invalid level", result["message"])

    def test_fit_wrapper(self) -> None:
        if not self.has_xtb:
            self.skipTest("Skipping xTB integration test (xtb not found on PATH)")

        energy = self.opt.fit(save_dir=self.test_dir, clean_xyz=True, level="crude")
        self.assertIsInstance(energy, float)
        self.assertGreaterEqual(energy, -1e6)

    # ---------- batch ----------
    def test_process_smiles_list_and_remove_dir(self) -> None:
        if not self.has_xtb:
            self.skipTest("Skipping xTB integration test (xtb not found on PATH)")

        smiles = ["CCO", "c1ccccc1"]
        results = GeoOpt.process_smiles_list(
            smiles,
            save_dir=self.test_dir,
            level="crude",
            n_jobs=1,
            xtb_omp_threads=1,
            clean=True,
            cleanup_save_dir=False,
            remove_save_dir_after=True,
        )
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), len(smiles))
        for r in results:
            self.assertIsInstance(r, dict)
            self.assertIn("status", r)
            self.assertEqual(
                r["status"],
                "success",
                f"Batch optimize failed for {r.get('smiles')}: {r.get('message')}",
            )
        self.assertFalse(
            os.path.exists(self.test_dir), "save_dir was not removed after batch run"
        )

    # ---------- single-point (SP) ----------
    def test_single_point_from_mol(self) -> None:
        if not self.has_xtb:
            self.skipTest("Skipping xTB SP test (xtb not found on PATH)")

        # Build an RDKit 3D mol first
        mol = self.opt.smiles_to_3D(self.smiles)
        self.assertIsNotNone(mol)
        res = self.opt.single_point(mol=mol, timeout=180, xtb_omp_threads=1)
        self.assertIsInstance(res, dict)
        self.assertEqual(res["status"], "success", f"SP failed: {res.get('message')}")
        self.assertIsInstance(res.get("energy_Eh"), float)

    def test_single_point_from_xyz(self) -> None:
        if not self.has_xtb:
            self.skipTest("Skipping xTB SP test (xtb not found on PATH)")

        # Save an xyz, then SP directly on it
        mol = self.opt.smiles_to_3D(self.smiles)
        self.assertIsNotNone(mol)
        xyz = os.path.join(self.test_dir, "input.xyz")
        self.opt.save_mol_to_xyz(mol, xyz)
        res = self.opt.single_point(xyz_path=xyz, timeout=180, gfn=2, xtb_omp_threads=1)
        self.assertIsInstance(res, dict)
        self.assertEqual(
            res["status"], "success", f"SP on XYZ failed: {res.get('message')}"
        )
        self.assertIsInstance(res.get("energy_Eh"), float)
        self.assertEqual(res.get("used_xyz"), xyz)

    def test_single_point_failure_without_xtb(self) -> None:
        # If xtb is unavailable, calling SP should return an error status with a message.
        if self.has_xtb:
            self.skipTest("xTB present; this negative test is only for missing xtb.")
        res = self.opt.single_point()
        self.assertEqual(res["status"], "error")
        self.assertIn("xTB executable", res["message"])


if __name__ == "__main__":
    unittest.main()
