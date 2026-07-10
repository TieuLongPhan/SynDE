# import os
# import shutil
# import tempfile
# import unittest

# from synde.energy.xtb.xtb_minimize import XTBMinimize

# try:
#     from shutil import which
# except Exception:
#     which = lambda x: None  # type: ignore

# HAS_XTB: bool = which("xtb") is not None


# class TestXTBMinimize(unittest.TestCase):
#     def setUp(self) -> None:
#         # Simple, quick SMILES
#         self.smiles = "CCC"  # propane
#         self.xtb_minimizer = XTBMinimize(self.smiles)
#         self.test_dir = tempfile.mkdtemp(prefix="xtb_min_test_")

#     def tearDown(self) -> None:
#         # Remove test directory entirely (safe temporary dir)
#         try:
#             if os.path.exists(self.test_dir):
#                 shutil.rmtree(self.test_dir)
#         except Exception:
#             # Best-effort cleanup
#             pass

#     def test_smiles_to_3D(self) -> None:
#         """Test RDKit embedding via the compatibility helper smiles_to_3D."""
#         mol = self.xtb_minimizer.smiles_to_3D(self.smiles)
#         self.assertIsNotNone(mol, "smiles_to_3D returned None for a valid SMILES")
#         # Check that a conformer exists
#         self.assertTrue(
#             mol.GetNumConformers() > 0, "Molecule has no conformers after embedding"
#         )

#     def test_save_mol_to_xyz(self) -> None:
#         """Test saving a molecule to an XYZ file using save_mol_to_xyz."""
#         mol = self.xtb_minimizer.smiles_to_3D(self.smiles)
#         self.assertIsNotNone(mol, "smiles_to_3D failed; cannot test save_mol_to_xyz")
#         out_path = os.path.join(self.test_dir, "test_molecule.xyz")
#         fname = self.xtb_minimizer.save_mol_to_xyz(mol, out_path)
#         self.assertTrue(os.path.exists(fname), "XYZ file was not created")
#         self.assertGreater(os.stat(fname).st_size, 0, "XYZ file is empty")

#     @unittest.skipUnless(
#         HAS_XTB, "Skipping xTB integration test (xtb not found on PATH)"
#     )
#     def test_optimize_method(self) -> None:
#         """
#         Test the new optimize(...) API. Requires xTB on PATH.
#         We request a very light level ('crude') to keep runtime small.
#         """
#         result = self.xtb_minimizer.optimize(
#             save_dir=self.test_dir,
#             level="crude",
#             timeout=300,
#             clean=True,
#             keep_intermediates=False,
#             xtb_omp_threads=1,
#         )
#         # Basic checks
#         self.assertIsInstance(result, dict)
#         self.assertIn("status", result)
#         self.assertEqual(
#             result["status"],
#             "success",
#             f"xTB optimize did not succeed: {result.get('message')}",
#         )
#         # If energy parsed, it should be a float
#         energy = result.get("energy_Eh")
#         if energy is not None:
#             self.assertIsInstance(energy, float)

#     @unittest.skipUnless(
#         HAS_XTB, "Skipping xTB integration test (xtb not found on PATH)"
#     )
#     def test_fit_wrapper(self) -> None:
#         """
#         Test the legacy fit(...) wrapper which returns a float energy and optionally
#         removes optimized files when clean_xyz=True.
#         """
#         energy = self.xtb_minimizer.fit(
#             save_dir=self.test_dir, clean_xyz=True, level="crude"
#         )
#         # fit returns a float (0.0 on failure)
#         self.assertIsInstance(energy, float)
#         self.assertGreaterEqual(energy, -1e6)

#     @unittest.skipUnless(
#         HAS_XTB, "Skipping xTB integration test (xtb not found on PATH)"
#     )
#     def test_process_smiles_list_and_remove_dir(self) -> None:
#         """
#         Test the batch API and the remove_save_dir_after behavior:
#         produce outputs and then remove the save_dir after.
#         """
#         smiles = ["CCO", "c1ccccc1"]
#         results = XTBMinimize.process_smiles_list(
#             smiles,
#             save_dir=self.test_dir,
#             level="crude",
#             n_jobs=1,
#             xtb_omp_threads=1,
#             clean=True,
#             cleanup_save_dir=False,
#             remove_save_dir_after=True,  # request deletion after run
#         )
#         # results should be a list with one dict per smiles
#         self.assertIsInstance(results, list)
#         self.assertEqual(len(results), len(smiles))
#         for r in results:
#             self.assertIsInstance(r, dict)
#             self.assertIn("status", r)
#             self.assertEqual(
#                 r["status"],
#                 "success",
#                 f"Batch optimize did not succeed for {r.get('smiles')}:"
#                 + f" {r.get('message')}",
#             )
#         self.assertFalse(
#             os.path.exists(self.test_dir),
#             "save_dir was not removed after batch run (remove_save_dir_after=True)",
#         )


# if __name__ == "__main__":
#     unittest.main()
