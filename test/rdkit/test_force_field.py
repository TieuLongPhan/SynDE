import unittest
from rdkit import Chem
from rdkit.Chem import AllChem
from synde.geometry.rdkit._force_field import ForceField


class TestForceField(unittest.TestCase):
    def setUp(self):
        # Create a simple molecule for testing.
        # IMPORTANT: add hydrogens BEFORE embedding so conformers include H atoms.
        mol = Chem.MolFromSmiles("CCO")
        self.molecule = Chem.AddHs(mol)
        # create multiple conformers so lowest-energy selection is meaningful
        AllChem.EmbedMultipleConfs(self.molecule, numConfs=3, randomSeed=42)

        # Instance of the OO ForceField
        self.ff = ForceField()

    def test_force_field_minimization_mmff94(self):
        result = self.ff.minimize(
            self.molecule, force_field_method="MMFF94", max_iter=100
        )
        self.assertIsInstance(result, Chem.Mol)
        self.assertGreater(result.GetNumConformers(), 0)

    def test_force_field_minimization_uff(self):
        result = self.ff.minimize(self.molecule, force_field_method="UFF", max_iter=100)
        self.assertIsInstance(result, Chem.Mol)
        self.assertGreater(result.GetNumConformers(), 0)

    def test_force_field_minimization_auto_iterations(self):
        result = self.ff.minimize(
            self.molecule, force_field_method="MMFF94", max_iter="auto"
        )
        self.assertIsInstance(result, Chem.Mol)
        self.assertGreater(result.GetNumConformers(), 0)

    def test_force_field_minimization_invalid_method(self):
        with self.assertRaises(ValueError):
            self.ff.minimize(self.molecule, force_field_method="INVALID_METHOD")

    def test_force_field_minimization_no_conformers(self):
        # Molecule without any conformers should raise ValueError
        mol_no_conf = Chem.AddHs(Chem.MolFromSmiles("C"))
        with self.assertRaises(ValueError):
            self.ff.minimize(mol_no_conf, force_field_method="MMFF94")

    def test_compute_force_field_energy_mmff94(self):
        # Ensure at least one conformer exists and is (optionally) optimized
        AllChem.MMFFOptimizeMoleculeConfs(self.molecule)
        energy = self.ff.compute_energy(self.molecule, 0, force_field_method="MMFF94")
        self.assertIsInstance(energy, float)

    def test_compute_force_field_energy_uff(self):
        AllChem.UFFOptimizeMoleculeConfs(self.molecule)
        energy = self.ff.compute_energy(self.molecule, 0, force_field_method="UFF")
        self.assertIsInstance(energy, float)

    def test_compute_force_field_energy_invalid_method(self):
        with self.assertRaises(ValueError):
            self.ff.compute_energy(
                self.molecule, 0, force_field_method="INVALID_METHOD"
            )

    def test_compute_force_field_energy_invalid_conformer_id(self):
        # Use an out-of-range conformer id
        with self.assertRaises(ValueError):
            self.ff.compute_energy(self.molecule, 999, force_field_method="MMFF94")

    def test_get_lowest_energy_conformer(self):
        # Make sure conformers exist and can be scored
        AllChem.MMFFOptimizeMoleculeConfs(self.molecule)
        lowest = self.ff.get_lowest_energy_conformer(
            self.molecule, force_field_method="MMFF94"
        )
        self.assertIsInstance(lowest, Chem.Mol)
        self.assertEqual(lowest.GetNumConformers(), 1)

    def test_get_lowest_energy_conformer_no_conformers(self):
        mol_no_conf = Chem.AddHs(Chem.MolFromSmiles("C"))
        with self.assertRaises(ValueError):
            self.ff.get_lowest_energy_conformer(
                mol_no_conf, force_field_method="MMFF94"
            )

    def test_get_lowest_energy_conformer_invalid_method(self):
        AllChem.MMFFOptimizeMoleculeConfs(self.molecule)
        with self.assertRaises(ValueError):
            self.ff.get_lowest_energy_conformer(
                self.molecule, force_field_method="INVALID_METHOD"
            )


if __name__ == "__main__":
    unittest.main()
