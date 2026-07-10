import unittest
from rdkit import Chem
from synde.energy.rdkit._conformer import ConformerGenerator


class TestConformerGenerator(unittest.TestCase):
    def setUp(self):
        self.molecule = Chem.MolFromSmiles("CCO")

    # ---- legacy static API (backward compatible) ----
    def test_mol_process_valid(self):
        minimized_mol, minimized_energy = ConformerGenerator._mol_process(
            self.molecule,
            num_conformers="auto",
            embedding_method="ETKDGv3",
            num_threads=1,
            random_coords_threshold=100,
            random_seed=42,
            force_field_method="MMFF94",
            max_iter="auto",
            return_energies=False,
        )
        self.assertIsInstance(minimized_mol, Chem.Mol)
        self.assertIsInstance(minimized_energy, float)
        self.assertGreater(minimized_energy, -float("inf"))

    def test_smiles_process_valid(self):
        smiles = "CCO"
        minimized_mol, minimized_energy = ConformerGenerator._smiles_process(
            smiles,
            num_conformers="auto",
            embedding_method="ETKDGv3",
            num_threads=1,
            random_coords_threshold=100,
            random_seed=42,
            force_field_method="MMFF94",
            max_iter="auto",
            return_energies=False,
        )
        self.assertIsInstance(minimized_mol, Chem.Mol)
        self.assertIsInstance(minimized_energy, float)
        self.assertGreater(minimized_energy, -float("inf"))

    def test_smiles_process_invalid(self):
        smiles = "InvalidSmilesString"
        minimized_mol, minimized_energy = ConformerGenerator._smiles_process(
            smiles,
            num_conformers="auto",
            embedding_method="ETKDGv3",
            num_threads=1,
            random_coords_threshold=100,
            random_seed=42,
            force_field_method="MMFF94",
            max_iter="auto",
            return_energies=False,
        )
        self.assertIsNone(minimized_mol)
        self.assertEqual(minimized_energy, 0.0)

    def test_rsmi_process_valid(self):
        rsmi = "CCO>>C=C.O"
        delta_e = ConformerGenerator._rsmi_process(
            rsmi,
            symbol=">>",
            num_conformers="auto",
            embedding_method="ETKDGv3",
            num_threads=1,
            random_coords_threshold=100,
            random_seed=42,
            force_field_method="MMFF94",
            max_iter="auto",
            return_energies=False,
        )
        self.assertIsInstance(delta_e, float)
        self.assertFalse(delta_e != delta_e)  # not NaN

    def test_rsmi_process_invalid(self):
        rsmi = "InvalidRSMIString"
        delta_e = ConformerGenerator._rsmi_process(
            rsmi,
            symbol=">>",
            num_conformers="auto",
            embedding_method="ETKDGv3",
            num_threads=1,
            random_coords_threshold=100,
            random_seed=42,
            force_field_method="MMFF94",
            max_iter="auto",
            return_energies=False,
        )
        self.assertTrue(delta_e != delta_e)  # NaN

    def test_rsmi_process_no_products_or_reactants(self):
        rsmi = "CCO>>"
        delta_e = ConformerGenerator._rsmi_process(
            rsmi,
            symbol=">>",
            num_conformers="auto",
            embedding_method="ETKDGv3",
            num_threads=1,
            random_coords_threshold=100,
            random_seed=42,
            force_field_method="MMFF94",
            max_iter="auto",
            return_energies=False,
        )
        self.assertTrue(delta_e != delta_e)  # NaN

    # ---- new instance API ----
    def test_instance_process_mol(self):
        cg = ConformerGenerator()
        mol_min, e_min = cg.process_mol(
            self.molecule, num_conformers=3, force_field_method="MMFF94"
        )
        self.assertIsInstance(mol_min, Chem.Mol)
        self.assertIsInstance(e_min, float)

    def test_instance_process_smiles(self):
        cg = ConformerGenerator()
        mol_min, e_min = cg.process_smiles(
            "CCO", num_conformers=2, force_field_method="MMFF94"
        )
        self.assertIsInstance(mol_min, Chem.Mol)
        self.assertIsInstance(e_min, float)

    def test_instance_rsmi_delta_e(self):
        cg = ConformerGenerator()
        de = cg.rsmi_delta_e(
            "CCO>>C=C.O", num_conformers=2, force_field_method="MMFF94"
        )
        self.assertIsInstance(de, float)
        self.assertFalse(de != de)  # not NaN


if __name__ == "__main__":
    unittest.main()
