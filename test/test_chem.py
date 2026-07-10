import unittest
from rdkit import Chem
from synde.chem import calc_mol_formula


class TestCalcMolFormula(unittest.TestCase):
    def test_with_mol_object(self):
        # Water: H2O
        mol = Chem.MolFromSmiles("O")
        self.assertEqual(calc_mol_formula(mol), "H2O")

    def test_with_smiles(self):
        # Ethanol: C2H6O
        self.assertEqual(calc_mol_formula("CCO"), "C2H6O")

    def test_none_input(self):
        with self.assertRaises(ValueError):
            calc_mol_formula(None)

    def test_invalid_smiles(self):
        with self.assertRaises(ValueError):
            calc_mol_formula("not_a_smiles")

    def test_unsupported_type(self):
        with self.assertRaises(TypeError):
            calc_mol_formula(123)


if __name__ == "__main__":
    unittest.main()
