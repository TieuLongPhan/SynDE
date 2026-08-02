import unittest

from synde.geometry.xtb.xtb_minimize import XTBMinimize


class TestXTBMinimize(unittest.TestCase):
    def test_smiles_validation_does_not_require_xtb_binary(self) -> None:
        molecule = XTBMinimize("CCO")._validate_smiles()

        self.assertGreater(molecule.GetNumAtoms(), 3)
