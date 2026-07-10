from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.rdMolDescriptors import CalcMolFormula as rdCalcMolFormula


def calc_mol_formula(mol):
    """
    Calculate the molecular formula of an RDKit molecule.

    :param mol: The molecule, provided as an RDKit Mol object or a SMILES string.
    :type mol: rdkit.Chem.Mol or str
    :return: The molecular formula.
    :rtype: str
    :raises ValueError: If mol is None or the SMILES string is invalid.
    :raises TypeError: If mol is of an unsupported type.
    """
    if mol is None:
        raise ValueError("Input molecule is None")

    # Convert SMILES string to RDKit Mol
    if isinstance(mol, str):
        mol_obj = Chem.MolFromSmiles(mol)
        if mol_obj is None:
            raise ValueError(f"Invalid SMILES string: {mol}")
    elif isinstance(mol, Mol):
        mol_obj = mol
    else:
        raise TypeError(f"Unsupported type for mol: {type(mol)}")

    # Compute formula and return
    return rdCalcMolFormula(mol_obj)
