from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.rdMolDescriptors import CalcMolFormula as rdCalcMolFormula

from synde.errors import SynDEDomainError

SUPPORTED_ELEMENTS = (
    "H",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Si",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
)
SUPPORTED_ELEMENT_SET = frozenset(SUPPORTED_ELEMENTS)
ISOTOPE_EXCLUSION_REASON = "isotopically_labelled"


def has_isotopically_labelled_atom(mol: Mol) -> bool:
    """Return whether any atom has a non-default isotope number."""

    return any(atom.GetIsotope() != 0 for atom in mol.GetAtoms())


def normalize_ordinary_explicit_hydrogens(mol: Mol) -> Mol:
    """Return a copy with ordinary explicit protium represented implicitly.

    Isotopically labelled atoms are deliberately not normalized into the active
    domain.  Callers that audit eligibility should detect them first and record
    :data:`ISOTOPE_EXCLUSION_REASON`; inference callers reject them.
    """

    if has_isotopically_labelled_atom(mol):
        labels = sorted(
            {atom.GetIsotope() for atom in mol.GetAtoms() if atom.GetIsotope()}
        )
        raise SynDEDomainError(
            "Isotopically labelled molecules are outside the SynDE model "
            f"domain; found mass numbers {labels}.",
            subject=Chem.MolToSmiles(mol),
            hint=(
                "Remove the isotope labels. SynDE descriptors are "
                "mass-independent, so the unlabelled structure yields the same "
                "prediction."
            ),
            details={"isotopes": labels, "reason": ISOTOPE_EXCLUSION_REASON},
        )
    parameters = Chem.RemoveHsParameters()
    # Atom-mapped hydrogen vertices encode reaction correspondence rather than
    # an alternative molecular representation and must retain that mapping.
    parameters.removeMapped = False
    # A bracketed protium can carry the slash/backslash that defines adjacent
    # double-bond stereo.  SynDE uses achiral 2D connectivity, so remove that
    # hydrogen as well and normalize to the same graph as an implicit-H input.
    parameters.removeDefiningBondStereo = True
    normalized = Chem.RemoveHs(Chem.Mol(mol), parameters, sanitize=True)
    Chem.SanitizeMol(normalized)
    return normalized


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
