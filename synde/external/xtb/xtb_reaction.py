from typing import List
from joblib import Parallel, delayed
from .xtb_minimize import XTBMinimize
from synkit.IO import setup_logging

logger = setup_logging()


class XTBReaction:
    def __init__(self, n_jobs: int = 1, verbose: int = 0) -> None:
        """
        Initialize an XTBReaction object for calculating reaction energies.

        Parameters:
        n_jobs (int): Number of jobs to run in parallel. Defaults to 1.
        verbose (int): Verbosity level. Defaults to 0.
        """
        self.n_jobs = n_jobs
        self.verbose = verbose

    @staticmethod
    def components_energy(smiles_list: List[str], level: str = "loose") -> float:
        """
        Calculate the total energy of a list of SMILES strings.

        Parameters:
        - smiles_list (List[str]): List of SMILES strings.
        - level (str): The optimization level; default is "loose".

        Returns:
        float: The total energy of the SMILES components.
        """
        return sum(
            XTBMinimize(rsmi).fit(clean_xyz=True, level=level) for rsmi in smiles_list
        )

    @staticmethod
    def delta_e_rsmi(
        rsmi: str,
        symbol: str = ">>",
        level: str = "loose",
        pred_type: str = "fw",
        shared_energy: float = None,
    ) -> float:
        """
        Calculate the energy difference (delta E) for a reaction given as SMILES.

        Parameters:
        - rsmi (str): Reaction SMILES string with reactants and products separated
          by a symbol.
        - symbol (str): Delimiter between reactants and products in the rSMI string.
        - level (str): The optimization level; default is "loose".
        - pred_type (str): Direction of prediction ('fw' for forward, 'bw' for backward).
        - shared_energy (float): Pre-calculated energy for shared reactants or products.

        Returns:
        float: The calculated delta energy of the reaction.
        """
        try:
            reactants, products = rsmi.split(symbol)
        except ValueError:
            logger.error("The rSMI string does not contain the specified delimiter.")
            return 0

        list_reactants = reactants.split(".")
        list_products = products.split(".")

        try:
            if shared_energy is not None:
                e_reactants = (
                    shared_energy
                    if pred_type == "fw"
                    else XTBReaction.components_energy(list_reactants, level)
                )
                e_products = (
                    XTBReaction.components_energy(list_products, level)
                    if pred_type == "fw"
                    else shared_energy
                )
            else:
                e_reactants = XTBReaction.components_energy(list_reactants, level)
                e_products = XTBReaction.components_energy(list_products, level)
        except Exception as e:
            logger.error(f"An error occurred during energy calculation: {e}")
            return 0

        return e_products - e_reactants

    def delta_e_parallel(
        self,
        list_rsmi: List[str],
        symbol: str = ">>",
        level: str = "loose",
        pred_type: str = "fw",
    ) -> List[float]:
        """
        Compute the delta E values for a list of reactions in parallel.

        Parameters:
        - list_rsmi (List[str]): A list of reaction SMILES strings.
        - symbol (str): Delimiter between reactants and products in the rSMI string.
        - level (str): The optimization level; default is "loose".
        - pred_type (str): Specify 'fw' if reactions share reactants,
        'bw' if reactions share products,  or None for standard processing.

        Returns:
        List[float]: A list of delta E values for the given reactions.
        """
        shared_energy = None
        if pred_type in ["fw", "bw"]:
            components = [
                rsmi.split(symbol)[0 if pred_type == "fw" else 1] for rsmi in list_rsmi
            ]
            components_flat = ".".join(components).split(".")
            shared_energy = self.components_energy(list(set(components_flat)), level)

        try:
            with Parallel(n_jobs=self.n_jobs, verbose=self.verbose) as parallel:
                delta_es = parallel(
                    delayed(XTBReaction.delta_e_rsmi)(
                        rsmi, symbol, level, pred_type, shared_energy
                    )
                    for rsmi in list_rsmi
                )
            return delta_es
        except Exception as e:
            logger.error(f"An error occurred during parallel computation: {e}")
            raise
