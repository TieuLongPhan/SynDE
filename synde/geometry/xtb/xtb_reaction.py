from typing import List
from joblib import Parallel, delayed
from .xtb_minimize import XTBMinimize
from synkit.IO import setup_logging

logger = setup_logging()


class XTBReaction:
    def __init__(self, n_jobs: int = 1, verbose: int = 0) -> None:
        """
        Initialize an XTBReaction object for calculating reaction energies.

        :param n_jobs: Number of jobs to run in parallel.
        :type n_jobs: int
        :param verbose: Verbosity level.
        :type verbose: int
        """
        self.n_jobs = n_jobs
        self.verbose = verbose

    @staticmethod
    def components_energy(smiles_list: List[str], level: str = "loose") -> float:
        """
        Calculate the total energy of a list of SMILES strings.

        :param smiles_list: List of SMILES strings.
        :type smiles_list: List[str]
        :param level: Optimization level.
        :type level: str
        :return: Total energy of the SMILES components.
        :rtype: float
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

        :param rsmi: Reaction SMILES with reactants and products separated by
            ``symbol``.
        :type rsmi: str
        :param symbol: Delimiter between reactants and products.
        :type symbol: str
        :param level: Optimization level.
        :type level: str
        :param pred_type: Direction of prediction (``fw`` or ``bw``).
        :type pred_type: str
        :param shared_energy: Pre-calculated energy for shared reactants or products.
        :type shared_energy: float | None
        :return: Calculated reaction energy difference.
        :rtype: float
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

        :param list_rsmi: Reaction SMILES strings.
        :type list_rsmi: List[str]
        :param symbol: Delimiter between reactants and products.
        :type symbol: str
        :param level: Optimization level.
        :type level: str
        :param pred_type: Use ``fw`` if reactions share reactants or ``bw`` if
            they share products.
        :type pred_type: str
        :return: Reaction energy differences.
        :rtype: List[float]
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
