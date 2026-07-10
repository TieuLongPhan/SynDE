from typing import List, Tuple, Optional
from synde.geometry.rdkit._conformer import ConformerGenerator
from synde.external.xtb.xtb_reaction import XTBReaction
from synkit.IO import setup_logging

logger = setup_logging()


class SFEnergy:
    def __init__(
        self,
        energy_type: str = "XTB",
        num_threads: int = 4,
        verbose: int = 0,
        random_seed: int = 42,
    ) -> None:
        """
        Initializes an instance of the SFEnergy class for processing chemical reactions
        based on energy calculations to determine synthetic feasibility.

        Parameters:
        - energy_type (str): The type of energy calculation method ('XTB', 'GRAPH', or others).
        Default 'XTB'.
        - num_threads (int): Number of threads to use for parallel calculations.
        Default 4.
        - verbose (int): Verbosity level for logging output.
        Default 0.
        - random_seed (int): Seed for random number generation.
        Default 42.
        """
        self.energy_type = energy_type
        self.num_threads = num_threads
        self.verbose = verbose
        self.random_seed = random_seed

    def sort_reactions(
        self,
        reactions: List[str],
        num_conformers: str = "auto",
        embedding_method: str = "ETKDGv3",
        random_coords_threshold: int = 100,
        force_field_method: str = "MMFF94",
        max_iter: str = "auto",
        sort: Optional[bool] = True,
    ) -> Tuple[List[str], List[float]]:
        """
        Processes and sorts a list of reaction SMILES strings based on their calculated
        energy values.

        Parameters:
        - reactions (List[str]): List of reaction SMILES strings.
        - num_conformers (str): Strategy for determining the number of conformers.
        Default 'auto'.
        - embedding_method (str): Embedding method. Default 'ETKDGv3'.
        - random_coords_threshold (int): Threshold for using random coordinates.
        Default 100.
        - force_field_method (str): Force field method used for energy calculations.
        Default 'MMFF94'.
        - max_iter (str): Maximum number of iterations during calculation.
        Default 'auto'.
        - sort (bool, optional): Whether to sort the results by energy in
        ascending order. Default True.

        Returns:
        - Tuple[List[str], List[float]]: Sorted list of reaction SMILES and their
        corresponding energy values.
        """
        if self.energy_type == "XTB":
            logger.info("Minimize energy using xTB")
            xtb = XTBReaction(n_jobs=self.num_threads, verbose=self.verbose)
            energies = xtb.delta_e_parallel(reactions, level="tight")
            results = list(zip(reactions, energies))
        elif self.energy_type in ("GRAPH", "SYN_V2", "SYN"):
            logger.info("Calculate reaction score using GraphEnergy v2")
            from synde.energy.graph_energy import GraphEnergy

            ge = GraphEnergy()
            results = []
            for rsmi in reactions:
                try:
                    res = ge.score_reaction(rsmi)
                    score = res.reaction_delta_score
                except Exception as e:
                    logger.error(f"Error calculating graph energy for {rsmi}: {e}")
                    score = 0.0
                results.append((rsmi, score))
        else:
            logger.info(f"Minimize energy using {force_field_method}")
            results = []
            for rsmi in reactions:
                logger.info(f"{rsmi}")
                delta_e = ConformerGenerator._rsmi_process(
                    rsmi,
                    symbol=">>",
                    num_conformers=num_conformers,
                    embedding_method=embedding_method,
                    num_threads=self.num_threads,
                    random_coords_threshold=random_coords_threshold,
                    random_seed=self.random_seed,
                    force_field_method=force_field_method,
                    max_iter=max_iter,
                    return_energies=False,
                )
                results.append((rsmi, round(delta_e, 2)))

        if sort:
            results.sort(key=lambda x: x[1])
        rsmi_list, energies = zip(*results) if results else ([], [])
        return list(rsmi_list), list(energies)
