from typing import List, Tuple
from synde.geometry.rdkit._conformer import ConformerGenerator
from synde.geometry.xtb.xtb_reaction import XTBReaction
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
        Configure reaction-energy scoring.

        :param energy_type: Energy calculation method, such as ``XTB`` or
            ``GRAPH``.
        :type energy_type: str
        :param num_threads: Number of parallel calculation threads.
        :type num_threads: int
        :param verbose: Logging verbosity.
        :type verbose: int
        :param random_seed: Random-number seed.
        :type random_seed: int
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
        sort: bool = True,
    ) -> Tuple[List[str], List[float]]:
        """
        Score reactions and optionally sort them by ascending energy.

        :param reactions: Reaction SMILES strings.
        :type reactions: List[str]
        :param num_conformers: Conformer-count strategy.
        :type num_conformers: str
        :param embedding_method: RDKit embedding method.
        :type embedding_method: str
        :param random_coords_threshold: Atom-count threshold for random coordinates.
        :type random_coords_threshold: int
        :param force_field_method: Force-field energy method.
        :type force_field_method: str
        :param max_iter: Maximum-iteration strategy.
        :type max_iter: str
        :param sort: Whether to sort results by ascending energy.
        :type sort: bool
        :return: Reaction SMILES strings and corresponding energy values.
        :rtype: Tuple[List[str], List[float]]
        """
        if self.energy_type == "XTB":
            logger.info("Minimize energy using xTB")
            xtb = XTBReaction(n_jobs=self.num_threads, verbose=self.verbose)
            energies = xtb.delta_e_parallel(reactions, level="tight")
            results = list(zip(reactions, energies))
        elif self.energy_type in ("GRAPH", "SYN_V2", "SYN", "ITS", "SYN_ITS"):
            logger.info("Calculate reaction score using SynDE graph scoring")
            from synde.energy.graph_energy import GraphEnergy

            ge = GraphEnergy()
            results = []
            for rsmi in reactions:
                try:
                    if self.energy_type in ("ITS", "SYN_ITS"):
                        res = ge.score_its(rsmi)
                        score = res.its_score
                    else:
                        res = ge.score_reaction(rsmi)
                        score = res.reaction_delta_score
                    if score is None:
                        raise ValueError(
                            f"SynDE returned {res.status!r}: {res.warnings!r}"
                        )
                except Exception as error:
                    logger.error("SynDE scoring failed for %s: %s", rsmi, error)
                    score = float("inf")
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
