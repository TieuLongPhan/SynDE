from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

from rdkit import Chem

from ._embeddings import Embeddings, EmbeddingConfig
from ._force_field import ForceField, ForceFieldConfig
from synkit.IO import setup_logging

logger = setup_logging()


@dataclass
class ConformerConfig:
    """
    High-level configuration for conformer generation & minimization.

    :param embedding: Optional EmbeddingConfig; if None, sensible defaults are used.
    :param forcefield: Optional ForceFieldConfig; if None, sensible defaults are used.
    """

    embedding: Optional[EmbeddingConfig] = None
    forcefield: Optional[ForceFieldConfig] = None


class ConformerGenerator:
    """
    High-level orchestrator for generating and optimizing conformers.

    This class composes :class:`Embeddings` (for 3D embedding) and
    :class:`ForceField` (for MMFF/UFF minimization + energy scoring) to provide:

      * Conformer generation for a molecule
      * Energy minimization
      * Lowest-energy conformer extraction
      * Reaction-SMILES (RSMI) product–reactant energy difference (ΔE)

    Example
    -------
    >>> cg = ConformerGenerator()
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> mol_min, e_min = cg.process_mol(mol, num_conformers=5,
    force_field_method="MMFF94")
    >>> isinstance(mol_min, Chem.Mol), isinstance(e_min, float)
    (True, True)
    """

    def __init__(self, config: Optional[ConformerConfig] = None) -> None:
        """
        Initialize the generator with optional configs.

        :param config: ConformerConfig with sub-configs for embedding and force-fields.
        """
        self._config = config or ConformerConfig()
        self._emb = Embeddings(self._config.embedding or EmbeddingConfig())
        self._ff = ForceField(self._config.forcefield or ForceFieldConfig())

    # --------- dunder & properties ---------
    def __repr__(self) -> str:
        return (
            f"<ConformerGenerator emb={self._emb.default_method!r} "
            f"ff={self._ff.default_method!r} threads={self._emb.num_threads}>"
        )

    @property
    def config(self) -> ConformerConfig:
        """Return the configuration dataclass."""
        return self._config

    @property
    def embeddings(self) -> Embeddings:
        """Access the embedded :class:`Embeddings` helper."""
        return self._emb

    @property
    def forcefield(self) -> ForceField:
        """Access the embedded :class:`ForceField` helper."""
        return self._ff

    @classmethod
    def show_help(cls) -> str:
        """Short usage summary."""
        return cls.__doc__ or "Conformer generator (embedding + minimization)."

    # --------- public API (instance) ---------
    def process_mol(
        self,
        molecule: Chem.Mol,
        num_conformers: Optional[Union[int, str]] = "auto",
        embedding_method: Optional[str] = None,
        num_threads: Optional[int] = None,
        random_coords_threshold: Optional[int] = None,
        random_seed: Optional[int] = None,
        force_field_method: Optional[str] = None,
        max_iter: Optional[Union[int, str]] = "auto",
        return_energies: bool = False,
        **kwargs: Any,
    ) -> Tuple[Optional[Chem.Mol], float]:
        """
        Generate and minimize conformers for ``molecule`` and return the
        lowest-energy conformer and its energy.

        :param molecule: RDKit molecule.
        :param num_conformers: Integer or "auto".
        :param embedding_method: Override embedding method (e.g., "ETKDGv3").
        :param num_threads: Override thread count for embedding/minimization.
        :param random_coords_threshold: Override random-init threshold.
        :param random_seed: RNG seed override for embedding.
        :param force_field_method: Override FF method ("MMFF94", "MMFF94s", "UFF").
        :param max_iter: Integer or "auto" for minimization iterations.
        :param return_energies: Ignored here (kept for signature parity).
        :param kwargs: Forwarded to RDKit minimizers if supported.
        :returns: (minimized_molecule, energy_kcal_per_mol); (None, 0.0) on error.
        """
        try:
            # 1) Embed conformers (adds Hs internally)
            mol3d = self._emb.embed(
                molecule,
                num_conformers=num_conformers,
                embedding_method=embedding_method,
                num_threads=num_threads,
                random_coords_threshold=random_coords_threshold,
                random_seed=random_seed,
            )

            # 2) Minimize all conformers
            ff_method = force_field_method or self._ff.default_method
            mol_min = self._ff.minimize(
                mol3d,
                force_field_method=ff_method,
                max_iter=max_iter,
                return_energies=False,
                num_threads=num_threads or self._ff.num_threads,
                **kwargs,
            )

            # 3) Extract lowest-energy conformer and compute its energy
            mol_low = self._ff.get_lowest_energy_conformer(mol_min, ff_method)
            e_low = self._ff.compute_energy(mol_low, 0, ff_method)
            return mol_low, float(e_low)
        except Exception as exc:
            logger.error("Error in process_mol: %s", exc)
            return None, 0.0

    def process_smiles(
        self,
        smiles: str,
        **kwargs: Any,
    ) -> Tuple[Optional[Chem.Mol], float]:
        """
        SMILES convenience wrapper around :meth:`process_mol`.

        :param smiles: SMILES string.
        :param kwargs: Forwarded to :meth:`process_mol`.
        :returns: (minimized_molecule, energy_kcal_per_mol);
        (None, 0.0) on error/invalid SMILES.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.error("Invalid SMILES string: %s", smiles)
                return None, 0.0
            return self.process_mol(mol, **kwargs)
        except Exception as exc:
            logger.error("Error in process_smiles %s: %s", smiles, exc)
            return None, 0.0

    def rsmi_delta_e(
        self,
        rsmi: str,
        symbol: str = ">>",
        **kwargs: Any,
    ) -> float:
        """
        Compute ΔE = sum(E_products) − sum(E_reactants) for a reaction SMILES.

        :param rsmi: Reaction SMILES (RSMI), e.g. "CCO.O>>C=C".
        :param symbol: separator between reactants and products (default '>>').
        :param kwargs: Forwarded to :meth:`process_smiles` for each component.
        :returns: ΔE (float). Returns NaN on parsing errors.
        """
        try:
            parts = rsmi.split(symbol)
            if len(parts) != 2:
                logger.error(
                    "Invalid RSMI string (missing or multiple separators): %s", rsmi
                )
                return float("nan")
            educt_block, product_block = parts
            if not educt_block or not product_block:
                logger.error("Invalid RSMI string: missing reactants or products.")
                return float("nan")

            educt_smiles = [s for s in educt_block.split(".") if s]
            product_smiles = [s for s in product_block.split(".") if s]

            e_educts = []
            for smi in educt_smiles:
                _, e = self.process_smiles(smi, **kwargs)
                e_educts.append(e)

            e_products = []
            for smi in product_smiles:
                _, e = self.process_smiles(smi, **kwargs)
                e_products.append(e)

            # sum of floats (zeros for failures already baked in)
            return float(sum(e_products) - sum(e_educts))
        except Exception as exc:
            logger.error("Error in rsmi_delta_e: %s", exc)
            return float("nan")

    # --------- legacy static shims for backward compatibility ---------
    @staticmethod
    def _mol_process(
        molecule: Chem.Mol,
        num_conformers: Optional[Union[int, str]] = "auto",
        embedding_method: str = "ETKDGv3",
        num_threads: int = 1,
        random_coords_threshold: int = 100,
        random_seed: int = 42,
        force_field_method: Optional[str] = "MMFF94",
        max_iter: Optional[Union[int, str]] = "auto",
        return_energies: bool = False,
        **kwargs: Any,
    ) -> Tuple[Optional[Chem.Mol], float]:
        """
        Legacy static API forwarding to :meth:`process_mol`.
        """
        cg = ConformerGenerator(
            ConformerConfig(
                embedding=EmbeddingConfig(
                    default_method=embedding_method,
                    num_threads=num_threads,
                    random_coords_threshold=random_coords_threshold,
                    random_seed=random_seed,
                ),
                forcefield=ForceFieldConfig(default_method=force_field_method),
            )
        )
        return cg.process_mol(
            molecule,
            num_conformers=num_conformers,
            embedding_method=embedding_method,
            num_threads=num_threads,
            random_coords_threshold=random_coords_threshold,
            random_seed=random_seed,
            force_field_method=force_field_method,
            max_iter=max_iter,
            return_energies=return_energies,
            **kwargs,
        )

    @staticmethod
    def _smiles_process(
        smiles: str,
        num_conformers: Optional[Union[int, str]] = "auto",
        embedding_method: str = "ETKDGv3",
        num_threads: int = 1,
        random_coords_threshold: int = 100,
        random_seed: int = 42,
        force_field_method: Optional[str] = "MMFF94",
        max_iter: Optional[Union[int, str]] = "auto",
        return_energies: bool = False,
        **kwargs: Any,
    ) -> Tuple[Optional[Chem.Mol], float]:
        """
        Legacy static API forwarding to :meth:`process_smiles`.
        """
        cg = ConformerGenerator(
            ConformerConfig(
                embedding=EmbeddingConfig(
                    default_method=embedding_method,
                    num_threads=num_threads,
                    random_coords_threshold=random_coords_threshold,
                    random_seed=random_seed,
                ),
                forcefield=ForceFieldConfig(default_method=force_field_method),
            )
        )
        return cg.process_smiles(
            smiles,
            num_conformers=num_conformers,
            embedding_method=embedding_method,
            num_threads=num_threads,
            random_coords_threshold=random_coords_threshold,
            random_seed=random_seed,
            force_field_method=force_field_method,
            max_iter=max_iter,
            return_energies=return_energies,
            **kwargs,
        )

    @staticmethod
    def _rsmi_process(
        rsmi: str,
        symbol: str = ">>",
        num_conformers: Optional[Union[int, str]] = "auto",
        embedding_method: str = "ETKDGv3",
        num_threads: int = 1,
        random_coords_threshold: int = 100,
        random_seed: int = 42,
        force_field_method: Optional[str] = "MMFF94",
        max_iter: Optional[Union[int, str]] = "auto",
        return_energies: bool = False,
        **kwargs: Any,
    ) -> float:
        """
        Legacy static API forwarding to :meth:`rsmi_delta_e`.
        """
        cg = ConformerGenerator(
            ConformerConfig(
                embedding=EmbeddingConfig(
                    default_method=embedding_method,
                    num_threads=num_threads,
                    random_coords_threshold=random_coords_threshold,
                    random_seed=random_seed,
                ),
                forcefield=ForceFieldConfig(default_method=force_field_method),
            )
        )
        return cg.rsmi_delta_e(
            rsmi,
            symbol=symbol,
            num_conformers=num_conformers,
            embedding_method=embedding_method,
            num_threads=num_threads,
            random_coords_threshold=random_coords_threshold,
            random_seed=random_seed,
            force_field_method=force_field_method,
            max_iter=max_iter,
            return_energies=return_energies,
            **kwargs,
        )
