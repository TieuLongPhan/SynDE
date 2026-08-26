from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from rdkit import Chem
from rdkit.Chem import rdDistGeom
from rdkit.Chem.rdDistGeom import ETDG, ETKDG, ETKDGv2, ETKDGv3, srETKDGv3, KDG

from synkit.IO import setup_logging

logger = setup_logging()


@dataclass
class EmbeddingConfig:
    """
    Configure RDKit conformer embedding.

    :param default_method: Default embedding method.
    :param max_num_conformers: Upper bound when num_conformers="auto".
    :param min_num_conformers: Lower bound when num_conformers="auto".
    :param decr_num_conformers: Per-atom decrement factor used in "auto" mode.
    :param num_threads: Default thread count passed to RDKit embedding.
    :param random_coords_threshold: Use random coordinate init if atom count > threshold.
    :param random_seed: RNG seed for reproducible embeddings.
    """

    default_method: str = "ETKDGv3"
    max_num_conformers: int = 10
    min_num_conformers: int = 2
    decr_num_conformers: float = 0.04
    num_threads: int = 1
    random_coords_threshold: int = 100
    random_seed: int = 42
    _available_methods: Dict[str, object] = field(
        default_factory=lambda: {
            "ETDG": ETDG(),
            "ETKDG": ETKDG(),
            "ETKDGv2": ETKDGv2(),
            "ETKDGv3": ETKDGv3(),
            "srETKDGv3": srETKDGv3(),
            "KDG": KDG(),
        }
    )


class Embeddings:
    """Generate RDKit conformers with a selected distance-geometry method."""

    def __init__(self, config: Optional[EmbeddingConfig] = None) -> None:
        """
        Create a conformer embedder.

        :param config: Embedding settings, or ``None`` for defaults.
        """
        self._config = config or EmbeddingConfig()

    def __repr__(self) -> str:
        return f"<Embeddings method={self.default_method!r} threads={self.num_threads}>"

    @property
    def config(self) -> EmbeddingConfig:
        """Return the configuration dataclass."""
        return self._config

    @property
    def default_method(self) -> str:
        """Default embedding method name (e.g., 'ETKDGv3')."""
        return self._config.default_method

    @property
    def num_threads(self) -> int:
        """Default number of threads used for embedding."""
        return self._config.num_threads

    @property
    def available_methods(self) -> List[str]:
        """List of supported embedding method names."""
        return list(self._config._available_methods.keys())

    @classmethod
    def show_help(cls) -> str:
        """Return a brief description of the class usage."""
        return cls.__doc__ or "Embeddings helper for RDKit conformer generation."

    def embed(
        self,
        molecule: Chem.Mol,
        num_conformers: Optional[Union[int, str]] = "auto",
        embedding_method: Optional[str] = None,
        num_threads: Optional[int] = None,
        random_coords_threshold: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> Chem.Mol:
        """
        Embed multiple conformers for ``molecule``.

        :param molecule: RDKit molecule (3D conformers will be added).
                         Hydrogen atoms are ensured (added if missing).
        :param num_conformers: Integer number of conformers or "auto".
        :param embedding_method: One of available_methods; if None, uses default.
        :param num_threads: Override default thread count for this call.
        :param random_coords_threshold: Override threshold for random coordinate init.
        :param random_seed: Override RNG seed.
        :return: New molecule with embedded conformers (same atoms, with Hs).
        :raises ValueError: If parameters are invalid or the method name is unsupported.
        """
        if isinstance(num_conformers, str) and num_conformers != "auto":
            raise ValueError("`num_conformers` must be an integer or 'auto'.")

        method = embedding_method or self.default_method
        params = self._get_embedding_params(method)  # raises ValueError if invalid

        nt = int(num_threads or self.num_threads)
        seed = int(random_seed or self._config.random_seed)
        rct = int(random_coords_threshold or self._config.random_coords_threshold)

        mol = Chem.Mol(molecule)
        mol = Chem.AddHs(mol)

        if num_conformers == "auto" or num_conformers is None:
            num_confs = self._get_num_conformers_from_molecule_size(
                mol,
                max_num=self._config.max_num_conformers,
                min_num=self._config.min_num_conformers,
                decr=self._config.decr_num_conformers,
            )
        else:
            num_confs = int(num_conformers)
            if num_confs <= 0:
                raise ValueError("`num_conformers` must be a positive integer.")

        params.numThreads = nt
        params.randomSeed = seed
        params.useRandomCoords = mol.GetNumAtoms() > rct

        try:
            conf_ids = rdDistGeom.EmbedMultipleConfs(
                mol, numConfs=num_confs, params=params
            )
        except Exception as exc:  # pragma: no cover (depends on RDKit backend issues)
            logger.warning(
                "Embedding raised an exception (%s). Falling back to 2D.", exc
            )
            Chem.rdDepictor.Compute2DCoords(mol)
            return mol

        if not conf_ids or len(conf_ids) == 0:
            logger.warning(
                "No conformers embedded; computing 2D coordinates as fallback."
            )
            Chem.rdDepictor.Compute2DCoords(mol)

        return mol

    @staticmethod
    def mol_embed(
        molecule: Chem.Mol,
        num_conformers: Optional[Union[int, str]] = "auto",
        embedding_method: str = "ETKDGv3",
        num_threads: int = 1,
        random_coords_threshold: int = 100,
        random_seed: int = 42,
    ) -> Chem.Mol:
        """Retain the original static embedding interface."""
        emb = Embeddings(
            EmbeddingConfig(
                default_method=embedding_method,
                num_threads=num_threads,
                random_coords_threshold=random_coords_threshold,
                random_seed=random_seed,
            )
        )
        return emb.embed(molecule, num_conformers=num_conformers)

    def _get_embedding_params(self, method_name: str):
        """
        Return an RDKit parameter object for the given method name.

        :raises ValueError: if method_name is unsupported.
        """
        try:
            return self._config._available_methods[method_name]
        except KeyError:
            raise ValueError(
                f"Unsupported embedding method {method_name!r}. "
                f"Supported: {self.available_methods}"
            )

    @staticmethod
    def _get_num_conformers_from_molecule_size(
        molecule: Chem.Mol,
        max_num: int,
        min_num: int,
        decr: float,
    ) -> int:
        """
        Suggest a conformer count from molecule size.

        :param molecule: RDKit molecule.
        :param max_num: Maximum conformers.
        :param min_num: Minimum conformers.
        :param decr: Per-atom decrement factor.
        :return: Integer conformer count within [min_num, max_num].
        """
        n = molecule.GetNumAtoms()
        suggested = max_num - int(n * decr)
        return max(min_num, min(max_num, suggested))
