from __future__ import annotations

from typing import Dict, Any, Optional, List
import warnings
import networkx as nx

from .params import SynParams
from .huckel2d import Huckel2D
from .syn_interact2d import SynInteract2D
from .rdkit_graph_builder import RDKitGraphBuilder


class SynEnergy:
    r"""
    Facade for Hückel-style π energies and cross-component interaction ranking.

    This class provides two families of helpers:

    1) **Whole-molecule π energy** in *β units* (toy PMO/Hückel). If you supply a
       mapping of β to Hartree (``beta_to_hartree``), energies are also returned
       in Hartree, kJ/mol, kcal/mol and eV.

    2) **Inter-component pair ranking** via :class:`SynInteract2D`.

    Parameters
    ----------
    params : Optional[SynParams]
        Hyperparameters controlling Hückel assembly and substituent effects.
        Defaults to ``SynParams()``.

    beta_to_hartree : Optional[float]
        Optional mapping from the Hückel resonance integral β (model units,
        typically -1 by default) to an actual energy in Hartree (Eh). If
        provided, the toy π energy in β units (``energy_beta``) is scaled by

            energy_Eh = energy_beta * (beta_to_hartree / params.beta)

        and subsequently converted to kJ/mol, kcal/mol and eV via the class
        constants below.

    Unit conversions
    ----------------
    - ``HARTREE_TO_KJMOL = 2625.499638``
    - ``HARTREE_TO_KCALMOL = 627.509474``
    - ``HARTREE_TO_EV = 27.211386245988``
    """

    # Unit conversions
    HARTREE_TO_KJMOL: float = 2625.499638
    HARTREE_TO_KCALMOL: float = 627.509474
    HARTREE_TO_EV: float = 27.211386245988

    def __init__(
        self,
        params: Optional[SynParams] = None,
        *,
        beta_to_hartree: Optional[float] = None,
    ) -> None:
        self._p = params or SynParams()
        self._h = Huckel2D(self._p)
        self._si = SynInteract2D(self._p)
        # If provided, interpret 1 model-β == beta_to_hartree [Eh]
        self._beta_to_eh: Optional[float] = beta_to_hartree
        if beta_to_hartree is not None:
            warnings.warn(
                "Legacy beta_to_hartree conversion is deprecated: energy_beta is "
                "a toy Hückel quantity, not a validated physical molecular energy. "
                "Use GraphEnergy score units or a named calibrated/reference method.",
                DeprecationWarning,
                stacklevel=2,
            )

    def __repr__(self) -> str:
        bt = self._beta_to_eh
        bt_str = "None" if bt is None else f"{bt:g} Eh"
        return (
            f"<SynEnergy beta={self._p.beta} use_extended={self._p.use_extended} "
            f"beta_to_hartree={bt_str}>"
        )

    # ---------------------------------------------------------------------
    # Energy helpers
    # ---------------------------------------------------------------------
    def _with_conversions(self, energy_beta: float) -> Dict[str, Optional[float]]:
        """
        Convert β-units energy to physical units if ``beta_to_hartree`` is set.

        Conversion uses scale = beta_to_hartree / params.beta, so that the
        model β is replaced by the physical β consistently.

        Returns a dict with keys: energy_Eh, energy_kJmol, energy_kcalmol,
        energy_eV. Values are floats if conversion is possible; otherwise None.
        """
        if self._beta_to_eh is None:
            return {
                "energy_Eh": None,
                "energy_kJmol": None,
                "energy_kcalmol": None,
                "energy_eV": None,
            }
        # guard against model beta == 0 (can't scale)
        if self._p.beta == 0:
            return {
                "energy_Eh": None,
                "energy_kJmol": None,
                "energy_kcalmol": None,
                "energy_eV": None,
            }
        scale = float(self._beta_to_eh) / float(self._p.beta)
        energy_eh = energy_beta * scale
        return {
            "energy_Eh": float(energy_eh),
            "energy_kJmol": float(energy_eh * self.HARTREE_TO_KJMOL),
            "energy_kcalmol": float(energy_eh * self.HARTREE_TO_KCALMOL),
            "energy_eV": float(energy_eh * self.HARTREE_TO_EV),
        }

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def energy_from_graph(self, G: nx.Graph) -> Dict[str, Any]:
        """
        Compute whole-molecule π energy via Hückel from a NetworkX graph.

        Returns a dictionary always containing ``energy_beta`` (toy units).
        If ``beta_to_hartree`` was provided at construction, the dict also
        contains converted physical energies (``energy_Eh``, ``energy_kJmol``,
        ``energy_kcalmol``, ``energy_eV``).
        """
        res: Dict[str, Any] = {
            "status": "error",
            "message": "",
            "n_pi": 0,
            "energy_beta": None,
            "energy_Eh": None,
            "energy_kJmol": None,
            "energy_kcalmol": None,
            "energy_eV": None,
            "E": None,
            "nodes": None,
            "alpha": None,
        }
        PG = self._h.pi_layer(G)
        if PG.number_of_nodes() == 0:
            res["message"] = "No π layer detected."
            return res

        initials = self._h.initial_effects(G, PG)
        alpha = self._h.propagate(PG, initials)
        H, nodes = self._h.build_huckel(PG, alpha)
        E, _ = self._h.solve(H)
        energy_beta = float(self._h.total_pi_energy(E))

        # add conversions if a β → Eh mapping is provided
        conv = self._with_conversions(energy_beta)

        res.update(
            {
                "status": "success",
                "message": "OK",
                "n_pi": PG.number_of_nodes(),
                "energy_beta": energy_beta,
                "E": E.tolist(),
                "nodes": nodes,
                "alpha": alpha,
                **conv,
            }
        )
        return res

    def energy_from_smiles(
        self, smiles: str, compute_gasteiger: bool = True
    ) -> Dict[str, Any]:
        """
        Compute π energy starting from a SMILES string.
        """
        G = RDKitGraphBuilder.from_smiles(smiles, compute_gasteiger=compute_gasteiger)
        return self.energy_from_graph(G)

    def rank_pairs_from_graph(
        self, G: nx.Graph, top_k: int = None, export: str = "basic"
    ) -> List[Dict[str, Any]]:
        """
        Rank cross-component atom pairs via :class:`SynInteract2D`.
        """
        return self._si.rank_pairs_from_graph(G, top_k=top_k, export=export)

    def rank_pairs_from_smiles(
        self, smiles: str, top_k: int = None, export: str = "basic"
    ) -> List[Dict[str, Any]]:
        """
        Rank pairs from a SMILES string (builds a graph internally).
        """
        G = RDKitGraphBuilder.from_smiles(smiles, compute_gasteiger=True)
        return self._si.rank_pairs_from_graph(G, top_k=top_k, export=export)
