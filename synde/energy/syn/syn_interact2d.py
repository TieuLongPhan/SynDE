from __future__ import annotations

from typing import Dict, List, Any, Tuple
import itertools
import numpy as np
import networkx as nx

from .params import SynParams
from .huckel2d import Huckel2D


class SynInteract2D:
    """
    (Formerly SynReact2D) — compute pairwise ΔE scores between π components.

    Usage:
        si = SynInteract2D(SynParams())
        rows = si.rank_pairs_from_graph(G)
    """

    def __init__(self, params: SynParams) -> None:
        self.p = params
        self.hm = Huckel2D(params)

    def _frontier(
        self,
        a: int,
        b: int,
        nodesA: List[int],
        nodesB: List[int],
        dA: Dict[str, Any],
        dB: Dict[str, Any],
    ) -> float:
        p = self.p
        IA = {nodesA[i]: i for i in range(len(nodesA))}
        IB = {nodesB[i]: i for i in range(len(nodesB))}
        ia, ib = IA[a], IB[b]

        gap_AB = max(p.min_gap, float(dB["E_L"] - dA["E_H"]))
        gap_BA = max(p.min_gap, float(dA["E_L"] - dB["E_H"]))

        S = (
            abs(dA["cH"][ia] * dB["cL"][ib]) / gap_AB
            + abs(dB["cH"][ib] * dA["cL"][ia]) / gap_BA
        )

        if p.use_extended:
            gap_h1 = max(p.min_gap, float(dB["E_L"] - dA.get("E_H1", dA["E_H"])))
            gap_l1 = max(p.min_gap, float(dA.get("E_L1", dA["E_L"]) - dB["E_H"]))
            S += p.w_front_ext * (
                abs(dA.get("cH1", dA["cH"])[ia] * dB["cL"][ib]) / gap_h1
                + abs(dB["cH"][ib] * dA.get("cL1", dA["cL"])[ia]) / gap_l1
            )
        return float(S)

    def _coul_pi(
        self,
        a: int,
        b: int,
        nodesA: List[int],
        nodesB: List[int],
        qA: np.ndarray,
        qB: np.ndarray,
    ) -> float:
        p = self.p
        IA = {nodesA[i]: i for i in range(len(nodesA))}
        IB = {nodesB[i]: i for i in range(len(nodesB))}
        return float(qA[IA[a]] * qB[IB[b]] / (p.eps * p.rC))

    def _coul_pc(self, a: int, b: int, GA: nx.Graph, GB: nx.Graph) -> float:
        p = self.p
        qA = float(GA.nodes[a].get("partial_charge", 0.0))
        qB = float(GB.nodes[b].get("partial_charge", 0.0))
        return float(qA * qB / (p.eps * p.rC))

    def _steric(
        self, a: int, b: int, GA: nx.Graph, GB: nx.Graph, PA: nx.Graph, PB: nx.Graph
    ) -> float:
        subsA = sum(1 for m in GA.neighbors(a) if m not in PA)
        subsB = sum(1 for m in GB.neighbors(b) if m not in PB)
        return float(self.p.steric_k * (subsA + subsB))

    def _deltaE(
        self,
        a: int,
        b: int,
        GA: nx.Graph,
        GB: nx.Graph,
        PA: nx.Graph,
        PB: nx.Graph,
        nodesA: List[int],
        nodesB: List[int],
        dA: Dict[str, Any],
        dB: Dict[str, Any],
        qA: np.ndarray,
        qB: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        S = self._frontier(a, b, nodesA, nodesB, dA, dB)
        C1 = self._coul_pi(a, b, nodesA, nodesB, qA, qB)
        C2 = self._coul_pc(a, b, GA, GB)
        St = self._steric(a, b, GA, GB, PA, PB)
        DE = (
            -self.p.w_front * S
            + self.p.w_coul_pi * C1
            + self.p.w_coul_pc * C2
            + self.p.w_steric * St
        )
        return float(DE), {
            "S_front": float(S),
            "Coul_pi": float(C1),
            "Coul_pc": float(C2),
            "Steric": float(St),
        }

    def rank_pairs_from_graph(
        self, G: nx.Graph, top_k: int = None, export: str = "basic"
    ) -> List[Dict[str, Any]]:
        """
        Rank cross-component atom pairs.

        :param G: NetworkX molecular graph.
        :param top_k: Keep only top_k (default params.top_pairs).
        :param export: 'basic' or 'all' for extended debug info.
        :returns: Sorted ascending list of rows (each row is dict).
        """
        comps = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        comps.sort(key=lambda g: -len(g.nodes()))
        results = []

        for i, j in itertools.combinations(range(len(comps)), 2):
            GA, GB = comps[i], comps[j]
            PA, PB = self.hm.pi_layer(GA), self.hm.pi_layer(GB)
            if PA.number_of_nodes() == 0 or PB.number_of_nodes() == 0:
                continue

            initialsA = self.hm.initial_effects(GA, PA)
            initialsB = self.hm.initial_effects(GB, PB)
            alphaA = self.hm.propagate(PA, initialsA)
            alphaB = self.hm.propagate(PB, initialsB)

            HA, nodesA = self.hm.build_huckel(PA, alphaA)
            EA, CA = self.hm.solve(HA)
            HB, nodesB = self.hm.build_huckel(PB, alphaB)
            EB, CB = self.hm.solve(HB)
            dA, dB = self.hm.descriptors(EA, CA), self.hm.descriptors(EB, CB)
            qA, qB = self.hm.pi_mulliken(EA, CA), self.hm.pi_mulliken(EB, CB)

            for a in nodesA:
                for b in nodesB:
                    DE, parts = self._deltaE(
                        a, b, GA, GB, PA, PB, nodesA, nodesB, dA, dB, qA, qB
                    )
                    row = {"compA": i, "compB": j, "a": a, "b": b, "DE": DE, **parts}
                    if export == "all":
                        row.update(
                            {
                                "alphaA": alphaA,
                                "alphaB": alphaB,
                                "initialsA": initialsA,
                                "initialsB": initialsB,
                                "pi_qA": dict(zip(nodesA, qA)),
                                "pi_qB": dict(zip(nodesB, qB)),
                                "E_A": EA.tolist(),
                                "E_B": EB.tolist(),
                                "descA": dA,
                                "descB": dB,
                            }
                        )
                    results.append(row)

        results.sort(key=lambda d: d["DE"])
        if top_k is None:
            top_k = self.p.top_pairs
        return results[:top_k]
