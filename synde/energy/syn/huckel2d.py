from __future__ import annotations

from typing import Dict, List, Tuple, Iterable, Any
import numpy as np
import networkx as nx

from .params import SynParams


class Huckel2D:
    """
    2D Hückel π-system builder, substituent detection and propagation.

    See SynParams for hyperparameters.

    Expected node attributes in the input graph G:
      - element: str (e.g. 'C', 'N')
      - aromatic: bool (True/False)
      - optional hcount: int
      - optional partial_charge: float
      - optional effects: list of dicts with keys 'kind' and 'strength'

    Expected edge attributes:
      - order: float (1.0, 2.0, ...)

    Public methods:
      - pi_layer(G) -> nx.Graph: return π-subgraph
      - initial_effects(G, PG) -> list of (node, kind, strength)
      - propagate(PG, initials) -> dict node->alpha
      - build_huckel(PG, alpha) -> (H, nodes)
      - solve(H) -> (E, C)
      - descriptors(E,C) -> dict
      - pi_mulliken(E,C) -> np.ndarray
      - total_pi_energy(E) -> float
    """

    def __init__(self, params: SynParams) -> None:
        self.p = params

    def _is_pi_bond(self, G: nx.Graph, u: int, v: int) -> bool:
        order = float(G[u][v].get("order", 1.0))
        arom = bool(
            G.nodes[u].get("aromatic", False) and G.nodes[v].get("aromatic", False)
        )
        return order >= 2.0 or arom

    def pi_layer(self, G: nx.Graph) -> nx.Graph:
        PG = nx.Graph()
        pi_nodes = {u for u, v in G.edges() if self._is_pi_bond(G, u, v)} | {
            v for u, v in G.edges() if self._is_pi_bond(G, u, v)
        }
        for n in pi_nodes:
            PG.add_node(n, **G.nodes[n])
        for u, v in G.edges():
            if u in pi_nodes and v in pi_nodes:
                PG.add_edge(u, v, **G[u][v])
        return PG

    def _is_carbonyl_carbon(self, G: nx.Graph, c: int) -> bool:
        if G.nodes[c].get("element") != "C":
            return False
        for nbr in G.neighbors(c):
            if (
                G.nodes[nbr].get("element") == "O"
                and float(G[c][nbr].get("order", 1.0)) >= 2.0
            ):
                return True
        return False

    def initial_effects(
        self, G: nx.Graph, PG: nx.Graph
    ) -> List[Tuple[int, str, float]]:
        p = self.p
        initials: List[Tuple[int, str, float]] = []
        halogens = {"F", "Cl", "Br", "I"}

        for n in PG.nodes():
            for m in G.neighbors(n):
                if m in PG:
                    continue
                elem = G.nodes[m].get("element", "C")
                order = float(G[n][m].get("order", 1.0))

                if (
                    p.sp3_plusH
                    and elem == "C"
                    and int(G.nodes[m].get("hcount", 0)) >= 2
                ):
                    initials.append((n, "+H", p.strength_plusH))

                if p.hetero_plusM and elem in ("O", "N") and order == 1.0:
                    initials.append((n, "+M", p.strength_plusM))

                if p.detect_halogen_I and elem in halogens:
                    initials.append((n, "-I", p.strength_minusI))

            if p.detect_carbonyl_alpha:
                for m in PG.neighbors(n):
                    if self._is_carbonyl_carbon(G, m):
                        initials.append((n, "-M", p.strength_minusM))

            for eff in G.nodes[n].get("effects") or []:
                initials.append((n, eff["kind"], float(eff.get("strength", 0.0))))

        return initials

    def propagate(
        self, PG: nx.Graph, initials: Iterable[Tuple[int, str, float]]
    ) -> Dict[int, float]:
        p = self.p
        shifts: Dict[int, float] = {n: 0.0 for n in PG.nodes()}
        att = {
            "+M": p.att_M,
            "-M": p.att_M,
            "+H": p.att_H,
            "+I": p.att_I,
            "-I": p.att_I,
        }
        sgn = {"+M": +1.0, "-M": -1.0, "+H": +1.0, "+I": +1.0, "-I": -1.0}

        for src, kind, strength in initials:
            if src not in PG:
                continue
            for t, d in nx.single_source_shortest_path_length(PG, src).items():
                shifts[t] += sgn.get(kind, 0.0) * strength * (att.get(kind, 1.0) ** d)
        return shifts

    def build_huckel(
        self, PG: nx.Graph, alpha: Dict[int, float]
    ) -> Tuple[np.ndarray, List[int]]:
        nodes = sorted(PG.nodes())
        n = len(nodes)
        H = np.zeros((n, n), dtype=float)
        idx = {nodes[i]: i for i in range(n)}
        for i, node in enumerate(nodes):
            H[i, i] = self.p.beta * float(alpha.get(node, 0.0))
        for u, v in PG.edges():
            i, j = idx[u], idx[v]
            H[i, j] = H[j, i] = self.p.beta
        return H, nodes

    @staticmethod
    def solve(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        E, C = np.linalg.eigh(H)
        idx = np.argsort(E)
        return E[idx], C[:, idx]

    def descriptors(self, E: np.ndarray, C: np.ndarray) -> Dict[str, Any]:
        n = C.shape[0]
        occ = max(1, n // 2)
        h = max(0, occ - 1)
        lumo = min(n - 1, occ)
        h1 = max(0, h - 1)
        l1 = min(n - 1, lumo + 1)
        d: Dict[str, Any] = {
            "h": h,
            "l": lumo,
            "cH": C[:, h],
            "cL": C[:, lumo],
            "E_H": E[h],
            "E_L": E[lumo],
        }
        if self.p.use_extended:
            d.update(
                {
                    "cH1": C[:, h1],
                    "cL1": C[:, l1],
                    "E_H1": E[h1],
                    "E_L1": E[l1],
                }
            )
        return d

    @staticmethod
    def pi_mulliken(E: np.ndarray, C: np.ndarray) -> np.ndarray:
        n = C.shape[0]
        occ = max(1, n // 2)
        h = max(0, occ - 1)
        pop = np.zeros(n)
        for m in range(0, h + 1):
            pop += 2.0 * (C[:, m] ** 2)
        return 1.0 - pop

    @staticmethod
    def total_pi_energy(E: np.ndarray) -> float:
        n = len(E)
        occ = max(1, n // 2)
        return float(np.sum(E[:occ]) * 2.0)
