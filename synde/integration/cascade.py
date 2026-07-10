"""Graph-first shortlist followed by optional xTB evaluation."""

from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from typing import Callable
from synde.energy.graph_energy import GraphEnergy


@dataclass(frozen=True)
class CascadeRow:
    reaction_smiles: str
    graph_score: float | None
    xtb_delta_e: float | None
    selected: bool
    status: str


@dataclass(frozen=True)
class CascadeReport:
    rows: tuple[CascadeRow, ...]
    shortlist_size: int
    graph_seconds: float
    xtb_seconds: float


class GraphXTBCascade:
    """Use graph scores to choose reactions for a named xTB evaluator."""

    def __init__(self, graph_energy: GraphEnergy | None = None) -> None:
        self.graph_energy = graph_energy or GraphEnergy()
        self._xtb_cache: dict[tuple[str, str], float] = {}

    def screen(
        self,
        reactions: list[str],
        *,
        top_k: int = 10,
        level: str = "loose",
        xtb_evaluator: Callable[[str, str], float] | None = None,
    ) -> CascadeReport:
        started = perf_counter()
        scored = []
        for reaction in reactions:
            try:
                scored.append(
                    (
                        reaction,
                        self.graph_energy.score_reaction(reaction).reaction_delta_score,
                        "success",
                    )
                )
            except Exception:
                scored.append((reaction, None, "graph_error"))
        graph_seconds = perf_counter() - started
        ranked = sorted(
            scored, key=lambda row: float("inf") if row[1] is None else row[1]
        )
        selected = {row[0] for row in ranked[:top_k] if row[1] is not None}
        started = perf_counter()
        rows = []
        for reaction, score, status in scored:
            value = None
            if reaction in selected:
                key = (level, reaction)
                try:
                    if key not in self._xtb_cache:
                        if xtb_evaluator is None:
                            from synde.external.xtb.xtb_reaction import XTBReaction

                            self._xtb_cache[key] = float(
                                XTBReaction.delta_e_rsmi(reaction, level=level)
                            )
                        else:
                            self._xtb_cache[key] = float(xtb_evaluator(reaction, level))
                    value = self._xtb_cache[key]
                except Exception:
                    status = "xtb_error"
            rows.append(
                CascadeRow(reaction, score, value, reaction in selected, status)
            )
        return CascadeReport(
            tuple(rows), len(selected), graph_seconds, perf_counter() - started
        )
