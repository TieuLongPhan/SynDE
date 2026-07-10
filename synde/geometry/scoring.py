"""Optional conformer-based geometry refinement; never run by graph-only APIs."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
import numpy as np
from .rdkit._conformer import ConformerGenerator
from synde.energy.molecule_scoring import MoleculeScorer
from synde.graph.builder import GraphBuilder


@dataclass(frozen=True)
class GeometryScoringConfig:
    num_conformers: int | str = "auto"
    force_field: str = "MMFF94"
    random_seed: int = 42
    gate: str = "always"


@dataclass(frozen=True)
class GeometryScoreResult:
    status: str
    geometry_corrected_score: float | None
    force_field_energy_kcal_mol: float | None
    components: dict[str, float]
    warnings: tuple[str, ...]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GeometryScorer:
    def __init__(
        self,
        config: GeometryScoringConfig | None = None,
        molecule_scorer: MoleculeScorer | None = None,
    ) -> None:
        self.config = config or GeometryScoringConfig()
        self.molecule_scorer = molecule_scorer or MoleculeScorer()

    def score_smiles(self, smiles: str) -> GeometryScoreResult:
        graph = GraphBuilder.from_smiles(smiles)
        base = self.molecule_scorer.score(graph)
        conformer, energy = ConformerGenerator().process_smiles(
            smiles,
            num_conformers=self.config.num_conformers,
            force_field_method=self.config.force_field,
            random_seed=self.config.random_seed,
        )
        provenance = {
            "mode": "geometry",
            "embedding_seed": self.config.random_seed,
            "force_field": self.config.force_field,
            "num_conformers": self.config.num_conformers,
        }
        if conformer is None:
            return GeometryScoreResult(
                "error", None, None, {}, ("CONFORMER_GENERATION_FAILED",), provenance
            )
        terms = self._nonbonded_terms(graph.graph, conformer)
        corrected = float(
            (base.score or 0.0) + terms["charge_distance"] + terms["repulsion"]
        )
        return GeometryScoreResult(
            "partial" if base.warnings else "success",
            corrected,
            float(energy),
            terms,
            base.warnings,
            provenance,
        )

    def should_run(self, normalized, *, score_margin: float | None = None) -> bool:
        """Apply explicit user/uncertainty/flexibility gating without embedding."""
        if self.config.gate == "always":
            return True
        if self.config.gate == "never":
            return False
        if self.config.gate != "auto":
            raise ValueError("gate must be 'always', 'never', or 'auto'.")
        flexible = sum(
            1
            for _, _, data in normalized.graph.edges(data=True)
            if float(data.get("order", 1.0)) == 1.0 and not data.get("in_ring", False)
        )
        return flexible > 0 or (score_margin is not None and score_margin < 0.1)

    @staticmethod
    def _nonbonded_terms(graph, mol) -> dict[str, float]:
        conf = mol.GetConformer()
        charge = 0.0
        repulsion = 0.0
        for i in graph.nodes:
            for j in graph.nodes:
                if j <= i or graph.has_edge(i, j):
                    continue
                pi = conf.GetAtomPosition(int(i))
                pj = conf.GetAtomPosition(int(j))
                r = max(
                    0.5, float(np.linalg.norm([pi.x - pj.x, pi.y - pj.y, pi.z - pj.z]))
                )
                qi = graph.nodes[i].get("partial_charge") or 0.0
                qj = graph.nodes[j].get("partial_charge") or 0.0
                charge += qi * qj / r
                repulsion += np.exp(-r)
        return {"charge_distance": float(charge), "repulsion": float(repulsion)}

    @staticmethod
    def nonbonded_terms_for_positions(
        graph, positions: dict[int, tuple[float, float, float]]
    ) -> dict[str, float]:
        """Distance-term helper for deterministic tests and supplied poses."""
        charge = 0.0
        repulsion = 0.0
        for i in graph.nodes:
            for j in graph.nodes:
                if j <= i or graph.has_edge(i, j):
                    continue
                r = max(
                    0.5, float(np.linalg.norm(np.subtract(positions[i], positions[j])))
                )
                qi = graph.nodes[i].get("partial_charge") or 0.0
                qj = graph.nodes[j].get("partial_charge") or 0.0
                charge += qi * qj / r
                repulsion += np.exp(-r)
        return {"charge_distance": float(charge), "repulsion": float(repulsion)}
