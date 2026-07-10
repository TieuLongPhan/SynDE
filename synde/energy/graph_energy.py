"""Public graph-first v2 facade."""

from __future__ import annotations
from .molecule_scoring import MoleculeScorer
from synde.graph.pair_scoring import GraphPairScorer
from synde.graph.builder import GraphBuilder
from .reaction_scoring import ReactionScorer
from .its_scoring import ITSScorer, ITSScoringConfig
from synde.geometry.scoring import GeometryScorer, GeometryScoringConfig


class GraphEnergy:
    def __init__(self) -> None:
        self.molecules = MoleculeScorer()
        self.pairs = GraphPairScorer()
        self.reactions = ReactionScorer(self.molecules)
        self.its = ITSScorer(self.molecules, self.pairs)
        self._cache = {}

    def score_molecule_from_smiles(self, smiles: str):
        graph = GraphBuilder.from_smiles(smiles)
        return self._cached_molecule(graph)

    def score_molecule_from_graph(self, graph):
        return self._cached_molecule(GraphBuilder.from_graph(graph))

    def rank_pairs_from_smiles(self, smiles: str, *, top_k: int = 50):
        return self.pairs.rank(GraphBuilder.from_smiles(smiles), top_k=top_k)

    def score_reaction(self, reaction_smiles: str):
        parts = reaction_smiles.split(">>")
        if len(parts) != 2:
            raise ValueError("Reaction SMILES must contain exactly one '>>'.")
        reactants = [
            GraphBuilder.from_smiles(item) for item in parts[0].split(".") if item
        ]
        products = [
            GraphBuilder.from_smiles(item) for item in parts[1].split(".") if item
        ]
        return self.reactions.score(reactants, products, reaction_smiles)

    def score_its(
        self, reaction_smiles: str, *, config: ITSScoringConfig | None = None
    ):
        """Return mapped reaction-centre feasibility terms in graph ``score`` units."""
        parts = reaction_smiles.split(">>")
        if len(parts) != 2:
            raise ValueError("Reaction SMILES must contain exactly one '>>'.")
        reactants = [
            GraphBuilder.from_smiles(item) for item in parts[0].split(".") if item
        ]
        products = [
            GraphBuilder.from_smiles(item) for item in parts[1].split(".") if item
        ]
        scorer = (
            self.its
            if config is None
            else ITSScorer(self.molecules, self.pairs, config)
        )
        return scorer.score(reactants, products, reaction_smiles)

    def score_molecule_with_geometry(
        self,
        smiles: str,
        *,
        config: GeometryScoringConfig | None = None,
        score_margin: float | None = None,
    ):
        normalized = GraphBuilder.from_smiles(smiles)
        scorer = GeometryScorer(config, self.molecules)
        if not scorer.should_run(normalized, score_margin=score_margin):
            return self._cached_molecule(normalized)
        return scorer.score_smiles(smiles)

    def _cached_molecule(self, graph):
        if graph.identity not in self._cache:
            self._cache[graph.identity] = self.molecules.score(graph)
        return self._cache[graph.identity]
