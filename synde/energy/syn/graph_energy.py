"""Public graph-first v2 facade."""

from __future__ import annotations
from .molecule_scoring import MoleculeScorer
from .pair_scoring import GraphPairScorer
from .rdkit_graph_builder import RDKitGraphBuilder
from .reaction_scoring import ReactionScorer
from .geometry_scoring import GeometryScorer, GeometryScoringConfig


class GraphEnergy:
    def __init__(self) -> None:
        self.molecules = MoleculeScorer()
        self.pairs = GraphPairScorer()
        self.reactions = ReactionScorer(self.molecules)
        self._cache = {}

    def score_molecule_from_smiles(self, smiles: str):
        graph = RDKitGraphBuilder.from_smiles_v2(smiles)
        return self._cached_molecule(graph)

    def score_molecule_from_graph(self, graph):
        return self._cached_molecule(RDKitGraphBuilder.from_graph_v2(graph))

    def rank_pairs_from_smiles(self, smiles: str, *, top_k: int = 50):
        return self.pairs.rank(RDKitGraphBuilder.from_smiles_v2(smiles), top_k=top_k)

    def score_reaction(self, reaction_smiles: str):
        parts = reaction_smiles.split(">>")
        if len(parts) != 2:
            raise ValueError("Reaction SMILES must contain exactly one '>>'.")
        reactants = [
            RDKitGraphBuilder.from_smiles_v2(item)
            for item in parts[0].split(".")
            if item
        ]
        products = [
            RDKitGraphBuilder.from_smiles_v2(item)
            for item in parts[1].split(".")
            if item
        ]
        return self.reactions.score(reactants, products, reaction_smiles)

    def score_molecule_with_geometry(
        self,
        smiles: str,
        *,
        config: GeometryScoringConfig | None = None,
        score_margin: float | None = None,
    ):
        normalized = RDKitGraphBuilder.from_smiles_v2(smiles)
        scorer = GeometryScorer(config, self.molecules)
        if not scorer.should_run(normalized, score_margin=score_margin):
            return self._cached_molecule(normalized)
        return scorer.score_smiles(smiles)

    def _cached_molecule(self, graph):
        if graph.identity not in self._cache:
            self._cache[graph.identity] = self.molecules.score(graph)
        return self._cache[graph.identity]
