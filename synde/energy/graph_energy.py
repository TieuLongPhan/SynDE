"""Public graph-first v2 facade."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .molecule_scoring import MoleculeScorer
from synde.graph.pair_scoring import GraphPairScorer
from synde.graph.builder import GraphBuilder
from .reaction_scoring import ReactionScorer
from .semiempirical_energy import GFN2SinglePointScorer
from .truncated_scc_energy import GFN2TwoCycleScorer
from .its_scoring import ITSScorer, ITSScoringConfig
from .theory_energy import TheoryEnergyScorer
from .orbital_theory_energy import OrbitalTheoryEnergyScorer
from .graph_theory_energy import GraphTheoryEnergyScorer
from .empirical_two_d_energy import EmpiricalTwoDEnergyScorer
from .first_order_two_d_energy import FirstOrderTwoDEnergyScorer
from .valence_energy import ValenceEnergyScorer
from .xtb_proxy_energy import GFN2XTBProxyScorer

if TYPE_CHECKING:  # pragma: no cover - static analysis only
    from synde.geometry.scoring import GeometryScoringConfig


class GraphEnergy:
    def __init__(self) -> None:
        self.molecules = MoleculeScorer()
        self.theory_molecules = TheoryEnergyScorer(base_scorer=self.molecules)
        self.orbital_theory_molecules = OrbitalTheoryEnergyScorer(
            base_scorer=self.molecules
        )
        self.graph_theory_molecules = GraphTheoryEnergyScorer(
            base_scorer=self.molecules
        )
        self.valence_molecules = ValenceEnergyScorer(base_scorer=self.molecules)
        self.empirical_two_d_molecules = EmpiricalTwoDEnergyScorer()
        self.first_order_two_d_molecules = FirstOrderTwoDEnergyScorer()
        self.xtb_proxy_molecules = GFN2XTBProxyScorer()
        self.gfn2_molecules = GFN2SinglePointScorer()
        self.gfn2_two_cycle_molecules = GFN2TwoCycleScorer()
        self.pairs = GraphPairScorer()
        self.reactions = ReactionScorer(self.molecules)
        self.its = ITSScorer(self.molecules, self.pairs)
        self._cache = {}
        self._theory_cache = {}
        self._orbital_theory_cache = {}
        self._graph_theory_cache = {}
        self._valence_cache = {}
        self._empirical_two_d_cache = {}
        self._first_order_two_d_cache = {}
        self._xtb_proxy_cache = {}
        self._gfn2_cache = {}
        self._gfn2_two_cycle_cache = {}

    def score_molecule_from_smiles(self, smiles: str):
        graph = GraphBuilder.from_smiles(smiles)
        return self._cached_molecule(graph)

    def score_molecule_from_graph(self, graph):
        return self._cached_molecule(GraphBuilder.from_graph(graph))

    def score_molecule_theory_from_smiles(self, smiles: str):
        """Score with the enhanced, still-unfitted theory-guided heuristic."""
        return self._cached_theory_molecule(GraphBuilder.from_smiles(smiles))

    def score_molecule_theory_from_graph(self, graph):
        """Score a NetworkX graph with the enhanced uncalibrated heuristic."""
        return self._cached_theory_molecule(GraphBuilder.from_graph(graph))

    def score_molecule_orbital_theory_from_smiles(self, smiles: str):
        """Score with the experimental donor/triple/cumulene orbital model."""
        return self._cached_orbital_theory_molecule(GraphBuilder.from_smiles(smiles))

    def score_molecule_orbital_theory_from_graph(self, graph):
        """Score a graph with the experimental orbital-level pi model."""
        return self._cached_orbital_theory_molecule(GraphBuilder.from_graph(graph))

    def score_molecule_graph_theory_from_smiles(self, smiles: str):
        """Score with the reduced graph-only v2 development candidate."""
        return self._cached_graph_theory_molecule(GraphBuilder.from_smiles(smiles))

    def score_molecule_graph_theory_from_graph(self, graph):
        """Score a graph after excluding geometry-dependent coarse terms."""
        return self._cached_graph_theory_molecule(GraphBuilder.from_graph(graph))

    def score_molecule_valence_from_smiles(self, smiles: str):
        """Score with the conventional non-overlapping valence decomposition."""
        return self._cached_valence_molecule(GraphBuilder.from_smiles(smiles))

    def score_molecule_valence_from_graph(self, graph):
        """Score a labeled molecular graph with the valence decomposition."""
        return self._cached_valence_molecule(GraphBuilder.from_graph(graph))

    def score_molecule_empirical_two_d_from_smiles(self, smiles: str):
        """Score with the interpretable Joback--Hückel graph-only ledger."""
        return self._cached_empirical_two_d_molecule(GraphBuilder.from_smiles(smiles))

    def score_molecule_empirical_two_d_from_graph(self, graph):
        """Score a labeled graph without coordinates or learned coefficients."""
        return self._cached_empirical_two_d_molecule(GraphBuilder.from_graph(graph))

    def score_molecule_first_order_two_d_from_smiles(self, smiles: str):
        """Score with the first-order average-bond/Hückel graph ledger."""
        return self._cached_first_order_two_d_molecule(GraphBuilder.from_smiles(smiles))

    def score_molecule_first_order_two_d_from_graph(self, graph):
        """Score a labeled graph using only fixed empirical 2D terms."""
        return self._cached_first_order_two_d_molecule(GraphBuilder.from_graph(graph))

    def score_molecule_xtb_proxy_from_smiles(self, smiles: str):
        """Estimate a raw GFN2-xTB-like total energy without running xTB."""
        return self._cached_xtb_proxy_molecule(GraphBuilder.from_smiles(smiles))

    def score_molecule_xtb_proxy_from_graph(self, graph):
        """Estimate the raw xTB-like proxy from a labeled molecular graph."""
        return self._cached_xtb_proxy_molecule(GraphBuilder.from_graph(graph))

    def score_molecule_gfn2_from_smiles(self, smiles: str):
        """Run an uncalibrated GFN2-xTB single point on deterministic geometry."""
        return self._cached_gfn2_molecule(GraphBuilder.from_smiles(smiles))

    def score_molecule_gfn2_from_graph(self, graph):
        """Run GFN2-xTB for a labeled molecular graph."""
        return self._cached_gfn2_molecule(GraphBuilder.from_graph(graph))

    def score_molecule_gfn2_two_cycle_from_smiles(self, smiles: str):
        """Evaluate the frozen two-cycle SCC plus explicit repulsion score."""
        return self._cached_gfn2_two_cycle_molecule(GraphBuilder.from_smiles(smiles))

    def score_molecule_gfn2_two_cycle_from_graph(self, graph):
        """Evaluate the frozen two-term score for a labeled molecular graph."""
        return self._cached_gfn2_two_cycle_molecule(GraphBuilder.from_graph(graph))

    def rank_pairs_from_smiles(self, smiles: str, *, top_k: int = 50):
        return self.pairs.rank(GraphBuilder.from_smiles(smiles), top_k=top_k)

    def score_reaction(self, reaction_smiles: str):
        reactant, product, _ = GraphBuilder.reaction_states_from_smiles(reaction_smiles)
        return self.reactions.score([reactant], [product], reaction_smiles)

    def score_its(
        self, reaction_smiles: str, *, config: ITSScoringConfig | None = None
    ):
        """Return mapped reaction-centre feasibility terms in graph ``score`` units."""
        reactant, product, native_its = GraphBuilder.reaction_states_from_smiles(
            reaction_smiles
        )
        scorer = (
            self.its
            if config is None
            else ITSScorer(self.molecules, self.pairs, config)
        )
        return scorer.score(
            [reactant], [product], reaction_smiles, native_its=native_its
        )

    def score_molecule_with_geometry(
        self,
        smiles: str,
        *,
        config: GeometryScoringConfig | None = None,
        score_margin: float | None = None,
    ):
        # Imported here so that loading the energy stack never pulls in the
        # conformer and xTB machinery, which inference does not use.
        from synde.geometry.scoring import GeometryScorer

        normalized = GraphBuilder.from_smiles(smiles)
        scorer = GeometryScorer(config, self.molecules)
        if not scorer.should_run(normalized, score_margin=score_margin):
            return self._cached_molecule(normalized)
        return scorer.score_smiles(smiles)

    def _cached_molecule(self, graph):
        if graph.identity not in self._cache:
            self._cache[graph.identity] = self.molecules.score(graph)
        return self._cache[graph.identity]

    def _cached_theory_molecule(self, graph):
        if graph.identity not in self._theory_cache:
            self._theory_cache[graph.identity] = self.theory_molecules.score(graph)
        return self._theory_cache[graph.identity]

    def _cached_orbital_theory_molecule(self, graph):
        if graph.identity not in self._orbital_theory_cache:
            self._orbital_theory_cache[graph.identity] = (
                self.orbital_theory_molecules.score(graph)
            )
        return self._orbital_theory_cache[graph.identity]

    def _cached_graph_theory_molecule(self, graph):
        if graph.identity not in self._graph_theory_cache:
            self._graph_theory_cache[graph.identity] = (
                self.graph_theory_molecules.score(graph)
            )
        return self._graph_theory_cache[graph.identity]

    def _cached_valence_molecule(self, graph):
        if graph.identity not in self._valence_cache:
            self._valence_cache[graph.identity] = self.valence_molecules.score(graph)
        return self._valence_cache[graph.identity]

    def _cached_empirical_two_d_molecule(self, graph):
        if graph.identity not in self._empirical_two_d_cache:
            self._empirical_two_d_cache[graph.identity] = (
                self.empirical_two_d_molecules.score(graph)
            )
        return self._empirical_two_d_cache[graph.identity]

    def _cached_first_order_two_d_molecule(self, graph):
        if graph.identity not in self._first_order_two_d_cache:
            self._first_order_two_d_cache[graph.identity] = (
                self.first_order_two_d_molecules.score(graph)
            )
        return self._first_order_two_d_cache[graph.identity]

    def _cached_xtb_proxy_molecule(self, graph):
        if graph.identity not in self._xtb_proxy_cache:
            self._xtb_proxy_cache[graph.identity] = self.xtb_proxy_molecules.score(
                graph
            )
        return self._xtb_proxy_cache[graph.identity]

    def _cached_gfn2_molecule(self, graph):
        if graph.identity not in self._gfn2_cache:
            self._gfn2_cache[graph.identity] = self.gfn2_molecules.score(graph)
        return self._gfn2_cache[graph.identity]

    def _cached_gfn2_two_cycle_molecule(self, graph):
        if graph.identity not in self._gfn2_two_cycle_cache:
            self._gfn2_two_cycle_cache[graph.identity] = (
                self.gfn2_two_cycle_molecules.score(graph)
            )
        return self._gfn2_two_cycle_cache[graph.identity]
