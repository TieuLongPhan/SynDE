"""Build the SYN v2 normalized graph contract from Synkit's MolToGraph."""

from __future__ import annotations
import networkx as nx
from .graph_schema import NormalizedMolecularGraph, normalize_graph


class GraphBuilder:
    @staticmethod
    def from_smiles(smiles: str, *, strict: bool = True) -> NormalizedMolecularGraph:
        from rdkit import Chem
        from synkit.Chem.Molecule.standardize import sanitize_and_canonicalize_smiles
        from synkit.IO.mol_to_graph import MolToGraph

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        canonical = sanitize_and_canonicalize_smiles(smiles)
        graph = MolToGraph(attr_profile="minimal", with_topology=True).transform(
            Chem.Mol(mol)
        )
        return GraphBuilder.from_synkit_graph(
            graph, strict=strict, canonical_smiles=canonical
        )

    @staticmethod
    def from_synkit_graph(
        graph: nx.Graph, *, strict: bool = True, canonical_smiles: str | None = None
    ) -> NormalizedMolecularGraph:
        from rdkit import Chem

        table = Chem.GetPeriodicTable()
        adapted = nx.Graph()
        for node, attrs in graph.nodes(data=True):
            data = dict(attrs)
            element = data.get("element")
            data.update(
                atomic_number=table.GetAtomicNumber(element) if element else 0,
                formal_charge=data.get("formal_charge", data.get("charge", 0)),
                total_hcount=data.get("total_hcount", data.get("hcount", 0)),
                radical_electrons=data.get("radical_electrons", data.get("radical", 0)),
                is_in_ring=data.get("is_in_ring", data.get("in_ring", False)),
            )
            adapted.add_node(node, **data)
        adapted.add_edges_from((u, v, dict(d)) for u, v, d in graph.edges(data=True))
        return normalize_graph(
            adapted,
            strict=strict,
            canonical_smiles=canonical_smiles,
            source="synkit-mol-to-graph",
        )

    @staticmethod
    def from_graph(graph: nx.Graph, *, strict: bool = True) -> NormalizedMolecularGraph:
        return normalize_graph(graph, strict=strict, source="networkx-v2")
