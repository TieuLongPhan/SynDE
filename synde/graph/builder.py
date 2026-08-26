"""Build the SYN v2 normalized graph contract from Synkit's MolToGraph."""

from __future__ import annotations
import networkx as nx
from synde.errors import SynDEInputError
from .graph_schema import NormalizedMolecularGraph, normalize_graph


class GraphBuilder:
    @staticmethod
    def from_smiles(smiles: str, *, strict: bool = True) -> NormalizedMolecularGraph:
        from rdkit import Chem
        from synkit.Chem.Molecule.standardize import sanitize_and_canonicalize_smiles
        from synkit.IO.mol_to_graph import MolToGraph
        from synde.chem import normalize_ordinary_explicit_hydrogens

        parser = Chem.SmilesParserParams()
        parser.removeHs = False
        mol = Chem.MolFromSmiles(smiles, parser)
        if mol is None:
            raise SynDEInputError(
                f"Invalid SMILES: {smiles!r} could not be parsed by RDKit.\n"
                "  Hint: check valences, ring-closure digits, and bracketed "
                "atoms. RDKit prints the specific parse failure above this "
                "traceback."
            )
        mol = normalize_ordinary_explicit_hydrogens(mol)
        normalized_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        canonical = sanitize_and_canonicalize_smiles(normalized_smiles)
        graph = MolToGraph(attr_profile="full", with_topology=True).transform(
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
    def reaction_states_from_smiles(
        reaction_smiles: str, *, strict: bool = True
    ) -> tuple[NormalizedMolecularGraph, NormalizedMolecularGraph, nx.Graph | None]:
        """Convert reaction SMILES with Synkit's native reaction graph converter.

        The first two values are SynDE-normalized reactant and product state
        graphs.  The third is Synkit's paired-attribute ITS graph when complete
        atom mapping is available; otherwise it is ``None``.
        """
        from synkit.IO import rsmi_to_graph
        from synkit.Graph.ITS import ITSConstruction

        reactant_raw, product_raw = rsmi_to_graph(
            reaction_smiles,
            drop_non_aam=False,
            use_index_as_atom_map=False,
        )
        if reactant_raw is None or product_raw is None:
            raise SynDEInputError(
                f"Invalid reaction SMILES: {reaction_smiles!r} could not be "
                "split into reactant and product graphs.\n"
                "  Hint: use 'reactants>>products' with parsable SMILES on "
                "both sides."
            )
        reactant = GraphBuilder.from_synkit_graph(reactant_raw, strict=strict)
        product = GraphBuilder.from_synkit_graph(product_raw, strict=strict)
        try:
            mapped_reactant = GraphBuilder._relabel_by_atom_map(reactant_raw)
            mapped_product = GraphBuilder._relabel_by_atom_map(product_raw)
            its = ITSConstruction().construct(mapped_reactant, mapped_product)
        except ValueError:
            its = None
        return reactant, product, its

    @staticmethod
    def _relabel_by_atom_map(graph: nx.Graph) -> nx.Graph:
        mapping: dict[object, int] = {}
        for node, attrs in graph.nodes(data=True):
            atom_map = attrs.get("atom_map")
            if atom_map in (None, 0):
                raise ValueError("REACTION_MAPPING_MISSING")
            atom_map = int(atom_map)
            if atom_map in mapping.values():
                raise ValueError("REACTION_NOT_BALANCED")
            mapping[node] = atom_map
        return nx.relabel_nodes(graph, mapping, copy=True)

    @staticmethod
    def from_graph(graph: nx.Graph, *, strict: bool = True) -> NormalizedMolecularGraph:
        return normalize_graph(graph, strict=strict, source="networkx-v2")
