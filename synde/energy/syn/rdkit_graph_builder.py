from __future__ import annotations

import networkx as nx

from .graph_schema import NormalizedMolecularGraph, normalize_graph


class RDKitGraphBuilder:
    """
    Optional RDKit-backed graph builder.

    This file intentionally keeps RDKit usage inside the function so the module
    can be imported even when RDKit is not present. Use RDKit only when needed.
    """

    @staticmethod
    def from_smiles(smiles: str, *, compute_gasteiger: bool = True) -> nx.Graph:
        """
        Build the minimal graph from SMILES.

        :raises ImportError: if RDKit is not installed.
        :raises ValueError: if SMILES cannot be parsed.
        """
        try:
            from rdkit import Chem  # type: ignore
            from rdkit.Chem import rdPartialCharges  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("RDKit is required for SMILES conversion") from exc

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        if compute_gasteiger:
            try:
                Chem.AddHs(mol)
                rdPartialCharges.ComputeGasteigerCharges(mol)
            except Exception:
                compute_gasteiger = False

        G = nx.Graph()
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            aromatic = bool(atom.GetIsAromatic())
            elem = atom.GetSymbol()
            hcount = atom.GetTotalNumHs()
            pc = 0.0
            if compute_gasteiger:
                try:
                    pc = float(atom.GetDoubleProp("_GasteigerCharge"))
                except Exception:
                    pc = 0.0
            G.add_node(
                idx, element=elem, aromatic=aromatic, hcount=hcount, partial_charge=pc
            )

        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            order = float(bond.GetBondTypeAsDouble())
            G.add_edge(i, j, order=order)
        return G

    @staticmethod
    def from_smiles_v2(
        smiles: str,
        *,
        compute_gasteiger: bool = True,
        strict: bool = True,
    ) -> NormalizedMolecularGraph:
        """Build a normalized v2 molecular graph from SMILES.

        The legacy :meth:`from_smiles` API remains unchanged.  This method
        preserves formal charge, radical count, hybridization, conjugation,
        stereo, atom maps, component identity, and optional finite Gasteiger
        charges needed by the graph-first v2 pipeline.
        """
        try:
            from rdkit import Chem  # type: ignore
            from rdkit.Chem import rdPartialCharges  # type: ignore
            from synkit.Chem.Molecule.graph_annotator import GraphAnnotator
            from synkit.Chem.Molecule.standardize import (
                sanitize_and_canonicalize_smiles,
            )
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "RDKit and synkit are required for v2 SMILES conversion"
            ) from exc

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        canonical_smiles = sanitize_and_canonicalize_smiles(smiles)
        if canonical_smiles is None:
            raise ValueError(f"SMILES could not be canonicalized: {smiles}")

        charge_mol = Chem.Mol(mol)
        if compute_gasteiger:
            try:
                rdPartialCharges.ComputeGasteigerCharges(charge_mol)
            except Exception:
                compute_gasteiger = False

        graph = nx.Graph()
        for atom in mol.GetAtoms():
            index = atom.GetIdx()
            partial_charge = None
            if compute_gasteiger:
                try:
                    partial_charge = float(
                        charge_mol.GetAtomWithIdx(index).GetProp("_GasteigerCharge")
                    )
                except Exception:
                    partial_charge = None
            atom_map = atom.GetAtomMapNum() or None
            graph.add_node(
                index,
                element=atom.GetSymbol(),
                atomic_number=atom.GetAtomicNum(),
                formal_charge=atom.GetFormalCharge(),
                aromatic=atom.GetIsAromatic(),
                hybridization=str(atom.GetHybridization()),
                total_hcount=atom.GetTotalNumHs(),
                radical_electrons=atom.GetNumRadicalElectrons(),
                atom_map=atom_map,
                partial_charge=partial_charge,
                is_in_ring=atom.IsInRing(),
            )
        for bond in mol.GetBonds():
            graph.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                order=float(bond.GetBondTypeAsDouble()),
                aromatic=bond.GetIsAromatic(),
                conjugated=bond.GetIsConjugated(),
                in_ring=bond.IsInRing(),
                stereo=str(bond.GetStereo()),
                bond_type=str(bond.GetBondType()),
            )

        # Synkit supplies graph topology annotations; in_place=False preserves
        # the freshly constructed source graph before the schema makes its own copy.
        annotated = GraphAnnotator(graph, in_place=False).annotate().graph
        return normalize_graph(
            annotated,
            strict=strict,
            canonical_smiles=canonical_smiles,
            source="rdkit-smiles-v2",
        )

    @staticmethod
    def from_graph_v2(
        graph: nx.Graph,
        *,
        strict: bool = True,
    ) -> NormalizedMolecularGraph:
        """Normalize a caller-supplied NetworkX molecular graph for v2."""
        return normalize_graph(graph, strict=strict, source="networkx-v2")
