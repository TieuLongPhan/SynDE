from __future__ import annotations

import unittest

import networkx as nx

from synde.energy.syn import RDKitGraphBuilder, assign_pi_systems, normalize_graph


class TestPiSystemAssigner(unittest.TestCase):
    def _assignment(self, smiles: str):
        return assign_pi_systems(RDKitGraphBuilder.from_smiles_v2(smiles))

    def test_reference_pi_electron_counts(self) -> None:
        cases = {
            "C=C": 2,
            "C=CC=C": 4,
            "c1ccccc1": 6,
            "n1ccccc1": 6,
            "c1cc[nH]c1": 6,
            "c1ccoc1": 6,
            "O=CC": 2,
        }
        for smiles, electrons in cases.items():
            with self.subTest(smiles=smiles):
                result = self._assignment(smiles)
                self.assertEqual(result.status, "success")
                self.assertEqual(result.electron_count, electrons)

    def test_pyridine_and_pyrrole_nitrogen_are_distinguished(self) -> None:
        pyridine = self._assignment("n1ccccc1")
        pyrrole = self._assignment("c1cc[nH]c1")

        pyridine_n = next(atom for atom in pyridine.atoms if atom.node == 0)
        pyrrole_n = next(atom for atom in pyrrole.atoms if atom.node == 3)
        self.assertEqual(pyridine_n.electrons, 1)
        self.assertEqual(pyrrole_n.electrons, 2)
        self.assertIn("pyridine-like", pyridine_n.reason)
        self.assertIn("pyrrole-like", pyrrole_n.reason)

    def test_butadiene_has_one_conjugated_pi_system(self) -> None:
        result = self._assignment("C=CC=C")

        self.assertEqual(len(result.systems), 1)
        self.assertEqual(len(result.systems[0].nodes), 4)
        self.assertEqual(len(result.systems[0].edges), 3)

    def test_plain_single_bond_does_not_join_pi_systems_without_conjugation(
        self,
    ) -> None:
        graph = nx.Graph()
        for node in range(4):
            graph.add_node(node, element="C")
        graph.add_edge(0, 1, order=2.0)
        graph.add_edge(1, 2, order=1.0, conjugated=False)
        graph.add_edge(2, 3, order=2.0)
        result = assign_pi_systems(normalize_graph(graph))

        self.assertEqual(len(result.systems), 2)
        self.assertEqual(
            sorted(system.electron_count for system in result.systems), [2, 2]
        )

    def test_triple_bond_is_flagged_not_silently_modeled(self) -> None:
        result = self._assignment("C#N")

        self.assertEqual(result.status, "unsupported")
        self.assertIn("PI_ORBITAL_MULTIPLICITY_UNSUPPORTED", result.warning_codes())
        self.assertEqual(result.electron_count, 0)

    def test_each_atom_has_a_diagnostic(self) -> None:
        normalized = RDKitGraphBuilder.from_smiles_v2("CCO")
        result = assign_pi_systems(normalized)

        self.assertEqual(len(result.atoms), normalized.graph.number_of_nodes())
        self.assertTrue(all(atom.reason for atom in result.atoms))
        self.assertTrue(all(atom.confidence for atom in result.atoms))


if __name__ == "__main__":
    unittest.main()
