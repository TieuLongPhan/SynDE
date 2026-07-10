from __future__ import annotations

import unittest

import networkx as nx
import numpy as np

from synde import (
    GeneralizedHuckel,
    GraphBuilder,
    assign_pi_systems,
    normalize_graph,
    solve_generalized_huckel,
)


class TestGeneralizedHuckel(unittest.TestCase):
    def _solve(self, smiles: str):
        assigned = assign_pi_systems(GraphBuilder.from_smiles(smiles))
        return solve_generalized_huckel(assigned)

    def test_explicit_pyrrole_occupation_uses_six_electrons(self) -> None:
        result = self._solve("c1cc[nH]c1")
        system = result.systems[0]

        self.assertEqual(system.electron_count, 6)
        np.testing.assert_allclose(system.occupations, [2.0, 2.0, 2.0, 0.0, 0.0])

    def test_pyridine_and_benzene_have_distinct_generalized_descriptors(self) -> None:
        benzene = self._solve("c1ccccc1")
        pyridine = self._solve("n1ccccc1")

        self.assertNotAlmostEqual(benzene.raw_pi_energy, pyridine.raw_pi_energy)
        self.assertNotAlmostEqual(
            benzene.systems[0].orbital_energies[0],
            pyridine.systems[0].orbital_energies[0],
        )

    def test_disconnected_systems_are_additive(self) -> None:
        ethene = self._solve("C=C")
        two_ethenes = self._solve("C=C.C=C")

        self.assertAlmostEqual(two_ethenes.raw_pi_energy, 2 * ethene.raw_pi_energy)
        self.assertAlmostEqual(
            two_ethenes.reference_pi_energy, 2 * ethene.reference_pi_energy
        )
        self.assertAlmostEqual(
            two_ethenes.pi_stabilization, 2 * ethene.pi_stabilization
        )

    def test_node_relabeling_preserves_energies_and_local_densities(self) -> None:
        graph = nx.Graph()
        for node in range(6):
            graph.add_node(node, element="C", aromatic=True)
        for node in range(6):
            graph.add_edge(node, (node + 1) % 6, order=1.5, aromatic=True)
        relabeled = nx.relabel_nodes(
            graph, {node: node + 100 for node in graph.nodes()}, copy=True
        )

        first = solve_generalized_huckel(assign_pi_systems(normalize_graph(graph)))
        second = solve_generalized_huckel(assign_pi_systems(normalize_graph(relabeled)))
        np.testing.assert_allclose(
            first.systems[0].orbital_energies, second.systems[0].orbital_energies
        )
        np.testing.assert_allclose(
            np.sort(first.systems[0].homo_density),
            np.sort(second.systems[0].homo_density),
        )

    def test_benzene_frontier_densities_are_symmetry_invariant(self) -> None:
        system = self._solve("c1ccccc1").systems[0]

        np.testing.assert_allclose(system.homo_density, np.full(6, 1 / 3))
        self.assertIsNotNone(system.lumo_density)
        np.testing.assert_allclose(system.lumo_density, np.full(6, 1 / 3))

    def test_subspace_density_is_invariant_to_degenerate_rotation(self) -> None:
        coefficients = np.array(
            [
                [1 / np.sqrt(2), 1 / np.sqrt(2)],
                [1 / np.sqrt(2), -1 / np.sqrt(2)],
            ]
        )
        angle = np.pi / 5
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        rotated = coefficients @ rotation

        np.testing.assert_allclose(
            GeneralizedHuckel.subspace_density(coefficients, (0, 1)),
            GeneralizedHuckel.subspace_density(rotated, (0, 1)),
        )


if __name__ == "__main__":
    unittest.main()
