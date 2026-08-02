import unittest

from synde.graph import (
    GraphBuilder,
    assign_orbital_pi,
    solve_generalized_huckel,
)


class TestOrbitalPiAssignment(unittest.TestCase):
    def assign(self, smiles: str):
        return assign_orbital_pi(GraphBuilder.from_smiles(smiles))

    def test_amide_and_ester_include_four_electron_donor_system(self):
        for smiles in ("CC(=O)N", "CC(=O)O"):
            with self.subTest(smiles=smiles):
                result = self.assign(smiles)
                systems = [
                    system for system in result.systems if len(system.nodes) == 3
                ]
                self.assertEqual(len(systems), 1)
                self.assertEqual(systems[0].electron_count, 4)

    def test_triple_bond_has_two_orthogonal_two_electron_systems(self):
        result = self.assign("C#N")
        self.assertEqual(result.status, "success")
        self.assertEqual(len(result.systems), 2)
        self.assertEqual([system.electron_count for system in result.systems], [2, 2])

    def test_allene_uses_two_orthogonal_pi_systems(self):
        result = self.assign("C=C=C")
        self.assertEqual(result.status, "success")
        self.assertEqual(len(result.systems), 2)
        self.assertEqual([system.electron_count for system in result.systems], [2, 2])
        allene_energy = solve_generalized_huckel(result).raw_pi_energy
        ethene_energy = solve_generalized_huckel(self.assign("C=C")).raw_pi_energy
        self.assertAlmostEqual(allene_energy, 2 * ethene_energy)

    def test_conjugated_nitrile_couples_only_one_triple_orbital(self):
        result = self.assign("C=CC#N")
        self.assertEqual(sorted(len(system.nodes) for system in result.systems), [2, 4])
        self.assertEqual(sum(system.electron_count for system in result.systems), 6)

    def test_sulfone_has_two_orthogonal_two_electron_pi_systems(self):
        result = self.assign("CS(=O)(=O)C")

        self.assertEqual(result.status, "success")
        self.assertEqual(len(result.systems), 2)
        self.assertEqual([system.electron_count for system in result.systems], [2, 2])

    def test_sulfonamide_donor_couples_symmetrically_to_sulfur_orbitals(self):
        result = self.assign("CS(=O)(=O)N")
        donor = next(
            orbital
            for orbital, attrs in result.pi_graph.nodes(data=True)
            if attrs["element"] == "N"
        )
        sulfur_targets = [
            orbital
            for orbital in result.pi_graph.neighbors(donor)
            if result.pi_graph.nodes[orbital]["element"] == "S"
        ]

        self.assertEqual(result.status, "success")
        self.assertEqual(len(sulfur_targets), 2)
        self.assertEqual(
            {
                round(result.pi_graph.edges[donor, target]["coupling_scale"], 12)
                for target in sulfur_targets
            },
            {round(2**-0.5, 12)},
        )


if __name__ == "__main__":
    unittest.main()
