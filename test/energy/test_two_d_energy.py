import unittest

from synde.energy import MoleculeScorer
from synde.energy.two_d_energy import _sigma_bond_strength
from synde.graph import GraphBuilder


class TestTwoDEnergy(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = MoleculeScorer()

    def components(self, smiles: str) -> dict[str, float]:
        return self.scorer.score(GraphBuilder.from_smiles(smiles)).components

    def test_components_are_explicit_and_sum_to_score(self) -> None:
        result = self.scorer.score(GraphBuilder.from_smiles("c1ccccc1"))

        self.assertEqual(
            set(result.components),
            {
                "atom_reference",
                "sigma_bond_energy",
                "pi_stabilization",
                "cyclic_pi_correction",
                "ring_strain",
                "charge_electrostatic",
                "branching_stabilization",
                "steric_congestion",
            },
        )
        self.assertAlmostEqual(result.score, sum(result.components.values()))

    def test_small_ring_has_more_strain_than_six_membered_ring(self) -> None:
        cyclopropane = self.components("C1CC1")
        cyclohexane = self.components("C1CCCCC1")

        self.assertGreater(cyclopropane["ring_strain"], cyclohexane["ring_strain"])

    def test_antiaromatic_cycle_receives_destabilizing_correction(self) -> None:
        cyclobutadiene = self.components("C1=CC=C1")

        self.assertGreater(cyclobutadiene["cyclic_pi_correction"], 0.0)

    def test_sigma_term_rewards_stronger_bond_inventory(self) -> None:
        self.assertGreater(
            _sigma_bond_strength("C", "H"), _sigma_bond_strength("C", "N")
        )

    def test_cyclobutane_strain_is_similar_to_cyclopropane(self) -> None:
        cyclopropane = self.components("C1CC1")["ring_strain"]
        cyclobutane = self.components("C1CCC1")["ring_strain"]
        self.assertGreater(cyclobutane, 0.9 * cyclopropane)

    def test_moderate_branching_gets_a_stabilization_term(self) -> None:
        straight = self.components("CCCC")["branching_stabilization"]
        branched = self.components("CC(C)C")["branching_stabilization"]
        self.assertEqual(straight, 0.0)
        self.assertLess(branched, straight)
