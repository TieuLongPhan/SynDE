import unittest

from synde.graph import GraphBuilder
from synde.energy.local_energy import local_score_components


class TestLocalEnergy(unittest.TestCase):
    def test_components_are_explicit_and_finite(self) -> None:
        components = local_score_components(GraphBuilder.from_smiles("CCO").graph)

        self.assertEqual(
            set(components),
            {"local_atom", "local_bond", "formal_charge", "ring_topology"},
        )
        self.assertTrue(all(isinstance(value, float) for value in components.values()))
