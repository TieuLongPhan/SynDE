import unittest

from synde.graph import ComponentFrontier, directional_fmo


class TestFrontier(unittest.TestCase):
    def test_directional_fmo_uses_local_densities(self) -> None:
        donor = ComponentFrontier(0, (1,), -1.0, 1.0, {1: 0.5}, {1: 0.2})
        acceptor = ComponentFrontier(1, (2,), -2.0, 0.5, {2: 0.1}, {2: 0.4})

        result = directional_fmo(donor, acceptor, 1, 2)

        self.assertTrue(result.valid)
        self.assertGreater(result.score, 0.0)
