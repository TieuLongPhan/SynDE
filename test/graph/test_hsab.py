import unittest

from synde.graph import ComponentFrontier, hsab_descriptor, local_hsab_compatibility


class TestHSAB(unittest.TestCase):
    def test_descriptor_and_local_compatibility_are_finite(self) -> None:
        donor = hsab_descriptor(
            ComponentFrontier(0, (1,), -1.0, 1.0, {1: 0.5}, {1: 0.2})
        )
        acceptor = hsab_descriptor(
            ComponentFrontier(1, (2,), -2.0, 0.5, {2: 0.1}, {2: 0.4})
        )

        self.assertIsNotNone(donor.hardness)
        self.assertGreater(local_hsab_compatibility(donor, acceptor, 1, 2), 0.0)
