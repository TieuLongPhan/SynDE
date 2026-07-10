import unittest
from synde.energy.syn.params import SynParams


class TestSynParams(unittest.TestCase):
    def test_defaults(self) -> None:
        p = SynParams()
        self.assertAlmostEqual(p.beta, -1.0)
        self.assertTrue(p.use_extended)
        self.assertEqual(p.top_pairs, 50)

    def test_mutability(self) -> None:
        p = SynParams(beta=-2.0, w_front=2.5, use_extended=False)
        self.assertAlmostEqual(p.beta, -2.0)
        self.assertAlmostEqual(p.w_front, 2.5)
        self.assertFalse(p.use_extended)

    def test_fields_exist(self) -> None:
        p = SynParams()
        # spot-check presence of a few critical fields
        for field in (
            "beta",
            "min_gap",
            "strength_plusM",
            "att_M",
            "use_extended",
            "w_coul_pi",
            "eps",
            "steric_k",
        ):
            self.assertTrue(hasattr(p, field))


if __name__ == "__main__":
    unittest.main()
