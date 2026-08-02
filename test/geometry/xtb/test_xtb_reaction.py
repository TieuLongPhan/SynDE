import unittest

from synde.geometry.xtb.xtb_reaction import XTBReaction


class TestXTBReaction(unittest.TestCase):
    def test_invalid_reaction_string_returns_zero_without_running_xtb(self) -> None:
        self.assertEqual(XTBReaction.delta_e_rsmi("not-a-reaction"), 0)
