# import unittest
# from synde.energy.xtb.xtb_reaction import XTBReaction


# class TestXTBReaction(unittest.TestCase):

#     def setUp(self):
#         self.xtb_reaction = XTBReaction(n_jobs=2, verbose=10)
#         self.test_rsmi = "C=C.[H][H]>>CC"
#         self.test_rsmi_2 = "C#C.[H][H].[H][H]>>CC"

#     def test_delta_e_rsmi(self):
#         # Test the calculation of delta E for a single reaction
#         delta_e = self.xtb_reaction.delta_e_rsmi(self.test_rsmi)
#         self.assertIsInstance(delta_e, float)

#     def test_delta_e_parallel(self):
#         test_rsmis = [self.test_rsmi, self.test_rsmi_2]
#         delta_es = self.xtb_reaction.delta_e_parallel(test_rsmis)
#         self.assertEqual(len(delta_es), 2)
#         self.assertTrue(all(isinstance(e, float) for e in delta_es))


# if __name__ == "__main__":
#     unittest.main()
