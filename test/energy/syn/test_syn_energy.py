# tests/test_syn_energy.py
import networkx as nx
from importlib.util import find_spec
import unittest
import warnings

from synde.energy.syn.params import SynParams
from synde.energy.syn.syn_energy import SynEnergy

HAS_RDKIT = find_spec("rdkit") is not None


def make_benzene_graph() -> nx.Graph:
    G = nx.Graph()
    for i in range(6):
        G.add_node(i, element="C", aromatic=True, hcount=1, partial_charge=0.0)
    for i in range(6):
        G.add_edge(i, (i + 1) % 6, order=1.0)
    return G


class TestSynEnergy(unittest.TestCase):
    def test_energy_beta_only_from_graph(self) -> None:
        se = SynEnergy(SynParams())
        G = make_benzene_graph()
        res = se.energy_from_graph(G)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["n_pi"], 6)
        # Toy PMO energy in β units for benzene with beta=-1 is -8.0
        self.assertAlmostEqual(res["energy_beta"], -8.0, places=6)
        # No conversions when beta_to_hartree was not provided
        self.assertIsNone(res["energy_Eh"])
        self.assertIsNone(res["energy_kJmol"])
        self.assertIsNone(res["energy_kcalmol"])
        self.assertIsNone(res["energy_eV"])

    def test_energy_with_conversions_from_graph(self) -> None:
        # Map β → -0.1 Eh; benzene energy_beta = -8 → -0.8 Eh
        se = SynEnergy(SynParams(), beta_to_hartree=-0.1)
        G = make_benzene_graph()
        res = se.energy_from_graph(G)
        self.assertEqual(res["status"], "success")
        self.assertAlmostEqual(res["energy_beta"], -8.0, places=6)
        self.assertAlmostEqual(res["energy_Eh"], -0.8, places=6)
        # Check derived units (within ~1 unit)
        self.assertAlmostEqual(
            res["energy_kJmol"],
            -0.8 * SynEnergy.HARTREE_TO_KJMOL,
            places=3,
        )
        self.assertAlmostEqual(
            res["energy_kcalmol"],
            -0.8 * SynEnergy.HARTREE_TO_KCALMOL,
            places=3,
        )
        self.assertAlmostEqual(
            res["energy_eV"],
            -0.8 * SynEnergy.HARTREE_TO_EV,
            places=6,
        )

    @unittest.skipUnless(HAS_RDKIT, "RDKit not available")
    def test_energy_from_smiles_with_mapping(self) -> None:
        # Same check through the SMILES entry point
        se = SynEnergy(SynParams(), beta_to_hartree=-0.1)
        res = se.energy_from_smiles("c1ccccc1", compute_gasteiger=False)
        self.assertEqual(res["status"], "success")
        self.assertAlmostEqual(res["energy_beta"], -8.0, places=6)
        self.assertAlmostEqual(res["energy_Eh"], -0.8, places=6)

    def test_repr_includes_mapping(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            se = SynEnergy(SynParams(), beta_to_hartree=-0.1234)
        r = repr(se)
        self.assertIn("beta_to_hartree", r)
        self.assertIn("0.1234", r)  # magnitude appears (sign may be shown as well)

    def test_legacy_conversion_warns(self) -> None:
        with self.assertWarns(DeprecationWarning):
            SynEnergy(SynParams(), beta_to_hartree=-0.1)


if __name__ == "__main__":
    unittest.main()
