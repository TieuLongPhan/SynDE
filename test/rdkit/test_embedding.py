import unittest
from rdkit import Chem
from synde.geometry.rdkit._embeddings import (
    Embeddings,
    EmbeddingConfig,
)


class TestEmbeddings(unittest.TestCase):
    def setUp(self):
        # Use a small molecule; do NOT add Hs here (the class ensures/does it)
        self.mol = Chem.MolFromSmiles("CCO")
        self.emb = Embeddings()  # defaults: ETKDGv3

    def test_embed_default(self):
        mol3d = self.emb.embed(self.mol, num_conformers=3)
        self.assertIsInstance(mol3d, Chem.Mol)
        self.assertGreaterEqual(mol3d.GetNumConformers(), 1)

    def test_embed_auto_num_confs(self):
        mol3d = self.emb.embed(self.mol, num_conformers="auto")
        self.assertGreaterEqual(mol3d.GetNumConformers(), 1)

    def test_embed_specific_method(self):
        # Try another preset method to ensure mapping works
        mol3d = self.emb.embed(self.mol, num_conformers=2, embedding_method="ETKDGv2")
        self.assertGreaterEqual(mol3d.GetNumConformers(), 1)

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            self.emb.embed(self.mol, num_conformers=2, embedding_method="NOT_A_METHOD")

    def test_invalid_num_conformers_raises(self):
        with self.assertRaises(ValueError):
            self.emb.embed(self.mol, num_conformers="bad")

        with self.assertRaises(ValueError):
            self.emb.embed(self.mol, num_conformers=0)

    def test_config_overrides(self):
        cfg = EmbeddingConfig(default_method="ETKDG", num_threads=2, random_seed=7)
        emb = Embeddings(cfg)
        mol3d = emb.embed(self.mol, num_conformers=2)
        self.assertGreaterEqual(mol3d.GetNumConformers(), 1)

    def test_legacy_static_wrapper(self):
        # Backward-compatible API
        mol3d = Embeddings.mol_embed(
            self.mol, num_conformers=2, embedding_method="ETDG"
        )
        self.assertGreaterEqual(mol3d.GetNumConformers(), 1)

    def test_output_has_hydrogens(self):
        # Class ensures hydrogens are present before embedding
        mol3d = self.emb.embed(self.mol, num_conformers=2)
        # At least some H atoms should be present
        h_count = sum(1 for a in mol3d.GetAtoms() if a.GetAtomicNum() == 1)
        self.assertGreater(h_count, 0)


if __name__ == "__main__":
    unittest.main()
