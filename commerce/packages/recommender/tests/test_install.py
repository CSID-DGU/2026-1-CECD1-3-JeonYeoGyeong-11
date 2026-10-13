"""install.py on a made-up encoder folder and a tiny release (no model download)."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from commerce.packages.recommender.install import InstallError, install_encoder, install_release_dir
from commerce.packages.recommender.seller_runtime import SellerRuntime
from commerce.packages.recommender.serving import SERVICE_ENCODER, canonical_npz, write_bundle
from commerce.packages.recommender.tests.test_seller_runtime import TINY, FakeText, release_for
from commerce.packages.recommender.text_encoder import artifact_hash


class EncoderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "download"
        (self.source / "1_Pooling").mkdir(parents=True)
        (self.source / "config.json").write_text('{"hidden_size": 8}', encoding="utf-8")
        (self.source / "tokenizer.json").write_text("{}", encoding="utf-8")
        (self.source / "1_Pooling" / "config.json").write_text("{}", encoding="utf-8")
        (self.source / ".cache").mkdir()
        (self.source / ".cache" / "download.lock").write_text("tool metadata", encoding="utf-8")
        self.expected = artifact_hash(self.source, SERVICE_ENCODER)

    def test_the_service_files_go_to_frozen_text_once(self):
        model_dir = self.root / "models"
        self.assertTrue(install_encoder(self.source, model_dir, expected_hash=self.expected).startswith("installed"))
        target = model_dir / "frozen_text"
        self.assertEqual(artifact_hash(target, SERVICE_ENCODER), self.expected)
        self.assertFalse((target / ".cache").exists())  # the download tool's files stay behind
        self.assertEqual(install_encoder(self.source, model_dir, expected_hash=self.expected), "already installed")

    def test_other_files_are_refused_and_nothing_changes(self):
        model_dir = self.root / "models"
        with self.assertRaises(InstallError):
            install_encoder(self.source, model_dir, expected_hash="0" * 64)
        self.assertFalse((model_dir / "frozen_text").exists())
        other = model_dir / "frozen_text"
        other.mkdir(parents=True)
        (other / "config.json").write_text('{"hidden_size": 4}', encoding="utf-8")
        with self.assertRaises(InstallError):
            install_encoder(self.source, model_dir, expected_hash=self.expected)
        self.assertEqual((other / "config.json").read_text(encoding="utf-8"), '{"hidden_size": 4}')
        self.assertEqual([p.name for p in model_dir.iterdir()], ["frozen_text"])  # no staging left


class ReleaseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        maker = SellerRuntime("maker", self.root / "m.sqlite", self.root / "maker", text=FakeText(), architectures=TINY)
        self.release, self.manifest, self.tensors = release_for(maker, "text_relation", "ic100-test")
        data = canonical_npz(self.tensors, [t["name"] for t in self.manifest["tensors"]])
        write_bundle(self.root / "release", self.release, self.manifest, data)

    def test_a_release_folder_installs_as_the_serving_base(self):
        model_dir = self.root / "seller"
        version = install_release_dir(model_dir, self.root / "release", "text_relation", text=FakeText(),
                                      architectures=TINY)
        self.assertEqual(version, "ic100-test")
        seller = SellerRuntime("seller-1", self.root / "features.sqlite", model_dir, text=FakeText(),
                               architectures=TINY)
        served = seller.export_shared_state(model_variant="text_relation")
        self.assertTrue(all(np.array_equal(served[k], self.tensors[k]) for k in self.tensors))

    def test_a_release_for_the_other_variant_is_refused(self):
        with self.assertRaises(InstallError):
            install_release_dir(self.root / "seller", self.root / "release", "text_only", text=FakeText(),
                                architectures=TINY)


if __name__ == "__main__":
    unittest.main()
