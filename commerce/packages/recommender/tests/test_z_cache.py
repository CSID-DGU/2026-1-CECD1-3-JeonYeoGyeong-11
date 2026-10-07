import dataclasses
from pathlib import Path
import tempfile
import unittest

import numpy as np

from commerce.packages.recommender.tests import write_tiny_encoder
from commerce.packages.recommender.text_encoder import EncoderSpec, FrozenTextEncoder
from commerce.packages.recommender.z_cache import ZCache, preprocessing_version

SPEC = EncoderSpec("tiny-test-bert", "0", max_length=12)
TEXTS = ["[NAME] oat drink 1 L", "[NAME] oat drink 2 L", "[NAME] 유기농 우유 1 L"]


class Cache(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.model_dir = write_tiny_encoder(self.root / "model")
        self.encoder = FrozenTextEncoder(self.model_dir, SPEC)

    def cache(self, name="z.sqlite", **kwargs):
        cache = ZCache(self.root / name, self.encoder, self.model_dir, **kwargs)
        self.addCleanup(cache.close)
        return cache

    def test_hits_skip_the_encoder_and_match_it(self):
        cache = self.cache()
        first = cache.vectors(TEXTS)
        self.assertEqual(cache.encoded, 3)
        again = cache.vectors(list(reversed(TEXTS)))
        self.assertEqual(cache.encoded, 3)
        np.testing.assert_array_equal(again[::-1], first)  # a hit returns the stored bytes
        # A fresh encode batches the texts differently; padding shape moves the last bits.
        np.testing.assert_allclose(first, self.encoder.encode(TEXTS), atol=1e-6)

    def test_survives_reopening(self):
        self.cache().vectors(TEXTS)
        reopened = self.cache()
        reopened.vectors(TEXTS)
        self.assertEqual(reopened.encoded, 0)

    def test_an_edited_text_is_the_only_miss(self):
        cache = self.cache()
        cache.vectors(TEXTS)
        cache.vectors(TEXTS[:2] + ["[NAME] 유기농 우유 2 L"])
        self.assertEqual(cache.encoded, 4)

    def test_duplicates_in_one_call_encode_once(self):
        cache = self.cache()
        z = cache.vectors([TEXTS[0], TEXTS[0]])
        self.assertEqual(cache.encoded, 1)
        np.testing.assert_array_equal(z[0], z[1])

    def test_new_artifact_or_preprocessing_misses(self):
        self.cache().vectors(TEXTS)
        other = self.cache(preprocessing="another builder")
        other.vectors(TEXTS)
        self.assertEqual(other.encoded, 3)
        longer = FrozenTextEncoder(self.model_dir, dataclasses.replace(SPEC, max_length=13))
        changed = ZCache(self.root / "z.sqlite", longer, self.model_dir)
        self.addCleanup(changed.close)
        changed.vectors(TEXTS)
        self.assertEqual(changed.encoded, 3)

    def test_empty_request(self):
        self.assertEqual(self.cache().vectors([]).shape, (0, 16))

    def test_the_cache_keys_on_the_preprocessing_version(self):
        self.assertEqual(self.cache().preprocessing_version, preprocessing_version())
