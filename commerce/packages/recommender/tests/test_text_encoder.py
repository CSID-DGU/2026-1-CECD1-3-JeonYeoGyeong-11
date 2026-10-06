import dataclasses
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from commerce.packages.recommender.tests import VOCAB, write_tiny_encoder
from commerce.packages.recommender.text_encoder import EncoderSpec, FrozenTextEncoder, artifact_hash

SPEC = EncoderSpec("tiny-test-bert", "0", max_length=12)
SHORT = "[NAME] oat drink"
LONG = "[NAME] oat drink original 2 L [CAT] 식품 > 두유"


def weights_digest(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class TinyModel(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = write_tiny_encoder(Path(directory.name))
        self.encoder = FrozenTextEncoder(self.dir, SPEC)


class Frozen(TinyModel):
    def test_no_parameter_trains_and_eval_mode_holds(self):
        self.assertFalse(self.encoder.model.training)
        self.assertEqual(sum(p.requires_grad for p in self.encoder.model.parameters()), 0)

    def test_encoding_leaves_weights_and_output_unchanged(self):
        before = weights_digest(self.encoder.model)
        first = self.encoder.encode([SHORT, LONG])
        second = self.encoder.encode([SHORT, LONG])
        self.assertEqual(weights_digest(self.encoder.model), before)
        np.testing.assert_array_equal(first, second)

    def test_no_graph_is_built(self):
        batch = self.encoder.tokenizer([SHORT], return_tensors="pt")
        with torch.inference_mode():
            hidden = self.encoder.model(**batch).last_hidden_state
        self.assertFalse(hidden.requires_grad)


class Pooling(TinyModel):
    def test_padding_does_not_move_a_vector(self):
        alone = self.encoder.encode([SHORT])[0]
        padded = self.encoder.encode([SHORT, LONG], batch_size=2)[0]  # SHORT is padded to LONG here
        np.testing.assert_allclose(alone, padded, atol=1e-6)

    def test_unit_length_float32_and_input_order(self):
        z = self.encoder.encode([LONG, SHORT, LONG])
        self.assertEqual((z.shape, z.dtype), ((3, 16), np.float32))
        np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1.0, atol=1e-6)
        np.testing.assert_array_equal(z[0], z[2])
        self.assertFalse(np.allclose(z[0], z[1]))

    def test_inputs_are_cut_at_max_length(self):
        self.assertGreater(len(self.encoder.token_ids([LONG])[0]), SPEC.max_length)
        cut = self.encoder.token_ids([LONG], max_length=SPEC.max_length)[0]
        self.assertEqual(len(cut), SPEC.max_length)
        self.assertEqual(cut[-1], self.encoder.tokenizer.sep_token_id)  # the closing token survives

    def test_non_finite_output_is_refused(self):
        with torch.no_grad():
            self.encoder.model.embeddings.word_embeddings.weight.fill_(float("nan"))
        with self.assertRaises(ValueError):
            self.encoder.encode([SHORT])

    def test_prefix_reaches_the_tokenizer(self):
        prefixed = FrozenTextEncoder(self.dir, dataclasses.replace(SPEC, prefix="milk "))
        self.assertEqual(prefixed.token_ids([SHORT])[0][1], VOCAB.index("milk"))



class ArtifactHash(TinyModel):
    def test_stable_and_ignores_download_metadata(self):
        first = artifact_hash(self.dir, SPEC)
        (self.dir / ".cache").mkdir()
        (self.dir / ".cache" / "note").write_text("download tool metadata", encoding="utf-8")
        self.assertEqual(artifact_hash(self.dir, SPEC), first)

    def test_changes_with_any_file_or_setting(self):
        base = artifact_hash(self.dir, SPEC)
        self.assertNotEqual(artifact_hash(self.dir, dataclasses.replace(SPEC, max_length=13)), base)
        self.assertNotEqual(artifact_hash(self.dir, dataclasses.replace(SPEC, prefix="query: ")), base)
        config = self.dir / "config.json"
        config.write_text(config.read_text(encoding="utf-8") + " ", encoding="utf-8")
        self.assertNotEqual(artifact_hash(self.dir, SPEC), base)
