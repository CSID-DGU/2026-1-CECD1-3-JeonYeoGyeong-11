"""b2 checks that run in CI: a tiny random BERT stands in for the real encoder weights."""
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
from transformers.utils import logging

logging.disable_progress_bar()

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[", "]", ">", "NAME", "DESC", "CAT", "AISLE", "DEPT",
         "oat", "milk", "drink", "original", "1", "2", "L", "ml", "우유", "유기농", "두유", "식품"]


def write_tiny_encoder(model_dir: Path, *, seed: int = 0) -> Path:
    """Word-level tokenizer and a one-layer 16-dimension BERT, saved like a downloaded model."""
    tokenizer = Tokenizer(models.WordLevel(vocab={w: i for i, w in enumerate(VOCAB)}, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]",
                            cls_token="[CLS]", sep_token="[SEP]").save_pretrained(model_dir)
    torch.manual_seed(seed)
    BertModel(BertConfig(vocab_size=len(VOCAB), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=32, max_position_embeddings=64)).save_pretrained(model_dir)
    return model_dir
