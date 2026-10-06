"""b2 checks that run in CI: a tiny random BERT stands in for the real encoder weights."""
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
from transformers.utils import logging

from commerce.packages.contracts.ids import purchase_event_id
from commerce.packages.data_adapters.baskets import basket_from_event, customer_visits

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


SELLER = "ic-client-990101"


def item_id(n: int) -> str:
    """Item number n of a test catalog. IDs start at 990001, outside the raw Instacart range."""
    return "ic-p-%d" % (990000 + n)


def visits_of(customer, baskets, gaps=None):
    """baskets: list of item-number lists, one per visit; gaps default to 7 days."""
    events, day = [], 0.0
    for rank, items in enumerate(baskets, start=1):
        if rank > 1:
            day += (gaps or {}).get(rank, 7.0)
        basket = "ic-o-%s-%d" % (customer, rank)
        events.append({
            "schema_version": "purchase_event.v1", "seller_id": SELLER, "source": "instacart",
            "seller_partition": "synthetic_partition", "customer_id_local": "ic-user-%s" % customer,
            "basket_id_local": basket, "purchase_event_id": purchase_event_id(SELLER, "instacart", basket),
            "time": {"kind": "relative_day", "value": day}, "order_rank": rank,
            "items": [{"item_id_local": item_id(i), "quantity_observed": None} for i in sorted(set(items))],
        })
    return customer_visits(basket_from_event(e) for e in events)
