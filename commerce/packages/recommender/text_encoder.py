"""Frozen pretrained text encoder: product text -> z (model.md §2).

The weights never train: the module stays in eval mode with gradients off, and
nothing here is part of any FL export. Pooling follows the model's own rule, the
mean over non-padding tokens, then L2 normalisation. No special token is added;
the text builder's markers are ordinary text for the tokenizer.
"""
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from commerce.packages.contracts.ids import canonical_json


@dataclass(frozen=True)
class EncoderSpec:
    model_id: str
    revision: str  # the Hugging Face commit the local files came from
    max_length: int  # tokens, special tokens included
    prefix: str = ""  # prepended to every input; e5 models expect "query: "
    pooling: str = "mean"  # mean over non-padding tokens
    normalize: bool = True  # L2 after pooling


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_files(model_dir: Path) -> list[Path]:
    """Every file of the installed artifact, skipping the download tool's own metadata."""
    return sorted(p for p in model_dir.rglob("*")
                  if p.is_file() and ".cache" not in p.relative_to(model_dir).parts)


def artifact_hash(model_dir: str | Path, spec: EncoderSpec) -> str:
    """text_artifact_hash: per-file SHA-256 by relative path plus the encoder settings."""
    model_dir = Path(model_dir)
    files = [[p.relative_to(model_dir).as_posix(), _sha256(p)] for p in artifact_files(model_dir)]
    return hashlib.sha256(canonical_json({"files": files, "encoder": asdict(spec)})).hexdigest()


class FrozenTextEncoder:
    def __init__(self, model_dir: str | Path, spec: EncoderSpec):
        if spec.pooling != "mean":
            raise ValueError("only mean pooling is implemented")
        self.spec = spec
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.dim = int(self.model.config.hidden_size)

    def token_ids(self, texts: Sequence[str], *, max_length: int | None = None) -> list[list[int]]:
        """Input ids as the model sees them, special tokens included."""
        return self.tokenizer([self.spec.prefix + t for t in texts], truncation=max_length is not None,
                              max_length=max_length)["input_ids"]

    @torch.inference_mode()
    def encode(self, texts: Sequence[str], *, batch_size: int = 256) -> np.ndarray:
        # Similar lengths share a batch so padding stays short; rows come back in input order.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        z = np.zeros((len(texts), self.dim), dtype=np.float32)
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            batch = self.tokenizer([self.spec.prefix + texts[i] for i in rows], truncation=True,
                                   max_length=self.spec.max_length, padding=True, return_tensors="pt")
            hidden = self.model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            if self.spec.normalize:
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
            z[rows] = pooled.float().numpy()
        if not np.isfinite(z).all():
            raise ValueError("the encoder produced a non-finite vector")
        return z
