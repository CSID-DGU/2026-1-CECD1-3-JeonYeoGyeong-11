"""npz payload codec for synthetic rounds. Pure functions: no HTTP, no state.

Rules are in docs/design/interfaces.md §5. The transfer length of an npz (ZIP and
npy headers included) is not the tensor length sum(prod(shape) * 4); both limits
are checked separately, and entries are inspected before any array is read.
"""
import io
import math
import zipfile
from typing import Mapping, Sequence

import numpy as np

from commerce.packages.contracts.errors import ContractError
from commerce.packages.contracts.types import TensorMap

MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
_NPY_HEADER_SLACK = 1024  # per-entry allowance for the npy header over the raw tensor bytes


class PayloadTooLarge(ValueError):
    """Transfer larger than MAX_PAYLOAD_BYTES. HTTP 413 at the edge; not a wire error code."""


def tensor_nbytes(specs: Sequence[Mapping]) -> int:
    return sum(math.prod(spec["shape"]) * 4 for spec in specs)


def encode_npz(tensors: TensorMap) -> bytes:
    buffer = io.BytesIO()
    np.savez(buffer, **tensors)
    return buffer.getvalue()


def _read_header(archive: zipfile.ZipFile, info: zipfile.ZipInfo):
    with archive.open(info) as fp:
        version = np.lib.format.read_magic(fp)
        if version == (1, 0):
            return np.lib.format.read_array_header_1_0(fp)
        if version == (2, 0):
            return np.lib.format.read_array_header_2_0(fp)
    raise ContractError("TENSOR_SET_MISMATCH")


def decode_npz(data: bytes, specs: Sequence[Mapping], *, max_nbytes: int = MAX_PAYLOAD_BYTES) -> TensorMap:
    """Return the arrays of `data`, which must be exactly the float32 tensors of `specs`."""
    if len(data) > max_nbytes:
        raise PayloadTooLarge("payload exceeds the transfer limit")
    expected = {spec["name"]: tuple(spec["shape"]) for spec in specs}
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ContractError("SCHEMA_INVALID") from None
    with archive:
        infos = archive.infolist()
        # Count and declared sizes are checked before anything is decompressed or parsed.
        if len(infos) != len(expected):
            raise ContractError("TENSOR_SET_MISMATCH")
        entries = {}
        for info in infos:
            name = info.filename[:-4] if info.filename.endswith(".npy") else None
            if name not in expected or name in entries:
                raise ContractError("TENSOR_SET_MISMATCH")
            if info.compress_type != zipfile.ZIP_STORED:
                raise ContractError("SCHEMA_INVALID")
            if info.file_size > math.prod(expected[name]) * 4 + _NPY_HEADER_SLACK:
                raise ContractError("TENSOR_SET_MISMATCH")
            entries[name] = info
        arrays: TensorMap = {}
        try:
            for name, info in entries.items():
                shape, _fortran, dtype = _read_header(archive, info)
                if dtype != np.dtype("<f4") or tuple(shape) != expected[name]:
                    raise ContractError("TENSOR_SET_MISMATCH")
                with archive.open(info) as fp:
                    array = np.lib.format.read_array(fp, allow_pickle=False)
                if not np.isfinite(array).all():
                    raise ContractError("SCHEMA_INVALID")
                arrays[name] = array
        except (ValueError, OSError, EOFError, zipfile.BadZipFile):
            raise ContractError("SCHEMA_INVALID") from None
    return arrays
