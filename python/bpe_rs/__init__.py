"""Byte-level BPE tokenizer written in Rust, compatible with tiktoken rank files.

>>> import bpe_rs
>>> enc = bpe_rs.get_encoding("cl100k_base")  # downloads the rank file once
>>> enc.encode("hello world")
[15339, 1917]
"""

from __future__ import annotations

import functools
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

from bpe_rs._bpe_rs import (
    CL100K_BASE_PATTERN,
    O200K_BASE_PATTERN,
    Encoding,
    __version__,
    _presets,
    train,
)

__all__ = [
    "CL100K_BASE_PATTERN",
    "O200K_BASE_PATTERN",
    "Encoding",
    "__version__",
    "cache_dir",
    "get_encoding",
    "list_encoding_names",
    "rank_file_path",
    "train",
]

_PRESETS = {name: (url, sha256) for name, url, sha256 in _presets()}


def cache_dir() -> Path:
    """Where downloaded rank files are kept.

    ``$BPE_RS_CACHE_DIR`` if set, otherwise ``$XDG_CACHE_HOME/bpe-rs``, otherwise
    ``~/.cache/bpe-rs``. The Rust crate and ``scripts/fetch_data.sh`` use the same rule.
    """
    if explicit := os.environ.get("BPE_RS_CACHE_DIR"):
        return Path(explicit)
    if xdg := os.environ.get("XDG_CACHE_HOME"):
        return Path(xdg) / "bpe-rs"
    return Path.home() / ".cache" / "bpe-rs"


def list_encoding_names() -> list[str]:
    """Names accepted by :func:`get_encoding`."""
    return list(_PRESETS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rank_file_path(name: str) -> Path:
    """Returns the cached rank file for a published encoding, downloading it if needed.

    The download is checked against the SHA-256 that tiktoken pins and written
    atomically, so an interrupted download never leaves a bad file in the cache.
    """
    try:
        url, expected = _PRESETS[name]
    except KeyError:
        known = ", ".join(_PRESETS)
        raise ValueError(f"unknown encoding {name!r}; expected one of: {known}") from None

    path = cache_dir() / f"{name}.tiktoken"
    if path.exists() and _sha256(path) == expected:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(f"hash mismatch for {url}: expected {expected}, got {actual}")
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{name}.", suffix=".part")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return path


@functools.cache
def get_encoding(name: str) -> Encoding:
    """Loads ``cl100k_base`` or ``o200k_base``, downloading the rank file on first use."""
    return Encoding.from_preset_file(name, rank_file_path(name))
