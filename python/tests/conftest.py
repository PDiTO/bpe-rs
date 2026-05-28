"""Shared fixtures.

The differential tests need tiktoken and the published rank files. When either is
unavailable (offline, tiktoken not installed) they are skipped, unless
BPE_RS_REQUIRE_DIFFERENTIAL=1 is set, in which case they fail. CI sets it so a network
hiccup can't quietly turn the most important tests into skips.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pytest

import bpe_rs

# Keep tiktoken's downloads next to ours so there is one cache to clear.
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(bpe_rs.cache_dir() / "tiktoken"))

ENCODING_NAMES = ["cl100k_base", "o200k_base"]


def _unavailable(reason: str) -> None:
    if os.environ.get("BPE_RS_REQUIRE_DIFFERENTIAL") == "1":
        pytest.fail(f"differential test prerequisites missing: {reason}")
    pytest.skip(reason)


@dataclass(frozen=True)
class Pair:
    """The same encoding loaded by both libraries."""

    name: str
    ours: bpe_rs.Encoding
    theirs: Any  # tiktoken.Encoding


_pairs: dict[str, Pair] = {}


def load_pair(name: str) -> Pair:
    if name in _pairs:
        return _pairs[name]
    try:
        import tiktoken
    except ImportError:
        _unavailable("tiktoken is not installed")
    try:
        ours = bpe_rs.get_encoding(name)
        theirs = tiktoken.get_encoding(name)
    except OSError as e:  # urllib's URLError and socket errors are OSErrors
        _unavailable(f"could not download {name}: {e}")
    _pairs[name] = Pair(name, ours, theirs)
    return _pairs[name]


@pytest.fixture(params=ENCODING_NAMES)
def pair(request: pytest.FixtureRequest) -> Pair:
    return load_pair(request.param)


@pytest.fixture(scope="session")
def cl100k() -> bpe_rs.Encoding:
    """Just our cl100k_base, for tests that don't compare against tiktoken."""
    try:
        return bpe_rs.get_encoding("cl100k_base")
    except OSError as e:
        pytest.skip(f"could not download cl100k_base: {e}")


# Count every string compared against tiktoken, and report it at the end of the run.
_compared: dict[str, int] = {}


def assert_same_tokens(pair: Pair, texts: list[str]) -> None:
    """Asserts both libraries encode every text identically, and counts the comparisons."""
    for text in texts:
        ours = pair.ours.encode_ordinary(text)
        theirs = pair.theirs.encode_ordinary(text)
        assert ours == theirs, (
            f"{pair.name} mismatch on {text[:200]!r}:\n  bpe_rs:   {ours}\n  tiktoken: {theirs}"
        )
    _compared[pair.name] = _compared.get(pair.name, 0) + len(texts)


def pytest_terminal_summary(terminalreporter: Any) -> None:
    if _compared:
        terminalreporter.section("differential test")
        for name, n in sorted(_compared.items()):
            terminalreporter.write_line(f"{name}: {n} strings encoded identically to tiktoken")
