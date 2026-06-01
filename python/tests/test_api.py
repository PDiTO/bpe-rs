"""The Python API itself: arguments, errors, file handling. Needs cl100k_base only."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

import bpe_rs

TEXTS = ["hello world", "Tokenizers aren't magic 🙂", "  indented\n\tcode()\n", "中文", ""]


@pytest.mark.parametrize("text", TEXTS)
def test_round_trip(cl100k: bpe_rs.Encoding, text: str) -> None:
    tokens = cl100k.encode(text)
    assert cl100k.decode(tokens) == text
    assert cl100k.decode_bytes(tokens) == text.encode()
    assert b"".join(cl100k.decode_single_token_bytes(t) for t in tokens) == text.encode()


def test_known_ids(cl100k: bpe_rs.Encoding) -> None:
    assert cl100k.encode("hello world") == [15339, 1917]
    assert cl100k.name == "cl100k_base"
    assert cl100k.n_vocab == 100277
    assert cl100k.special_tokens["<|endoftext|>"] == 100257
    assert repr(cl100k) == '<Encoding "cl100k_base">'


def test_special_token_arguments(cl100k: bpe_rs.Encoding) -> None:
    text = "a<|endoftext|>b<|fim_prefix|>"
    with pytest.raises(ValueError, match="disallowed special token"):
        cl100k.encode(text)
    everything = cl100k.encode(text, allowed_special="all")
    assert 100257 in everything and 100258 in everything
    for allowed in ({"<|endoftext|>"}, ["<|endoftext|>"], frozenset({"<|endoftext|>"})):
        tokens = cl100k.encode(text, allowed_special=allowed, disallowed_special=())
        assert 100257 in tokens and 100258 not in tokens
    # <|fim_prefix|> is still disallowed by default when only <|endoftext|> is allowed.
    with pytest.raises(ValueError):
        cl100k.encode(text, allowed_special={"<|endoftext|>"})
    with pytest.raises(ValueError, match='expected "all"'):
        cl100k.encode(text, allowed_special="everything")
    with pytest.raises(TypeError):
        cl100k.encode(text, allowed_special=42)


def test_decode_errors(cl100k: bpe_rs.Encoding) -> None:
    # The first 256 cl100k ids are the single bytes. 0xC3 on its own is the first half
    # of a two-byte character like "é", so it is not valid UTF-8.
    lead_byte = next(t for t in range(256) if cl100k.decode_single_token_bytes(t) == b"\xc3")
    assert cl100k.decode([lead_byte]) == "\ufffd"
    assert cl100k.decode([lead_byte], errors="ignore") == ""
    with pytest.raises(UnicodeDecodeError):
        cl100k.decode([lead_byte], errors="strict")
    assert cl100k.decode(cl100k.encode("é"), errors="strict") == "é"
    with pytest.raises(KeyError):
        cl100k.decode_single_token_bytes(100261)  # a gap between special tokens
    with pytest.raises(KeyError):
        cl100k.decode([1, 2, 10**9])
    with pytest.raises(OverflowError):
        cl100k.decode([-1])


def test_huge_whitespace_run_raises_instead_of_panicking(cl100k: bpe_rs.Encoding) -> None:
    # fancy-regex can't split a run of ~1M spaces (tiktoken raises a PanicException).
    with pytest.raises(ValueError, match="pre-tokenization failed at byte 5"):
        cl100k.encode("hello" + " " * 1_100_000 + "x")


def test_split_shows_the_pieces(cl100k: bpe_rs.Encoding) -> None:
    assert cl100k.split("I'm here  now") == ["I", "'m", " here", " ", " now"]


def test_batch(cl100k: bpe_rs.Encoding) -> None:
    expected = [cl100k.encode_ordinary(t) for t in TEXTS]
    assert cl100k.encode_ordinary_batch(TEXTS) == expected
    assert cl100k.encode_ordinary_batch(TEXTS, num_threads=2) == expected
    assert cl100k.encode_batch(TEXTS, num_threads=1) == expected
    assert cl100k.encode_ordinary_batch([]) == []
    with pytest.raises(ValueError, match="num_threads"):
        cl100k.encode_ordinary_batch(TEXTS, num_threads=0)


def test_threads_share_one_encoding(cl100k: bpe_rs.Encoding) -> None:
    text = "The quick brown fox jumps over the lazy dog. " * 200
    expected = cl100k.encode(text)
    results: list[list[int]] = []

    def work() -> None:
        results.append(cl100k.encode(text))

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [expected] * 8


def test_save_and_load(cl100k: bpe_rs.Encoding, tmp_path: Path) -> None:
    path = tmp_path / "copy.tiktoken"
    cl100k.save(path)
    assert path.read_bytes() == bpe_rs.rank_file_path("cl100k_base").read_bytes()

    loaded = bpe_rs.Encoding.from_tiktoken_file(
        path, "cl100k_base", special_tokens=cl100k.special_tokens
    )
    assert loaded.name == "copy"
    text = "same vocabulary, same tokens <|endoftext|>"
    assert loaded.encode(text, allowed_special="all") == cl100k.encode(text, allowed_special="all")


def test_load_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        bpe_rs.Encoding.from_tiktoken_file(tmp_path / "missing.tiktoken")
    bad = tmp_path / "bad.tiktoken"
    bad.write_text("IQ== 0\nnot base64!! 1\n")
    with pytest.raises(ValueError, match="line 2"):
        bpe_rs.Encoding.from_tiktoken_file(bad)
    incomplete = tmp_path / "incomplete.tiktoken"
    incomplete.write_text("IQ== 0\n")
    with pytest.raises(ValueError, match="every single byte"):
        bpe_rs.Encoding.from_tiktoken_file(incomplete)


def test_get_encoding(cl100k: bpe_rs.Encoding) -> None:
    assert bpe_rs.get_encoding("cl100k_base") is cl100k
    assert bpe_rs.list_encoding_names() == ["cl100k_base", "o200k_base"]
    with pytest.raises(ValueError, match="unknown encoding"):
        bpe_rs.get_encoding("gpt2")


def test_cache_dir_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BPE_RS_CACHE_DIR", str(tmp_path))
    assert bpe_rs.cache_dir() == tmp_path
    monkeypatch.delenv("BPE_RS_CACHE_DIR")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert bpe_rs.cache_dir() == tmp_path / "bpe-rs"


def test_patterns_are_exported() -> None:
    assert r"\p{N}{1,3}+" in bpe_rs.CL100K_BASE_PATTERN
    assert r"[\r\n/]*" in bpe_rs.O200K_BASE_PATTERN
