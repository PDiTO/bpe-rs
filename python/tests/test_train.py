"""Training from Python. Runs offline."""

from __future__ import annotations

from pathlib import Path

import pytest

import bpe_rs

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "The lazy dog sleeps; the quick fox doesn't.",
    "def tokenize(text):\n    return text.split()\n",
    "Numbers like 12345 and 2026 show up too, as does ünïcödé and 日本語.",
] * 25


def test_trained_encoding_round_trips() -> None:
    enc = bpe_rs.train(CORPUS, 400)
    assert 256 < enc.n_vocab <= 400
    for text in [*CORPUS[:4], "unseen text 🙂 with new words", ""]:
        assert enc.decode(enc.encode(text)) == text


def test_training_compresses_the_training_text() -> None:
    enc = bpe_rs.train(CORPUS, 500)
    text = CORPUS[0]
    assert len(enc.encode(text)) < len(text.encode()) / 3


def test_single_string_and_list_inputs() -> None:
    text = "".join(CORPUS[:4])
    assert bpe_rs.train(text, 300).encode(text) == bpe_rs.train([text], 300).encode(text)


def test_training_is_deterministic() -> None:
    a = bpe_rs.train(CORPUS, 450)
    b = bpe_rs.train(CORPUS, 450)
    probe = " ".join(CORPUS[:4])
    assert a.encode(probe) == b.encode(probe)
    assert [a.decode_single_token_bytes(t) for t in range(256, a.n_vocab)] == [
        b.decode_single_token_bytes(t) for t in range(256, b.n_vocab)
    ]


def test_first_merge_is_the_most_frequent_pair() -> None:
    # One piece per document, so pair counts are easy to reason about.
    enc = bpe_rs.train(["xyxyxy", "xyz", "ab"], 257, pattern=r"[\s\S]+")
    assert enc.decode_single_token_bytes(256) == b"xy"


def test_special_tokens_follow_the_learned_vocab() -> None:
    enc = bpe_rs.train(CORPUS, 300, special_tokens=["<|bos|>", "<|eos|>"])
    assert enc.special_tokens == {"<|bos|>": 300, "<|eos|>": 301}
    assert enc.encode("<|bos|>hi<|eos|>", allowed_special="all")[0] == 300


def test_pattern_by_name_or_regex() -> None:
    by_name = bpe_rs.train(CORPUS, 300, pattern="o200k_base")
    assert by_name.pattern == bpe_rs.O200K_BASE_PATTERN
    custom = bpe_rs.train(CORPUS, 300, pattern=r"\S+|\s+", name="words")
    assert custom.name == "words"
    assert custom.split("a  b") == ["a", "  ", "b"]


def test_save_and_reload(tmp_path: Path) -> None:
    enc = bpe_rs.train(CORPUS, 350)
    path = tmp_path / "trained.tiktoken"
    enc.save(path)
    assert len(path.read_text().splitlines()) == enc.n_vocab
    loaded = bpe_rs.Encoding.from_tiktoken_file(path)
    probe = "The quick brown fox, again."
    assert loaded.encode(probe) == enc.encode(probe)


def test_min_frequency_stops_early() -> None:
    enc = bpe_rs.train(["abc abd"], 10_000, min_frequency=2)
    # "ab" is the only pair that occurs twice, so it is the only merge.
    assert enc.n_vocab == 257
    assert enc.decode_single_token_bytes(256) == b"ab"


def test_bad_arguments() -> None:
    with pytest.raises(ValueError, match="at least 256"):
        bpe_rs.train(CORPUS, 100)
    with pytest.raises(ValueError, match="pattern"):
        bpe_rs.train(CORPUS, 300, pattern="(")
    with pytest.raises(TypeError):
        bpe_rs.train([1, 2, 3], 300)
