"""bpe_rs must produce exactly the same token ids as tiktoken.

Runs for cl100k_base and o200k_base over hand-written edge cases, real source code and
seeded random strings, and also checks special-token handling, decoding and metadata.
"""

from __future__ import annotations

import random

import pytest
from conftest import Pair, assert_same_tokens
from corpus import GENERATORS, HANDWRITTEN, generated, real_text

SEEDS = range(10)
STRINGS_PER_BATCH = 100


@pytest.mark.parametrize("text", HANDWRITTEN, ids=[f"hand{i:03d}" for i in range(len(HANDWRITTEN))])
def test_handwritten(pair: Pair, text: str) -> None:
    assert_same_tokens(pair, [text])


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("kind", list(GENERATORS))
def test_generated(pair: Pair, kind: str, seed: int) -> None:
    assert_same_tokens(pair, generated(kind, STRINGS_PER_BATCH, seed))


@pytest.fixture(scope="module")
def real_chunks() -> list[str]:
    chunks = real_text()
    assert len(chunks) > 100
    return chunks


def test_real_source_code(pair: Pair, real_chunks: list[str]) -> None:
    assert_same_tokens(pair, real_chunks)


def test_batch_matches_tiktoken(pair: Pair, real_chunks: list[str]) -> None:
    texts = HANDWRITTEN + real_chunks + generated("prose", 500, seed=99)
    ours = pair.ours.encode_ordinary_batch(texts)
    assert ours == pair.theirs.encode_ordinary_batch(texts)
    assert pair.ours.encode_batch(texts, disallowed_special=(), num_threads=3) == ours


SPECIAL_TEXTS = [
    "<|endoftext|>",
    "hello<|endoftext|>world",
    "<|endoftext|><|endoftext|>",
    " <|endoftext|> ",
    "a\n<|endoftext|>\nb",
    "<|endofprompt|>tail",
    "<|endoftext|<|endoftext|>",
    "<|fim_prefix|>def f():<|fim_suffix|>    return 1<|fim_middle|>",
]


@pytest.mark.parametrize("text", SPECIAL_TEXTS)
def test_special_tokens(pair: Pair, text: str) -> None:
    ours, theirs = pair.ours, pair.theirs

    assert ours.encode(text, allowed_special="all") == theirs.encode(text, allowed_special="all")

    only_eot = {"<|endoftext|>"}
    assert ours.encode(text, allowed_special=only_eot, disallowed_special=()) == theirs.encode(
        text, allowed_special=only_eot, disallowed_special=()
    )
    assert ours.encode(text, disallowed_special=()) == theirs.encode(text, disallowed_special=())
    assert ours.encode(text, disallowed_special=None) == theirs.encode(
        text, disallowed_special=None
    )

    # Default arguments: any special-token text is an error, in both libraries.
    def outcome(enc) -> list[int] | type[Exception]:
        try:
            return enc.encode(text)
        except ValueError:
            return ValueError

    assert outcome(ours) == outcome(theirs)


@pytest.mark.parametrize("disallowed", [{"hello"}, ["wor", "zzz"], {"<|endoftext|>"}])
def test_disallowing_arbitrary_strings(pair: Pair, disallowed: set[str] | list[str]) -> None:
    # tiktoken raises for any disallowed string, not only for real special tokens.
    for text in ["hello world", "say hello", "nothing to see", "<|endoftext|>"]:
        outcomes = []
        for enc in (pair.ours, pair.theirs):
            try:
                outcomes.append(enc.encode(text, disallowed_special=disallowed))
            except ValueError:
                outcomes.append("ValueError")
        assert outcomes[0] == outcomes[1], text


def test_special_token_batches(pair: Pair) -> None:
    kwargs = {"allowed_special": "all"}
    assert pair.ours.encode_batch(SPECIAL_TEXTS, **kwargs) == pair.theirs.encode_batch(
        SPECIAL_TEXTS, **kwargs
    )
    with pytest.raises(ValueError):
        pair.ours.encode_batch(SPECIAL_TEXTS)


@pytest.mark.parametrize("text", ["a\ud800b", "\udc00", "😀", "x\ud83dy", "\ud800\ud800"], ids=repr)
def test_lone_surrogates_are_replaced_like_tiktoken(pair: Pair, text: str) -> None:
    assert pair.ours.encode(text) == pair.theirs.encode(text)


def test_metadata(pair: Pair) -> None:
    assert pair.ours.n_vocab == pair.theirs.n_vocab
    assert pair.ours.max_token_value == pair.theirs.max_token_value
    assert pair.ours.special_tokens_set == pair.theirs.special_tokens_set
    assert pair.ours.pattern == pair.theirs._pat_str


def test_every_token_decodes_to_the_same_bytes(pair: Pair) -> None:
    for token in range(pair.theirs.n_vocab):
        try:
            expected = pair.theirs.decode_single_token_bytes(token)
        except KeyError:
            with pytest.raises(KeyError):
                pair.ours.decode_single_token_bytes(token)
            continue
        assert pair.ours.decode_single_token_bytes(token) == expected, token


def test_decoding_random_token_sequences(pair: Pair) -> None:
    rng = random.Random(7)
    valid = list(pair.theirs._mergeable_ranks.values())
    for _ in range(2000):
        tokens = rng.choices(valid, k=rng.randint(1, 20))
        assert pair.ours.decode_bytes(tokens) == pair.theirs.decode_bytes(tokens)
        # Random tokens often cut UTF-8 sequences apart, so this exercises the
        # replacement rules, not just the happy path.
        assert pair.ours.decode(tokens) == pair.theirs.decode(tokens)
        assert pair.ours.decode(tokens, errors="ignore") == pair.theirs.decode(
            tokens, errors="ignore"
        )
