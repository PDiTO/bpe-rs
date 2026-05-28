"""Type stubs for the compiled extension module."""

from collections.abc import Collection, Sequence
from os import PathLike
from typing import Literal, final

__version__: str
CL100K_BASE_PATTERN: str
O200K_BASE_PATTERN: str

@final
class Encoding:
    """A byte-level BPE encoding: split pattern, mergeable ranks and special tokens.

    Encoding and decoding release the GIL. The batch methods run on a Rust thread pool.
    """

    @staticmethod
    def from_tiktoken_file(
        path: str | PathLike[str],
        pattern: str = ...,
        special_tokens: dict[str, int] | None = None,
        name: str | None = None,
    ) -> Encoding:
        """Loads a tiktoken rank file.

        ``pattern`` is a split regex, or the name of a published encoding
        (``"cl100k_base"``, ``"o200k_base"``) to use its pattern. It defaults to the
        cl100k_base pattern. ``name`` defaults to the file name without its extension.
        """

    @staticmethod
    def from_preset_file(name: str, path: str | PathLike[str]) -> Encoding:
        """Loads a published encoding's rank file with its pattern and special tokens."""

    @property
    def name(self) -> str: ...
    @property
    def pattern(self) -> str: ...
    @property
    def n_vocab(self) -> int:
        """Largest token id plus one, special tokens included (same as tiktoken)."""

    @property
    def max_token_value(self) -> int: ...
    @property
    def special_tokens(self) -> dict[str, int]: ...
    @property
    def special_tokens_set(self) -> frozenset[str]: ...
    def encode_ordinary(self, text: str) -> list[int]:
        """Encodes text, treating special-token text as plain text."""

    def encode(
        self,
        text: str,
        *,
        allowed_special: Literal["all"] | Collection[str] = ...,
        disallowed_special: Literal["all"] | Collection[str] = "all",
    ) -> list[int]:
        """Encodes text with tiktoken's special-token rules.

        By default any special-token text raises ``ValueError``. Tokens in
        ``allowed_special`` are encoded as their special id; pass
        ``disallowed_special=()`` to encode special-token text as plain text instead.
        """

    def encode_ordinary_batch(
        self, texts: Sequence[str], *, num_threads: int | None = None
    ) -> list[list[int]]:
        """Encodes many texts in parallel. ``num_threads=None`` uses every core."""

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        num_threads: int | None = None,
        allowed_special: Literal["all"] | Collection[str] = ...,
        disallowed_special: Literal["all"] | Collection[str] = "all",
    ) -> list[list[int]]: ...
    def decode_bytes(self, tokens: Sequence[int]) -> bytes: ...
    def decode(self, tokens: Sequence[int], errors: str = "replace") -> str:
        """Decodes tokens to text. Invalid UTF-8 is handled according to ``errors``."""

    def decode_single_token_bytes(self, token: int) -> bytes:
        """Raises ``KeyError`` for ids that are not in the vocabulary."""

    def split(self, text: str) -> list[str]:
        """Splits text with the pre-tokenization regex, without merging."""

    def save(self, path: str | PathLike[str]) -> None:
        """Writes the mergeable ranks as a tiktoken rank file.

        Special tokens and the pattern are not part of that format.
        """

def train(
    texts: str | Sequence[str],
    vocab_size: int,
    *,
    pattern: str = ...,
    special_tokens: Sequence[str] = (),
    min_frequency: int = 1,
    name: str = "trained",
) -> Encoding:
    """Learns a byte-level BPE vocabulary of up to ``vocab_size`` mergeable tokens.

    ``texts`` can be one string or a sequence of documents; documents are
    pre-tokenized in parallel. Special tokens get ids after the learned vocabulary.
    """

def _presets() -> list[tuple[str, str, str]]: ...
