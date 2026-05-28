"""Test strings for the differential tests.

Three sources:

* ``HANDWRITTEN``: cases aimed at the corners of the split patterns and the merge loop.
* ``real_text()``: chunks of the Python standard library's own source, plus this repo's
  Rust source. Real code with real comments, docstrings and whitespace.
* ``generated()``: seeded random strings from several generators.
"""

from __future__ import annotations

import random
import sysconfig
from collections.abc import Iterator
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

HANDWRITTEN: list[str] = [
    # Empty and tiny
    "",
    " ",
    "a",
    "\n",
    "\t",
    "\x00",
    "hello",
    "hello world",
    "Hello, World!",
    # Whitespace runs: the \s+(?!\S) and \s*[\r\n] alternatives
    "  ",
    "   ",
    "a  b",
    "a   b",
    "a    b",
    "a \n b",
    "a\n\nb",
    "a\n\n\nb",
    "a\r\nb",
    "a\r\n\r\nb",
    "a \t b",
    "trailing   ",
    "   leading",
    " \n \n ",
    "\n\n\n\n",
    "\r\r\r",
    "x\u00a0y",
    "x\u00a0\u00a0y",
    "x\u3000y",
    "x\u2028y\u2029z",
    "x\u200bz",
    " " * 1000,
    "\n" * 300,
    "\t" * 50 + "code",
    " " * 17 + "\n" + " " * 4 + "return",
    # Contractions, including uppercase and odd apostrophes
    "I'm here",
    "don't",
    "DON'T",
    "Don'T",
    "they'll",
    "we've",
    "she'd",
    "it's",
    "It'S",
    "'s",
    "'",
    "''",
    "'ll've'd",
    "O'Neil's",
    "rock'n'roll",
    "can’t",  # typographic apostrophe
    "'hello'",
    # Numbers: \p{N}{1,3}
    "1",
    "12",
    "123",
    "1234",
    "12345",
    "1234567890",
    "3.14159",
    "1,000,000",
    "-42",
    "+1 (555) 010-9999",
    "0x1F",
    "1e-10",
    "２０２６",
    "٣٤٥٦",
    "½ ¾ ⅓",
    "Ⅻ",
    "2026-09-25T17:34:00Z",
    "v1.2.3-rc.4",
    # Case and camel case (o200k splits on case changes)
    "camelCase",
    "PascalCase",
    "snake_case",
    "SCREAMING_SNAKE",
    "HTTPServer",
    "XMLHttpRequest",
    "iPhone",
    "McDonald's",
    "ÉCOLE école",
    "ǅungla",
    # Code
    "def f(x):\n    return x ** 2\n",
    'fn main() {\n    println!("{}", 1 + 2);\n}\n',
    "for (let i = 0; i < n; i++) { sum += a[i]; }",
    '{"key": [1, 2, 3], "nested": {"a": null, "b": true}}',
    '<div class="x"><p>Hello</p></div>',
    "SELECT id, name FROM users WHERE age >= 21 ORDER BY name;",
    r"^(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+)@example\.com$",
    "if [[ -f ~/.bashrc ]]; then source ~/.bashrc; fi",
    "#include <stdio.h>\nint main(void) { return 0; }",
    "x = a->b ?: c ?? d;",
    "    \n\tindented\n\t\tmore\n",
    "https://example.com/path?query=1&other=two#frag",
    "user.name+tag@example.co.uk",
    "/usr/local/bin/python3",
    "C:\\Windows\\System32\\drivers\\etc\\hosts",
    "## Heading\n\n- item one\n- item two\n\n```py\nprint(1)\n```\n",
    "a/b/c//d///e",
    "path/to/file.txt\n/another/path\n",
    # Punctuation runs
    "...",
    "!!!???",
    "-----",
    "=" * 500,
    "-" * 1000,
    "*_*_*_*",
    "«quoted» „German“ 「Japanese」",
    "(((nested)))",
    "a,b;c:d.e!f?g",
    # Emoji, including ZWJ sequences, flags, skin tones and keycaps
    "🙂",
    "🙂🙂🙂",
    "I ❤️ Rust",
    "👨‍👩‍👧‍👦",
    "🇺🇸🇯🇵🇩🇪",
    "👍🏽",
    "1️⃣ 2️⃣",
    "🧑🏿‍🚀 in 🚀",
    "emoji🙂inside",
    # CJK
    "你好，世界",
    "我爱自然语言处理。",
    "東京は日本の首都です。",
    "ひらがなとカタカナ",
    "한국어 텍스트입니다",
    "中文English混合text",
    # Other scripts
    "Привет, мир!",
    "Γειά σου Κόσμε",
    "مرحبا بالعالم",
    "שלום עולם",
    "नमस्ते दुनिया",
    "สวัสดีชาวโลก",
    "გამარჯობა",
    "Tiếng Việt có dấu",
    # Combining marks
    "e\u0301",
    "cafe\u0301",
    "Z̤͔ͧ̑̓ä͖̭̈̇lͮ̒ͫǧ̗͚̚o̙̔ͮ̇͐̇",
    "a\u0300\u0301\u0302\u0303\u0304",
    "\u0301leading combining mark",
    # Odd code points
    "\ufeffBOM",
    "\uffff\ufffe",
    "\U0010ffff",
    "\ue000private use",
    "\x7f\x1b[31mred\x1b[0m",
    "tab\tseparated\tvalues",
    # Special-token lookalikes (encoded as ordinary text here)
    "<|endoftext|>",
    "<|endoftext",
    "endoftext|>",
    "<|fim_prefix|>x<|fim_suffix|>",
    "<<|endoftext|>>",
    # Long pieces, which take the heap path in the merge
    "pneumonoultramicroscopicsilicovolcanoconiosis",
    "a" * 5000,
    "ab" * 2500,
    "abcdefghijklmnopqrstuvwxyz" * 100,
    "Supercalifragilisticexpialidocious" * 30,
    "\u4e00" * 800,
    "9" * 1000,
    "0123456789" * 50,
    "".join(chr(0x41 + (i * 7) % 26) for i in range(3000)),
]


def _chunks(text: str, size: int) -> Iterator[str]:
    for start in range(0, len(text), size):
        yield text[start : start + size]


def real_text(max_files: int = 40, chunk_chars: int = 3000) -> list[str]:
    """Chunks of real source code: part of the stdlib, and this repo's Rust source."""
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    files = sorted(stdlib.glob("*.py"))[:max_files]
    files += sorted((REPO / "src").rglob("*.rs"))
    out = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        out.extend(_chunks(text, chunk_chars))
    return out


_WORDS = (
    "the of and to in is it you that he was for on are with as I his they be at one "
    "have this from or had by hot word but what some we can out other were all there "
    "when up use your how said an each she which do their time if will way about many "
    "then them write would like so these her long make thing see him two has look more "
    "day could go come did number sound no most people my over know water than call "
    "first who may down side been now find tokenizer merge rank byte pair encoding"
).split()

_PUNCT = list(".,;:!?-_/\\\"'()[]{}<>|@#$%^&*=+~`")
_SPACES = [" ", " ", " ", "  ", "\n", "\n\n", "\t", "\r\n", " \n", "\u00a0"]


def _random_classes(rng: random.Random) -> str:
    """Characters drawn from the classes the split patterns distinguish."""
    pools = [
        (8, lambda: rng.choice("abcdefghijklmnopqrstuvwxyz")),
        (3, lambda: rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")),
        (3, lambda: rng.choice(_SPACES)),
        (2, lambda: rng.choice("0123456789")),
        (2, lambda: rng.choice(_PUNCT)),
        (1, lambda: chr(rng.randint(0x4E00, 0x9FFF))),
        (1, lambda: chr(rng.randint(0x1F300, 0x1FAFF))),
        (1, lambda: chr(rng.randint(0x0300, 0x036F))),
        (1, lambda: chr(rng.randint(0x0400, 0x04FF))),
        (1, lambda: chr(rng.randint(0x0600, 0x06FF))),
    ]
    weights = [w for w, _ in pools]
    n = rng.randint(1, 120)
    return "".join(rng.choices(pools, weights)[0][1]() for _ in range(n))


def _random_codepoints(rng: random.Random) -> str:
    """Any Unicode scalar value (no surrogates), biased towards the BMP."""
    out = []
    for _ in range(rng.randint(1, 60)):
        hi = 0xFFFF if rng.random() < 0.8 else 0x10FFFF
        cp = rng.randint(0, hi)
        if 0xD800 <= cp <= 0xDFFF:
            cp = 0xFFFD
        out.append(chr(cp))
    return "".join(out)


def _random_prose(rng: random.Random) -> str:
    """Word salad with random case, punctuation, contractions and numbers."""
    parts = []
    for _ in range(rng.randint(1, 40)):
        word = rng.choice(_WORDS)
        r = rng.random()
        if r < 0.1:
            word = word.upper()
        elif r < 0.25:
            word = word.capitalize()
        if rng.random() < 0.08:
            word += rng.choice(["'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "'LL"])
        if rng.random() < 0.08:
            word = str(rng.randint(0, 10 ** rng.randint(1, 8)))
        parts.append(word)
        parts.append(rng.choice(_SPACES) if rng.random() < 0.85 else rng.choice(_PUNCT) + " ")
    return "".join(parts)


def _random_bytes(rng: random.Random) -> str:
    """Arbitrary bytes decoded leniently, the way untrusted input often arrives."""
    data = bytes(rng.randrange(256) for _ in range(rng.randint(1, 100)))
    return data.decode("utf-8", errors="replace")


def _random_whitespace_digits(rng: random.Random) -> str:
    """Stress the whitespace and number rules specifically."""
    alphabet = [" ", "  ", "\n", "\r", "\t", "\u3000", "1", "22", "333", "x", "'", "."]
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 80)))


GENERATORS = {
    "classes": _random_classes,
    "codepoints": _random_codepoints,
    "prose": _random_prose,
    "bytes": _random_bytes,
    "whitespace_digits": _random_whitespace_digits,
}


def generated(kind: str, count: int, seed: int) -> list[str]:
    rng = random.Random(f"{kind}-{seed}")
    make = GENERATORS[kind]
    return [make(rng) for _ in range(count)]
