"""Throughput of bpe_rs vs tiktoken on the same text.

    uv run maturin develop --uv --release
    uv run python benchmarks/bench_tiktoken.py

Two corpora:

* prose: Pride and Prejudice from Project Gutenberg (downloaded to the cache once)
* code:  the first ~1 MB of the Python standard library's own source
* long piece: 64 KB of random letters with no spaces, which the split pattern turns
  into a single pre-token. This is the worst case for a quadratic merge loop.

For each encoding and corpus it measures a single ``encode_ordinary`` call on the whole
text, then ``encode_ordinary_batch`` over ~4 KB documents with the same thread count
for both libraries. Every number is the median of several runs, reported as MB/s of
UTF-8 input. Token ids are checked for equality before anything is timed.
"""

from __future__ import annotations

import argparse
import os
import platform
import random
import statistics
import string
import sysconfig
import time
import urllib.request
from collections.abc import Callable
from functools import partial
from pathlib import Path

import tiktoken

import bpe_rs

PROSE_URL = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"


def prose() -> str:
    path = bpe_rs.cache_dir() / "corpus" / "pride_and_prejudice.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(PROSE_URL, timeout=60) as response:
            path.write_bytes(response.read())
    return path.read_text(encoding="utf-8")


def code(limit: int = 1_000_000) -> str:
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    parts, size = [], 0
    for path in sorted(stdlib.glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        parts.append(text)
        size += len(text.encode())
        if size >= limit:
            break
    return "".join(parts)


def long_piece(size: int = 64 * 1024) -> str:
    rng = random.Random(0)
    return "".join(rng.choice(string.ascii_letters) for _ in range(size))


def documents(text: str, size: int = 4096) -> list[str]:
    """Cuts text into ~size-byte documents at line boundaries."""
    docs, current, current_size = [], [], 0
    for line in text.splitlines(keepends=True):
        current.append(line)
        current_size += len(line.encode())
        if current_size >= size:
            docs.append("".join(current))
            current, current_size = [], 0
    if current:
        docs.append("".join(current))
    return docs


def median_seconds(fn: Callable[[], object], repeat: int) -> float:
    fn()  # warm up: lazy regex compilation, thread pool start-up, caches
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--encodings", nargs="+", default=["cl100k_base", "o200k_base"])
    args = parser.parse_args()

    corpora = {"prose": prose(), "code": code(), "long piece": long_piece()}
    print(
        f"python {platform.python_version()}, tiktoken {tiktoken.__version__}, "
        f"bpe_rs {bpe_rs.__version__}, {platform.machine()}, "
        f"{os.cpu_count()} cores, batch threads = {args.threads}, median of {args.repeat}\n"
    )
    print("| encoding | corpus | mode | tiktoken MB/s | bpe_rs MB/s | ratio |")
    print("|---|---|---|---:|---:|---:|")

    for name in args.encodings:
        ours = bpe_rs.get_encoding(name)
        theirs = tiktoken.get_encoding(name)
        for corpus_name, text in corpora.items():
            mb = len(text.encode()) / 1e6
            docs = documents(text)
            assert ours.encode_ordinary(text) == theirs.encode_ordinary(text)
            assert ours.encode_ordinary_batch(docs) == theirs.encode_ordinary_batch(docs)

            cases = {
                "single thread": (
                    partial(theirs.encode_ordinary, text),
                    partial(ours.encode_ordinary, text),
                ),
                f"batch, {args.threads} threads": (
                    partial(theirs.encode_ordinary_batch, docs, num_threads=args.threads),
                    partial(ours.encode_ordinary_batch, docs, num_threads=args.threads),
                ),
            }
            if corpus_name == "long piece":
                del cases[f"batch, {args.threads} threads"]  # one piece, nothing to batch
            for mode, (tk, br) in cases.items():
                tk_mbs = mb / median_seconds(tk, args.repeat)
                br_mbs = mb / median_seconds(br, args.repeat)
                print(
                    f"| {name} | {corpus_name} ({mb:.2f} MB) | {mode} | "
                    f"{tk_mbs:.1f} | {br_mbs:.1f} | {br_mbs / tk_mbs:.2f}x |"
                )


if __name__ == "__main__":
    main()
