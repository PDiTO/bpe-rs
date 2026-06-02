<!-- logo -->

# bpe-rs

A byte-level BPE tokenizer written in Rust, with Python bindings. It reads tiktoken rank
files, so it can load `cl100k_base` and `o200k_base`, and it gives the same token ids as
tiktoken. A differential test checks that on about 5,500 strings per encoding. It can
also train its own vocabulary from text and save it in the same format.

```text
$ bpe encode --show "Tokenizers aren't magic"
   3404  "Token"
  12509  "izers"
   7784  " aren"
    956  "'t"
  11204  " magic"
```

## Why I built this

I use tokenizers every day, mostly without looking at them. A string goes in, a list of
integers comes out, and the model only ever sees the integers. I knew the outline of BPE,
but I couldn't have told you exactly why `" aren't"` becomes two tokens under cl100k and
one under o200k, or why the order merges were learned in changes the output, or how
tiktoken gets through tens of megabytes a second when the textbook algorithm is
quadratic.

So I wrote one. The rule I set myself was that it had to match tiktoken exactly, id for
id. "Roughly right" isn't something you can test. "Identical to tiktoken on thousands of
nasty strings" is, and it forced me to pin down every rule instead of the ones I
already understood.

## What I looked at

- `tiktoken/core.py` for the special-token rules. These are stricter than I expected. By
  default, text that looks like `<|endoftext|>` is an error, not a token.
- `tiktoken_ext/openai_public.py` for the exact split regexes and special token ids.
- tiktoken's Rust core, `src/lib.rs` in the 0.14 source distribution, for its merge loop
  and its notes on regex performance. I read this after writing my own merge code, which
  turned out to be useful, see below.
- The fancy-regex source, once I hit two problems with it. Batch encoding wouldn't
  scale, and there's a hard limit on how long a whitespace run can be.

## How a string becomes tokens

Encoding happens in three steps.

1. **Special tokens.** Allowed special tokens such as `<|endoftext|>` are found and cut
   out of the text. Each becomes one id.
2. **Pre-tokenization.** The text in between is split with a regex into pieces. This is
   where the two encodings differ most. cl100k splits `aren't` into `aren` and `'t`.
   o200k keeps the contraction on the word, and it also splits `camelCase` into `camel`
   and `Case`. Merges never cross a piece boundary.
3. **Merging.** Each piece is looked up whole. On English prose and on Python source,
   93 to 94 percent of pieces are a single token in both vocabularies, so this lookup
   usually ends the work. Otherwise the piece starts as one symbol per byte,
   and the adjacent pair whose concatenation has the lowest rank is merged, over and
   over, until no adjacent pair is in the vocabulary.

### Why merge order matters

A rank is a merge priority. Lower ranks were learned earlier and win. With a toy
vocabulary of the 256 single bytes plus two merges:

| vocabulary | `"abc"` encodes to |
|---|---|
| `ab` = 256, `bc` = 257 | `[ab, c]` |
| `bc` = 256, `ab` = 257 | `[a, bc]` |

Same tokens, different order, different output. One consequence surprised me. A token
in the vocabulary is only reachable if its parts are. If the vocabulary contains
`abc` but neither `ab` nor `bc`, then `"abc"` encodes as three single bytes, even though
`abc` is right there. Both cases are unit tests in `src/merge.rs`.

tiktoken keys its ranks by byte strings rather than by pairs of token ids. So a rank file
is only a map from bytes to integers, and the merge rules fall out of that map. This crate
does the same. That's what lets it load any tiktoken file.

## The merge algorithm

The obvious implementation keeps a list of symbols, scans the whole list for the
lowest-ranked pair, merges it, and scans again. That's `O(n * m)` for `n` bytes and `m`
merges, so quadratic in the worst case.

`src/merge.rs` has that version, `encode_piece_scan`, and a second one,
`encode_piece_heap`, that does `O(n log n)` work:

- Symbols live in a doubly linked list stored as two arrays, `prev` and `next`, indexed by
  the byte offset where each symbol starts. A merge is two array writes.
- Every possible merge sits in a binary min-heap keyed by `(rank, start)`, so the lowest
  rank comes out first and ties go to the leftmost pair, as in tiktoken.
- Merging two symbols only creates two new adjacent pairs, the merged symbol with its
  left neighbour and with its right neighbour. Only those two go on the heap.
- Entries that went stale are not removed from the heap. When one is popped, the code
  checks that its left symbol still starts at `start` and that the symbol after it still
  ends at `end`. Symbols only grow, so if both checks pass the pair covers the same bytes
  as when it was pushed, and has the same rank.

Worked example, with the byte vocabulary plus `ab` = 256, `bc` = 257, `abc` = 258,
`cd` = 259, encoding the piece `"abcd"`:

The heap starts with `ab`(0..2), `bc`(1..3) and `cd`(2..4), where the numbers are byte
offsets.

| step | popped | action | symbols after |
|---|---|---|---|
| 1 | `ab`(0..2) | valid, merge. `abc`(0..3) is in the vocab, push it | `ab c d` |
| 2 | `bc`(1..3) | stale, nothing starts at 1 any more. Skip | `ab c d` |
| 3 | `abc`(0..3) | valid, merge. `abcd` is not in the vocab | `abc d` |
| 4 | `cd`(2..4) | stale, nothing starts at 2. Skip | `abc d` |

The result is `[258, 100]`. This exact case is the `readme_worked_example` test.

Which one is faster depends on the piece length. Most pieces are a few bytes long, and
there the flat scan wins, because it touches one small vector and nothing else. Here are
criterion medians for pieces cut from Pride and Prejudice with everything but letters
removed, using the cl100k vocabulary:

| piece length (bytes) | scan | heap |
|---:|---:|---:|
| 8 | 0.12 µs | 0.17 µs |
| 32 | 0.56 µs | 0.82 µs |
| 96 | 2.52 µs | 2.81 µs |
| 128 | 4.15 µs | 3.91 µs |
| 256 | 15.0 µs | 8.5 µs |
| 1,024 | 192 µs | 41 µs |
| 4,096 | 2.60 ms | 0.25 ms |
| 16,384 | 42.3 ms | 2.06 ms |

The crossover is somewhere between 96 and 128 bytes, so `encode_piece` uses the scan below
128 bytes and the heap above. When I then read tiktoken's current source, I found it does
the same thing. It has a heap-based `_byte_pair_merge_large` and switches to it at 100
bytes. So the hybrid is not my invention, but it was nice to arrive at the same threshold
from a benchmark.

A property test checks that both versions agree with a brute-force reference on random
vocabularies and random strings.

## Training

`Trainer` learns a vocabulary from text:

1. Split every document with the same regex the encoder uses, and count distinct pieces.
   From then on the trainer works on distinct pieces with a count, not on the raw text.
2. Start each piece as single bytes, ids 0 to 255.
3. Merge the most frequent adjacent pair, give it the next id, repeat until the
   vocabulary is big enough.

Recounting every pair after every merge would mean a full pass per merge. Instead the
trainer keeps a count for every pair, an index from each pair to the pieces it occurs in,
and a max-heap of `(count, pair)`. A merge visits only the pieces that contain the pair,
works out their pair counts before and after, and applies the difference. When a count
drops, the old heap entry is left in place and gets corrected when it's popped. Ties go to
the pair with smaller ids, so the result doesn't depend on thread scheduling.

A property test trains the same random corpora with this and with a naive version that
recounts everything after each merge, and checks the vocabularies are identical.

Documents are pre-tokenized in parallel. On the Python standard library's non-test
source, 632 files and 11.2 MB, `bpe train` learns 32,000 tokens in 0.37 s. The output is a
normal tiktoken rank file.

## Design decisions and trade-offs

**fancy-regex with the published patterns.** Both split patterns use `(?!\S)`, a
lookahead, so the `regex` crate can't run them. I used fancy-regex, like tiktoken, and
copied the patterns exactly. The cost is speed. For cl100k, splitting alone runs at 41
MB/s, and the full encode at 31 MB/s, so the regex is about three quarters of the time. A
hand-written splitter would be faster, and I'd rather have exact matches.

**One compiled regex per thread.** My first batch encoder barely scaled. On a 3 MB text,
one thread encoded at 30 MB/s and twelve threads managed only 59 MB/s. The notes in
tiktoken's source explain it. The regex engine keeps scratch space in a pool inside each
compiled regex, and the threads fight over it. tiktoken hands each thread a clone of the
regex. I did the same and it didn't help. It turns out that in fancy-regex 0.19, clones
still share the inner regexes that own those pools. Compiling a separate copy per thread
slot, lazily, took twelve threads to about 200 MB/s on the same text. The price is
memory. Each compiled copy is roughly a megabyte, and an encoding keeps up to two per
core.

**Errors instead of panics.** fancy-regex caps its backtracking stack at a million
entries, and `\s+(?!\S)` uses one entry per character of a whitespace run. A string with
about a million spaces in a row can't be split. tiktoken panics on that input, which
Python sees as `PanicException`. Here it's `Error::PreTokenize` in Rust and `ValueError`
in Python. So every encode method returns a `Result`, which is a bit noisier to call.

**tiktoken's special-token rules, including the odd ones.** Any string in
`disallowed_special` raises, even if it isn't a special token. `disallowed_special=None`
turns the check off. An allowed special token is found even when a longer, non-allowed one
overlaps it. None of this matters for everyday use, but the differential tests cover it,
so it matches.

**Python bindings release the GIL.** Single calls release it while encoding. Batch calls
run on a rayon pool. You can pass `num_threads` like in tiktoken. Each distinct value gets
its own pool, kept for the life of the process. The wheel is built for the stable ABI, so
one wheel covers Python 3.10 and newer.

**Not included.** Only `cl100k_base` and `o200k_base` are built in. Other tiktoken files
such as `r50k_base` load fine if you pass their pattern yourself. There's no
`encode_to_numpy`, no decoding with offsets, and no support for gpt2's older
`vocab.bpe` + `encoder.json` format.

## Benchmarks

Measured on an Apple Silicon laptop, an M4 Pro with 12 cores, with other apps running.
Batch numbers varied by as much as a third between runs when the machine was busy.
Single-thread numbers stayed within about 5 percent.

### Python, bpe_rs vs tiktoken

`benchmarks/bench_tiktoken.py`, Python 3.13, tiktoken 0.14.0, median of 10 runs. "prose"
is Pride and Prejudice from Project Gutenberg. "code" is the first megabyte of the Python
standard library source. "long piece" is 64 KB of random letters with no spaces, which the
regex turns into a single piece. Batch mode cuts the text into roughly 4 KB documents and
gives both libraries 12 threads.

| encoding | corpus | mode | tiktoken MB/s | bpe_rs MB/s | ratio |
|---|---|---|---:|---:|---:|
| cl100k_base | prose, 0.76 MB | single thread | 29.5 | 28.2 | 0.96x |
| cl100k_base | prose, 0.76 MB | batch, 12 threads | 54.0 | 133.8 | 2.48x |
| cl100k_base | code, 1.00 MB | single thread | 23.1 | 23.2 | 1.00x |
| cl100k_base | code, 1.00 MB | batch, 12 threads | 49.2 | 115.3 | 2.35x |
| cl100k_base | long piece, 0.07 MB | single thread | 15.0 | 12.8 | 0.85x |
| o200k_base | prose, 0.76 MB | single thread | 41.1 | 42.6 | 1.04x |
| o200k_base | prose, 0.76 MB | batch, 12 threads | 81.4 | 180.4 | 2.22x |
| o200k_base | code, 1.00 MB | single thread | 33.7 | 32.3 | 0.96x |
| o200k_base | code, 1.00 MB | batch, 12 threads | 66.3 | 153.3 | 2.31x |
| o200k_base | long piece, 0.07 MB | single thread | 20.7 | 21.5 | 1.04x |

Single-threaded, it's a tie, within a few percent either way. That's what I'd expect with
the same regex engine, the same pattern and the same scan-or-heap merge. On the cl100k
long piece tiktoken is about 15 percent faster. Its heap version remembers each merged token's rank
as it goes, so it skips the final lookups my version does, and its per-symbol state sits in
one array instead of my separate arrays. I haven't profiled enough to say which of those
matters.

Batched, bpe_rs is 2.2 to 2.5 times faster. tiktoken's batch methods run a Python
`ThreadPoolExecutor` over single calls, and its throughput flattens out around 4 threads.
In a separate run on four copies of Pride and Prejudice, tiktoken went 21, 38, 45, 54, 53
MB/s at 1, 2, 4, 8 and 12 threads, and bpe_rs went 29, 51, 90, 143, 142. My guess is that
tiktoken is hitting the same shared regex pools I described above, since my own
clone-based version stalled at about the same speed. I haven't proven that.

### Rust, criterion

`cargo bench` on Pride and Prejudice, 0.77 MB, medians:

| benchmark | cl100k_base | o200k_base |
|---|---:|---:|
| split only, no merging | 41.0 MB/s | 70.7 MB/s |
| `encode_ordinary`, 1 thread | 31.3 MB/s | 46.3 MB/s |
| `encode_ordinary_batch`, 12 threads | 186 MB/s | 292 MB/s |
| `decode_bytes` | 299 MB/s | 293 MB/s |

Training on the same text takes 19.5 ms for a 1,000-token vocabulary and 23.3 ms for
5,000 tokens. The extra 4,000 merges cost under 4 ms, so most of the time goes to
splitting and counting pieces.

## Quickstart

### Rust

The crate isn't on crates.io. Use it as a path or git dependency. The rank files aren't
bundled, so fetch them first:

```sh
scripts/fetch_data.sh   # into ~/.cache/bpe-rs, checked against tiktoken's SHA-256 hashes
```

```rust
use bpe_rs::presets::CL100K_BASE;
use bpe_rs::{Encoding, SpecialTokens, Trainer};

fn main() -> bpe_rs::Result<()> {
    let enc = Encoding::from_preset_file(&CL100K_BASE, CL100K_BASE.cached_path())?;

    let ids = enc.encode_ordinary("hello world")?;
    assert_eq!(ids, [15339, 1917]);
    assert_eq!(enc.decode(&ids)?, "hello world");

    // tiktoken's defaults: special-token text is an error unless you allow it.
    assert!(enc.encode("<|endoftext|>", SpecialTokens::None, SpecialTokens::All).is_err());
    assert_eq!(enc.encode_with_special_tokens("<|endoftext|>")?, [100257]);

    // Train a small vocabulary and save it as a tiktoken rank file.
    let docs = ["the cat sat on the mat", "the dog sat on the log"];
    let trained = Trainer::new(300).train("demo", &docs)?;
    trained.save_tiktoken_file("demo.tiktoken")?;
    Ok(())
}
```

The command-line tool:

```sh
cargo install --path .
bpe encode --show -e o200k_base "Tokenizers aren't magic"
echo "15339 1917" | bpe decode
bpe train --vocab-size 8000 -o mine.tiktoken corpus/*.txt
bpe encode --rank-file mine.tiktoken --pattern cl100k_base "some text"
bpe bench -e cl100k_base some_big_file.txt
```

### Python

Build from source with [uv](https://docs.astral.sh/uv/) and maturin:

```sh
uv sync
uv run maturin develop --uv --release
```

```python
import bpe_rs

enc = bpe_rs.get_encoding("cl100k_base")  # downloads and caches the rank file once
enc.encode("hello world")  # [15339, 1917]
enc.encode("<|endoftext|>", allowed_special="all")  # [100257]
enc.encode_batch(["many", "texts"], num_threads=8)
enc.decode([15339, 1917])  # 'hello world'
enc.split("I'm here  now")  # ['I', "'m", ' here', ' ', ' now']

trained = bpe_rs.train(["some", "documents"], vocab_size=1000, pattern="o200k_base")
trained.save("mine.tiktoken")
again = bpe_rs.Encoding.from_tiktoken_file("mine.tiktoken", pattern="o200k_base")
```

The method names and keyword arguments follow tiktoken's `Encoding`. Type stubs are in
`python/bpe_rs/_bpe_rs.pyi`.

## Tests

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
scripts/fetch_data.sh
cargo test

uv sync
uv run ruff check .
uv run ruff format --check .
uv run maturin develop --uv --release
uv run pytest
```

These are the commands CI runs, except that CI uses `uv sync --locked` and also builds a
wheel with `uv run maturin build`.

**Rust, 54 tests.** Unit tests for the rank file parser, both merge versions, special
tokens, training and the CLI. Known-answer tests against the real vocabularies, including
a check that saving cl100k gives back a byte-identical file. Nine proptest properties,
256 cases each by default:

- `decode(encode(x)) == x` for arbitrary Unicode, for arbitrary bytes read as lossy
  UTF-8, and with special tokens mixed in.
- Encoding is deterministic, and batch encoding matches one-at-a-time encoding.
- Pieces from the splitter concatenate back to the input.
- Scan, heap and a brute-force reference agree on random tiny vocabularies. This is the
  "merges respect rank order" check.
- Incremental training matches naive training, and trained vocabularies round-trip and
  survive save and load.

The encoding properties run against a vocabulary trained on this crate's own source, and
against cl100k and o200k when the rank files are present. Set `PROPTEST_CASES=5000` for a longer
run.

**Python, 479 tests.** 452 of them compare against tiktoken, for both encodings:

- 154 hand-written edge cases, such as whitespace runs, contractions in both cases,
  digit groups, CJK, emoji with ZWJ sequences and flags, combining marks, code, and
  special-token lookalikes
- 5,000 seeded random strings from five generators
- the first 40 modules of the standard library plus this repo's Rust source, cut into
  405 chunks of 3,000 characters
- special-token handling, lone surrogates, metadata, the bytes of every token id, and
  decoding of 2,000 random token sequences

In total 5,559 strings per encoding go through `encode_ordinary` in both libraries.
pytest prints the count at the end. It moves a little when the Rust source changes,
since that source is part of the corpus. If tiktoken or the rank files can't be
downloaded, those tests skip. With `BPE_RS_REQUIRE_DIFFERENTIAL=1`, as in CI, they fail
instead.

## Layout

```text
src/
  merge.rs         scan and heap merge for a single piece
  encoding.rs      Encoding: splitting, special tokens, encode, decode, batch
  train.rs         Trainer with incremental pair counts
  rank_file.rs     tiktoken rank file parsing and writing
  presets.rs       cl100k_base / o200k_base patterns, special tokens, URLs, hashes
  error.rs
  bin/bpe.rs       the `bpe` CLI
tests/             known-answer tests and proptest properties
benches/encode.rs  criterion benchmarks
python/
  src/lib.rs       PyO3 bindings
  bpe_rs/          Python package, type stubs, downloader
  tests/           pytest, including the tiktoken differential tests
benchmarks/        Python benchmark against tiktoken
scripts/           fetch_data.sh
```

## License

MIT. See [LICENSE](LICENSE).
