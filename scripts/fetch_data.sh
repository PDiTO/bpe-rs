#!/usr/bin/env bash
# Downloads the cl100k_base and o200k_base rank files into the bpe-rs cache directory
# and checks them against the SHA-256 hashes tiktoken pins. Files that are already
# present and valid are left alone.
#
# Cache directory: $BPE_RS_CACHE_DIR, else $XDG_CACHE_HOME/bpe-rs, else ~/.cache/bpe-rs.
set -euo pipefail

if [[ -n "${BPE_RS_CACHE_DIR:-}" ]]; then
  cache="$BPE_RS_CACHE_DIR"
elif [[ -n "${XDG_CACHE_HOME:-}" ]]; then
  cache="$XDG_CACHE_HOME/bpe-rs"
else
  cache="$HOME/.cache/bpe-rs"
fi
mkdir -p "$cache"

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

fetch() {
  local name="$1" url="$2" expected="$3"
  local dest="$cache/$name"
  if [[ -f "$dest" && "$(sha256 "$dest")" == "$expected" ]]; then
    echo "ok       $dest"
    return
  fi
  echo "fetching $url"
  curl --fail --silent --show-error --location --retry 3 -o "$dest.part" "$url"
  local actual
  actual="$(sha256 "$dest.part")"
  if [[ "$actual" != "$expected" ]]; then
    rm -f "$dest.part"
    echo "hash mismatch for $name: expected $expected, got $actual" >&2
    exit 1
  fi
  mv "$dest.part" "$dest"
  echo "saved    $dest"
}

base="https://openaipublic.blob.core.windows.net/encodings"
fetch cl100k_base.tiktoken "$base/cl100k_base.tiktoken" \
  223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7
fetch o200k_base.tiktoken "$base/o200k_base.tiktoken" \
  446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d

# Benchmark text: Pride and Prejudice from Project Gutenberg. Gutenberg occasionally
# edits its headers, so this one is not hash-checked; the benchmarks only need
# realistic English prose.
corpus="$cache/corpus/pride_and_prejudice.txt"
if [[ -f "$corpus" ]]; then
  echo "ok       $corpus"
else
  mkdir -p "$cache/corpus"
  echo "fetching Pride and Prejudice (benchmark corpus)"
  curl --fail --silent --show-error --location --retry 3 -o "$corpus.part" \
    https://www.gutenberg.org/cache/epub/1342/pg1342.txt
  mv "$corpus.part" "$corpus"
  echo "saved    $corpus"
fi
