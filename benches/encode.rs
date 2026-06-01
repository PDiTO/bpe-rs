//! Criterion benchmarks.
//!
//! Needs the rank files and the benchmark corpus in the cache directory; run
//! `scripts/fetch_data.sh` first. Groups that are missing their inputs are skipped with a
//! note on stderr.
//!
//! ```text
//! cargo bench                  # everything
//! cargo bench -- merge         # just the scan-vs-heap comparison
//! ```

use std::hint::black_box;
use std::time::Duration;

use bpe_rs::merge::{encode_piece_heap, encode_piece_scan};
use bpe_rs::presets::{self, CL100K_BASE, O200K_BASE, Preset};
use bpe_rs::{Encoding, Trainer};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

fn corpus() -> Option<String> {
    let path = presets::cache_dir().join("corpus/pride_and_prejudice.txt");
    match std::fs::read_to_string(&path) {
        Ok(text) => Some(text),
        Err(_) => {
            eprintln!(
                "skipping: {} not found (run scripts/fetch_data.sh)",
                path.display()
            );
            None
        }
    }
}

fn load(preset: &Preset) -> Option<Encoding> {
    let path = preset.cached_path();
    if !path.exists() {
        eprintln!(
            "skipping {}: rank file not found (run scripts/fetch_data.sh)",
            preset.name
        );
        return None;
    }
    Some(Encoding::from_preset_file(preset, path).unwrap())
}

/// Cuts text into ~4 KB documents at line boundaries.
fn documents(text: &str) -> Vec<&str> {
    let mut docs = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        let cut = rest
            .as_bytes()
            .iter()
            .skip(4096)
            .position(|&b| b == b'\n')
            .map_or(rest.len(), |i| 4096 + i + 1);
        let (doc, tail) = rest.split_at(cut);
        docs.push(doc);
        rest = tail;
    }
    docs
}

fn encode(c: &mut Criterion) {
    let Some(text) = corpus() else { return };
    let docs = documents(&text);
    for preset in [&CL100K_BASE, &O200K_BASE] {
        let Some(enc) = load(preset) else { continue };
        let mut group = c.benchmark_group(format!("encode/{}", preset.name));
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.measurement_time(Duration::from_secs(8));
        group.bench_function("split_only", |b| {
            b.iter(|| enc.split(black_box(&text)).unwrap().len())
        });
        group.bench_function("encode_ordinary", |b| {
            b.iter(|| enc.encode_ordinary(black_box(&text)))
        });
        group.bench_function("encode_ordinary_batch", |b| {
            b.iter(|| enc.encode_ordinary_batch(black_box(&docs)))
        });
        let tokens = enc.encode_ordinary(&text).unwrap();
        group.bench_function("decode_bytes", |b| {
            b.iter(|| enc.decode_bytes(black_box(&tokens)).unwrap())
        });
        group.finish();
    }
}

/// Scan vs heap on single pieces of increasing length. The pieces are the corpus with
/// everything but letters removed, which gives long "words" that still merge the way
/// English does. This is what `merge::HEAP_THRESHOLD` is based on.
fn merge(c: &mut Criterion) {
    let (Some(text), Some(enc)) = (corpus(), load(&CL100K_BASE)) else {
        return;
    };
    let letters: String = text
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .take(20_000)
        .collect();

    let mut group = c.benchmark_group("merge");
    group.measurement_time(Duration::from_secs(3));
    for len in [4, 8, 16, 32, 64, 96, 128, 192, 256, 512, 1024, 4096, 16384] {
        let piece = &letters.as_bytes()[..len];
        group.throughput(Throughput::Bytes(len as u64));
        let mut out = Vec::with_capacity(len);
        group.bench_with_input(BenchmarkId::new("scan", len), piece, |b, piece| {
            b.iter(|| {
                out.clear();
                encode_piece_scan(enc.ranks(), black_box(piece), &mut out);
            })
        });
        group.bench_with_input(BenchmarkId::new("heap", len), piece, |b, piece| {
            b.iter(|| {
                out.clear();
                encode_piece_heap(enc.ranks(), black_box(piece), &mut out);
            })
        });
    }
    group.finish();
}

fn train(c: &mut Criterion) {
    let Some(text) = corpus() else { return };
    let docs = documents(&text);
    let mut group = c.benchmark_group("train");
    group.sample_size(10);
    group.throughput(Throughput::Bytes(text.len() as u64));
    for vocab_size in [1_000, 5_000] {
        group.bench_with_input(
            BenchmarkId::new("pride_and_prejudice", vocab_size),
            &vocab_size,
            |b, &n| b.iter(|| Trainer::new(n).train_ranks(black_box(&docs)).unwrap()),
        );
    }
    group.finish();
}

criterion_group!(benches, encode, merge, train);
criterion_main!(benches);
