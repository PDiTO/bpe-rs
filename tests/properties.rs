//! Property tests. Run more cases with `PROPTEST_CASES=10000 cargo test --release`.

mod common;

use std::collections::HashMap;
use std::sync::LazyLock;

use bpe_rs::merge::{encode_piece, encode_piece_heap, encode_piece_scan};
use bpe_rs::presets::{CL100K_BASE, O200K_BASE, O200K_BASE_PATTERN};
use bpe_rs::{Encoding, Rank, Ranks, SpecialTokens, Trainer, rank_file};
use proptest::prelude::*;
use proptest::sample::select;

// ---------------------------------------------------------------------------------------
// Fixtures

/// A vocabulary trained on this crate's own source, so the tests below have realistic
/// multi-byte merges to work with even when the published rank files are not cached.
static TRAINED: LazyLock<Encoding> = LazyLock::new(|| {
    let sources = [
        include_str!("../src/encoding.rs"),
        include_str!("../src/merge.rs"),
        include_str!("../src/train.rs"),
        include_str!("../src/rank_file.rs"),
    ];
    let docs: Vec<&str> = sources.iter().flat_map(|s| s.split("\n\n")).collect();
    Trainer::new(1500)
        .special_tokens(["<|endoftext|>", "<|sep|>"])
        .train("self", &docs)
        .unwrap()
});

static CL100K: LazyLock<Option<Encoding>> = LazyLock::new(|| common::load_preset(&CL100K_BASE));
static O200K: LazyLock<Option<Encoding>> = LazyLock::new(|| common::load_preset(&O200K_BASE));

fn encodings() -> Vec<&'static Encoding> {
    let mut all = vec![&*TRAINED];
    all.extend(CL100K.as_ref());
    all.extend(O200K.as_ref());
    all
}

// ---------------------------------------------------------------------------------------
// Strategies

/// Characters weighted towards the classes the split patterns care about.
fn interesting_char() -> impl Strategy<Value = char> {
    prop_oneof![
        4 => proptest::char::range('a', 'z'),
        2 => proptest::char::range('A', 'Z'),
        2 => select(vec![' ', ' ', ' ', '\n', '\r', '\t', '\u{a0}', '\u{3000}']),
        2 => proptest::char::range('0', '9'),
        2 => select("'.,;:!?-_/\\\"()[]{}<>|@#$%^&*=+~`".chars().collect::<Vec<_>>()),
        1 => proptest::char::range('\u{4e00}', '\u{9fff}'),   // CJK
        1 => proptest::char::range('\u{1f300}', '\u{1faff}'), // emoji
        1 => proptest::char::range('\u{0300}', '\u{036f}'),   // combining marks
        1 => proptest::char::range('\u{0400}', '\u{04ff}'),   // Cyrillic
        1 => any::<char>(),
    ]
}

fn text() -> impl Strategy<Value = String> {
    prop::collection::vec(interesting_char(), 0..300).prop_map(|cs| cs.into_iter().collect())
}

/// Arbitrary bytes, made into a string the way a caller with bad input would.
fn lossy_text() -> impl Strategy<Value = String> {
    prop::collection::vec(any::<u8>(), 0..300)
        .prop_map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
}

/// A small random vocabulary over a four-letter alphabet, ranked in generation order.
fn tiny_vocab() -> impl Strategy<Value = Ranks> {
    prop::collection::vec("[abcd]{2,6}", 1..60).prop_map(|tokens| {
        let mut ranks: Ranks = (0..=255u8).map(|b| (vec![b], Rank::from(b))).collect();
        for token in tokens {
            let next = ranks.len() as Rank;
            ranks.entry(token.into_bytes()).or_insert(next);
        }
        ranks
    })
}

// ---------------------------------------------------------------------------------------
// Reference implementations: slow, obviously correct

/// BPE merge by brute force: recompute every adjacent pair's rank, merge the lowest
/// (leftmost on ties), repeat.
fn naive_merge(ranks: &Ranks, piece: &[u8]) -> Vec<Rank> {
    let mut parts: Vec<Vec<u8>> = piece.iter().map(|&b| vec![b]).collect();
    loop {
        let best = (0..parts.len().saturating_sub(1))
            .filter_map(|i| {
                let joined = [parts[i].as_slice(), &parts[i + 1]].concat();
                ranks.get(&joined).map(|&rank| (rank, i))
            })
            .min();
        let Some((_, i)) = best else { break };
        let right = parts.remove(i + 1);
        parts[i].extend(right);
    }
    parts.iter().map(|p| ranks[p]).collect()
}

/// BPE training that recounts every pair from scratch after each merge, with the same
/// tie-breaking rule as `Trainer`.
fn naive_train(texts: &[String], vocab_size: usize, pattern: &str) -> Ranks {
    let splitter = Encoding::new("split", pattern, byte_ranks(), std::iter::empty()).unwrap();
    let mut counts: HashMap<&str, i64> = HashMap::new();
    for text in texts {
        for piece in splitter.split(text) {
            *counts.entry(piece).or_default() += 1;
        }
    }
    let mut words: Vec<(Vec<u32>, i64)> = counts
        .into_iter()
        .map(|(p, c)| (p.bytes().map(u32::from).collect(), c))
        .collect();

    let mut vocab: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
    while vocab.len() < vocab_size {
        let mut pairs: HashMap<(u32, u32), i64> = HashMap::new();
        for (symbols, count) in &words {
            for w in symbols.windows(2) {
                *pairs.entry((w[0], w[1])).or_default() += count;
            }
        }
        let Some((&pair, _)) = pairs
            .iter()
            .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
        else {
            break;
        };
        let bytes = [vocab[pair.0 as usize].as_slice(), &vocab[pair.1 as usize]].concat();
        let new_id = match vocab.iter().position(|v| *v == bytes) {
            Some(id) => id as u32,
            None => {
                vocab.push(bytes);
                (vocab.len() - 1) as u32
            }
        };
        for (symbols, _) in &mut words {
            let mut merged = Vec::with_capacity(symbols.len());
            let mut i = 0;
            while i < symbols.len() {
                if i + 1 < symbols.len() && (symbols[i], symbols[i + 1]) == pair {
                    merged.push(new_id);
                    i += 2;
                } else {
                    merged.push(symbols[i]);
                    i += 1;
                }
            }
            *symbols = merged;
        }
    }
    vocab
        .into_iter()
        .enumerate()
        .map(|(i, b)| (b, i as Rank))
        .collect()
}

fn byte_ranks() -> Ranks {
    (0..=255u8).map(|b| (vec![b], Rank::from(b))).collect()
}

// ---------------------------------------------------------------------------------------
// Properties

proptest! {
    #[test]
    fn decode_inverts_encode(s in text()) {
        for enc in encodings() {
            let tokens = enc.encode_ordinary(&s);
            prop_assert_eq!(enc.decode(&tokens).unwrap(), s.as_str(), "{}", enc.name());
        }
    }

    #[test]
    fn decode_inverts_encode_on_lossy_bytes(s in lossy_text()) {
        for enc in encodings() {
            let tokens = enc.encode_ordinary(&s);
            prop_assert_eq!(enc.decode_bytes(&tokens).unwrap(), s.as_bytes());
        }
    }

    #[test]
    fn decode_inverts_encode_with_special_tokens(
        parts in prop::collection::vec((text(), prop::bool::ANY), 0..6)
    ) {
        let mut s = String::new();
        for (chunk, add_special) in &parts {
            s.push_str(chunk);
            if *add_special {
                s.push_str("<|endoftext|>");
            }
        }
        for enc in encodings() {
            let tokens = enc.encode_with_special_tokens(&s);
            let eot = enc.special_token("<|endoftext|>").unwrap();
            let expected = parts.iter().filter(|(_, special)| *special).count();
            prop_assert_eq!(tokens.iter().filter(|&&t| t == eot).count(), expected);
            prop_assert_eq!(enc.decode(&tokens).unwrap(), s.as_str());

            let default = enc.encode(&s, SpecialTokens::None, SpecialTokens::All);
            prop_assert_eq!(default.is_err(), expected > 0);
        }
    }

    #[test]
    fn encoding_is_deterministic_and_batch_agrees(texts in prop::collection::vec(text(), 0..20)) {
        for enc in encodings() {
            let sequential: Vec<Vec<Rank>> = texts.iter().map(|t| enc.encode_ordinary(t)).collect();
            let again: Vec<Vec<Rank>> = texts.iter().map(|t| enc.encode_ordinary(t)).collect();
            prop_assert_eq!(&sequential, &again);
            prop_assert_eq!(&enc.encode_ordinary_batch(&texts), &sequential);
        }
    }

    #[test]
    fn pieces_concatenate_to_the_input(s in text()) {
        for enc in encodings() {
            prop_assert_eq!(enc.split(&s).concat(), s.as_str());
        }
    }

    #[test]
    fn merges_follow_rank_order(ranks in tiny_vocab(), piece in "[abcd]{0,300}") {
        let expected = naive_merge(&ranks, piece.as_bytes());
        let (mut scan, mut heap, mut auto) = (Vec::new(), Vec::new(), Vec::new());
        encode_piece_scan(&ranks, piece.as_bytes(), &mut scan);
        encode_piece_heap(&ranks, piece.as_bytes(), &mut heap);
        encode_piece(&ranks, piece.as_bytes(), &mut auto);
        prop_assert_eq!(&scan, &expected);
        prop_assert_eq!(&heap, &expected);
        prop_assert_eq!(&auto, &expected);
    }

    #[test]
    fn heap_matches_scan_on_real_vocab(s in text()) {
        for enc in encodings() {
            let (mut scan, mut heap) = (Vec::new(), Vec::new());
            encode_piece_scan(enc.ranks(), s.as_bytes(), &mut scan);
            encode_piece_heap(enc.ranks(), s.as_bytes(), &mut heap);
            prop_assert_eq!(scan, heap);
        }
    }

    #[test]
    fn incremental_training_matches_naive_training(
        docs in prop::collection::vec("[ab cd\n]{0,40}", 1..8),
        extra in 0usize..60,
    ) {
        let fast = Trainer::new(256 + extra).train_ranks(&docs).unwrap();
        let slow = naive_train(&docs, 256 + extra, bpe_rs::presets::CL100K_BASE_PATTERN);
        prop_assert_eq!(fast, slow);
    }

    #[test]
    fn trained_vocab_round_trips(
        docs in prop::collection::vec(text(), 1..10),
        probe in text(),
        extra in 0usize..200,
    ) {
        let enc = Trainer::new(256 + extra)
            .pattern(O200K_BASE_PATTERN)
            .train("t", &docs)
            .unwrap();
        for s in docs.iter().chain([&probe]) {
            prop_assert_eq!(enc.decode(&enc.encode_ordinary(s)).unwrap(), s.as_str());
        }

        // Saving and reloading gives back the same vocabulary.
        let mut buf = Vec::new();
        rank_file::write(enc.ranks(), &mut buf).unwrap();
        prop_assert_eq!(&rank_file::parse(&buf).unwrap(), enc.ranks());
    }
}
