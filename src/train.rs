//! Learning a byte-level BPE vocabulary from text.
//!
//! The procedure is the classic one:
//!
//! 1. Split every document with the same pre-tokenization regex the encoder will use,
//!    and count how often each distinct piece occurs. From here on the trainer works on
//!    distinct pieces ("words") with a count, not on the raw text.
//! 2. Start every word as a sequence of single-byte tokens (ids 0 to 255).
//! 3. Find the adjacent pair with the highest total count, give its concatenation the
//!    next id, and replace it everywhere. Repeat until the vocabulary is big enough.
//!
//! Recounting every pair after every merge would cost a full pass over the corpus per
//! merge. Instead the trainer keeps:
//!
//! * `pair_counts`: the current count of every adjacent pair,
//! * `pair_words`: for each pair, the words it (possibly) occurs in, and
//! * a max-heap of `(count, pair)` entries.
//!
//! A merge only visits the words that contain the merged pair. For each of them it
//! computes the pair counts before and after the merge and applies the difference, so
//! work per merge is proportional to the words that actually changed. Heap entries go
//! stale when counts drop; they are corrected when popped rather than searched for.
//!
//! Ties between pairs with equal counts go to the pair with the smaller token ids, so
//! training is deterministic regardless of thread scheduling.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::encoding::{compile_pattern, pieces};
use crate::presets::CL100K_BASE_PATTERN;
use crate::{Encoding, Error, Rank, Ranks, Result};

/// Configures and runs BPE training.
///
/// ```
/// use bpe_rs::train::Trainer;
///
/// let corpus = ["the cat sat on the mat", "the dog sat on the log"];
/// let enc = Trainer::new(300).train("demo", &corpus).unwrap();
/// let tokens = enc.encode_ordinary("the cat sat");
/// assert_eq!(enc.decode(&tokens).unwrap(), "the cat sat");
/// ```
#[derive(Debug, Clone)]
pub struct Trainer {
    vocab_size: usize,
    pattern: String,
    special_tokens: Vec<String>,
    min_frequency: u64,
}

impl Trainer {
    /// A trainer that learns up to `vocab_size` mergeable tokens (the 256 single bytes
    /// included), splitting text with the cl100k_base pattern.
    pub fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            pattern: CL100K_BASE_PATTERN.to_owned(),
            special_tokens: Vec::new(),
            min_frequency: 1,
        }
    }

    /// Uses a different pre-tokenization regex, for example
    /// [`crate::presets::O200K_BASE_PATTERN`].
    pub fn pattern(mut self, pattern: impl Into<String>) -> Self {
        self.pattern = pattern.into();
        self
    }

    /// Special tokens to add after the learned vocabulary, in the order given.
    pub fn special_tokens<I, S>(mut self, tokens: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.special_tokens = tokens.into_iter().map(Into::into).collect();
        self
    }

    /// Stops early once the most frequent pair occurs fewer than `n` times. The default
    /// of 1 keeps merging as long as any pair is left.
    pub fn min_frequency(mut self, n: u64) -> Self {
        self.min_frequency = n.max(1);
        self
    }

    /// Learns mergeable ranks from `texts`.
    ///
    /// Each text is pre-tokenized on its own, and texts are processed in parallel, so
    /// passing a corpus as many documents is faster than one big string. The result
    /// has at most `vocab_size` entries; fewer if the corpus runs out of pairs.
    pub fn train_ranks<S: AsRef<str> + Sync>(&self, texts: &[S]) -> Result<Ranks> {
        if self.vocab_size < 256 {
            return Err(Error::VocabSizeTooSmall(self.vocab_size));
        }
        let regex = compile_pattern(&self.pattern)?;

        let piece_counts: FxHashMap<&str, u64> = texts
            .par_iter()
            .fold(FxHashMap::default, |mut counts, text| {
                for piece in pieces(&regex, text.as_ref()) {
                    *counts.entry(piece).or_default() += 1;
                }
                counts
            })
            .reduce(FxHashMap::default, |mut a, mut b| {
                if a.len() < b.len() {
                    std::mem::swap(&mut a, &mut b);
                }
                for (piece, n) in b {
                    *a.entry(piece).or_default() += n;
                }
                a
            });

        let words = piece_counts
            .into_iter()
            .map(|(piece, count)| Word {
                symbols: piece.bytes().map(u32::from).collect(),
                count: count as i64,
            })
            .collect();

        Ok(learn(words, self.vocab_size, self.min_frequency as i64))
    }

    /// Learns a vocabulary and wraps it in an [`Encoding`] with this trainer's pattern.
    /// Special tokens get the ids right after the last learned token.
    pub fn train<S: AsRef<str> + Sync>(&self, name: &str, texts: &[S]) -> Result<Encoding> {
        let ranks = self.train_ranks(texts)?;
        let first_special = ranks.len() as Rank;
        let specials = self
            .special_tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), first_special + i as Rank));
        Encoding::new(name, &self.pattern, ranks, specials)
    }
}

type Pair = (u32, u32);

struct Word {
    symbols: Vec<u32>,
    count: i64,
}

/// A heap entry. Higher counts come out first; among equal counts, the smaller pair.
#[derive(PartialEq, Eq)]
struct Candidate {
    count: i64,
    pair: Pair,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            .then_with(|| other.pair.cmp(&self.pair))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn learn(mut words: Vec<Word>, vocab_size: usize, min_frequency: i64) -> Ranks {
    // Token id -> bytes, and the reverse. Ids are ranks: the order merges were learned.
    let mut vocab: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
    let mut ids: FxHashMap<Vec<u8>, u32> = vocab
        .iter()
        .enumerate()
        .map(|(i, b)| (b.clone(), i as u32))
        .collect();

    let mut pair_counts: FxHashMap<Pair, i64> = FxHashMap::default();
    let mut pair_words: FxHashMap<Pair, Vec<u32>> = FxHashMap::default();
    for (w, word) in words.iter().enumerate() {
        for pair in word.symbols.windows(2).map(|p| (p[0], p[1])) {
            *pair_counts.entry(pair).or_default() += word.count;
            pair_words.entry(pair).or_default().push(w as u32);
        }
    }
    let mut heap: BinaryHeap<Candidate> = pair_counts
        .iter()
        .map(|(&pair, &count)| Candidate { count, pair })
        .collect();

    let mut delta: FxHashMap<Pair, i64> = FxHashMap::default();
    let mut merged_symbols: Vec<u32> = Vec::new();

    while vocab.len() < vocab_size {
        let Some(Candidate { count, pair }) = heap.pop() else {
            break;
        };
        let current = pair_counts.get(&pair).copied().unwrap_or(0);
        if current != count {
            // Stale entry: the count changed after this was pushed.
            if current > 0 {
                heap.push(Candidate {
                    count: current,
                    pair,
                });
            }
            continue;
        }
        if count < min_frequency {
            break;
        }

        let bytes = [vocab[pair.0 as usize].as_slice(), &vocab[pair.1 as usize]].concat();
        // Two different pairs can spell the same bytes ("a"+"bc" and "ab"+"c"). A rank
        // file maps bytes to ranks, so they share one id rather than getting two.
        let new_id = *ids.entry(bytes).or_insert_with_key(|bytes| {
            vocab.push(bytes.clone());
            (vocab.len() - 1) as u32
        });

        let mut affected = pair_words.remove(&pair).unwrap_or_default();
        affected.sort_unstable();
        affected.dedup();

        delta.clear();
        for &w in &affected {
            let word = &mut words[w as usize];
            if !merge_word(&word.symbols, pair, new_id, &mut merged_symbols) {
                continue; // stale index: this word no longer contains the pair
            }
            for p in word.symbols.windows(2) {
                *delta.entry((p[0], p[1])).or_default() -= word.count;
            }
            for p in merged_symbols.windows(2) {
                let p = (p[0], p[1]);
                *delta.entry(p).or_default() += word.count;
                if p.0 == new_id || p.1 == new_id {
                    pair_words.entry(p).or_default().push(w);
                }
            }
            std::mem::swap(&mut word.symbols, &mut merged_symbols);
        }

        for (&p, &d) in &delta {
            if d == 0 {
                continue;
            }
            let count = pair_counts.entry(p).or_default();
            *count += d;
            let count = *count;
            if count <= 0 {
                pair_counts.remove(&p);
            } else if d > 0 {
                // Decreases leave an entry in the heap that is too high, which gets
                // fixed when it is popped. Increases need a fresh entry.
                heap.push(Candidate { count, pair: p });
            }
        }
    }

    vocab
        .into_iter()
        .enumerate()
        .map(|(rank, bytes)| (bytes, rank as Rank))
        .collect()
}

/// Writes `symbols` with every non-overlapping occurrence of `pair` (left to right)
/// replaced by `new_id` into `out`. Returns false, leaving `out` unspecified, if the pair
/// does not occur.
fn merge_word(symbols: &[u32], pair: Pair, new_id: u32, out: &mut Vec<u32>) -> bool {
    out.clear();
    let mut found = false;
    let mut i = 0;
    while i < symbols.len() {
        if i + 1 < symbols.len() && (symbols[i], symbols[i + 1]) == pair {
            out.push(new_id);
            found = true;
            i += 2;
        } else {
            out.push(symbols[i]);
            i += 1;
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token_of(ranks: &Ranks, rank: Rank) -> Vec<u8> {
        ranks
            .iter()
            .find(|&(_, &r)| r == rank)
            .map(|(b, _)| b.clone())
            .unwrap()
    }

    #[test]
    fn rejects_tiny_vocab() {
        assert!(matches!(
            Trainer::new(10).train_ranks(&["x"]),
            Err(Error::VocabSizeTooSmall(10))
        ));
    }

    #[test]
    fn base_vocab_is_the_256_bytes() {
        let ranks = Trainer::new(256).train_ranks(&["hello"]).unwrap();
        assert_eq!(ranks.len(), 256);
        for b in 0..=255u8 {
            assert_eq!(ranks[[b].as_slice()], Rank::from(b));
        }
    }

    #[test]
    fn learns_the_most_frequent_pair_first() {
        // With a pattern that keeps everything in one piece, "ab" occurs three times and
        // wins; then "abab" pairs ("ab","ab") twice; and so on.
        let trainer = Trainer::new(259).pattern(r"[\s\S]+");
        let ranks = trainer.train_ranks(&["abababx"]).unwrap();
        assert_eq!(token_of(&ranks, 256), b"ab");
        assert_eq!(token_of(&ranks, 257), b"abab");
    }

    #[test]
    fn counts_are_weighted_by_piece_frequency() {
        // Pieces are counted once and weighted, so " ab" occurring three times beats
        // "cd" and "ef", which each occur once but twice within the piece.
        let text = "cdcd efef ab ab ab";
        let ranks = Trainer::new(257).train_ranks(&[text]).unwrap();
        assert_eq!(token_of(&ranks, 256), b" a");
    }

    #[test]
    fn repeated_symbols_merge_left_to_right() {
        let mut out = Vec::new();
        assert!(merge_word(&[1, 1, 1], (1, 1), 9, &mut out));
        assert_eq!(out, [9, 1]);
        assert!(merge_word(&[1, 1, 1, 1], (1, 1), 9, &mut out));
        assert_eq!(out, [9, 9]);
        assert!(!merge_word(&[1, 2, 1], (2, 2), 9, &mut out));
    }

    #[test]
    fn stops_when_pairs_run_out() {
        let ranks = Trainer::new(10_000).train_ranks(&["abc"]).unwrap();
        // "abc" has two merges available: one pair, then the pair it forms.
        assert_eq!(ranks.len(), 258);
    }

    #[test]
    fn min_frequency_stops_early() {
        let ranks = Trainer::new(10_000)
            .min_frequency(2)
            .train_ranks(&["x aa aa bc"])
            .unwrap();
        // " aa" occurs twice, so its pairs qualify; "bc" occurs once and does not.
        assert!(ranks.contains_key(b" a".as_slice()));
        assert!(!ranks.contains_key(b" b".as_slice()));
        assert!(!ranks.contains_key(b"bc".as_slice()));
    }

    #[test]
    fn training_is_deterministic() {
        let docs: Vec<String> = (0..200)
            .map(|i| format!("document {i}: the quick brown fox {} jumps", i * 37 % 11))
            .collect();
        let a = Trainer::new(400).train_ranks(&docs).unwrap();
        let b = Trainer::new(400).train_ranks(&docs).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn duplicate_byte_strings_share_an_id() {
        // Force ("a","bc") and ("ab","c") style collisions: the ranks must stay a
        // bijection between byte strings and ids.
        let docs = ["abc abc abc ab ab bc bc bc bc a"];
        let ranks = Trainer::new(300)
            .pattern(r"[\s\S]+")
            .train_ranks(&docs)
            .unwrap();
        let mut seen: Vec<Rank> = ranks.values().copied().collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..ranks.len() as Rank).collect::<Vec<_>>());
    }

    #[test]
    fn trained_encoding_round_trips_and_compresses() {
        let docs: Vec<String> = (0..50)
            .map(|i| format!("fn main() {{ println!(\"hello {i}\"); }}\n"))
            .collect();
        let enc = Trainer::new(320)
            .special_tokens(["<|eot|>"])
            .train("code", &docs)
            .unwrap();
        // The corpus is repetitive enough to run out of pairs before 320 tokens.
        let learned = enc.ranks().len();
        assert!(learned <= 320);
        assert_eq!(enc.special_token("<|eot|>"), Some(learned as Rank));
        let text = "fn main() { println!(\"hello 7\"); }\n";
        let tokens = enc.encode_ordinary(text);
        assert_eq!(enc.decode(&tokens).unwrap(), text);
        assert!(tokens.len() * 3 < text.len(), "{} tokens", tokens.len());
    }
}
