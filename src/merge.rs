//! Byte-pair merging of a single pre-tokenized piece.
//!
//! Given the bytes of one piece (say `" tokenizer"`), BPE starts with one symbol per byte
//! and repeatedly merges the adjacent pair whose concatenation has the lowest rank in the
//! vocabulary. When no adjacent pair is in the vocabulary any more, the remaining symbols
//! are the tokens. When several pairs share the lowest rank (they then spell the same
//! bytes), the leftmost one is merged first. That matches tiktoken.
//!
//! Two implementations live here and produce identical output:
//!
//! * [`encode_piece_scan`] keeps a flat vector of symbol boundaries and rescans it for the
//!   minimum after every merge. That is `O(n * m)` for `n` bytes and `m` merges, which is
//!   quadratic in the worst case, but for the short pieces that make up almost all real
//!   text it is hard to beat: no allocation beyond one small vector, and everything sits
//!   in cache. This is the same approach tiktoken uses.
//!
//! * [`encode_piece_heap`] keeps the symbols in a doubly linked list (stored as `prev` and
//!   `next` arrays indexed by byte offset) and every candidate merge in a binary min-heap
//!   keyed by `(rank, start)`. A merge only changes the two pairs that touch the merged
//!   symbol, so each merge is `O(log n)` and a whole piece is `O(n log n)`. Stale heap
//!   entries are not removed eagerly. They are recognised and skipped when popped.
//!
//! [`encode_piece`] picks between them based on piece length.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::{Rank, Ranks};

/// Pieces at least this long use the heap. Shorter pieces use the linear scan.
///
/// The value comes from the `merge` group in `benches/encode.rs`; see the README for
/// the measurements. Both paths give identical results, so this only affects speed.
pub const HEAP_THRESHOLD: usize = 128;

/// Sentinel rank for "these two symbols cannot be merged". Real ranks may not use it.
pub(crate) const NO_MERGE: Rank = Rank::MAX;

/// Encodes one piece, appending its token ids to `out`.
///
/// Every single byte must be present in `ranks`. [`crate::Encoding`] checks this when it
/// is constructed; calling this with an incomplete vocabulary panics.
#[inline]
pub fn encode_piece(ranks: &Ranks, piece: &[u8], out: &mut Vec<Rank>) {
    if piece.len() < HEAP_THRESHOLD {
        encode_piece_scan(ranks, piece, out);
    } else {
        encode_piece_heap(ranks, piece, out);
    }
}

#[inline]
fn rank_of(ranks: &Ranks, bytes: &[u8]) -> Rank {
    ranks.get(bytes).copied().unwrap_or(NO_MERGE)
}

#[inline]
fn token_for(ranks: &Ranks, bytes: &[u8]) -> Rank {
    match ranks.get(bytes) {
        Some(&rank) => rank,
        None => panic!("no rank for bytes {bytes:?}; the vocabulary must contain every byte"),
    }
}

/// Linear-scan merge. See the module docs.
pub fn encode_piece_scan(ranks: &Ranks, piece: &[u8], out: &mut Vec<Rank>) {
    let n = piece.len();
    if n == 0 {
        return;
    }

    // parts[i] = (start offset of symbol i, rank of merging symbol i with symbol i + 1).
    // The final entry is a sentinel holding the end offset, so symbol i always spans
    // parts[i].0..parts[i + 1].0.
    let mut parts: Vec<(usize, Rank)> = Vec::with_capacity(n + 1);
    for i in 0..n - 1 {
        parts.push((i, rank_of(ranks, &piece[i..i + 2])));
    }
    parts.push((n - 1, NO_MERGE));
    parts.push((n, NO_MERGE));

    loop {
        let mut best = NO_MERGE;
        let mut best_i = 0;
        for (i, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
            if rank < best {
                best = rank;
                best_i = i;
            }
        }
        if best == NO_MERGE {
            break;
        }

        // Merge symbol best_i with its right neighbour by dropping the boundary between
        // them, then refresh the two pair ranks that involve the merged symbol.
        parts.remove(best_i + 1);
        parts[best_i].1 = match parts.get(best_i + 2) {
            Some(&(end, _)) => rank_of(ranks, &piece[parts[best_i].0..end]),
            None => NO_MERGE,
        };
        if best_i > 0 {
            parts[best_i - 1].1 = rank_of(ranks, &piece[parts[best_i - 1].0..parts[best_i + 1].0]);
        }
    }

    out.extend(
        parts
            .windows(2)
            .map(|w| token_for(ranks, &piece[w[0].0..w[1].0])),
    );
}

/// A merge that was possible when it was pushed: the symbol starting at `start` followed
/// by the symbol ending at `end`. Field order gives the heap order: lowest rank first,
/// then leftmost.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Candidate {
    rank: Rank,
    start: u32,
    end: u32,
}

/// Marks a byte offset that is no longer the start of a symbol.
const DEAD: u32 = u32::MAX;

/// Heap-based merge. See the module docs.
///
/// # Panics
///
/// Panics if the piece is 4 GiB or longer.
pub fn encode_piece_heap(ranks: &Ranks, piece: &[u8], out: &mut Vec<Rank>) {
    let n = piece.len();
    if n == 0 {
        return;
    }
    let n32 = u32::try_from(n).expect("piece longer than u32::MAX bytes");

    // Symbols are identified by their start offset. For a live symbol starting at i,
    // next[i] is where the following symbol starts (n for the last one) and prev[i] is
    // where the preceding one starts (DEAD for the first). Offsets that are no longer
    // symbol starts have next[i] == DEAD.
    let mut next: Vec<u32> = (1..=n32).collect();
    let mut prev: Vec<u32> = (0..n32)
        .map(|i| if i == 0 { DEAD } else { i - 1 })
        .collect();

    let mut heap = BinaryHeap::with_capacity(n);
    let push = |heap: &mut BinaryHeap<Reverse<Candidate>>, start: u32, end: u32| {
        let rank = rank_of(ranks, &piece[start as usize..end as usize]);
        if rank != NO_MERGE {
            heap.push(Reverse(Candidate { rank, start, end }));
        }
    };
    for i in 0..n32 - 1 {
        push(&mut heap, i, i + 2);
    }

    while let Some(Reverse(Candidate { start, end, .. })) = heap.pop() {
        // The candidate is still valid only if `start` is a live symbol and the symbol
        // after it still ends at `end`. Symbols only ever grow, so if either neighbour
        // has merged with something else since this entry was pushed, one of these
        // checks fails. If both pass, the pair covers exactly the same bytes and so
        // still has the same rank.
        let mid = next[start as usize];
        if mid == DEAD || mid >= n32 || next[mid as usize] != end {
            continue;
        }

        next[start as usize] = end;
        next[mid as usize] = DEAD;
        if end < n32 {
            prev[end as usize] = start;
        }

        let before = prev[start as usize];
        if before != DEAD {
            push(&mut heap, before, end);
        }
        if end < n32 {
            push(&mut heap, start, next[end as usize]);
        }
    }

    let mut i = 0;
    while i < n {
        let j = next[i] as usize;
        out.push(token_for(ranks, &piece[i..j]));
        i = j;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn byte_vocab(extra: &[&str]) -> Ranks {
        let mut ranks: Ranks = (0..=255u8).map(|b| (vec![b], Rank::from(b))).collect();
        for (i, token) in extra.iter().enumerate() {
            ranks.insert(token.as_bytes().to_vec(), 256 + i as Rank);
        }
        ranks
    }

    fn both(ranks: &Ranks, piece: &str) -> Vec<Rank> {
        let mut scan = Vec::new();
        let mut heap = Vec::new();
        encode_piece_scan(ranks, piece.as_bytes(), &mut scan);
        encode_piece_heap(ranks, piece.as_bytes(), &mut heap);
        assert_eq!(scan, heap, "scan and heap disagree on {piece:?}");
        scan
    }

    #[test]
    fn empty_and_single_byte() {
        let ranks = byte_vocab(&[]);
        assert_eq!(both(&ranks, ""), Vec::<Rank>::new());
        assert_eq!(both(&ranks, "a"), vec![Rank::from(b'a')]);
    }

    #[test]
    fn no_merges_means_one_token_per_byte() {
        let ranks = byte_vocab(&[]);
        assert_eq!(both(&ranks, "abc"), vec![97, 98, 99]);
    }

    #[test]
    fn lower_rank_wins() {
        // "ab" is ranked before "bc", so "abc" becomes [ab, c].
        let ranks = byte_vocab(&["ab", "bc"]);
        assert_eq!(both(&ranks, "abc"), vec![256, 99]);

        // Swap the ranks and the other pair wins: [a, bc].
        let ranks = byte_vocab(&["bc", "ab"]);
        assert_eq!(both(&ranks, "abc"), vec![97, 256]);
    }

    #[test]
    fn merges_chain_through_intermediate_tokens() {
        // ab -> abc needs "ab" first, then "abc" is formed from [ab, c].
        let ranks = byte_vocab(&["ab", "abc", "abcd"]);
        assert_eq!(both(&ranks, "abcd"), vec![258]);
    }

    #[test]
    fn unreachable_token_is_not_used() {
        // "abc" is in the vocab, but neither "ab" nor "bc" is, so it can never be formed.
        let ranks = byte_vocab(&["abc"]);
        assert_eq!(both(&ranks, "abc"), vec![97, 98, 99]);
    }

    #[test]
    fn ties_go_to_the_leftmost_pair() {
        // "aaa": both (a, a) pairs have the same rank; the left one merges first.
        let ranks = byte_vocab(&["aa"]);
        assert_eq!(both(&ranks, "aaa"), vec![256, 97]);
        assert_eq!(both(&ranks, "aaaaa"), vec![256, 256, 97]);
    }

    #[test]
    fn readme_worked_example() {
        // The example walked through in the README.
        let ranks = byte_vocab(&["ab", "bc", "abc", "cd"]);
        assert_eq!(both(&ranks, "abcd"), vec![258, Rank::from(b'd')]);
    }

    #[test]
    fn long_runs_agree() {
        let ranks = byte_vocab(&["aa", "aaaa", "ab", "aab", "ba", "aaaaaaaa"]);
        let piece: String = (0..5000)
            .map(|i| if i % 7 == 3 { 'b' } else { 'a' })
            .collect();
        let tokens = both(&ranks, &piece);
        assert!(tokens.len() < piece.len() / 2);
    }
}
