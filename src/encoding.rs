//! The [`Encoding`] type: pre-tokenization, special tokens, encoding and decoding.

use std::cmp::Reverse;
use std::fmt;
use std::path::Path;
use std::sync::OnceLock;

use aho_corasick::{AhoCorasick, MatchKind};
use fancy_regex::Regex;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::merge::{self, NO_MERGE};
use crate::presets::Preset;
use crate::{Error, Rank, Ranks, Result, rank_file};

/// A set of special tokens, used to say which ones are allowed or disallowed in a call
/// to [`Encoding::encode`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SpecialTokens<'a> {
    /// No special tokens.
    #[default]
    None,
    /// Every special token the encoding defines.
    All,
    /// Just these strings. As in tiktoken, an allowed string that is not one of the
    /// encoding's special tokens has no effect, while a disallowed string is an error
    /// wherever it appears, special token or not.
    Only(&'a [&'a str]),
}

impl SpecialTokens<'_> {
    fn contains(self, token: &str) -> bool {
        match self {
            SpecialTokens::None => false,
            SpecialTokens::All => true,
            SpecialTokens::Only(tokens) => tokens.contains(&token),
        }
    }
}

/// A byte-level BPE encoding: a split pattern, a table of mergeable ranks and a set of
/// special tokens.
///
/// Encoding a string happens in three steps:
///
/// 1. Special tokens that the caller allowed are found and cut out of the text.
/// 2. The text between them is split into pieces with the pre-tokenization regex, so
///    merges never cross, for example, a word boundary.
/// 3. Each piece is looked up whole in the vocabulary, and if it is not there it is
///    merged byte pair by byte pair ([`crate::merge`]).
///
/// `Encoding` is `Send + Sync`. The batch methods share one instance across threads.
pub struct Encoding {
    name: String,
    pattern: String,
    /// Per-thread copies of the split regex, compiled on first use. See [`Encoding::regex`].
    regexes: Box<[OnceLock<Regex>]>,
    encoder: Ranks,
    decoder: FxHashMap<Rank, Box<[u8]>>,
    /// Sorted by rank.
    special_tokens: Vec<(String, Rank)>,
    /// Finds any special token. `None` if there are none.
    all_specials: Option<SpecialMatcher>,
    max_token_value: Rank,
}

/// Finds special tokens in text: the leftmost match, and the longest one if several
/// start at the same place.
struct SpecialMatcher {
    automaton: AhoCorasick,
    /// Token id for each automaton pattern.
    ranks: Vec<Rank>,
}

impl SpecialMatcher {
    fn new<'a>(tokens: impl IntoIterator<Item = &'a (String, Rank)>) -> Option<Self> {
        let (patterns, ranks): (Vec<&str>, Vec<Rank>) =
            tokens.into_iter().map(|(t, r)| (t.as_str(), *r)).unzip();
        if patterns.is_empty() {
            return None;
        }
        let automaton = AhoCorasick::builder()
            .match_kind(MatchKind::LeftmostLongest)
            .build(patterns)
            .expect("a handful of literal special tokens always fits in an automaton");
        Some(Self { automaton, ranks })
    }

    /// `(start, end, rank)` of the first special token at or after byte `from`.
    fn find(&self, text: &str, from: usize) -> Option<(usize, usize, Rank)> {
        let input = aho_corasick::Input::new(text).span(from..text.len());
        let m = self.automaton.find(input)?;
        Some((m.start(), m.end(), self.ranks[m.pattern().as_usize()]))
    }
}

impl fmt::Debug for Encoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Encoding")
            .field("name", &self.name)
            .field("pattern", &self.pattern)
            .field("mergeable_tokens", &self.encoder.len())
            .field("special_tokens", &self.special_tokens)
            .finish_non_exhaustive()
    }
}

/// How many copies of the split regex each encoding can hold. See [`Encoding::regex`].
/// Each compiled copy of the cl100k or o200k pattern costs somewhere around a megabyte
/// once it has been used, so this is kept to a small multiple of the core count.
fn regex_slots() -> usize {
    let cores = std::thread::available_parallelism().map_or(8, |n| n.get());
    (2 * cores).clamp(2, 64)
}

pub(crate) fn compile_pattern(pattern: &str) -> Result<Regex> {
    Regex::new(pattern).map_err(|e| Error::Pattern(Box::new(e)))
}

/// Splits `text` into pre-tokenization pieces. `base` is added to byte offsets in
/// errors, for when `text` is a slice of something bigger.
///
/// Matching can fail: fancy-regex caps its backtracking stack at a million entries, and
/// `\s+(?!\S)` needs one entry per character of a whitespace run. So a run of about a
/// million spaces can't be split. tiktoken panics on the same input; this returns an
/// error instead.
pub(crate) fn pieces<'r, 't>(
    regex: &'r Regex,
    text: &'t str,
    base: usize,
) -> impl Iterator<Item = Result<&'t str>> + 'r
where
    't: 'r,
{
    let mut last_end = 0;
    regex.find_iter(text).map(move |m| match m {
        Ok(m) => {
            last_end = m.end();
            Ok(m.as_str())
        }
        Err(e) => Err(Error::PreTokenize {
            offset: base + last_end,
            source: Box::new(e),
        }),
    })
}

impl Encoding {
    /// Builds an encoding from its parts.
    ///
    /// Fails if the pattern does not compile, if any single byte is missing from `ranks`
    /// (every input must be encodable), or if ranks collide.
    pub fn new(
        name: impl Into<String>,
        pattern: &str,
        ranks: Ranks,
        special_tokens: impl IntoIterator<Item = (String, Rank)>,
    ) -> Result<Self> {
        let regex = compile_pattern(pattern)?;

        for byte in 0..=255u8 {
            if !ranks.contains_key([byte].as_slice()) {
                return Err(Error::MissingByte(byte));
            }
        }

        let mut decoder: FxHashMap<Rank, Box<[u8]>> =
            FxHashMap::with_capacity_and_hasher(ranks.len(), Default::default());
        for (bytes, &rank) in &ranks {
            if rank == NO_MERGE {
                return Err(Error::ReservedRank(rank));
            }
            if decoder.insert(rank, bytes.as_slice().into()).is_some() {
                return Err(Error::DuplicateRank(rank));
            }
        }

        let mut specials: Vec<(String, Rank)> = special_tokens.into_iter().collect();
        specials.sort_by_key(|&(_, rank)| rank);
        for (i, (token, rank)) in specials.iter().enumerate() {
            if token.is_empty() || specials[..i].iter().any(|(t, _)| t == token) {
                return Err(Error::InvalidSpecialToken(token.clone()));
            }
            if *rank == NO_MERGE {
                return Err(Error::ReservedRank(*rank));
            }
            if decoder.insert(*rank, token.as_bytes().into()).is_some() {
                return Err(Error::DuplicateRank(*rank));
            }
        }

        let all_specials = SpecialMatcher::new(&specials);
        let max_token_value = decoder.keys().copied().max().unwrap_or(0);

        // Keep the copy compiled for validation as the first slot; the rest are
        // compiled lazily by whichever threads end up using them.
        let regexes: Box<[OnceLock<Regex>]> = (0..regex_slots()).map(|_| OnceLock::new()).collect();
        let _ = regexes[0].set(regex);

        Ok(Self {
            name: name.into(),
            pattern: pattern.to_owned(),
            regexes,
            encoder: ranks,
            decoder,
            special_tokens: specials,
            all_specials,
            max_token_value,
        })
    }

    /// Loads a tiktoken rank file and builds an encoding around it.
    pub fn from_tiktoken_file(
        name: impl Into<String>,
        path: impl AsRef<Path>,
        pattern: &str,
        special_tokens: impl IntoIterator<Item = (String, Rank)>,
    ) -> Result<Self> {
        Self::new(name, pattern, rank_file::load(path)?, special_tokens)
    }

    /// Builds one of the published encodings from its rank table.
    pub fn from_preset(preset: &Preset, ranks: Ranks) -> Result<Self> {
        let specials = preset
            .special_tokens
            .iter()
            .map(|&(t, r)| (t.to_owned(), r));
        Self::new(preset.name, preset.pattern, ranks, specials)
    }

    /// Builds one of the published encodings from a rank file on disk.
    pub fn from_preset_file(preset: &Preset, path: impl AsRef<Path>) -> Result<Self> {
        Self::from_preset(preset, rank_file::load(path)?)
    }

    /// The split regex for the current thread.
    ///
    /// Matching needs scratch space, which the regex engine keeps in a pool inside each
    /// compiled regex. With a dozen threads sharing one regex, that pool becomes a point
    /// of contention and batch encoding stops scaling. tiktoken hands each thread a
    /// clone for the same reason, but with fancy-regex 0.19 clones still share the
    /// inner delegate regexes (and their pools), so here each slot compiles its own copy
    /// the first time a thread lands on it.
    fn regex(&self) -> &Regex {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT_SLOT: AtomicUsize = AtomicUsize::new(0);
        thread_local! {
            static SLOT: usize = NEXT_SLOT.fetch_add(1, Ordering::Relaxed);
        }
        let slot = SLOT.with(|&slot| slot) % self.regexes.len();
        self.regexes[slot]
            .get_or_init(|| compile_pattern(&self.pattern).expect("pattern compiled once already"))
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// The pre-tokenization regex this encoding splits text with.
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// The mergeable (non-special) tokens and their ranks.
    pub fn ranks(&self) -> &Ranks {
        &self.encoder
    }

    /// Special tokens and their ids, in id order.
    pub fn special_tokens(&self) -> impl ExactSizeIterator<Item = (&str, Rank)> {
        self.special_tokens.iter().map(|(t, r)| (t.as_str(), *r))
    }

    /// The id of a special token, if the encoding defines it.
    pub fn special_token(&self, token: &str) -> Option<Rank> {
        self.special_tokens
            .iter()
            .find(|(t, _)| t == token)
            .map(|&(_, r)| r)
    }

    /// The largest token id, special tokens included.
    pub fn max_token_value(&self) -> Rank {
        self.max_token_value
    }

    /// `max_token_value() + 1`, which is what tiktoken reports as `n_vocab`.
    pub fn n_vocab(&self) -> usize {
        self.max_token_value() as usize + 1
    }

    /// Splits text the way the encoder does before merging. Useful for seeing why a
    /// string tokenizes the way it does.
    pub fn split<'t>(&self, text: &'t str) -> Result<Vec<&'t str>> {
        pieces(self.regex(), text, 0).collect()
    }

    /// Encodes text, treating any special-token text as ordinary text.
    ///
    /// The only possible error is [`Error::PreTokenize`], for inputs the split regex
    /// cannot handle (see [`Error::PreTokenize`]).
    pub fn encode_ordinary(&self, text: &str) -> Result<Vec<Rank>> {
        let mut out = Vec::with_capacity(text.len() / 4 + 1);
        self.encode_ordinary_into(text, 0, &mut out)?;
        Ok(out)
    }

    /// Encodes `text` (which starts at byte `base` of the caller's input) onto `out`.
    fn encode_ordinary_into(&self, text: &str, base: usize, out: &mut Vec<Rank>) -> Result<()> {
        for piece in pieces(self.regex(), text, base) {
            let bytes = piece?.as_bytes();
            match self.encoder.get(bytes) {
                Some(&rank) => out.push(rank),
                None => merge::encode_piece(&self.encoder, bytes, out),
            }
        }
        Ok(())
    }

    /// Encodes text with tiktoken's special-token rules.
    ///
    /// Occurrences of `allowed` special tokens are encoded as their special id. If the
    /// text contains a `disallowed` special token the call fails. As in tiktoken,
    /// `disallowed = SpecialTokens::All` means "every special token not in `allowed`",
    /// and special-token text that is neither allowed nor disallowed is encoded as
    /// ordinary text.
    ///
    /// tiktoken's defaults (`allowed_special=set()`, `disallowed_special="all"`) are
    /// `SpecialTokens::None, SpecialTokens::All`.
    pub fn encode(
        &self,
        text: &str,
        allowed: SpecialTokens<'_>,
        disallowed: SpecialTokens<'_>,
    ) -> Result<Vec<Rank>> {
        self.check_disallowed(text, allowed, disallowed)?;
        self.encode_allowing(text, allowed)
    }

    /// Encodes text, turning every special token it contains into its special id.
    pub fn encode_with_special_tokens(&self, text: &str) -> Result<Vec<Rank>> {
        self.encode_allowing(text, SpecialTokens::All)
    }

    /// Fails with the leftmost disallowed string in `text`, if there is one.
    fn check_disallowed(
        &self,
        text: &str,
        allowed: SpecialTokens<'_>,
        disallowed: SpecialTokens<'_>,
    ) -> Result<()> {
        let leftmost = |candidates: &mut dyn Iterator<Item = &str>| {
            candidates
                .filter_map(|t| {
                    text.find(t)
                        .map(|pos| (pos, Reverse(t.len()), t.to_owned()))
                })
                .min()
        };
        let hit = match disallowed {
            SpecialTokens::None => None,
            SpecialTokens::All => leftmost(
                &mut self
                    .special_tokens
                    .iter()
                    .map(|(t, _)| t.as_str())
                    .filter(|t| !allowed.contains(t)),
            ),
            SpecialTokens::Only(strings) => leftmost(&mut strings.iter().copied()),
        };
        match hit {
            Some((_, _, token)) => Err(Error::DisallowedSpecialToken(token)),
            None => Ok(()),
        }
    }

    fn encode_allowing(&self, text: &str, allowed: SpecialTokens<'_>) -> Result<Vec<Rank>> {
        // Search only for the allowed tokens, so that an allowed token is found even if a
        // longer, not-allowed one starts at the same place.
        let subset;
        let matcher = match allowed {
            SpecialTokens::None => None,
            SpecialTokens::All => self.all_specials.as_ref(),
            SpecialTokens::Only(list) => {
                subset = SpecialMatcher::new(
                    self.special_tokens
                        .iter()
                        .filter(|(t, _)| list.contains(&t.as_str())),
                );
                subset.as_ref()
            }
        };
        let Some(matcher) = matcher else {
            return self.encode_ordinary(text);
        };

        let mut out = Vec::with_capacity(text.len() / 4 + 1);
        let mut start = 0;
        loop {
            let found = matcher.find(text, start);
            let end = found.map_or(text.len(), |(s, _, _)| s);
            self.encode_ordinary_into(&text[start..end], start, &mut out)?;
            match found {
                Some((_, special_end, rank)) => {
                    out.push(rank);
                    start = special_end;
                }
                None => return Ok(out),
            }
        }
    }

    /// Encodes many texts in parallel on the rayon thread pool.
    pub fn encode_ordinary_batch<S: AsRef<str> + Sync>(
        &self,
        texts: &[S],
    ) -> Result<Vec<Vec<Rank>>> {
        texts
            .par_iter()
            .map(|t| self.encode_ordinary(t.as_ref()))
            .collect()
    }

    /// Parallel [`Encoding::encode`]. Fails if any text contains a disallowed token.
    pub fn encode_batch<S: AsRef<str> + Sync>(
        &self,
        texts: &[S],
        allowed: SpecialTokens<'_>,
        disallowed: SpecialTokens<'_>,
    ) -> Result<Vec<Vec<Rank>>> {
        texts
            .par_iter()
            .map(|t| self.encode(t.as_ref(), allowed, disallowed))
            .collect()
    }

    /// The bytes of a single token, special tokens included.
    pub fn decode_single_token_bytes(&self, token: Rank) -> Result<&[u8]> {
        self.decoder
            .get(&token)
            .map(AsRef::as_ref)
            .ok_or(Error::UnknownToken(token))
    }

    /// Concatenates the bytes of each token. The result is not necessarily valid UTF-8:
    /// a multi-byte character can be split across tokens.
    pub fn decode_bytes(&self, tokens: &[Rank]) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(tokens.len() * 4);
        for &token in tokens {
            out.extend_from_slice(self.decode_single_token_bytes(token)?);
        }
        Ok(out)
    }

    /// Decodes tokens to a string, replacing invalid UTF-8 with U+FFFD.
    pub fn decode(&self, tokens: &[Rank]) -> Result<String> {
        let bytes = self.decode_bytes(tokens)?;
        Ok(match String::from_utf8(bytes) {
            Ok(s) => s,
            Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
        })
    }

    /// Writes the mergeable ranks to a tiktoken rank file. Special tokens and the
    /// pattern are not part of that format.
    pub fn save_tiktoken_file(&self, path: impl AsRef<Path>) -> Result<()> {
        rank_file::save(&self.encoder, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE_PATTERN: &str = r"\s?\S+|\s+(?!\S)|\s+";

    fn tiny(extra: &[&str]) -> Encoding {
        let mut ranks: Ranks = (0..=255u8).map(|b| (vec![b], Rank::from(b))).collect();
        for (i, token) in extra.iter().enumerate() {
            ranks.insert(token.as_bytes().to_vec(), 256 + i as Rank);
        }
        let specials = [("<|end|>".to_owned(), 1000), ("<|pad|>".to_owned(), 1001)];
        Encoding::new("tiny", SIMPLE_PATTERN, ranks, specials).unwrap()
    }

    #[test]
    fn is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Encoding>();
    }

    #[test]
    fn encodes_and_decodes() {
        let enc = tiny(&["he", "ll", "hell", "hello", " w", " wo"]);
        let tokens = enc.encode_ordinary("hello world").unwrap();
        assert_eq!(tokens[0], 259, "whole piece `hello` is a token");
        assert_eq!(enc.decode(&tokens).unwrap(), "hello world");
    }

    #[test]
    fn split_follows_the_pattern() {
        let enc = tiny(&[]);
        assert_eq!(enc.split("a bb  c").unwrap(), ["a", " bb", " ", " c"]);
    }

    #[test]
    fn special_tokens_default_to_an_error() {
        let enc = tiny(&[]);
        let err = enc
            .encode("hi <|end|>", SpecialTokens::None, SpecialTokens::All)
            .unwrap_err();
        assert!(matches!(err, Error::DisallowedSpecialToken(ref t) if t == "<|end|>"));
    }

    #[test]
    fn allowed_special_tokens_become_one_id() {
        let enc = tiny(&[]);
        let tokens = enc
            .encode(
                "a<|end|>b",
                SpecialTokens::Only(&["<|end|>"]),
                SpecialTokens::All,
            )
            .unwrap();
        assert_eq!(tokens, [97, 1000, 98]);
        assert_eq!(
            enc.encode_with_special_tokens("<|pad|><|end|>").unwrap(),
            [1001, 1000]
        );
    }

    #[test]
    fn unlisted_special_tokens_are_ordinary_text_when_not_disallowed() {
        let enc = tiny(&[]);
        let text = "x<|pad|>y<|end|>";
        let tokens = enc
            .encode(text, SpecialTokens::Only(&["<|end|>"]), SpecialTokens::None)
            .unwrap();
        assert_eq!(*tokens.last().unwrap(), 1000);
        assert!(!tokens.contains(&1001));
        assert_eq!(enc.decode(&tokens).unwrap(), text);
        assert_eq!(enc.encode_ordinary(text).unwrap().len(), text.len());
    }

    #[test]
    fn disallowing_a_subset_only_checks_that_subset() {
        let enc = tiny(&[]);
        assert!(
            enc.encode(
                "<|pad|>",
                SpecialTokens::None,
                SpecialTokens::Only(&["<|end|>"])
            )
            .is_ok()
        );
        assert!(
            enc.encode(
                "<|end|>",
                SpecialTokens::None,
                SpecialTokens::Only(&["<|end|>"])
            )
            .is_err()
        );
    }

    #[test]
    fn overlapping_special_tokens_follow_tiktoken() {
        let ranks: Ranks = (0..=255u8).map(|b| (vec![b], Rank::from(b))).collect();
        let specials = [("<s>".to_owned(), 300), ("<s>x".to_owned(), 301)];
        let enc = Encoding::new("x", SIMPLE_PATTERN, ranks, specials).unwrap();
        let only_short = SpecialTokens::Only(&["<s>"]);
        let only_long = SpecialTokens::Only(&["<s>x"]);

        // The longer token is preferred when both are allowed...
        assert_eq!(enc.encode_with_special_tokens("<s>x").unwrap(), [301]);
        // ...but an allowed token is still found when a longer, unlisted one overlaps it.
        let tokens = enc.encode("<s>x", only_short, SpecialTokens::None).unwrap();
        assert_eq!(tokens, [300, u32::from(b'x')]);
        // Disallowed text is found even inside a longer special token.
        assert!(enc.encode("<s>x", SpecialTokens::None, only_short).is_err());
        assert!(enc.encode("<s>x", only_long, SpecialTokens::All).is_err());
    }

    #[test]
    fn any_disallowed_string_is_an_error() {
        // tiktoken raises for every string in disallowed_special, special token or not.
        let enc = tiny(&[]);
        let err = enc
            .encode(
                "say hello",
                SpecialTokens::None,
                SpecialTokens::Only(&["hello"]),
            )
            .unwrap_err();
        assert!(matches!(err, Error::DisallowedSpecialToken(ref t) if t == "hello"));
    }

    #[test]
    fn decoding_split_utf8_is_lossy() {
        let enc = tiny(&[]);
        let tokens = enc.encode_ordinary("é").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(enc.decode_bytes(&tokens[..1]).unwrap(), [0xc3]);
        assert_eq!(enc.decode(&tokens[..1]).unwrap(), "\u{fffd}");
    }

    #[test]
    fn unknown_tokens_are_an_error() {
        let enc = tiny(&[]);
        assert!(matches!(
            enc.decode(&[5000]),
            Err(Error::UnknownToken(5000))
        ));
    }

    #[test]
    fn constructor_validates() {
        let full: Ranks = (0..=255u8).map(|b| (vec![b], Rank::from(b))).collect();
        let none = std::iter::empty();

        let mut missing = full.clone();
        missing.remove([7u8].as_slice());
        assert!(matches!(
            Encoding::new("x", SIMPLE_PATTERN, missing, none),
            Err(Error::MissingByte(7))
        ));

        let mut dup = full.clone();
        dup.insert(b"ab".to_vec(), 3);
        assert!(matches!(
            Encoding::new("x", SIMPLE_PATTERN, dup, std::iter::empty()),
            Err(Error::DuplicateRank(3))
        ));

        assert!(matches!(
            Encoding::new("x", "(", full.clone(), std::iter::empty()),
            Err(Error::Pattern(_))
        ));

        let clash = [("<s>".to_owned(), 5)];
        assert!(matches!(
            Encoding::new("x", SIMPLE_PATTERN, full.clone(), clash),
            Err(Error::DuplicateRank(5))
        ));

        let twice = [("<s>".to_owned(), 300), ("<s>".to_owned(), 301)];
        assert!(matches!(
            Encoding::new("x", SIMPLE_PATTERN, full, twice),
            Err(Error::InvalidSpecialToken(_))
        ));
    }

    #[test]
    fn batch_matches_sequential() {
        let enc = tiny(&["he", "ll", "hell", "hello"]);
        let texts: Vec<String> = (0..200).map(|i| format!("hello {i} <|end|>")).collect();
        let batch = enc.encode_ordinary_batch(&texts).unwrap();
        for (text, tokens) in texts.iter().zip(&batch) {
            assert_eq!(&enc.encode_ordinary(text).unwrap(), tokens);
        }
        assert!(
            enc.encode_batch(&texts, SpecialTokens::None, SpecialTokens::All)
                .is_err()
        );
        let with_special = enc
            .encode_batch(&texts, SpecialTokens::All, SpecialTokens::None)
            .unwrap();
        assert!(with_special.iter().all(|t| *t.last().unwrap() == 1000));
    }

    #[test]
    fn enormous_whitespace_runs_are_an_error_not_a_panic() {
        let bytes: Ranks = (0..=255u8).map(|b| (vec![b], Rank::from(b))).collect();
        let enc = Encoding::new("x", crate::presets::CL100K_BASE_PATTERN, bytes, []).unwrap();
        let text = format!("abc{}x", " ".repeat(1_100_000));
        match enc.encode_ordinary(&text) {
            Err(Error::PreTokenize { offset, .. }) => assert_eq!(offset, 3),
            other => panic!("expected a pre-tokenization error, got {other:?}"),
        }
        // A long run that stays under the limit is fine.
        let text = format!("abc{}x", " ".repeat(100_000));
        assert_eq!(enc.encode_ordinary(&text).unwrap().len(), text.len());
    }

    #[test]
    fn reports_vocab_size_like_tiktoken() {
        let enc = tiny(&["ab"]);
        assert_eq!(enc.max_token_value(), 1001);
        assert_eq!(enc.n_vocab(), 1002);
        assert_eq!(enc.special_token("<|pad|>"), Some(1001));
        assert_eq!(
            enc.special_tokens().collect::<Vec<_>>(),
            [("<|end|>", 1000), ("<|pad|>", 1001)]
        );
    }
}
