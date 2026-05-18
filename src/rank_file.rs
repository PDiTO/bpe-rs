//! Reading and writing the tiktoken rank file format.
//!
//! A rank file is plain text with one token per line: the token's bytes in standard
//! base64, a single space, and the token's rank as a decimal integer.
//!
//! ```text
//! IQ== 0
//! Ig== 1
//! IHRoZQ== 279
//! ```
//!
//! Special tokens are not part of the file. tiktoken configures them in code, and so
//! does this crate (see [`crate::presets`]).

use std::fs;
use std::io::{self, Write};
use std::path::Path;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{Error, Rank, Result};

/// Mergeable token bytes mapped to their rank.
pub type Ranks = FxHashMap<Vec<u8>, Rank>;

/// Parses the contents of a tiktoken rank file.
///
/// Blank lines are ignored. Duplicate tokens and duplicate ranks are rejected, since
/// either would make encoding or decoding ambiguous.
pub fn parse(contents: &[u8]) -> Result<Ranks> {
    let mut ranks = Ranks::default();
    let mut seen_ranks = FxHashSet::default();

    for (index, line) in contents.split(|&b| b == b'\n').enumerate() {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let err = |reason: String| Error::RankFile {
            line: index + 1,
            reason,
        };

        // Split on runs of whitespace (which also drops a trailing \r), like tiktoken's
        // `line.split()`, and require exactly two fields.
        let mut fields = line
            .split(|b| b.is_ascii_whitespace())
            .filter(|f| !f.is_empty());
        let (Some(token_b64), Some(rank_text), None) =
            (fields.next(), fields.next(), fields.next())
        else {
            return Err(err("expected `<base64 token> <rank>`".into()));
        };

        let token = BASE64
            .decode(token_b64)
            .map_err(|e| err(format!("invalid base64: {e}")))?;
        let rank: Rank = std::str::from_utf8(rank_text)
            .ok()
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| err("rank is not a valid unsigned 32-bit integer".into()))?;

        if !seen_ranks.insert(rank) {
            return Err(err(format!("rank {rank} appears more than once")));
        }
        if ranks.insert(token, rank).is_some() {
            return Err(err("token appears more than once".into()));
        }
    }
    Ok(ranks)
}

/// Reads and parses a tiktoken rank file from disk.
pub fn load(path: impl AsRef<Path>) -> Result<Ranks> {
    parse(&fs::read(path)?)
}

/// Writes ranks in tiktoken format, one token per line in ascending rank order.
pub fn write<W: Write>(ranks: &Ranks, mut writer: W) -> io::Result<()> {
    let mut entries: Vec<(&[u8], Rank)> = ranks.iter().map(|(k, &v)| (k.as_slice(), v)).collect();
    entries.sort_unstable_by_key(|&(_, rank)| rank);
    for (token, rank) in entries {
        writeln!(writer, "{} {rank}", BASE64.encode(token))?;
    }
    writer.flush()
}

/// Writes ranks to `path` in tiktoken format.
pub fn save(ranks: &Ranks, path: impl AsRef<Path>) -> Result<()> {
    let file = fs::File::create(path)?;
    write(ranks, io::BufWriter::new(file))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tokens_and_ranks() {
        let ranks = parse(b"IQ== 0\nIg== 1\nIHRoZQ== 279\n").unwrap();
        assert_eq!(ranks.len(), 3);
        assert_eq!(ranks[b"!".as_slice()], 0);
        assert_eq!(ranks[b"\"".as_slice()], 1);
        assert_eq!(ranks[b" the".as_slice()], 279);
    }

    #[test]
    fn tolerates_crlf_blank_lines_and_extra_spaces() {
        let ranks = parse(b"IQ== 0\r\n\r\n  \nIg==  1 \n").unwrap();
        assert_eq!(ranks.len(), 2);
    }

    #[test]
    fn rejects_bad_lines() {
        for (input, needle) in [
            (&b"IQ==0\n"[..], "expected"),
            (b" 5\n", "expected"),
            (b"IQ== 0 extra\n", "expected"),
            (b"!!!! 0\n", "base64"),
            (b"IQ== -1\n", "rank"),
            (b"IQ== 0\nIg== 0\n", "more than once"),
            (b"IQ== 0\nIQ== 1\n", "more than once"),
        ] {
            let message = parse(input).unwrap_err().to_string();
            assert!(
                message.contains(needle),
                "{message:?} should mention {needle:?}"
            );
        }
    }

    #[test]
    fn reports_the_offending_line_number() {
        let err = parse(b"IQ== 0\nIg== 1\nnope\n").unwrap_err();
        assert!(matches!(err, Error::RankFile { line: 3, .. }), "{err:?}");
    }

    #[test]
    fn write_then_parse_round_trips() {
        let mut ranks = Ranks::default();
        for b in 0..=255u8 {
            ranks.insert(vec![b], Rank::from(b));
        }
        ranks.insert(b"hello".to_vec(), 256);
        ranks.insert(vec![0xff, 0x00, 0x80], 257);

        let mut buf = Vec::new();
        write(&ranks, &mut buf).unwrap();
        assert!(
            buf.starts_with(b"AA== 0\nAQ== 1\n"),
            "output should be sorted by rank"
        );
        assert_eq!(parse(&buf).unwrap(), ranks);
    }
}
