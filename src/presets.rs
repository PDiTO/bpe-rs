//! The published OpenAI encodings: split patterns, special tokens and where to get the
//! rank files.
//!
//! The rank files themselves are not bundled. They are several megabytes each and are
//! published by OpenAI at the URLs below. `scripts/fetch_data.sh` and the Python package
//! download them on demand into [`cache_dir`].

use std::env;
use std::path::PathBuf;

use crate::Rank;

/// Everything needed to rebuild one of tiktoken's encodings except the rank file.
#[derive(Debug, Clone, Copy)]
pub struct Preset {
    pub name: &'static str,
    /// The pre-tokenization regex, exactly as tiktoken publishes it.
    pub pattern: &'static str,
    pub special_tokens: &'static [(&'static str, Rank)],
    /// Where OpenAI publishes the rank file.
    pub url: &'static str,
    /// SHA-256 of the rank file, as pinned by tiktoken.
    pub sha256: &'static str,
}

pub const ENDOFTEXT: &str = "<|endoftext|>";
pub const FIM_PREFIX: &str = "<|fim_prefix|>";
pub const FIM_MIDDLE: &str = "<|fim_middle|>";
pub const FIM_SUFFIX: &str = "<|fim_suffix|>";
pub const ENDOFPROMPT: &str = "<|endofprompt|>";

/// Split pattern used by `cl100k_base` (GPT-3.5 and GPT-4).
///
/// tiktoken rewrote the original pattern with possessive quantifiers (`?+`, `++`, `*+`)
/// to cut down on backtracking. The `(?!\S)` lookahead is why this needs a backtracking
/// engine rather than the `regex` crate.
pub const CL100K_BASE_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s";

/// Split pattern used by `o200k_base` (GPT-4o and later).
///
/// Unlike cl100k, words are split on case changes (so `camelCase` becomes two pieces)
/// and contractions are attached to the preceding word.
pub const O200K_BASE_PATTERN: &str = concat!(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    "|",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    "|",
    r"\p{N}{1,3}",
    "|",
    r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
    "|",
    r"\s*[\r\n]+",
    "|",
    r"\s+(?!\S)",
    "|",
    r"\s+",
);

pub const CL100K_BASE: Preset = Preset {
    name: "cl100k_base",
    pattern: CL100K_BASE_PATTERN,
    special_tokens: &[
        (ENDOFTEXT, 100257),
        (FIM_PREFIX, 100258),
        (FIM_MIDDLE, 100259),
        (FIM_SUFFIX, 100260),
        (ENDOFPROMPT, 100276),
    ],
    url: "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
    sha256: "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
};

pub const O200K_BASE: Preset = Preset {
    name: "o200k_base",
    pattern: O200K_BASE_PATTERN,
    special_tokens: &[(ENDOFTEXT, 199999), (ENDOFPROMPT, 200018)],
    url: "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
    sha256: "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
};

/// All presets this crate knows about.
pub const ALL: &[Preset] = &[CL100K_BASE, O200K_BASE];

/// Looks up a preset by its tiktoken name, e.g. `"cl100k_base"`.
pub fn by_name(name: &str) -> Option<&'static Preset> {
    ALL.iter().find(|p| p.name == name)
}

/// Directory where downloaded rank files are cached.
///
/// `$BPE_RS_CACHE_DIR` if set, otherwise `$XDG_CACHE_HOME/bpe-rs`, otherwise
/// `~/.cache/bpe-rs`. The Python package and `scripts/fetch_data.sh` use the same rule.
pub fn cache_dir() -> PathBuf {
    let non_empty = |key: &str| env::var_os(key).filter(|v| !v.is_empty());
    if let Some(dir) = non_empty("BPE_RS_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(dir) = non_empty("XDG_CACHE_HOME") {
        return PathBuf::from(dir).join("bpe-rs");
    }
    let home = non_empty("HOME")
        .or_else(|| non_empty("USERPROFILE"))
        .unwrap_or_else(|| ".".into());
    PathBuf::from(home).join(".cache").join("bpe-rs")
}

impl Preset {
    /// Where this preset's rank file lives inside [`cache_dir`].
    pub fn cached_path(&self) -> PathBuf {
        cache_dir().join(format!("{}.tiktoken", self.name))
    }
}
