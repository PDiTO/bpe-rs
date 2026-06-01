use crate::Rank;

/// Everything that can go wrong when building, running or training an encoding.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("rank file line {line}: {reason}")]
    RankFile { line: usize, reason: String },

    #[error("invalid pre-tokenization pattern: {0}")]
    Pattern(#[from] Box<fancy_regex::Error>),

    #[error("vocabulary has no token for byte {0:#04x}; every single byte must be encodable")]
    MissingByte(u8),

    #[error("rank {0} is assigned to more than one token")]
    DuplicateRank(Rank),

    #[error("rank {0} is reserved and cannot be used")]
    ReservedRank(Rank),

    #[error("special token {0:?} is empty or listed more than once")]
    InvalidSpecialToken(String),

    #[error("unknown token id {0}")]
    UnknownToken(Rank),

    #[error("text contains the disallowed special token {0:?}")]
    DisallowedSpecialToken(String),

    /// The split regex gave up on the input. In practice this means a run of roughly a
    /// million whitespace characters, which overflows fancy-regex's backtracking stack.
    #[error("pre-tokenization failed at byte {offset}: {source}")]
    PreTokenize {
        offset: usize,
        source: Box<fancy_regex::Error>,
    },

    #[error("vocab_size must be at least 256, got {0}")]
    VocabSizeTooSmall(usize),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
