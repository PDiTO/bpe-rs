//! Byte-level BPE tokenizer compatible with tiktoken rank files.

mod error;
pub mod rank_file;

pub use error::{Error, Result};
pub use rank_file::Ranks;

/// A token id. In tiktoken's terminology the id of a mergeable token is its rank:
/// lower ranks were learned earlier and are merged first.
pub type Rank = u32;
