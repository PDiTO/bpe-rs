//! Byte-level BPE tokenizer compatible with tiktoken rank files.

mod encoding;
mod error;
pub mod merge;
pub mod presets;
pub mod rank_file;
pub mod train;

pub use encoding::{Encoding, SpecialTokens};
pub use error::{Error, Result};
pub use rank_file::Ranks;
pub use train::Trainer;

/// A token id. In tiktoken's terminology the id of a mergeable token is its rank:
/// lower ranks were learned earlier and are merged first.
pub type Rank = u32;
