//! `bpe`: encode, decode, train and benchmark from the command line.

use std::io::{self, BufWriter, Read, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use bpe_rs::presets;
use bpe_rs::{Encoding, Rank, SpecialTokens, Trainer, rank_file};
use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "bpe",
    version,
    about = "Byte-level BPE tokenizer (tiktoken compatible)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Encode text (argument or stdin) and print the token ids.
    Encode {
        #[command(flatten)]
        encoding: EncodingArgs,
        /// Encode special-token text like <|endoftext|> as the special token.
        #[arg(long)]
        allow_special: bool,
        /// Print one token per line with the bytes it stands for.
        #[arg(long)]
        show: bool,
        /// Text to encode. Reads stdin if omitted.
        text: Option<String>,
    },
    /// Decode token ids (arguments or whitespace-separated on stdin) to text.
    Decode {
        #[command(flatten)]
        encoding: EncodingArgs,
        ids: Vec<Rank>,
    },
    /// Learn a vocabulary from text files and write it as a tiktoken rank file.
    Train {
        /// Number of mergeable tokens to learn, including the 256 single bytes.
        #[arg(long)]
        vocab_size: usize,
        /// Split pattern: `cl100k_base`, `o200k_base` or a regex.
        #[arg(long, default_value = "cl100k_base")]
        pattern: String,
        /// Stop once the best pair occurs fewer than this many times.
        #[arg(long, default_value_t = 1)]
        min_frequency: u64,
        /// Where to write the rank file.
        #[arg(long, short)]
        output: PathBuf,
        /// Training text. Each file is one document; files are processed in parallel.
        #[arg(required = true)]
        files: Vec<PathBuf>,
    },
    /// Measure encoding and decoding throughput on some text files.
    Bench {
        #[command(flatten)]
        encoding: EncodingArgs,
        /// Size of the documents the text is cut into for the batch measurement.
        #[arg(long, default_value_t = 4096)]
        doc_bytes: usize,
        /// Minimum time to spend on each measurement.
        #[arg(long, default_value_t = 2.0)]
        seconds: f64,
        #[arg(required = true)]
        files: Vec<PathBuf>,
    },
}

#[derive(Args)]
struct EncodingArgs {
    /// Published encoding to use. Its rank file is read from the cache directory
    /// (see scripts/fetch_data.sh).
    #[arg(long, short, default_value = "cl100k_base")]
    encoding: String,
    /// Load this rank file instead, e.g. one written by `bpe train`. No special tokens.
    #[arg(long, conflicts_with = "encoding")]
    rank_file: Option<PathBuf>,
    /// Split pattern to use with --rank-file: `cl100k_base`, `o200k_base` or a regex.
    #[arg(long, requires = "rank_file", default_value = "cl100k_base")]
    pattern: String,
}

fn resolve_pattern(pattern: &str) -> &str {
    presets::by_name(pattern).map_or(pattern, |p| p.pattern)
}

impl EncodingArgs {
    fn load(&self) -> Result<Encoding> {
        if let Some(path) = &self.rank_file {
            let pattern = resolve_pattern(&self.pattern);
            return Encoding::from_tiktoken_file("custom", path, pattern, [])
                .with_context(|| format!("loading {}", path.display()));
        }
        let preset = presets::by_name(&self.encoding).with_context(|| {
            format!(
                "unknown encoding {:?}; expected cl100k_base or o200k_base",
                self.encoding
            )
        })?;
        let path = preset.cached_path();
        if !path.exists() {
            bail!(
                "{} not found; run scripts/fetch_data.sh or pass --rank-file",
                path.display()
            );
        }
        Encoding::from_preset_file(preset, &path)
            .with_context(|| format!("loading {}", path.display()))
    }
}

fn read_stdin() -> Result<String> {
    let mut text = String::new();
    io::stdin().read_to_string(&mut text)?;
    Ok(text)
}

fn show_bytes(bytes: &[u8]) -> String {
    match std::str::from_utf8(bytes) {
        Ok(s) => format!("{s:?}"),
        Err(_) => format!("b\"{}\"", bytes.escape_ascii()),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut out = BufWriter::new(io::stdout().lock());

    match cli.command {
        Command::Encode {
            encoding,
            allow_special,
            show,
            text,
        } => {
            let enc = encoding.load()?;
            let text = match text {
                Some(t) => t,
                None => read_stdin()?,
            };
            let allowed = if allow_special {
                SpecialTokens::All
            } else {
                SpecialTokens::None
            };
            let tokens = enc.encode(&text, allowed, SpecialTokens::All)?;
            if show {
                for &t in &tokens {
                    writeln!(
                        out,
                        "{t:>7}  {}",
                        show_bytes(enc.decode_single_token_bytes(t)?)
                    )?;
                }
            } else {
                let ids: Vec<String> = tokens.iter().map(ToString::to_string).collect();
                writeln!(out, "{}", ids.join(" "))?;
            }
        }

        Command::Decode { encoding, ids } => {
            let enc = encoding.load()?;
            let ids = if ids.is_empty() {
                read_stdin()?
                    .split_whitespace()
                    .map(|s| s.parse().with_context(|| format!("not a token id: {s:?}")))
                    .collect::<Result<Vec<Rank>>>()?
            } else {
                ids
            };
            out.write_all(enc.decode(&ids)?.as_bytes())?;
        }

        Command::Train {
            vocab_size,
            pattern,
            min_frequency,
            output,
            files,
        } => {
            let docs = files
                .iter()
                .map(|f| {
                    std::fs::read_to_string(f).with_context(|| format!("reading {}", f.display()))
                })
                .collect::<Result<Vec<_>>>()?;
            let bytes: usize = docs.iter().map(String::len).sum();
            let started = Instant::now();
            let ranks = Trainer::new(vocab_size)
                .pattern(resolve_pattern(&pattern))
                .min_frequency(min_frequency)
                .train_ranks(&docs)?;
            let elapsed = started.elapsed();
            rank_file::save(&ranks, &output)?;
            writeln!(
                out,
                "learned {} tokens ({} merges) from {:.1} MB in {:.2}s; wrote {}",
                ranks.len(),
                ranks.len() - 256,
                bytes as f64 / 1e6,
                elapsed.as_secs_f64(),
                output.display()
            )?;
        }

        Command::Bench {
            encoding,
            doc_bytes,
            seconds,
            files,
        } => {
            let enc = encoding.load()?;
            let mut text = String::new();
            for f in &files {
                text.push_str(
                    &std::fs::read_to_string(f)
                        .with_context(|| format!("reading {}", f.display()))?,
                );
            }
            bench(
                &mut out,
                &enc,
                &text,
                doc_bytes,
                Duration::from_secs_f64(seconds),
            )?;
        }
    }
    out.flush()?;
    Ok(())
}

/// Cuts text into documents of roughly `size` bytes, at line boundaries.
fn split_docs(text: &str, size: usize) -> Vec<&str> {
    let mut docs = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let mut end = (start + size).min(text.len());
        match text.as_bytes()[end..].iter().position(|&b| b == b'\n') {
            Some(i) if end < text.len() => end += i + 1,
            _ => end = text.len(),
        }
        docs.push(&text[start..end]);
        start = end;
    }
    docs
}

/// Runs `f` repeatedly for at least `budget` and returns the mean time per run.
fn time_it(budget: Duration, mut f: impl FnMut()) -> Duration {
    f(); // warm up
    let started = Instant::now();
    let mut runs = 0u32;
    while started.elapsed() < budget || runs < 3 {
        f();
        runs += 1;
    }
    started.elapsed() / runs
}

fn bench(
    out: &mut impl Write,
    enc: &Encoding,
    text: &str,
    doc_bytes: usize,
    budget: Duration,
) -> Result<()> {
    let mb = text.len() as f64 / 1e6;
    let docs = split_docs(text, doc_bytes);
    let tokens = enc.encode_ordinary(text);
    writeln!(
        out,
        "{}: {:.2} MB, {} tokens ({:.2} bytes/token), {} docs, {} threads",
        enc.name(),
        mb,
        tokens.len(),
        text.len() as f64 / tokens.len() as f64,
        docs.len(),
        rayon::current_num_threads()
    )?;

    let single = time_it(budget, || {
        std::hint::black_box(enc.encode_ordinary(text));
    });
    let batch = time_it(budget, || {
        std::hint::black_box(enc.encode_ordinary_batch(&docs));
    });
    let decode = time_it(budget, || {
        std::hint::black_box(enc.decode_bytes(&tokens).unwrap());
    });
    for (label, t) in [
        ("encode, 1 thread", single),
        ("encode_batch", batch),
        ("decode", decode),
    ] {
        writeln!(out, "  {label:<18} {:>9.1} MB/s", mb / t.as_secs_f64())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use bpe_rs::presets::CL100K_BASE_PATTERN;

    #[test]
    fn split_docs_covers_the_text_at_line_boundaries() {
        let text = "aaa\nbbbb\ncc\n\nd";
        let docs = split_docs(text, 4);
        assert_eq!(docs.concat(), text);
        assert!(docs[..docs.len() - 1].iter().all(|d| d.ends_with('\n')));
        assert_eq!(split_docs("", 10), Vec::<&str>::new());
    }

    #[test]
    fn pattern_names_resolve_to_presets() {
        assert_eq!(resolve_pattern("cl100k_base"), CL100K_BASE_PATTERN);
        assert_eq!(resolve_pattern(r"\w+"), r"\w+");
    }
}
