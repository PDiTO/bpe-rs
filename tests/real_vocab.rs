//! Known-answer tests against the published cl100k_base and o200k_base vocabularies.
//!
//! Expected ids were produced by tiktoken 0.14. The much larger differential test lives
//! in `python/tests/test_differential.py`. These tests skip (and say so on stderr) if the
//! rank files are not in the cache.

mod common;

use bpe_rs::merge::{encode_piece_heap, encode_piece_scan};
use bpe_rs::presets::{CL100K_BASE, O200K_BASE};
use bpe_rs::{Encoding, Rank, SpecialTokens};
use common::load_preset;

const MIXED: &str = "I'm can't HELLO'S camelCaseWord  123456 \n\n  x";
const CJK_EMOJI: &str = "你好，世界 🌍🚀";

fn check(enc: &Encoding, text: &str, expected: &[Rank]) {
    let got = enc
        .encode(text, SpecialTokens::All, SpecialTokens::None)
        .unwrap();
    assert_eq!(got, expected, "{} on {text:?}", enc.name());
    assert_eq!(enc.decode(&got).unwrap(), text);
}

#[test]
fn cl100k_known_answers() {
    let Some(enc) = load_preset(&CL100K_BASE) else {
        return;
    };
    assert_eq!(enc.n_vocab(), 100_277);
    check(&enc, "hello world", &[15339, 1917]);
    check(
        &enc,
        MIXED,
        &[
            40, 2846, 649, 956, 38757, 1623, 13575, 50252, 4301, 11116, 220, 220, 4513, 10961,
            4815, 220, 865,
        ],
    );
    check(
        &enc,
        CJK_EMOJI,
        &[
            57668, 53901, 3922, 3574, 244, 98220, 11410, 234, 235, 9468, 248, 222,
        ],
    );
    check(&enc, "<|endoftext|>", &[100257]);
}

#[test]
fn o200k_known_answers() {
    let Some(enc) = load_preset(&O200K_BASE) else {
        return;
    };
    assert_eq!(enc.n_vocab(), 200_019);
    check(&enc, "hello world", &[24912, 2375]);
    check(
        &enc,
        MIXED,
        &[
            15390, 8535, 58527, 2699, 31233, 83330, 6187, 12929, 220, 220, 7633, 19354, 1202, 220,
            1215,
        ],
    );
    check(
        &enc,
        CJK_EMOJI,
        &[177519, 979, 28428, 130321, 235, 112927, 222],
    );
    check(&enc, "<|endoftext|>", &[199999]);
}

#[test]
fn heap_and_scan_agree_on_long_pieces() {
    let Some(enc) = load_preset(&CL100K_BASE) else {
        return;
    };
    // Long pieces are rare in practice but easy to build: one enormous "word", a long
    // run of digits split into threes by the pattern, a long base64-ish blob.
    let blob: String = (0..3000u32)
        .map(|i| {
            char::from(
                b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"[(i * 7919 % 52) as usize],
            )
        })
        .collect();
    for piece in [
        "a".repeat(2000),
        "ab".repeat(1000),
        "supercalifragilisticexpialidocious".repeat(40),
        blob,
        "日本語のテキスト".repeat(100),
    ] {
        let (mut scan, mut heap) = (Vec::new(), Vec::new());
        encode_piece_scan(enc.ranks(), piece.as_bytes(), &mut scan);
        encode_piece_heap(enc.ranks(), piece.as_bytes(), &mut heap);
        assert_eq!(scan, heap);
        assert_eq!(enc.decode_bytes(&scan).unwrap(), piece.as_bytes());
    }
}

#[test]
fn save_and_reload_preserves_the_vocabulary() {
    let Some(enc) = load_preset(&CL100K_BASE) else {
        return;
    };
    let dir = std::env::temp_dir().join(format!("bpe-rs-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("cl100k_copy.tiktoken");
    enc.save_tiktoken_file(&path).unwrap();
    let original = std::fs::read(CL100K_BASE.cached_path()).unwrap();
    let written = std::fs::read(&path).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(
        original, written,
        "saved file should be byte-identical to the original"
    );
}
