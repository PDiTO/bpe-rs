#![allow(dead_code)]

use bpe_rs::Encoding;
use bpe_rs::presets::Preset;

/// Loads a published encoding from the cache, or returns `None` (with a note on stderr)
/// if the rank file has not been downloaded. Run `scripts/fetch_data.sh` to fetch them.
pub fn load_preset(preset: &Preset) -> Option<Encoding> {
    let path = preset.cached_path();
    if !path.exists() {
        eprintln!(
            "skipping: {} not found (run scripts/fetch_data.sh)",
            path.display()
        );
        return None;
    }
    Some(Encoding::from_preset_file(preset, &path).expect("cached rank file should load"))
}
