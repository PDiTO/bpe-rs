//! Python bindings for bpe-rs. The Python-facing API mirrors tiktoken's `Encoding`
//! closely enough that the differential tests can call both the same way.

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use bpe_rs::presets::{self, CL100K_BASE_PATTERN, O200K_BASE_PATTERN};
use bpe_rs::{Encoding, Error, Rank, SpecialTokens, Trainer};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyFrozenSet, PyString};
use rayon::ThreadPool;

fn to_py_err(err: Error) -> PyErr {
    match err {
        Error::Io(e) => PyErr::from(e),
        Error::UnknownToken(t) => PyKeyError::new_err(format!("unknown token id {t}")),
        Error::DisallowedSpecialToken(token) => PyValueError::new_err(format!(
            "Encountered text corresponding to disallowed special token {token:?}.\n\
             To encode it as a special token, pass it in `allowed_special`. To encode it \
             as normal text, remove it from `disallowed_special` (or pass \
             `disallowed_special=()` to disable the check entirely)."
        )),
        other => PyValueError::new_err(other.to_string()),
    }
}

/// Borrows the UTF-8 contents of a Python string.
///
/// Python strings can hold lone surrogates, which are not valid UTF-8. tiktoken's
/// answer is to round-trip the text through UTF-16 with `errors="replace"`, turning
/// each lone surrogate into U+FFFD. We do the same so results match exactly.
fn text_of<'a>(s: &'a Bound<'_, PyString>) -> PyResult<Cow<'a, str>> {
    match s.to_str() {
        Ok(text) => Ok(Cow::Borrowed(text)),
        Err(_) => {
            let fixed = s
                .call_method1("encode", ("utf-16", "surrogatepass"))?
                .call_method1("decode", ("utf-16", "replace"))?;
            Ok(Cow::Owned(fixed.extract::<String>()?))
        }
    }
}

/// tiktoken's `allowed_special` / `disallowed_special` arguments: the string `"all"` or
/// a collection of token strings.
enum SpecialArg {
    All,
    Only(Vec<String>),
}

impl SpecialArg {
    fn parse(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(s) = obj.cast::<PyString>() {
            return match s.to_str()? {
                "all" => Ok(SpecialArg::All),
                other => Err(PyValueError::new_err(format!(
                    "expected \"all\" or a collection of special tokens, got the string {other:?}"
                ))),
            };
        }
        let tokens = obj
            .try_iter()?
            .map(|item| item?.extract::<String>())
            .collect::<PyResult<Vec<_>>>()?;
        Ok(SpecialArg::Only(tokens))
    }

    fn with_borrowed<R>(&self, f: impl FnOnce(SpecialTokens<'_>) -> R) -> R {
        match self {
            SpecialArg::All => f(SpecialTokens::All),
            SpecialArg::Only(tokens) if tokens.is_empty() => f(SpecialTokens::None),
            SpecialArg::Only(tokens) => {
                let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
                f(SpecialTokens::Only(&refs))
            }
        }
    }
}

/// Rayon pools keyed by thread count, built on first use and kept for the life of
/// the process. `num_threads=None` uses rayon's global pool.
fn pool(num_threads: usize) -> PyResult<Arc<ThreadPool>> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<ThreadPool>>>> = OnceLock::new();
    let mut pools = POOLS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(pool) = pools.get(&num_threads) {
        return Ok(Arc::clone(pool));
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let pool = Arc::new(pool);
    pools.insert(num_threads, Arc::clone(&pool));
    Ok(pool)
}

fn run_on<R: Send>(num_threads: Option<usize>, f: impl FnOnce() -> R + Send) -> PyResult<R> {
    match num_threads {
        None => Ok(f()),
        Some(0) => Err(PyValueError::new_err("num_threads must be at least 1")),
        Some(n) => Ok(pool(n)?.install(f)),
    }
}

fn resolve_pattern(pattern: &str) -> &str {
    presets::by_name(pattern).map_or(pattern, |p| p.pattern)
}

/// A byte-level BPE encoding.
#[pyclass(module = "bpe_rs", name = "Encoding", frozen)]
struct PyEncoding {
    inner: Encoding,
}

#[pymethods]
impl PyEncoding {
    /// Loads a tiktoken rank file.
    #[staticmethod]
    #[pyo3(signature = (path, pattern = CL100K_BASE_PATTERN, special_tokens = None, name = None))]
    fn from_tiktoken_file(
        py: Python<'_>,
        path: PathBuf,
        pattern: &str,
        special_tokens: Option<HashMap<String, Rank>>,
        name: Option<String>,
    ) -> PyResult<Self> {
        let name = name.unwrap_or_else(|| {
            path.file_stem()
                .map_or_else(|| "custom".into(), |s| s.to_string_lossy().into_owned())
        });
        let pattern = resolve_pattern(pattern);
        let specials = special_tokens.unwrap_or_default();
        let inner = py
            .detach(|| Encoding::from_tiktoken_file(name, &path, pattern, specials))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Loads the rank file for one of the published encodings (`cl100k_base`,
    /// `o200k_base`) and applies that encoding's pattern and special tokens.
    #[staticmethod]
    fn from_preset_file(py: Python<'_>, name: &str, path: PathBuf) -> PyResult<Self> {
        let preset = presets::by_name(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown encoding {name:?}")))?;
        let inner = py
            .detach(|| Encoding::from_preset_file(preset, &path))
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

    #[getter]
    fn pattern(&self) -> &str {
        self.inner.pattern()
    }

    #[getter]
    fn n_vocab(&self) -> usize {
        self.inner.n_vocab()
    }

    #[getter]
    fn max_token_value(&self) -> Rank {
        self.inner.max_token_value()
    }

    #[getter]
    fn special_tokens<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (token, rank) in self.inner.special_tokens() {
            dict.set_item(token, rank)?;
        }
        Ok(dict)
    }

    #[getter]
    fn special_tokens_set<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyFrozenSet>> {
        let tokens: Vec<&str> = self.inner.special_tokens().map(|(t, _)| t).collect();
        PyFrozenSet::new(py, tokens)
    }

    fn encode_ordinary(&self, py: Python<'_>, text: &Bound<'_, PyString>) -> PyResult<Vec<Rank>> {
        let text = text_of(text)?;
        Ok(py.detach(|| self.inner.encode_ordinary(&text)))
    }

    #[pyo3(signature = (text, *, allowed_special = None, disallowed_special = None))]
    fn encode(
        &self,
        py: Python<'_>,
        text: &Bound<'_, PyString>,
        allowed_special: Option<&Bound<'_, PyAny>>,
        disallowed_special: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Rank>> {
        let (allowed, disallowed) = parse_special_args(allowed_special, disallowed_special)?;
        let text = text_of(text)?;
        py.detach(|| {
            allowed.with_borrowed(|allowed| {
                disallowed.with_borrowed(|disallowed| self.inner.encode(&text, allowed, disallowed))
            })
        })
        .map_err(to_py_err)
    }

    #[pyo3(signature = (texts, *, num_threads = None))]
    fn encode_ordinary_batch(
        &self,
        py: Python<'_>,
        texts: Vec<Bound<'_, PyString>>,
        num_threads: Option<usize>,
    ) -> PyResult<Vec<Vec<Rank>>> {
        let texts = texts.iter().map(text_of).collect::<PyResult<Vec<_>>>()?;
        py.detach(|| run_on(num_threads, || self.inner.encode_ordinary_batch(&texts)))
    }

    #[pyo3(signature = (texts, *, num_threads = None, allowed_special = None, disallowed_special = None))]
    fn encode_batch(
        &self,
        py: Python<'_>,
        texts: Vec<Bound<'_, PyString>>,
        num_threads: Option<usize>,
        allowed_special: Option<&Bound<'_, PyAny>>,
        disallowed_special: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<Vec<Rank>>> {
        let (allowed, disallowed) = parse_special_args(allowed_special, disallowed_special)?;
        let texts = texts.iter().map(text_of).collect::<PyResult<Vec<_>>>()?;
        py.detach(|| {
            run_on(num_threads, || {
                allowed.with_borrowed(|allowed| {
                    disallowed.with_borrowed(|disallowed| {
                        self.inner.encode_batch(&texts, allowed, disallowed)
                    })
                })
            })
        })?
        .map_err(to_py_err)
    }

    fn decode_bytes<'py>(
        &self,
        py: Python<'py>,
        tokens: Vec<Rank>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = py
            .detach(|| self.inner.decode_bytes(&tokens))
            .map_err(to_py_err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    #[pyo3(signature = (tokens, errors = "replace"))]
    fn decode<'py>(
        &self,
        py: Python<'py>,
        tokens: Vec<Rank>,
        errors: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        if errors == "replace" {
            let text = py
                .detach(|| self.inner.decode(&tokens))
                .map_err(to_py_err)?;
            return Ok(PyString::new(py, &text).into_any());
        }
        self.decode_bytes(py, tokens)?
            .call_method1("decode", ("utf-8", errors))
    }

    fn decode_single_token_bytes<'py>(
        &self,
        py: Python<'py>,
        token: Rank,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self
            .inner
            .decode_single_token_bytes(token)
            .map_err(to_py_err)?;
        Ok(PyBytes::new(py, bytes))
    }

    /// Splits text into the pieces BPE runs on, without merging.
    fn split(&self, text: &Bound<'_, PyString>) -> PyResult<Vec<String>> {
        let text = text_of(text)?;
        Ok(self
            .inner
            .split(&text)
            .into_iter()
            .map(str::to_owned)
            .collect())
    }

    /// Writes the mergeable ranks as a tiktoken rank file.
    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        py.detach(|| self.inner.save_tiktoken_file(&path))
            .map_err(to_py_err)
    }

    fn __repr__(&self) -> String {
        format!("<Encoding {:?}>", self.inner.name())
    }
}

fn parse_special_args(
    allowed: Option<&Bound<'_, PyAny>>,
    disallowed: Option<&Bound<'_, PyAny>>,
) -> PyResult<(SpecialArg, SpecialArg)> {
    let allowed = match allowed {
        Some(obj) => SpecialArg::parse(obj)?,
        None => SpecialArg::Only(Vec::new()),
    };
    let disallowed = match disallowed {
        Some(obj) => SpecialArg::parse(obj)?,
        None => SpecialArg::All,
    };
    Ok((allowed, disallowed))
}

/// Learns a BPE vocabulary from `texts` and returns it as an `Encoding`.
#[pyfunction]
#[pyo3(signature = (
    texts, vocab_size, *, pattern = CL100K_BASE_PATTERN, special_tokens = Vec::new(),
    min_frequency = 1, name = "trained"
))]
fn train(
    py: Python<'_>,
    texts: &Bound<'_, PyAny>,
    vocab_size: usize,
    pattern: &str,
    special_tokens: Vec<String>,
    min_frequency: u64,
    name: &str,
) -> PyResult<PyEncoding> {
    let texts: Vec<Bound<'_, PyString>> = match texts.cast::<PyString>() {
        Ok(single) => vec![single.clone()],
        Err(_) => texts.extract()?,
    };
    let texts = texts.iter().map(text_of).collect::<PyResult<Vec<_>>>()?;
    let trainer = Trainer::new(vocab_size)
        .pattern(resolve_pattern(pattern))
        .special_tokens(special_tokens)
        .min_frequency(min_frequency);
    let inner = py
        .detach(|| trainer.train(name, &texts))
        .map_err(to_py_err)?;
    Ok(PyEncoding { inner })
}

/// Name, URL, SHA-256 of the published encodings, for the Python-side downloader.
#[pyfunction]
fn _presets() -> Vec<(&'static str, &'static str, &'static str)> {
    presets::ALL
        .iter()
        .map(|p| (p.name, p.url, p.sha256))
        .collect()
}

#[pymodule]
fn _bpe_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEncoding>()?;
    m.add_function(wrap_pyfunction!(train, m)?)?;
    m.add_function(wrap_pyfunction!(_presets, m)?)?;
    m.add("CL100K_BASE_PATTERN", CL100K_BASE_PATTERN)?;
    m.add("O200K_BASE_PATTERN", O200K_BASE_PATTERN)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
