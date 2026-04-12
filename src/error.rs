//! Crate-wide error type.
//!
//! Every fallible function in this crate returns `Result<_, Error>`. The
//! [`Error`] enum is a sum over every upstream failure mode the crate can
//! encounter, plus a small number of domain-specific variants for shape and
//! configuration invariants that are not expressible in the type system.
//!
//! The enum is hand-rolled per the crate's style guide: no `thiserror`, no
//! `anyhow`. Every variant wraps (or describes) a concrete cause and
//! implements `From` for ergonomic `?` propagation.

use std::fmt;

/// All failure modes this crate can report.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Standard library I/O failure (filesystem, sockets, stdin/out).
    Io(std::io::Error),
    /// JSON serialization or deserialization failure.
    Json(serde_json::Error),
    /// `HuggingFace` Hub download or authentication failure.
    Hub(Box<hf_hub::api::sync::ApiError>),
    /// Safetensors container parse or metadata failure.
    Safetensors(safetensors::SafeTensorError),
    /// Tokenizer construction or encoding failure.
    Tokenizer(TokenizerError),
    /// Underlying candle tensor operation failure.
    Candle(candle_core::Error),
    /// A tensor had a shape that did not match the required shape.
    Shape {
        /// Human-readable name of the tensor or operand.
        what: &'static str,
        /// The shape that was expected by the caller.
        expected: Vec<usize>,
        /// The shape that was actually observed.
        actual: Vec<usize>,
    },
    /// A configuration value was outside its valid range or self-inconsistent.
    Config {
        /// Human-readable description of the invariant that was violated.
        reason: String,
    },
    /// A training-time invariant was violated (NaN loss, dead run, etc.).
    Train {
        /// Human-readable description of the failure.
        reason: String,
    },
    /// A boundary precondition was violated (missing resource, bad hookpoint).
    Boundary {
        /// Human-readable description of what went wrong.
        reason: String,
    },
}

/// Opaque wrapper around tokenizers errors so the crate can implement
/// [`std::error::Error`] without depending on the upstream trait hierarchy.
#[derive(Debug)]
pub struct TokenizerError(String);

impl TokenizerError {
    /// Construct a tokenizer error from any displayable cause.
    #[must_use]
    pub fn new(cause: impl fmt::Display) -> Self {
        Self(cause.to_string())
    }
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for TokenizerError {}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Json(e) => write!(f, "json error: {e}"),
            Self::Hub(e) => write!(f, "huggingface hub error: {e}"),
            Self::Safetensors(e) => write!(f, "safetensors error: {e}"),
            Self::Tokenizer(e) => write!(f, "tokenizer error: {e}"),
            Self::Candle(e) => write!(f, "candle error: {e}"),
            Self::Shape {
                what,
                expected,
                actual,
            } => write!(
                f,
                "shape mismatch for {what}: expected {expected:?}, got {actual:?}"
            ),
            Self::Config { reason } => write!(f, "configuration error: {reason}"),
            Self::Train { reason } => write!(f, "training error: {reason}"),
            Self::Boundary { reason } => write!(f, "boundary error: {reason}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Json(e) => Some(e),
            Self::Hub(e) => Some(e.as_ref()),
            Self::Safetensors(e) => Some(e),
            Self::Tokenizer(e) => Some(e),
            Self::Candle(e) => Some(e),
            Self::Shape { .. }
            | Self::Config { .. }
            | Self::Train { .. }
            | Self::Boundary { .. } => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

impl From<hf_hub::api::sync::ApiError> for Error {
    fn from(e: hf_hub::api::sync::ApiError) -> Self {
        Self::Hub(Box::new(e))
    }
}

impl From<safetensors::SafeTensorError> for Error {
    fn from(e: safetensors::SafeTensorError) -> Self {
        Self::Safetensors(e)
    }
}

impl From<candle_core::Error> for Error {
    fn from(e: candle_core::Error) -> Self {
        Self::Candle(e)
    }
}

impl From<TokenizerError> for Error {
    fn from(e: TokenizerError) -> Self {
        Self::Tokenizer(e)
    }
}
