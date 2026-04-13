//! Lightweight command-line argument parsing.
//!
//! Provides a `--key value` and `--key=value` parser with typed
//! accessors and defaults.  Positional arguments (those not preceded
//! by a `--` flag) are collected separately.

use std::str::FromStr;

use crate::error::Error;

/// Parsed command-line arguments split into named flags (`--key value`)
/// and positional arguments.
#[derive(Debug)]
pub struct Args {
    flags: Vec<(String, String)>,
    positional: Vec<String>,
}

impl Args {
    /// Parse arguments from the process command line, skipping the
    /// program name.  Supports `--key value` and `--key=value` syntax.
    #[must_use]
    pub fn parse() -> Self {
        let raw: Vec<String> = std::env::args().skip(1).collect();
        let (flags, positional, _) = raw.iter().fold(
            (
                Vec::<(String, String)>::new(),
                Vec::<String>::new(),
                None::<String>,
            ),
            |(flags, positional, pending), arg| {
                // Ownership prevents using map_or_else here: each branch
                // must consume flags/positional exclusively.
                if let Some(key) = pending {
                    (
                        flags
                            .into_iter()
                            .chain(std::iter::once((key, arg.clone())))
                            .collect(),
                        positional,
                        None,
                    )
                } else if let Some(stripped) = arg.strip_prefix("--") {
                    if let Some((k, v)) = stripped.split_once('=') {
                        (
                            flags
                                .into_iter()
                                .chain(std::iter::once((k.to_string(), v.to_string())))
                                .collect(),
                            positional,
                            None,
                        )
                    } else {
                        (flags, positional, Some(stripped.to_string()))
                    }
                } else {
                    (
                        flags,
                        positional
                            .into_iter()
                            .chain(std::iter::once(arg.clone()))
                            .collect(),
                        None,
                    )
                }
            },
        );
        Self { flags, positional }
    }

    /// Look up a flag value by key, returning `None` if absent.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&str> {
        self.flags
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Look up a flag value by key, parsing it into `T`, or return the
    /// provided default when the flag is absent.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] if the value is present but cannot be
    /// parsed as `T`.
    pub fn get_or<T: FromStr>(&self, key: &str, default: T) -> Result<T, Error>
    where
        T::Err: std::fmt::Display,
    {
        self.get(key).map_or_else(
            || Ok(default),
            |v| {
                v.parse::<T>().map_err(|e| Error::Config {
                    reason: format!("invalid value for --{key}: {e}"),
                })
            },
        )
    }

    /// Get a positional argument by zero-based index.
    #[must_use]
    pub fn positional(&self, index: usize) -> Option<&str> {
        self.positional.get(index).map(String::as_str)
    }
}
