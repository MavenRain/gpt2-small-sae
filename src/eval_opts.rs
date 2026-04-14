//! Shared CLI options and batch construction for evaluation binaries.
//!
//! The `eval_pretrained`, `patch_eval`, and `feature_hist` binaries all
//! accept the same flag set (`--layer`, `--expansion`, `--batch-size`,
//! `--ctx-len`, `--batches`, `--corpus`, `--checkpoint`, `--output`) and
//! build activation batches in exactly the same way.  This module
//! consolidates both pieces so each binary can delegate to
//! [`SharedEvalOpts::parse`] and [`build_batches`] instead of inlining
//! ~100 lines of duplicated setup.
//!
//! Binaries that need additional flags (e.g., `feature_hist`'s
//! histogram rendering options) compose a local struct that holds a
//! [`SharedEvalOpts`] alongside its own fields.

use candle_core::{DType, Device, Tensor};

use crate::cli::Args;
use crate::config::{BatchSize, ContextLength};
use crate::dataset::tokenize_corpus;
use crate::error::Error;

/// Vocabulary upper bound for random-token fallback batches.  GPT-2
/// small has 50257 entries, so 50256 is the inclusive maximum.
const VOCAB_UPPER: f32 = 50256.0;

/// Shared evaluation options used by every binary that runs GPT-2
/// through an SAE over an activation corpus.
#[derive(Clone, Debug)]
pub struct SharedEvalOpts {
    layer: usize,
    expansion: usize,
    batch_size: usize,
    ctx_len: usize,
    num_batches: usize,
    corpus: Option<std::path::PathBuf>,
    checkpoint: String,
    output: Option<String>,
}

impl SharedEvalOpts {
    /// Parse the shared flag set from a pre-built [`Args`].
    ///
    /// Defaults: `layer=8`, `expansion=8`, `batch-size=4`,
    /// `ctx-len=128`, `batches=8`, `checkpoint=sae_checkpoint.safetensors`.
    /// The first positional argument is accepted as a legacy alias for
    /// `--checkpoint`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] if any flag is present but fails to
    /// parse as the expected numeric type.
    pub fn parse(args: &Args) -> Result<Self, Error> {
        Ok(Self {
            layer: args.get_or("layer", 8_usize)?,
            expansion: args.get_or("expansion", 8_usize)?,
            batch_size: args.get_or("batch-size", 4_usize)?,
            ctx_len: args.get_or("ctx-len", 128_usize)?,
            num_batches: args.get_or("batches", 8_usize)?,
            corpus: args.get("corpus").map(std::path::PathBuf::from),
            checkpoint: args
                .get("checkpoint")
                .or_else(|| args.positional(0))
                .map_or_else(|| "sae_checkpoint.safetensors".to_string(), String::from),
            output: args.get("output").map(String::from),
        })
    }

    /// GPT-2 hookpoint layer index (0-indexed).
    #[must_use]
    pub fn layer(&self) -> usize {
        self.layer
    }

    /// SAE dictionary expansion factor relative to the model dimension.
    #[must_use]
    pub fn expansion(&self) -> usize {
        self.expansion
    }

    /// Batch size (number of sequences per forward pass).
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Context length (tokens per sequence).
    #[must_use]
    pub fn ctx_len(&self) -> usize {
        self.ctx_len
    }

    /// Number of activation batches to collect for evaluation.
    #[must_use]
    pub fn num_batches(&self) -> usize {
        self.num_batches
    }

    /// Optional corpus path.  `None` means the binary will fall back to
    /// fresh random token IDs.
    #[must_use]
    pub fn corpus(&self) -> Option<&std::path::Path> {
        self.corpus.as_deref()
    }

    /// SAE checkpoint path (safetensors file).
    #[must_use]
    pub fn checkpoint(&self) -> &str {
        &self.checkpoint
    }

    /// Optional JSON output path for structured results.
    #[must_use]
    pub fn output(&self) -> Option<&str> {
        self.output.as_deref()
    }

    /// True when `--corpus` was supplied; used to decide whether the
    /// binary needs to download the GPT-2 tokenizer.
    #[must_use]
    pub fn needs_tokenizer(&self) -> bool {
        self.corpus.is_some()
    }
}

/// Construct evaluation batches for a [`SharedEvalOpts`]: tokenized
/// corpus slices when `--corpus` was provided, or fresh random token-ID
/// batches otherwise.
///
/// Emits progress lines on stderr matching the format previously
/// inlined in each binary.
///
/// # Errors
///
/// - [`Error::Boundary`] when the corpus file cannot be read, when the
///   tokenizer is missing despite a corpus being requested, or when the
///   corpus is too short to produce a single batch.
/// - Any error bubbled from [`tokenize_corpus`] or candle tensor
///   construction.
pub fn build_batches(
    opts: &SharedEvalOpts,
    maybe_tokenizer: Option<&tokenizers::Tokenizer>,
    device: &Device,
) -> Result<Vec<Tensor>, Error> {
    opts.corpus().map_or_else(
        || {
            eprintln!(
                "eval batches: {} x {} x {} (random tokens)",
                opts.num_batches, opts.batch_size, opts.ctx_len
            );
            (0..opts.num_batches)
                .map(|_| {
                    Tensor::rand(
                        0.0_f32,
                        VOCAB_UPPER,
                        (opts.batch_size, opts.ctx_len),
                        device,
                    )
                    .and_then(|t| t.to_dtype(DType::U32))
                    .map_err(Error::from)
                })
                .collect::<Result<Vec<_>, _>>()
        },
        |path| {
            eprintln!("corpus: {}", path.display());
            let tokenizer = maybe_tokenizer.ok_or_else(|| Error::Boundary {
                reason: "tokenizer not loaded".into(),
            })?;
            let text = std::fs::read_to_string(path).map_err(|e| Error::Boundary {
                reason: format!("failed to read corpus {}: {e}", path.display()),
            })?;
            let bs = BatchSize::new(opts.batch_size)?;
            let cl = ContextLength::new(opts.ctx_len)?;
            let batches = tokenize_corpus(&text, tokenizer, bs, cl, device)?;
            if batches.is_empty() {
                Err(Error::Boundary {
                    reason: format!(
                        "corpus too short ({} tokens needed)",
                        opts.batch_size * opts.ctx_len
                    ),
                })
            } else {
                eprintln!(
                    "eval batches: {} x {} x {}",
                    batches.len(),
                    opts.batch_size,
                    opts.ctx_len,
                );
                Ok(batches)
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_device() -> Device {
        Device::Cpu
    }

    #[test]
    fn build_batches_random_fallback() -> Result<(), Error> {
        let opts = SharedEvalOpts {
            layer: 8,
            expansion: 8,
            batch_size: 2,
            ctx_len: 16,
            num_batches: 3,
            corpus: None,
            checkpoint: "sae_checkpoint.safetensors".to_string(),
            output: None,
        };
        let device = cpu_device();
        let batches = build_batches(&opts, None, &device)?;
        assert_eq!(batches.len(), 3);
        batches.iter().try_for_each(|t| {
            assert_eq!(t.dims(), &[2, 16]);
            assert_eq!(t.dtype(), DType::U32);
            Ok::<_, Error>(())
        })?;
        Ok(())
    }

    #[test]
    fn build_batches_missing_tokenizer_errors() {
        let opts = SharedEvalOpts {
            layer: 8,
            expansion: 8,
            batch_size: 2,
            ctx_len: 16,
            num_batches: 1,
            corpus: Some(std::path::PathBuf::from("/nonexistent/corpus.txt")),
            checkpoint: "sae_checkpoint.safetensors".to_string(),
            output: None,
        };
        let device = cpu_device();
        let result = build_batches(&opts, None, &device);
        assert!(result.is_err());
    }

    #[test]
    fn accessors_round_trip() {
        let opts = SharedEvalOpts {
            layer: 4,
            expansion: 16,
            batch_size: 8,
            ctx_len: 256,
            num_batches: 32,
            corpus: Some(std::path::PathBuf::from("corpus.txt")),
            checkpoint: "custom.safetensors".to_string(),
            output: Some("out.json".to_string()),
        };
        assert_eq!(opts.layer(), 4);
        assert_eq!(opts.expansion(), 16);
        assert_eq!(opts.batch_size(), 8);
        assert_eq!(opts.ctx_len(), 256);
        assert_eq!(opts.num_batches(), 32);
        assert_eq!(
            opts.corpus().map(std::path::Path::to_path_buf),
            Some(std::path::PathBuf::from("corpus.txt"))
        );
        assert_eq!(opts.checkpoint(), "custom.safetensors");
        assert_eq!(opts.output(), Some("out.json"));
        assert!(opts.needs_tokenizer());
    }
}
