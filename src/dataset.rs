//! Corpus tokenization and batching.
//!
//! Turns a raw text corpus into fixed-size token-ID batches suitable
//! for feeding through the GPT-2 forward pass.  The text is encoded
//! with the GPT-2 tokenizer, sliced into non-overlapping windows of
//! `ctx_len` tokens, and grouped into batches of `batch_size`.  Any
//! trailing tokens that do not fill a complete batch are discarded.

use candle_core::{DType, Device, Tensor};

use crate::config::{BatchSize, ContextLength};
use crate::error::{Error, TokenizerError};

/// Tokenize `text` and chunk into batches of shape `(batch_size, ctx_len)`
/// with dtype `u32`.
///
/// Returns an empty `Vec` if the corpus is too short to fill even one
/// batch.
///
/// # Errors
///
/// Returns [`Error::Tokenizer`] if encoding fails, or [`Error::Candle`]
/// on tensor construction failure.
pub fn tokenize_corpus(
    text: &str,
    tokenizer: &tokenizers::Tokenizer,
    batch_size: BatchSize,
    ctx_len: ContextLength,
    device: &Device,
) -> Result<Vec<Tensor>, Error> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| Error::Tokenizer(TokenizerError::new(e)))?;
    let all_ids: &[u32] = encoding.get_ids();
    let bs = batch_size.as_usize();
    let cl = ctx_len.as_usize();
    let tokens_per_batch = bs * cl;
    let n_batches = all_ids.len() / tokens_per_batch;
    (0..n_batches)
        .map(|i| {
            let start = i * tokens_per_batch;
            let chunk = &all_ids[start..start + tokens_per_batch];
            Tensor::from_slice(chunk, (bs, cl), device)
                .and_then(|t| t.to_dtype(DType::U32))
                .map_err(Error::from)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    /// Build a minimal Tokenizer that whitespace-splits input and looks up
    /// single-char words in an 8-token vocab.  Unknown chars map to `[UNK]`.
    fn toy_tokenizer() -> Result<tokenizers::Tokenizer, Error> {
        let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "WhitespaceSplit"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "a": 0, "b": 1, "c": 2, "d": 3,
                    "e": 4, "f": 5, "g": 6, "h": 7,
                    "[UNK]": 8
                },
                "unk_token": "[UNK]"
            }
        }"#;
        tokenizers::Tokenizer::from_str(json).map_err(|e| Error::Tokenizer(TokenizerError::new(e)))
    }

    /// Build a whitespace-delimited string of `n` repeated tokens.
    fn make_text(n: usize) -> String {
        (0..n)
            .map(|i| {
                let ch = (b'a' + u8::try_from(i % 8).unwrap_or(0)) as char;
                ch.to_string()
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn tokenize_chunks_exact_batch() -> Result<(), Error> {
        let tokenizer = toy_tokenizer()?;
        let text = make_text(8);
        let batches = tokenize_corpus(
            &text,
            &tokenizer,
            BatchSize::new(2)?,
            ContextLength::new(4)?,
            &Device::Cpu,
        )?;
        assert_eq!(batches.len(), 1);
        let b = batches.first().ok_or_else(|| Error::Boundary {
            reason: "missing".into(),
        })?;
        assert_eq!(b.dims(), &[2, 4]);
        assert_eq!(b.dtype(), DType::U32);
        Ok(())
    }

    #[test]
    fn tokenize_multiple_batches() -> Result<(), Error> {
        let tokenizer = toy_tokenizer()?;
        let text = make_text(24);
        let batches = tokenize_corpus(
            &text,
            &tokenizer,
            BatchSize::new(2)?,
            ContextLength::new(4)?,
            &Device::Cpu,
        )?;
        assert_eq!(batches.len(), 3);
        batches.iter().try_for_each(|b| {
            assert_eq!(b.dims(), &[2, 4]);
            Ok::<_, Error>(())
        })
    }

    #[test]
    fn tokenize_drops_trailing_incomplete_batch() -> Result<(), Error> {
        let tokenizer = toy_tokenizer()?;
        // 10 tokens, but tokens_per_batch = 2 * 4 = 8; only one batch fits.
        let text = make_text(10);
        let batches = tokenize_corpus(
            &text,
            &tokenizer,
            BatchSize::new(2)?,
            ContextLength::new(4)?,
            &Device::Cpu,
        )?;
        assert_eq!(batches.len(), 1);
        Ok(())
    }

    #[test]
    fn tokenize_empty_when_corpus_too_short() -> Result<(), Error> {
        let tokenizer = toy_tokenizer()?;
        let text = make_text(3);
        let batches = tokenize_corpus(
            &text,
            &tokenizer,
            BatchSize::new(2)?,
            ContextLength::new(4)?,
            &Device::Cpu,
        )?;
        assert!(batches.is_empty());
        Ok(())
    }
}
