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
