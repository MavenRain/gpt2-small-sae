//! Residual stream activation pipeline as a
//! [`comp_cat_rs::effect::stream::Stream`].
//!
//! The core primitive is [`activation_stream`], which takes a frozen GPT-2
//! model and a vector of pre-tokenized batches and lazily yields flattened
//! residual stream tensors of shape `(batch * seq_len, d_model)`.  The
//! training loop folds over this stream with an SAE training step.

use std::sync::Arc;

use candle_core::Tensor;
use comp_cat_rs::effect::io::Io;
use comp_cat_rs::effect::stream::Stream;

use crate::error::Error;
use crate::gpt2::Gpt2;

/// Build a lazy activation stream from a frozen [`Gpt2`] model and
/// pre-tokenized batches of shape `(batch, seq_len)`.
///
/// Each element yielded is a 2-D tensor `(batch * seq_len, d_model)` ready
/// to feed into [`crate::sae::Sae::forward`].
#[must_use]
pub fn activation_stream(gpt2: Gpt2, batches: Vec<Tensor>) -> Stream<Error, Tensor> {
    Stream::unfold(
        (gpt2, batches, 0_usize),
        Arc::new(|(gpt2, batches, idx)| {
            if idx >= batches.len() {
                Io::pure(None)
            } else {
                Io::suspend(move || {
                    let input = batches.get(idx).ok_or_else(|| Error::Boundary {
                        reason: format!("activation batch index {idx} out of range"),
                    })?;
                    let resid = gpt2.residual_stream(input)?;
                    let dims = resid.dims().to_vec();
                    let flat = match dims.as_slice() {
                        [b, s, d] => resid.reshape((b * s, *d)).map_err(Error::from),
                        other => Err(Error::Shape {
                            what: "residual stream output",
                            expected: vec![0, 0, 768],
                            actual: other.to_vec(),
                        }),
                    }?;
                    Ok(Some((flat, (gpt2, batches, idx + 1))))
                })
            }
        }),
    )
}
