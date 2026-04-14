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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LayerIndex;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};

    const D_MODEL: usize = 768;

    fn fresh_gpt2() -> Result<Gpt2, Error> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        Gpt2::new(vb, LayerIndex::new(0, 12)?)
    }

    #[allow(clippy::cast_possible_truncation)]
    fn id_batch(batch: usize, seq: usize) -> Result<Tensor, Error> {
        let data: Vec<u32> = (0..(batch * seq) as u32).collect();
        Tensor::from_vec(data, (batch, seq), &Device::Cpu).map_err(Error::from)
    }

    #[test]
    fn empty_batches_yield_nothing() -> Result<(), Error> {
        let gpt2 = fresh_gpt2()?;
        let stream = activation_stream(gpt2, Vec::new());
        let collected = stream.collect().run()?;
        assert!(collected.is_empty());
        Ok(())
    }

    #[test]
    fn single_batch_flattens_to_2d() -> Result<(), Error> {
        let gpt2 = fresh_gpt2()?;
        let batch = id_batch(2, 3)?;
        let stream = activation_stream(gpt2, vec![batch]);
        let collected = stream.collect().run()?;
        assert_eq!(collected.len(), 1);
        let elem = collected.first().ok_or_else(|| Error::Boundary {
            reason: "missing first element".into(),
        })?;
        assert_eq!(elem.dims(), &[2 * 3, D_MODEL]);
        Ok(())
    }

    #[test]
    fn multiple_batches_all_yielded() -> Result<(), Error> {
        let gpt2 = fresh_gpt2()?;
        let batches = vec![id_batch(1, 4)?, id_batch(1, 4)?, id_batch(1, 4)?];
        let stream = activation_stream(gpt2, batches);
        let collected = stream.collect().run()?;
        assert_eq!(collected.len(), 3);
        collected.iter().try_for_each(|t| {
            assert_eq!(t.dims(), &[4, D_MODEL]);
            Ok::<_, Error>(())
        })
    }

    #[test]
    fn rejects_1d_batch() -> Result<(), Error> {
        let gpt2 = fresh_gpt2()?;
        let bad = Tensor::from_vec(vec![0u32, 1, 2, 3], 4, &Device::Cpu)?;
        let stream = activation_stream(gpt2, vec![bad]);
        assert!(stream.collect().run().is_err());
        Ok(())
    }
}
