//! Sparse autoencoder model: encoder, decoder, forward pass, and loss.
//!
//! The SAE is a single-hidden-layer linear model with a `ReLU` nonlinearity
//! in the feature space and a "shift-by-`b_dec`" tied bias on the input:
//!
//! ```text
//! features = relu((x - b_dec) @ w_enc + b_enc)
//! recon    = features @ w_dec + b_dec
//! ```
//!
//! The training objective is mean squared reconstruction error plus an L1
//! penalty on the feature activations:
//!
//! ```text
//! mse   = mean((recon - x)^2)
//! l1    = mean(sum_i |features_i|) * l1_coefficient
//! total = mse + l1
//! ```
//!
//! `Sae` holds only [`Tensor`]s.  When the backing [`VarBuilder`] is a
//! [`candle_nn::VarMap`] the tensors are trainable (the caller retains the
//! `VarMap` and passes its vars to the optimizer); when the backing builder
//! is a safetensors file the tensors are frozen.

use std::path::Path;

use candle_core::{D, DType, Device, Tensor};
use candle_nn::{Init, VarBuilder};

use crate::config::{L1Coefficient, ModelDim, SaeDim};
use crate::error::Error;

/// A sparse autoencoder over a fixed-width residual stream.
#[derive(Debug, Clone)]
#[must_use]
pub struct Sae {
    w_enc: Tensor,
    b_enc: Tensor,
    w_dec: Tensor,
    b_dec: Tensor,
    model_dim: ModelDim,
    sae_dim: SaeDim,
}

/// Result of an [`Sae::forward`] call: per-token feature activations and the
/// reconstructed residual stream.
#[derive(Debug, Clone)]
#[must_use]
pub struct Forward {
    features: Tensor,
    reconstruction: Tensor,
}

/// Additive decomposition of the training objective returned by
/// [`Sae::loss`].
#[derive(Debug, Clone)]
#[must_use]
pub struct Loss {
    total: Tensor,
    mse: Tensor,
    l1: Tensor,
}

impl Sae {
    /// Build an [`Sae`] from a [`VarBuilder`].
    ///
    /// When the builder is backed by a [`candle_nn::VarMap`] the returned
    /// SAE owns fresh trainable parameters; when it is backed by a
    /// safetensors file the SAE wraps the loaded tensors.  Parameter names
    /// are `w_enc`, `b_enc`, `w_dec`, `b_dec`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Candle`] if the builder cannot produce the requested
    /// tensor shapes.
    #[allow(clippy::needless_pass_by_value, clippy::cast_precision_loss)]
    pub fn new(vb: VarBuilder, model_dim: ModelDim, sae_dim: SaeDim) -> Result<Self, Error> {
        let d_model = model_dim.as_usize();
        let d_sae = sae_dim.as_usize();
        let enc_stdev = (d_model as f64).sqrt().recip();
        let dec_stdev = (d_sae as f64).sqrt().recip();
        let w_enc = vb.get_with_hints(
            (d_model, d_sae),
            "w_enc",
            Init::Randn {
                mean: 0.0,
                stdev: enc_stdev,
            },
        )?;
        let b_enc = vb.get_with_hints((d_sae,), "b_enc", Init::Const(0.0))?;
        let w_dec = vb.get_with_hints(
            (d_sae, d_model),
            "w_dec",
            Init::Randn {
                mean: 0.0,
                stdev: dec_stdev,
            },
        )?;
        let b_dec = vb.get_with_hints((d_model,), "b_dec", Init::Const(0.0))?;
        Ok(Self {
            w_enc,
            b_enc,
            w_dec,
            b_dec,
            model_dim,
            sae_dim,
        })
    }

    /// Load an [`Sae`] from a safetensors file on disk.
    ///
    /// The file must contain four tensors named `w_enc`, `b_enc`, `w_dec`,
    /// and `b_dec` with shapes `(model_dim, sae_dim)`, `(sae_dim,)`,
    /// `(sae_dim, model_dim)`, `(model_dim,)`.  Any other dtype is coerced
    /// to `f32`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Boundary`] if a named tensor is missing, or
    /// [`Error::Shape`] if dimensions do not match.
    pub fn from_safetensors(
        path: &Path,
        model_dim: ModelDim,
        sae_dim: SaeDim,
        device: &Device,
    ) -> Result<Self, Error> {
        let tensors = candle_core::safetensors::load(path, device)?;
        let d_model = model_dim.as_usize();
        let d_sae = sae_dim.as_usize();
        let load = |name: &'static str, expected: &[usize]| -> Result<Tensor, Error> {
            tensors
                .get(name)
                .ok_or_else(|| Error::Boundary {
                    reason: format!("missing tensor `{name}` in {}", path.display()),
                })
                .and_then(|t| {
                    (t.dims() == expected)
                        .then(|| t.clone())
                        .ok_or_else(|| Error::Shape {
                            what: name,
                            expected: expected.to_vec(),
                            actual: t.dims().to_vec(),
                        })
                })
                .and_then(|t| t.to_dtype(DType::F32).map_err(Error::from))
        };
        let w_enc = load("w_enc", &[d_model, d_sae])?;
        let b_enc = load("b_enc", &[d_sae])?;
        let w_dec = load("w_dec", &[d_sae, d_model])?;
        let b_dec = load("b_dec", &[d_model])?;
        Ok(Self {
            w_enc,
            b_enc,
            w_dec,
            b_dec,
            model_dim,
            sae_dim,
        })
    }

    /// Encoder/decoder forward pass.
    ///
    /// `x` must have shape `(..., model_dim)`; any leading dimensions are
    /// flattened for the matmul and restored on the outputs.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if the trailing dimension does not match, or
    /// [`Error::Candle`] on tensor operation failure.
    pub fn forward(&self, x: &Tensor) -> Result<Forward, Error> {
        let dims = x.dims().to_vec();
        let d_model = self.model_dim.as_usize();
        let d_sae = self.sae_dim.as_usize();
        let &last = dims.last().ok_or_else(|| Error::Shape {
            what: "sae forward input",
            expected: vec![d_model],
            actual: Vec::new(),
        })?;
        (last == d_model)
            .then_some(())
            .ok_or_else(|| Error::Shape {
                what: "sae forward input trailing dim",
                expected: vec![d_model],
                actual: vec![last],
            })?;
        let leading: usize = dims.iter().take(dims.len() - 1).product();
        let flat = x.reshape((leading, d_model))?;
        let shifted = flat.broadcast_sub(&self.b_dec)?;
        let pre = shifted.matmul(&self.w_enc)?.broadcast_add(&self.b_enc)?;
        let features_flat = pre.relu()?;
        let recon_flat = features_flat
            .matmul(&self.w_dec)?
            .broadcast_add(&self.b_dec)?;
        let feat_shape: Vec<usize> = dims[..dims.len() - 1]
            .iter()
            .copied()
            .chain(std::iter::once(d_sae))
            .collect();
        let features = features_flat.reshape(feat_shape)?;
        let reconstruction = recon_flat.reshape(dims)?;
        Ok(Forward {
            features,
            reconstruction,
        })
    }

    /// Forward pass plus the full decomposed training loss.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`Self::forward`].
    pub fn loss(&self, x: &Tensor, l1_coefficient: L1Coefficient) -> Result<Loss, Error> {
        let forward = self.forward(x)?;
        let mse = forward.reconstruction.sub(x)?.sqr()?.mean_all()?;
        let l1 = forward
            .features
            .abs()?
            .sum(D::Minus1)?
            .mean_all()?
            .affine(l1_coefficient.as_f64(), 0.0)?;
        let total = mse.add(&l1)?;
        Ok(Loss { total, mse, l1 })
    }

    /// Residual stream width accepted by this SAE.
    pub fn model_dim(&self) -> ModelDim {
        self.model_dim
    }

    /// Dictionary width of this SAE.
    pub fn sae_dim(&self) -> SaeDim {
        self.sae_dim
    }

    /// Encoder weight, shape `(model_dim, sae_dim)`.
    #[must_use]
    pub fn w_enc(&self) -> &Tensor {
        &self.w_enc
    }

    /// Encoder bias, shape `(sae_dim,)`.
    #[must_use]
    pub fn b_enc(&self) -> &Tensor {
        &self.b_enc
    }

    /// Decoder weight, shape `(sae_dim, model_dim)`.
    #[must_use]
    pub fn w_dec(&self) -> &Tensor {
        &self.w_dec
    }

    /// Decoder bias, shape `(model_dim,)`.
    #[must_use]
    pub fn b_dec(&self) -> &Tensor {
        &self.b_dec
    }
}

impl Forward {
    /// Per-token feature activations, shape `(..., sae_dim)`.
    #[must_use]
    pub fn features(&self) -> &Tensor {
        &self.features
    }

    /// Reconstructed residual stream, same shape as the forward input.
    #[must_use]
    pub fn reconstruction(&self) -> &Tensor {
        &self.reconstruction
    }
}

impl Loss {
    /// Total loss (MSE plus scaled L1).
    #[must_use]
    pub fn total(&self) -> &Tensor {
        &self.total
    }

    /// Reconstruction MSE component.
    #[must_use]
    pub fn mse(&self) -> &Tensor {
        &self.mse
    }

    /// L1 sparsity penalty component, already scaled by the coefficient.
    #[must_use]
    pub fn l1(&self) -> &Tensor {
        &self.l1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    const MODEL_DIM: usize = 4;
    const SAE_DIM: usize = 8;

    fn cpu() -> Device {
        Device::Cpu
    }

    fn fresh_sae(device: &Device) -> Result<Sae, Error> {
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        Sae::new(vb, ModelDim::new(MODEL_DIM)?, SaeDim::new(SAE_DIM)?)
    }

    fn sae_from_tensors(tensors: HashMap<String, Tensor>, device: &Device) -> Result<Sae, Error> {
        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
        Sae::new(vb, ModelDim::new(MODEL_DIM)?, SaeDim::new(SAE_DIM)?)
    }

    /// Hand-crafted SAE with all parameters set to zero, so every forward
    /// pass produces zero features and a zero reconstruction regardless
    /// of the input.
    fn zero_sae(device: &Device) -> Result<Sae, Error> {
        let tensors: HashMap<String, Tensor> = [
            (
                "w_enc".to_string(),
                Tensor::zeros((MODEL_DIM, SAE_DIM), DType::F32, device)?,
            ),
            (
                "b_enc".to_string(),
                Tensor::zeros((SAE_DIM,), DType::F32, device)?,
            ),
            (
                "w_dec".to_string(),
                Tensor::zeros((SAE_DIM, MODEL_DIM), DType::F32, device)?,
            ),
            (
                "b_dec".to_string(),
                Tensor::zeros((MODEL_DIM,), DType::F32, device)?,
            ),
        ]
        .into_iter()
        .collect();
        sae_from_tensors(tensors, device)
    }

    /// Hand-crafted SAE with `b_enc = 1` and zero weights, so every
    /// forward pass produces an all-ones feature activation regardless
    /// of the input (making the L1 penalty analytically predictable).
    fn unit_bias_sae(device: &Device) -> Result<Sae, Error> {
        let tensors: HashMap<String, Tensor> = [
            (
                "w_enc".to_string(),
                Tensor::zeros((MODEL_DIM, SAE_DIM), DType::F32, device)?,
            ),
            (
                "b_enc".to_string(),
                Tensor::ones((SAE_DIM,), DType::F32, device)?,
            ),
            (
                "w_dec".to_string(),
                Tensor::zeros((SAE_DIM, MODEL_DIM), DType::F32, device)?,
            ),
            (
                "b_dec".to_string(),
                Tensor::zeros((MODEL_DIM,), DType::F32, device)?,
            ),
        ]
        .into_iter()
        .collect();
        sae_from_tensors(tensors, device)
    }

    // -- forward shape & error handling ------------------------------------

    #[test]
    fn forward_shape_3d() -> Result<(), Error> {
        let device = cpu();
        let sae = fresh_sae(&device)?;
        let x = Tensor::zeros((2, 3, MODEL_DIM), DType::F32, &device)?;
        let out = sae.forward(&x)?;
        assert_eq!(out.features().dims(), &[2, 3, SAE_DIM]);
        assert_eq!(out.reconstruction().dims(), &[2, 3, MODEL_DIM]);
        Ok(())
    }

    #[test]
    fn forward_shape_2d() -> Result<(), Error> {
        let device = cpu();
        let sae = fresh_sae(&device)?;
        let x = Tensor::zeros((5, MODEL_DIM), DType::F32, &device)?;
        let out = sae.forward(&x)?;
        assert_eq!(out.features().dims(), &[5, SAE_DIM]);
        assert_eq!(out.reconstruction().dims(), &[5, MODEL_DIM]);
        Ok(())
    }

    #[test]
    fn forward_wrong_trailing_dim_errors() -> Result<(), Error> {
        let device = cpu();
        let sae = fresh_sae(&device)?;
        let bad = Tensor::zeros((2, 3, MODEL_DIM + 1), DType::F32, &device)?;
        assert!(matches!(sae.forward(&bad), Err(Error::Shape { .. })));
        Ok(())
    }

    // -- forward numerical invariants --------------------------------------

    #[test]
    fn forward_features_non_negative() -> Result<(), Error> {
        let device = cpu();
        let sae = fresh_sae(&device)?;
        let x = Tensor::randn(0.0f32, 1.0f32, (2, 3, MODEL_DIM), &device)?;
        let out = sae.forward(&x)?;
        let min_feature = out.features().min_all()?.to_scalar::<f32>()?;
        assert!(
            min_feature >= 0.0,
            "features must be ReLU-positive, got min {min_feature}"
        );
        Ok(())
    }

    #[test]
    fn forward_zero_params_produces_zero_outputs() -> Result<(), Error> {
        let device = cpu();
        let sae = zero_sae(&device)?;
        let x = Tensor::from_vec(
            vec![1.0f32, -2.0, 3.0, -4.0, 5.0, 6.0, 7.0, 8.0],
            (2, MODEL_DIM),
            &device,
        )?;
        let out = sae.forward(&x)?;
        let feat_max = out.features().abs()?.max_all()?.to_scalar::<f32>()?;
        let recon_max = out.reconstruction().abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(feat_max.abs() < 1e-6);
        assert!(recon_max.abs() < 1e-6);
        Ok(())
    }

    // -- loss decomposition ------------------------------------------------

    #[test]
    fn loss_total_equals_mse_plus_l1() -> Result<(), Error> {
        let device = cpu();
        let sae = fresh_sae(&device)?;
        let x = Tensor::randn(0.0f32, 1.0f32, (2, MODEL_DIM), &device)?;
        let loss = sae.loss(&x, L1Coefficient::new(0.01)?)?;
        let total = loss.total().to_scalar::<f32>()?;
        let mse = loss.mse().to_scalar::<f32>()?;
        let l1 = loss.l1().to_scalar::<f32>()?;
        assert!((total - (mse + l1)).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn loss_mse_matches_mean_sq_for_zero_sae() -> Result<(), Error> {
        let device = cpu();
        let sae = zero_sae(&device)?;
        // x = [1, 2, 3, 4] -> mean(x^2) = (1 + 4 + 9 + 16) / 4 = 7.5
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, MODEL_DIM), &device)?;
        let loss = sae.loss(&x, L1Coefficient::new(0.01)?)?;
        let mse = loss.mse().to_scalar::<f32>()?;
        assert!((mse - 7.5).abs() < 1e-4);
        Ok(())
    }

    #[test]
    fn loss_l1_zero_when_features_zero() -> Result<(), Error> {
        let device = cpu();
        let sae = zero_sae(&device)?;
        let x = Tensor::randn(0.0f32, 1.0f32, (2, MODEL_DIM), &device)?;
        let loss = sae.loss(&x, L1Coefficient::new(1.0)?)?;
        let l1 = loss.l1().to_scalar::<f32>()?;
        assert!(l1.abs() < 1e-6);
        Ok(())
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn loss_l1_scales_linearly_with_coefficient() -> Result<(), Error> {
        let device = cpu();
        let sae = unit_bias_sae(&device)?;
        // Features == 1 everywhere, so sum over sae_dim per token = SAE_DIM,
        // mean over tokens = SAE_DIM, scaled L1 = SAE_DIM * coefficient.
        let x = Tensor::zeros((2, MODEL_DIM), DType::F32, &device)?;
        let low = sae.loss(&x, L1Coefficient::new(0.01)?)?;
        let high = sae.loss(&x, L1Coefficient::new(0.02)?)?;
        let low_l1 = low.l1().to_scalar::<f32>()?;
        let high_l1 = high.l1().to_scalar::<f32>()?;
        let expected_low = (SAE_DIM as f32) * 0.01;
        let expected_high = (SAE_DIM as f32) * 0.02;
        assert!((low_l1 - expected_low).abs() < 1e-5);
        assert!((high_l1 - expected_high).abs() < 1e-5);
        assert!(2.0f32.mul_add(-low_l1, high_l1).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn loss_components_nonnegative() -> Result<(), Error> {
        let device = cpu();
        let sae = fresh_sae(&device)?;
        let x = Tensor::randn(0.0f32, 1.0f32, (2, MODEL_DIM), &device)?;
        let loss = sae.loss(&x, L1Coefficient::new(0.01)?)?;
        let mse = loss.mse().to_scalar::<f32>()?;
        let l1 = loss.l1().to_scalar::<f32>()?;
        assert!(mse >= 0.0);
        assert!(l1 >= 0.0);
        Ok(())
    }
}
