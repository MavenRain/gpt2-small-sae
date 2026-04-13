//! SAE training loop: pure-functional `AdamW` optimizer, learning rate
//! schedule, decoder weight normalization, and a training stream that
//! yields per-step metrics.
//!
//! The optimizer state is immutable; each call to [`adam_step`] consumes
//! the current [`AdamState`] and produces a fresh one with updated
//! moments.  Parameter updates go through [`candle_core::Var::set`]
//! (interior mutability), so the [`crate::sae::Sae`] tensors reflect
//! the change automatically.

use std::collections::BinaryHeap;
use std::sync::Arc;

use candle_core::{D, DType, Device, Tensor, Var, backprop::GradStore};
use candle_nn::VarMap;
use comp_cat_rs::effect::io::Io;
use comp_cat_rs::effect::stream::Stream;

use crate::config::{L1Coefficient, LearningRate, ModelDim, SaeDim};
use crate::error::Error;
use crate::metrics::{DeadFraction, L0, Mse, TrainMetrics, VarianceExplained};
use crate::sae::Sae;

// ---------------------------------------------------------------------------
// AdamConfig
// ---------------------------------------------------------------------------

/// `AdamW` hyperparameters (excluding the learning rate, which is a
/// separate [`LearningRate`] newtype).
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct AdamConfig {
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
}

impl AdamConfig {
    /// Standard defaults: beta1 = 0.9, beta2 = 0.999, eps = 1e-8,
    /// `weight_decay` = 0.01.
    pub fn standard() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }

    /// First moment exponential decay rate.
    #[must_use]
    pub fn beta1(self) -> f64 {
        self.beta1
    }

    /// Second moment exponential decay rate.
    #[must_use]
    pub fn beta2(self) -> f64 {
        self.beta2
    }

    /// Numerical stability constant added to the denominator.
    #[must_use]
    pub fn eps(self) -> f64 {
        self.eps
    }

    /// L2 regularization coefficient applied before the gradient step.
    #[must_use]
    pub fn weight_decay(self) -> f64 {
        self.weight_decay
    }
}

// ---------------------------------------------------------------------------
// LrSchedule
// ---------------------------------------------------------------------------

/// Learning rate schedule: linear warmup followed by cosine decay to a
/// minimum floor.
///
/// Given a step `t`, `warmup_steps` `W`, `total_steps` `T`, peak
/// learning rate `lr_peak`, and minimum `lr_min`:
///
/// - Warmup (`t < W`): `lr = lr_peak * t / W`
/// - Cosine decay (`t >= W`): `lr = lr_min + 0.5 * (lr_peak - lr_min) * (1 + cos(pi * (t - W) / (T - W)))`
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct LrSchedule {
    lr_peak: f64,
    lr_min: f64,
    warmup_steps: u64,
    total_steps: u64,
}

impl LrSchedule {
    /// Construct a schedule.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] if `warmup_steps >= total_steps`, or if
    /// either learning rate is non-positive or non-finite, or if
    /// `lr_min > lr_peak`.
    pub fn new(
        lr_peak: LearningRate,
        lr_min: LearningRate,
        warmup_steps: u64,
        total_steps: u64,
    ) -> Result<Self, Error> {
        (warmup_steps < total_steps)
            .then_some(())
            .ok_or_else(|| Error::Config {
                reason: format!(
                    "warmup_steps ({warmup_steps}) must be less than \
                     total_steps ({total_steps})"
                ),
            })?;
        (lr_min.as_f64() <= lr_peak.as_f64())
            .then_some(())
            .ok_or_else(|| Error::Config {
                reason: format!(
                    "lr_min ({}) must not exceed lr_peak ({})",
                    lr_min.as_f64(),
                    lr_peak.as_f64()
                ),
            })?;
        Ok(Self {
            lr_peak: lr_peak.as_f64(),
            lr_min: lr_min.as_f64(),
            warmup_steps,
            total_steps,
        })
    }

    /// Construct a flat schedule (constant learning rate, no warmup).
    pub fn constant(lr: LearningRate, total_steps: u64) -> Self {
        Self {
            lr_peak: lr.as_f64(),
            lr_min: lr.as_f64(),
            warmup_steps: 0,
            total_steps: total_steps.max(1),
        }
    }

    /// Evaluate the schedule at step `t`.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn at_step(&self, t: u64) -> f64 {
        if self.warmup_steps > 0 && t < self.warmup_steps {
            self.lr_peak * (t as f64) / (self.warmup_steps as f64)
        } else {
            let decay_steps = self.total_steps.saturating_sub(self.warmup_steps).max(1);
            let progress = (t.saturating_sub(self.warmup_steps) as f64) / (decay_steps as f64);
            let cosine = (1.0 + (std::f64::consts::PI * progress).cos()) * 0.5;
            (self.lr_peak - self.lr_min).mul_add(cosine, self.lr_min)
        }
    }
}

// ---------------------------------------------------------------------------
// ResampleConfig
// ---------------------------------------------------------------------------

/// Dead feature resampling schedule.  Features whose cumulative
/// activation count is zero at a resample step get re-initialized with
/// directions drawn from high-reconstruction-error tokens.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct ResampleConfig {
    interval: u64,
}

impl ResampleConfig {
    /// Construct a resample config.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] if `interval` is zero.
    pub fn new(interval: u64) -> Result<Self, Error> {
        (interval > 0)
            .then_some(Self { interval })
            .ok_or(Error::Config {
                reason: "resample interval must be nonzero".into(),
            })
    }

    /// Whether resampling should fire at the given step.
    #[must_use]
    pub fn should_resample(self, step: u64) -> bool {
        step > 0 && step % self.interval == 0
    }
}

// ---------------------------------------------------------------------------
// AdamState
// ---------------------------------------------------------------------------

/// Immutable `AdamW` optimizer state.  Each call to [`adam_step`]
/// consumes the current state and produces a new one; the caller
/// never holds a stale reference.
#[derive(Debug)]
pub struct AdamState {
    first_moments: Vec<Tensor>,
    second_moments: Vec<Tensor>,
    vars: Vec<Var>,
    step: u64,
}

impl AdamState {
    /// Initialize with zero moments for every trainable variable in
    /// the map.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Candle`] if tensor allocation fails.
    pub fn init(varmap: &VarMap) -> Result<Self, Error> {
        let vars = varmap.all_vars();
        let first_moments = vars
            .iter()
            .map(|v| Tensor::zeros_like(v.as_tensor()).map_err(Error::from))
            .collect::<Result<Vec<_>, _>>()?;
        let second_moments = vars
            .iter()
            .map(|v| Tensor::zeros_like(v.as_tensor()).map_err(Error::from))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            first_moments,
            second_moments,
            vars,
            step: 0,
        })
    }

    /// The trainable variables tracked by this optimizer.
    #[must_use]
    pub fn vars(&self) -> &[Var] {
        &self.vars
    }

    /// Reset all first and second moments to zero and restart the step
    /// counter.  The variable references are preserved so that weight
    /// changes made via [`Var::set`] before this call persist.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Candle`] if tensor allocation fails.
    pub fn reset_moments(self) -> Result<Self, Error> {
        let first_moments = self
            .vars
            .iter()
            .map(|v| Tensor::zeros_like(v.as_tensor()).map_err(Error::from))
            .collect::<Result<Vec<_>, _>>()?;
        let second_moments = self
            .vars
            .iter()
            .map(|v| Tensor::zeros_like(v.as_tensor()).map_err(Error::from))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            first_moments,
            second_moments,
            vars: self.vars,
            step: 0,
        })
    }
}

// ---------------------------------------------------------------------------
// adam_step
// ---------------------------------------------------------------------------

/// Perform one `AdamW` optimization step.
///
/// Consumes `state`, updates every parameter via [`Var::set`]
/// (interior mutability), and returns a new [`AdamState`] with
/// updated moments and step counter.  No mutable bindings are used.
///
/// # Errors
///
/// Returns [`Error::Train`] if a variable has no gradient in `grads`,
/// or [`Error::Candle`] on tensor operation failure.
#[allow(clippy::cast_possible_truncation)]
pub fn adam_step(
    state: AdamState,
    grads: &GradStore,
    lr: f64,
    config: AdamConfig,
) -> Result<AdamState, Error> {
    let AdamState {
        first_moments,
        second_moments,
        vars,
        step,
    } = state;
    let step = step + 1;
    // step is bounded by num_training_steps (well below i32::MAX).
    let bc1 = 1.0 - config.beta1.powi(step as i32);
    let bc2 = 1.0 - config.beta2.powi(step as i32);
    let (new_m, new_v): (Vec<Tensor>, Vec<Tensor>) = vars
        .iter()
        .zip(first_moments)
        .zip(second_moments)
        .map(|((var, m), v)| {
            let grad = grads.get(var.as_tensor()).ok_or_else(|| Error::Train {
                reason: "missing gradient for training variable".into(),
            })?;
            // m_new = beta1 * m + (1 - beta1) * grad
            let m_new = m
                .affine(config.beta1, 0.0)?
                .add(&grad.affine(1.0 - config.beta1, 0.0)?)?;
            // v_new = beta2 * v + (1 - beta2) * grad^2
            let v_new = v
                .affine(config.beta2, 0.0)?
                .add(&grad.sqr()?.affine(1.0 - config.beta2, 0.0)?)?;
            // Bias-corrected estimates.
            let m_hat = m_new.affine(1.0 / bc1, 0.0)?;
            let v_hat = v_new.affine(1.0 / bc2, 0.0)?;
            // Weight decay: param * (1 - lr * wd).
            let decayed = var
                .as_tensor()
                .affine(lr.mul_add(-config.weight_decay, 1.0), 0.0)?;
            // Adam update: lr * m_hat / (sqrt(v_hat) + eps).
            let denom = v_hat.sqrt()?.affine(1.0, config.eps)?;
            let update = m_hat.broadcast_div(&denom)?.affine(lr, 0.0)?;
            var.set(&decayed.sub(&update)?)?;
            Ok((m_new, v_new))
        })
        .collect::<Result<Vec<(Tensor, Tensor)>, Error>>()?
        .into_iter()
        .unzip();
    Ok(AdamState {
        first_moments: new_m,
        second_moments: new_v,
        vars,
        step,
    })
}

// ---------------------------------------------------------------------------
// resume_from_checkpoint
// ---------------------------------------------------------------------------

/// Load SAE weights from a safetensors checkpoint into an existing
/// [`VarMap`], overwriting the current parameter values.
///
/// This enables resuming training from a prior checkpoint: the
/// `VarMap`'s [`Var`]s share storage with the [`Sae`] tensors, so
/// setting them updates the model in place.  The optimizer state
/// (moments) is not restored; call this before [`AdamState::init`]
/// so the optimizer starts fresh with the loaded weights.
///
/// # Errors
///
/// Returns [`Error::Train`] if the `VarMap` mutex is poisoned or a
/// checkpoint tensor name does not match any model variable.  Returns
/// [`Error::Candle`] if tensor loading or dtype conversion fails.
pub fn resume_from_checkpoint(
    varmap: &VarMap,
    path: &std::path::Path,
    device: &Device,
) -> Result<(), Error> {
    let tensors = candle_core::safetensors::load(path, device)?;
    let data = varmap.data().lock().map_err(|e| Error::Train {
        reason: format!("failed to lock VarMap: {e}"),
    })?;
    tensors.iter().try_for_each(|(name, tensor)| {
        data.get(name)
            .ok_or_else(|| Error::Train {
                reason: format!("checkpoint tensor `{name}` not found in model"),
            })
            .and_then(|var| var.set(&tensor.to_dtype(DType::F32)?).map_err(Error::from))
    })
}

// ---------------------------------------------------------------------------
// normalize_decoder
// ---------------------------------------------------------------------------

/// Project decoder weight rows to unit L2 norm.
///
/// The decoder weight has shape `(sae_dim, model_dim)`.  Each row is
/// a feature direction in the residual stream.  Normalizing prevents
/// the encoder from scaling features by shrinking decoder norms.
///
/// The decoder [`Var`] is identified among training variables by its
/// unique shape (transposed relative to the encoder weight).
///
/// # Errors
///
/// Returns [`Error::Train`] if no variable matches the expected
/// decoder shape, or [`Error::Candle`] on tensor operation failure.
pub fn normalize_decoder(vars: &[Var], sae_dim: SaeDim, model_dim: ModelDim) -> Result<(), Error> {
    let dec_shape = [sae_dim.as_usize(), model_dim.as_usize()];
    vars.iter()
        .find(|v| v.as_tensor().dims() == dec_shape)
        .ok_or_else(|| Error::Train {
            reason: "decoder weight not found among training variables".into(),
        })
        .and_then(|dec_var| {
            let w = dec_var.as_tensor();
            let norms = w.sqr()?.sum(D::Minus1)?.sqrt()?.clamp(1e-8, 1e30)?;
            let normalized = w.broadcast_div(&norms.unsqueeze(1)?)?;
            dec_var.set(&normalized).map_err(Error::from)
        })
}

// ---------------------------------------------------------------------------
// resample_dead_features
// ---------------------------------------------------------------------------

/// Reinitialize dead features using directions from high-error tokens.
///
/// For each feature whose cumulative activation count is zero:
///
/// 1.  Run the current batch through the SAE and rank tokens by
///     reconstruction error (descending).
/// 2.  Pick the highest-error tokens as replacement directions:
///     `dir = normalize(x_token - b_dec)`.
/// 3.  Set the decoder row to `dir`, the encoder column to
///     `dir * 0.2 * avg_alive_encoder_norm`, and the encoder bias to
///     zero.
///
/// Returns the number of features resampled (0 when none are dead).
///
/// # Errors
///
/// Returns [`Error::Train`] if a required weight variable cannot be
/// identified by shape, or [`Error::Candle`] on tensor operation
/// failure.
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
pub fn resample_dead_features(
    sae: &Sae,
    vars: &[Var],
    batch: &Tensor,
    feature_counts: &Tensor,
    model_dim: ModelDim,
    sae_dim: SaeDim,
) -> Result<u64, Error> {
    let d_model = model_dim.as_usize();
    let d_sae = sae_dim.as_usize();
    let counts_vec: Vec<f32> = feature_counts.to_vec1()?;
    let dead_indices: Vec<usize> = counts_vec
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c == 0.0)
        .map(|(i, _)| i)
        .collect();

    if dead_indices.is_empty() {
        Ok(0)
    } else {
        let n_dead = dead_indices.len();

        // Bounds-checked f32 slice access.
        let at = |slice: &[f32], i: usize| -> Result<f32, Error> {
            slice.get(i).copied().ok_or_else(|| Error::Train {
                reason: format!("resample index {i} out of bounds (len {})", slice.len()),
            })
        };

        // Flatten batch to 2D (n_tokens, model_dim).
        let n_tokens: usize = batch.dims().iter().rev().skip(1).product();
        let flat_batch = batch.reshape((n_tokens, d_model))?;

        // Per-token reconstruction error (sum of squared residuals).
        let fwd = sae.forward(&flat_batch)?;
        let per_token_error: Vec<f32> = flat_batch
            .sub(fwd.reconstruction())?
            .sqr()?
            .sum(D::Minus1)?
            .to_vec1()?;

        // Select the highest-error tokens via a max-heap on bit
        // representations (monotone for non-negative f32).
        let chosen: Vec<usize> = {
            let pairs: Vec<(u32, usize)> = per_token_error
                .iter()
                .enumerate()
                .map(|(i, &e)| (e.to_bits(), i))
                .collect();
            BinaryHeap::from(pairs)
                .into_sorted_vec()
                .into_iter()
                .rev()
                .take(n_dead)
                .map(|(_, i)| i)
                .collect()
        };

        if chosen.is_empty() {
            Ok(0)
        } else {
            // Flat batch data and decoder bias for direction computation.
            let flat_batch_data: Vec<f32> = flat_batch.reshape(n_tokens * d_model)?.to_vec1()?;
            let b_dec_vec: Vec<f32> = sae.b_dec().to_vec1()?;

            // Replacement directions: normalized(x_token - b_dec),
            // cycling through chosen tokens when n_dead > n_tokens.
            let n_chosen = chosen.len();
            let directions: Vec<Vec<f32>> =
                (0..n_dead)
                    .map(|dir_idx| {
                        let tok_idx = chosen.get(dir_idx % n_chosen).copied().ok_or_else(|| {
                            Error::Train {
                                reason: "chosen token index out of bounds".into(),
                            }
                        })?;
                        let start = tok_idx * d_model;
                        let shifted: Vec<f32> = (0..d_model)
                            .map(|j| Ok(at(&flat_batch_data, start + j)? - at(&b_dec_vec, j)?))
                            .collect::<Result<_, Error>>()?;
                        let norm = shifted.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                        Ok(shifted.iter().map(|&x| x / norm).collect())
                    })
                    .collect::<Result<_, Error>>()?;

            // Identify weight variables by shape.
            let enc_var = vars
                .iter()
                .find(|v| v.as_tensor().dims() == [d_model, d_sae])
                .ok_or_else(|| Error::Train {
                    reason: "encoder weight not found among training variables".into(),
                })?;
            let dec_var = vars
                .iter()
                .find(|v| v.as_tensor().dims() == [d_sae, d_model])
                .ok_or_else(|| Error::Train {
                    reason: "decoder weight not found among training variables".into(),
                })?;
            let benc_var = vars
                .iter()
                .find(|v| v.as_tensor().dims() == [d_sae] && v.as_tensor().dims() != [d_model])
                .ok_or_else(|| Error::Train {
                    reason: "encoder bias not found among training variables".into(),
                })?;

            // Read current weights into flat CPU vectors.
            let orig_encoder: Vec<f32> = enc_var.as_tensor().reshape(d_model * d_sae)?.to_vec1()?;
            let orig_decoder: Vec<f32> = dec_var.as_tensor().reshape(d_sae * d_model)?.to_vec1()?;
            let orig_enc_bias: Vec<f32> = benc_var.as_tensor().to_vec1()?;

            // Average L2 norm of alive encoder columns (fallback 1.0).
            let alive_count = counts_vec.iter().filter(|c| **c > 0.0).count().max(1);
            let avg_alive_norm: f32 = counts_vec
                .iter()
                .enumerate()
                .filter(|&(_, &c)| c > 0.0)
                .try_fold(0.0_f32, |total, (j, _)| {
                    let col_sq = (0..d_model).try_fold(0.0_f32, |acc, i| {
                        at(&orig_encoder, i * d_sae + j).map(|v| v.mul_add(v, acc))
                    })?;
                    Ok::<_, Error>(total + col_sq.sqrt())
                })?
                / alive_count as f32;
            let scale = 0.2 * avg_alive_norm;

            // Lookup: feature index -> Option<direction index>.
            let feature_to_dir: Vec<Option<usize>> = (0..d_sae)
                .map(|feat| dead_indices.iter().position(|&d| d == feat))
                .collect();

            // Bounds-checked direction lookup.
            let dir_val = |di: usize, elem: usize| -> Result<f32, Error> {
                directions
                    .get(di)
                    .and_then(|d| d.get(elem))
                    .copied()
                    .ok_or_else(|| Error::Train {
                        reason: "direction index out of bounds".into(),
                    })
            };

            // Build replacement tensors without mutation.
            let enc_data: Vec<f32> = (0..d_model * d_sae)
                .map(|flat| {
                    let row = flat / d_sae;
                    let col = flat % d_sae;
                    feature_to_dir.get(col).copied().flatten().map_or_else(
                        || at(&orig_encoder, flat),
                        |di| dir_val(di, row).map(|v| v * scale),
                    )
                })
                .collect::<Result<_, Error>>()?;
            let dec_data: Vec<f32> = (0..d_sae * d_model)
                .map(|flat| {
                    let row = flat / d_model;
                    let col = flat % d_model;
                    feature_to_dir
                        .get(row)
                        .copied()
                        .flatten()
                        .map_or_else(|| at(&orig_decoder, flat), |di| dir_val(di, col))
                })
                .collect::<Result<_, Error>>()?;
            let benc_data: Vec<f32> = (0..d_sae)
                .map(|feat| {
                    feature_to_dir
                        .get(feat)
                        .copied()
                        .flatten()
                        .map_or_else(|| at(&orig_enc_bias, feat), |_| Ok(0.0))
                })
                .collect::<Result<_, Error>>()?;

            // Write back to vars.
            let device = enc_var.as_tensor().device().clone();
            enc_var.set(&Tensor::from_slice(&enc_data, (d_model, d_sae), &device)?)?;
            dec_var.set(&Tensor::from_slice(&dec_data, (d_sae, d_model), &device)?)?;
            benc_var.set(&Tensor::from_slice(&benc_data, (d_sae,), &device)?)?;

            #[allow(clippy::cast_possible_truncation)]
            Ok(n_dead as u64)
        }
    }
}

// ---------------------------------------------------------------------------
// TrainState (private)
// ---------------------------------------------------------------------------

/// Internal state threaded through the training stream's unfold.
struct TrainState {
    sae: Sae,
    adam: AdamState,
    activations: Vec<Tensor>,
    feature_counts: Tensor,
    batch_idx: usize,
    step_idx: u64,
    num_steps: u64,
    l1_coefficient: L1Coefficient,
    lr_schedule: LrSchedule,
    adam_config: AdamConfig,
    model_dim: ModelDim,
    sae_dim: SaeDim,
    resample_config: Option<ResampleConfig>,
}

// ---------------------------------------------------------------------------
// training_stream
// ---------------------------------------------------------------------------

/// Build a lazy training stream that yields [`TrainMetrics`] for each
/// optimization step.
///
/// The stream cycles through the pre-collected `activations` for
/// `num_steps` steps, performing one gradient update per step.  All
/// parameter updates use [`Var::set`] (interior mutability), so a
/// clone of `sae` obtained before calling this function will reflect
/// the trained weights after the stream is fully consumed.
///
/// # Errors
///
/// Returns [`Error::Candle`] if the initial feature-count tensor
/// cannot be allocated on `device`.
#[allow(clippy::too_many_arguments)]
pub fn training_stream(
    sae: Sae,
    adam: AdamState,
    activations: Vec<Tensor>,
    device: &Device,
    l1_coefficient: L1Coefficient,
    lr_schedule: LrSchedule,
    adam_config: AdamConfig,
    model_dim: ModelDim,
    sae_dim: SaeDim,
    num_steps: u64,
    resample_config: Option<ResampleConfig>,
) -> Result<Stream<Error, TrainMetrics>, Error> {
    let feature_counts = Tensor::zeros((sae_dim.as_usize(),), DType::F32, device)?;
    let state = TrainState {
        sae,
        adam,
        activations,
        feature_counts,
        batch_idx: 0,
        step_idx: 0,
        num_steps,
        l1_coefficient,
        lr_schedule,
        adam_config,
        model_dim,
        sae_dim,
        resample_config,
    };
    Ok(Stream::unfold(
        state,
        Arc::new(|state: TrainState| {
            if state.step_idx >= state.num_steps {
                Io::pure(None)
            } else {
                Io::suspend(move || train_step(state))
            }
        }),
    ))
}

// ---------------------------------------------------------------------------
// train_step (private)
// ---------------------------------------------------------------------------

/// Execute a single training step: forward, loss, backward, optimize,
/// normalize, compute metrics.
fn train_step(state: TrainState) -> Result<Option<(TrainMetrics, TrainState)>, Error> {
    // Cycle through activation batches.
    let n_act = state.activations.len().max(1);
    let batch = state
        .activations
        .get(state.batch_idx % n_act)
        .ok_or_else(|| Error::Train {
            reason: "empty activation buffer".into(),
        })?;

    // Forward pass (pre-update).
    let fwd = state.sae.forward(batch)?;

    // Loss: MSE + scaled L1.
    let mse_loss = fwd.reconstruction().sub(batch)?.sqr()?.mean_all()?;
    let l1_loss = fwd
        .features()
        .abs()?
        .sum(D::Minus1)?
        .mean_all()?
        .affine(state.l1_coefficient.as_f64(), 0.0)?;
    let total_loss = mse_loss.add(&l1_loss)?;

    // Backward pass.
    let grads = total_loss.backward()?;

    // Scheduled learning rate for this step.
    let lr = state.lr_schedule.at_step(state.step_idx);

    // Optimizer step.
    let adam = adam_step(state.adam, &grads, lr, state.adam_config)?;

    // Decoder weight normalization.
    normalize_decoder(&adam.vars, state.sae_dim, state.model_dim)?;

    // Metrics from the pre-update forward pass.
    let l0 = L0::compute(fwd.features())?;
    let mse = Mse::compute(fwd.reconstruction(), batch)?;
    let var_expl = VarianceExplained::compute(fwd.reconstruction(), batch)?;
    let active = fwd
        .features()
        .gt(0.0_f32)?
        .to_dtype(DType::F32)?
        .transpose(0, 1)?
        .sum(D::Minus1)?;
    let feature_counts = state.feature_counts.add(&active)?;
    let dead = DeadFraction::compute(&feature_counts)?;

    // Conditional dead feature resampling.
    let should_resample = state
        .resample_config
        .is_some_and(|rc| rc.should_resample(state.step_idx));
    let (adam, feature_counts, resampled) = if should_resample {
        let n = resample_dead_features(
            &state.sae,
            adam.vars(),
            batch,
            &feature_counts,
            state.model_dim,
            state.sae_dim,
        )?;
        let reset_adam = adam.reset_moments()?;
        let reset_counts = Tensor::zeros(
            (state.sae_dim.as_usize(),),
            DType::F32,
            feature_counts.device(),
        )?;
        (reset_adam, reset_counts, n)
    } else {
        (adam, feature_counts, 0)
    };

    let metrics = TrainMetrics::new(state.step_idx, l0, mse, var_expl, dead, lr, resampled);

    let new_state = TrainState {
        sae: state.sae,
        adam,
        activations: state.activations,
        feature_counts,
        batch_idx: state.batch_idx + 1,
        step_idx: state.step_idx + 1,
        num_steps: state.num_steps,
        l1_coefficient: state.l1_coefficient,
        lr_schedule: state.lr_schedule,
        adam_config: state.adam_config,
        model_dim: state.model_dim,
        sae_dim: state.sae_dim,
        resample_config: state.resample_config,
    };

    Ok(Some((metrics, new_state)))
}
