//! Training and evaluation metrics as newtypes over `f64`, computed from
//! candle [`Tensor`]s.
//!
//! Each metric has a single constructor path that runs the relevant tensor
//! reduction.  Consumers hold the resulting newtype, which is opaque and can
//! only be turned back into `f64` through its explicit getter, preventing
//! accidental arithmetic between different metrics.

use candle_core::{D, Tensor};

use crate::error::Error;

/// L0 "sparsity": average number of nonzero feature activations per token.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct L0(f64);

/// Reconstruction mean squared error, averaged over tokens and features.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct Mse(f64);

/// Fraction of residual stream variance explained by the reconstruction.
/// `1 - SSE / SST`, clamped to `[-1, 1]` in pathological runs.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct VarianceExplained(f64);

/// Fraction of features that never fired during the evaluation window.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct DeadFraction(f64);

/// All metrics captured for a single training step or evaluation window.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct TrainMetrics {
    step: u64,
    l0: L0,
    mse: Mse,
    variance_explained: VarianceExplained,
    dead_fraction: DeadFraction,
    learning_rate: f64,
    resampled: u64,
}

impl L0 {
    /// Compute mean active features per token from a feature tensor of shape
    /// `(batch * ctx, sae_dim)`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Candle`] on tensor operation failure.
    pub fn compute(features: &Tensor) -> Result<Self, Error> {
        let active = features.gt(0.0_f32)?.to_dtype(candle_core::DType::F32)?;
        let per_token = active.sum(D::Minus1)?;
        let mean = per_token.mean_all()?.to_scalar::<f32>()?;
        Ok(Self(f64::from(mean)))
    }

    /// The raw value.
    #[must_use]
    pub fn as_f64(self) -> f64 {
        self.0
    }
}

impl Mse {
    /// Compute elementwise mean squared error between `recon` and `target`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Candle`] on tensor operation failure.
    pub fn compute(recon: &Tensor, target: &Tensor) -> Result<Self, Error> {
        let diff = recon.sub(target)?;
        let sq = diff.sqr()?;
        let mean = sq.mean_all()?.to_scalar::<f32>()?;
        Ok(Self(f64::from(mean)))
    }

    /// The raw value.
    #[must_use]
    pub fn as_f64(self) -> f64 {
        self.0
    }
}

impl VarianceExplained {
    /// Fraction of target variance explained by the reconstruction.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Candle`] on tensor operation failure.
    pub fn compute(recon: &Tensor, target: &Tensor) -> Result<Self, Error> {
        let mean = target.mean_all()?;
        let centered = target.broadcast_sub(&mean)?;
        let sst = centered.sqr()?.sum_all()?.to_scalar::<f32>()?;
        let sse = recon.sub(target)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
        let value = f64::from(1.0 - sse / sst.max(f32::EPSILON));
        Ok(Self(value.clamp(-1.0, 1.0)))
    }

    /// The raw value.
    #[must_use]
    pub fn as_f64(self) -> f64 {
        self.0
    }
}

impl DeadFraction {
    /// Given per-feature activation counts of shape `(sae_dim,)`, compute the
    /// fraction that are exactly zero.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Candle`] on tensor operation failure.
    pub fn compute(per_feature_counts: &Tensor) -> Result<Self, Error> {
        let zero_mask = per_feature_counts
            .eq(0.0_f32)?
            .to_dtype(candle_core::DType::F32)?;
        let mean = zero_mask.mean_all()?.to_scalar::<f32>()?;
        Ok(Self(f64::from(mean)))
    }

    /// The raw value.
    #[must_use]
    pub fn as_f64(self) -> f64 {
        self.0
    }
}

impl TrainMetrics {
    /// Assemble a metrics record.
    pub fn new(
        step: u64,
        l0: L0,
        mse: Mse,
        variance_explained: VarianceExplained,
        dead_fraction: DeadFraction,
        learning_rate: f64,
        resampled: u64,
    ) -> Self {
        Self {
            step,
            l0,
            mse,
            variance_explained,
            dead_fraction,
            learning_rate,
            resampled,
        }
    }

    /// The training step this record corresponds to.
    #[must_use]
    pub fn step(&self) -> u64 {
        self.step
    }

    /// L0 sparsity.
    pub fn l0(&self) -> L0 {
        self.l0
    }

    /// Mean squared reconstruction error.
    pub fn mse(&self) -> Mse {
        self.mse
    }

    /// Variance explained by the reconstruction.
    pub fn variance_explained(&self) -> VarianceExplained {
        self.variance_explained
    }

    /// Fraction of dead features.
    pub fn dead_fraction(&self) -> DeadFraction {
        self.dead_fraction
    }

    /// The learning rate used for this step.
    #[must_use]
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Number of features resampled at this step (0 on non-resample steps).
    #[must_use]
    pub fn resampled(&self) -> u64 {
        self.resampled
    }

    /// Render the metrics as a JSONL line.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Json`] on serialization failure.
    pub fn to_jsonl(&self) -> Result<String, Error> {
        let record = serde_json::json!({
            "step": self.step,
            "l0": self.l0.as_f64(),
            "mse": self.mse.as_f64(),
            "var_explained": self.variance_explained.as_f64(),
            "dead_fraction": self.dead_fraction.as_f64(),
            "lr": self.learning_rate,
            "resampled": self.resampled,
        });
        Ok(format!("{}\n", serde_json::to_string(&record)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    // -- L0 -----------------------------------------------------------------

    #[test]
    fn l0_all_zeros() -> Result<(), Error> {
        let features = Tensor::zeros((4, 8), DType::F32, &Device::Cpu)?;
        let l0 = L0::compute(&features)?;
        assert!((l0.as_f64() - 0.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn l0_all_positive() -> Result<(), Error> {
        let features = Tensor::ones((4, 8), DType::F32, &Device::Cpu)?;
        let l0 = L0::compute(&features)?;
        assert!((l0.as_f64() - 8.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn l0_half_active() -> Result<(), Error> {
        // 2 tokens, 4 features: first two features active, last two zero.
        let data: Vec<f32> = vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0];
        let features = Tensor::from_vec(data, (2, 4), &Device::Cpu)?;
        let l0 = L0::compute(&features)?;
        assert!((l0.as_f64() - 2.0).abs() < 1e-6);
        Ok(())
    }

    // -- Mse ----------------------------------------------------------------

    #[test]
    fn mse_identical_tensors() -> Result<(), Error> {
        let t = Tensor::ones((4, 8), DType::F32, &Device::Cpu)?;
        let mse = Mse::compute(&t, &t)?;
        assert!((mse.as_f64() - 0.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn mse_known_value() -> Result<(), Error> {
        let a = Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?;
        let b = Tensor::ones((2, 2), DType::F32, &Device::Cpu)?;
        let mse = Mse::compute(&a, &b)?;
        // MSE of all-zeros vs all-ones = 1.0
        assert!((mse.as_f64() - 1.0).abs() < 1e-6);
        Ok(())
    }

    // -- VarianceExplained --------------------------------------------------

    #[test]
    fn var_explained_perfect_reconstruction() -> Result<(), Error> {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let t = Tensor::from_vec(data, (2, 2), &Device::Cpu)?;
        let ve = VarianceExplained::compute(&t, &t)?;
        assert!((ve.as_f64() - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn var_explained_zero_reconstruction() -> Result<(), Error> {
        // Reconstructing with zeros should explain no variance.
        let target_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let target = Tensor::from_vec(target_data, (2, 2), &Device::Cpu)?;
        let recon = Tensor::zeros((2, 2), DType::F32, &Device::Cpu)?;
        let ve = VarianceExplained::compute(&recon, &target)?;
        assert!(ve.as_f64() < 1.0);
        Ok(())
    }

    // -- DeadFraction -------------------------------------------------------

    #[test]
    fn dead_fraction_all_dead() -> Result<(), Error> {
        let counts = Tensor::zeros(8, DType::F32, &Device::Cpu)?;
        let df = DeadFraction::compute(&counts)?;
        assert!((df.as_f64() - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn dead_fraction_none_dead() -> Result<(), Error> {
        let counts = Tensor::ones(8, DType::F32, &Device::Cpu)?;
        let df = DeadFraction::compute(&counts)?;
        assert!((df.as_f64() - 0.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn dead_fraction_half_dead() -> Result<(), Error> {
        let data: Vec<f32> = vec![0.0, 0.0, 1.0, 1.0];
        let counts = Tensor::from_vec(data, 4, &Device::Cpu)?;
        let df = DeadFraction::compute(&counts)?;
        assert!((df.as_f64() - 0.5).abs() < 1e-6);
        Ok(())
    }

    // -- TrainMetrics -------------------------------------------------------

    #[test]
    fn train_metrics_getters() {
        let m = TrainMetrics::new(
            42,
            L0(10.0),
            Mse(0.5),
            VarianceExplained(0.9),
            DeadFraction(0.1),
            3e-4,
            5,
        );
        assert_eq!(m.step(), 42);
        assert!((m.l0().as_f64() - 10.0).abs() < 1e-12);
        assert!((m.mse().as_f64() - 0.5).abs() < 1e-12);
        assert!((m.variance_explained().as_f64() - 0.9).abs() < 1e-12);
        assert!((m.dead_fraction().as_f64() - 0.1).abs() < 1e-12);
        assert!((m.learning_rate() - 3e-4).abs() < 1e-12);
        assert_eq!(m.resampled(), 5);
    }

    #[test]
    fn train_metrics_to_jsonl() -> Result<(), Error> {
        let m = TrainMetrics::new(
            10,
            L0(5.0),
            Mse(0.1),
            VarianceExplained(0.95),
            DeadFraction(0.02),
            1e-4,
            0,
        );
        let line = m.to_jsonl()?;
        assert!(line.ends_with('\n'));
        let parsed: serde_json::Value = serde_json::from_str(line.trim())?;
        assert_eq!(parsed["step"], 10);
        assert_eq!(parsed["l0"], 5.0);
        assert_eq!(parsed["mse"], 0.1);
        Ok(())
    }
}
