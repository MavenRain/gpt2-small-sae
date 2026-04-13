//! Newtype wrappers for hyperparameters and dimensions.
//!
//! Raw `usize`/`f64` values carry no domain meaning, so every hyperparameter
//! that flows across a function boundary is wrapped in a newtype.  This makes
//! illegal combinations (e.g. passing an `SaeDim` where a `ModelDim` is
//! expected) impossible to write, and centralizes bounds checking in the
//! constructors.
//!
//! The conversions to raw scalars are all explicit getters (`as_usize`,
//! `as_f64`); there are no `Into<usize>` impls by design, so the transitions
//! between typed and untyped code are grep-able.

use crate::error::Error;

/// Hidden size of the model's residual stream.  For GPT-2 small this is 768.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct ModelDim(usize);

impl ModelDim {
    /// The GPT-2 small residual stream width.
    pub const GPT2_SMALL: Self = Self(768);

    /// Construct a [`ModelDim`] from a raw size.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] if `size` is zero.
    pub fn new(size: usize) -> Result<Self, Error> {
        (size > 0).then_some(Self(size)).ok_or(Error::Config {
            reason: "model dimension must be nonzero".into(),
        })
    }

    /// The raw dimension as a `usize`.
    #[must_use]
    pub fn as_usize(self) -> usize {
        self.0
    }
}

/// Number of features in the SAE dictionary.  Typically a small integer
/// multiple of [`ModelDim`] (e.g. 8x, 16x, 32x, 64x).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct SaeDim(usize);

impl SaeDim {
    /// Construct an [`SaeDim`] from a raw size.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] if `size` is zero.
    pub fn new(size: usize) -> Result<Self, Error> {
        (size > 0).then_some(Self(size)).ok_or(Error::Config {
            reason: "sae dimension must be nonzero".into(),
        })
    }

    /// Construct an [`SaeDim`] as an integer multiple of a [`ModelDim`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] if `factor` is zero.
    pub fn from_expansion(model_dim: ModelDim, factor: usize) -> Result<Self, Error> {
        (factor > 0)
            .then(|| Self(model_dim.as_usize() * factor))
            .ok_or(Error::Config {
                reason: "sae expansion factor must be nonzero".into(),
            })
    }

    /// The raw dimension as a `usize`.
    #[must_use]
    pub fn as_usize(self) -> usize {
        self.0
    }
}

/// Zero-based index of a transformer block.  For GPT-2 small, valid indices
/// are `0..12`.  The residual stream exposed to the SAE is the output of the
/// block at this index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct LayerIndex(usize);

impl LayerIndex {
    /// Construct a [`LayerIndex`] bounded by the model's depth.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] if `layer >= depth`.
    pub fn new(layer: usize, depth: usize) -> Result<Self, Error> {
        (layer < depth).then_some(Self(layer)).ok_or(Error::Config {
            reason: format!("layer index {layer} out of range for depth {depth}"),
        })
    }

    /// The raw index as a `usize`.
    #[must_use]
    pub fn as_usize(self) -> usize {
        self.0
    }
}

/// L1 sparsity penalty coefficient applied to the feature activations.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct L1Coefficient(f64);

impl L1Coefficient {
    /// Construct an [`L1Coefficient`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] if the value is not positive and finite.
    pub fn new(value: f64) -> Result<Self, Error> {
        (value > 0.0 && value.is_finite())
            .then_some(Self(value))
            .ok_or(Error::Config {
                reason: format!("l1 coefficient must be positive and finite, got {value}"),
            })
    }

    /// The raw coefficient as an `f64`.
    #[must_use]
    pub fn as_f64(self) -> f64 {
        self.0
    }
}

/// `AdamW` learning rate.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub struct LearningRate(f64);

impl LearningRate {
    /// Construct a [`LearningRate`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] if the value is not positive and finite.
    pub fn new(value: f64) -> Result<Self, Error> {
        (value > 0.0 && value.is_finite())
            .then_some(Self(value))
            .ok_or(Error::Config {
                reason: format!("learning rate must be positive and finite, got {value}"),
            })
    }

    /// The raw learning rate as an `f64`.
    #[must_use]
    pub fn as_f64(self) -> f64 {
        self.0
    }
}

/// Number of tokens processed in a single forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct BatchSize(usize);

impl BatchSize {
    /// Construct a [`BatchSize`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] if `value` is zero.
    pub fn new(value: usize) -> Result<Self, Error> {
        (value > 0).then_some(Self(value)).ok_or(Error::Config {
            reason: "batch size must be nonzero".into(),
        })
    }

    /// The raw size as a `usize`.
    #[must_use]
    pub fn as_usize(self) -> usize {
        self.0
    }
}

/// Length of the token window processed by a single forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub struct ContextLength(usize);

impl ContextLength {
    /// Construct a [`ContextLength`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] if `value` is zero.
    pub fn new(value: usize) -> Result<Self, Error> {
        (value > 0).then_some(Self(value)).ok_or(Error::Config {
            reason: "context length must be nonzero".into(),
        })
    }

    /// The raw length as a `usize`.
    #[must_use]
    pub fn as_usize(self) -> usize {
        self.0
    }
}

/// SAE training configuration bundle.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct SaeTrainConfig {
    model_dim: ModelDim,
    sae_dim: SaeDim,
    l1_coefficient: L1Coefficient,
    learning_rate: LearningRate,
    batch_size: BatchSize,
    context_length: ContextLength,
}

impl SaeTrainConfig {
    /// Assemble a training configuration.
    pub fn new(
        model_dim: ModelDim,
        sae_dim: SaeDim,
        l1_coefficient: L1Coefficient,
        learning_rate: LearningRate,
        batch_size: BatchSize,
        context_length: ContextLength,
    ) -> Self {
        Self {
            model_dim,
            sae_dim,
            l1_coefficient,
            learning_rate,
            batch_size,
            context_length,
        }
    }

    /// Residual stream width.
    pub fn model_dim(&self) -> ModelDim {
        self.model_dim
    }

    /// SAE dictionary width.
    pub fn sae_dim(&self) -> SaeDim {
        self.sae_dim
    }

    /// L1 penalty coefficient.
    pub fn l1_coefficient(&self) -> L1Coefficient {
        self.l1_coefficient
    }

    /// `AdamW` learning rate.
    pub fn learning_rate(&self) -> LearningRate {
        self.learning_rate
    }

    /// Batch size in sequences.
    pub fn batch_size(&self) -> BatchSize {
        self.batch_size
    }

    /// Token window length.
    pub fn context_length(&self) -> ContextLength {
        self.context_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- ModelDim -----------------------------------------------------------

    #[test]
    fn model_dim_valid() -> Result<(), Error> {
        let d = ModelDim::new(768)?;
        assert_eq!(d.as_usize(), 768);
        Ok(())
    }

    #[test]
    fn model_dim_zero_rejected() {
        assert!(ModelDim::new(0).is_err());
    }

    #[test]
    fn model_dim_gpt2_small_constant() {
        assert_eq!(ModelDim::GPT2_SMALL.as_usize(), 768);
    }

    // -- SaeDim -------------------------------------------------------------

    #[test]
    fn sae_dim_valid() -> Result<(), Error> {
        let d = SaeDim::new(6144)?;
        assert_eq!(d.as_usize(), 6144);
        Ok(())
    }

    #[test]
    fn sae_dim_zero_rejected() {
        assert!(SaeDim::new(0).is_err());
    }

    #[test]
    fn sae_dim_from_expansion() -> Result<(), Error> {
        let dim = SaeDim::from_expansion(ModelDim::GPT2_SMALL, 8)?;
        assert_eq!(dim.as_usize(), 768 * 8);
        Ok(())
    }

    #[test]
    fn sae_dim_zero_expansion_rejected() {
        assert!(SaeDim::from_expansion(ModelDim::GPT2_SMALL, 0).is_err());
    }

    // -- LayerIndex ---------------------------------------------------------

    #[test]
    fn layer_index_valid_range() -> Result<(), Error> {
        assert_eq!(LayerIndex::new(0, 12)?.as_usize(), 0);
        assert_eq!(LayerIndex::new(11, 12)?.as_usize(), 11);
        Ok(())
    }

    #[test]
    fn layer_index_at_depth_rejected() {
        assert!(LayerIndex::new(12, 12).is_err());
    }

    #[test]
    fn layer_index_beyond_depth_rejected() {
        assert!(LayerIndex::new(100, 12).is_err());
    }

    // -- L1Coefficient ------------------------------------------------------

    #[test]
    fn l1_coefficient_valid() -> Result<(), Error> {
        let c = L1Coefficient::new(5e-4)?;
        assert!((c.as_f64() - 5e-4).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn l1_coefficient_zero_rejected() {
        assert!(L1Coefficient::new(0.0).is_err());
    }

    #[test]
    fn l1_coefficient_negative_rejected() {
        assert!(L1Coefficient::new(-1.0).is_err());
    }

    #[test]
    fn l1_coefficient_infinity_rejected() {
        assert!(L1Coefficient::new(f64::INFINITY).is_err());
    }

    #[test]
    fn l1_coefficient_nan_rejected() {
        assert!(L1Coefficient::new(f64::NAN).is_err());
    }

    // -- LearningRate -------------------------------------------------------

    #[test]
    fn learning_rate_valid() -> Result<(), Error> {
        let r = LearningRate::new(3e-4)?;
        assert!((r.as_f64() - 3e-4).abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn learning_rate_zero_rejected() {
        assert!(LearningRate::new(0.0).is_err());
    }

    #[test]
    fn learning_rate_nan_rejected() {
        assert!(LearningRate::new(f64::NAN).is_err());
    }

    // -- BatchSize ----------------------------------------------------------

    #[test]
    fn batch_size_valid() -> Result<(), Error> {
        let b = BatchSize::new(4)?;
        assert_eq!(b.as_usize(), 4);
        Ok(())
    }

    #[test]
    fn batch_size_zero_rejected() {
        assert!(BatchSize::new(0).is_err());
    }

    // -- ContextLength ------------------------------------------------------

    #[test]
    fn context_length_valid() -> Result<(), Error> {
        let c = ContextLength::new(128)?;
        assert_eq!(c.as_usize(), 128);
        Ok(())
    }

    #[test]
    fn context_length_zero_rejected() {
        assert!(ContextLength::new(0).is_err());
    }

    // -- SaeTrainConfig -----------------------------------------------------

    #[test]
    fn sae_train_config_roundtrip() {
        let cfg = SaeTrainConfig::new(
            ModelDim::GPT2_SMALL,
            SaeDim(6144),
            L1Coefficient(5e-4),
            LearningRate(3e-4),
            BatchSize(4),
            ContextLength(128),
        );
        assert_eq!(cfg.model_dim(), ModelDim::GPT2_SMALL);
        assert_eq!(cfg.sae_dim().as_usize(), 6144);
        assert_eq!(cfg.batch_size().as_usize(), 4);
        assert_eq!(cfg.context_length().as_usize(), 128);
    }
}
