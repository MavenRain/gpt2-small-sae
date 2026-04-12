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
