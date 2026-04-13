//! Effectful boundary primitives wrapped in
//! [`comp_cat_rs::effect::io::Io`].
//!
//! Every side effect in this crate (network download, filesystem read, device
//! acquisition, console output) lives behind an [`Io`] program constructed
//! here.  Binary entry points compose these programs with combinators and
//! call [`Io::run`] exactly once at the top level.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use comp_cat_rs::effect::io::Io;

use crate::config::LayerIndex;
use crate::error::{Error, TokenizerError};
use crate::gpt2::Gpt2;
use crate::metrics::TrainMetrics;
use crate::sae::Sae;

const GPT2_REPO: &str = "openai-community/gpt2";

/// Acquire the best available compute device.
#[must_use]
pub fn acquire_device() -> Io<Error, Device> {
    Io::suspend(|| {
        #[cfg(feature = "metal")]
        {
            Device::new_metal(0).map_err(Error::from)
        }
        #[cfg(not(feature = "metal"))]
        {
            Ok(Device::Cpu)
        }
    })
}

/// Download (or fetch from cache) the GPT-2 small safetensors weights.
#[must_use]
pub fn download_gpt2_weights() -> Io<Error, Vec<u8>> {
    Io::suspend(|| {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(GPT2_REPO.to_string());
        let path = repo.get("model.safetensors")?;
        std::fs::read(path).map_err(Error::from)
    })
}

/// Download (or fetch from cache) the GPT-2 tokenizer.
#[must_use]
pub fn download_tokenizer() -> Io<Error, tokenizers::Tokenizer> {
    Io::suspend(|| {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(GPT2_REPO.to_string());
        let path = repo.get("tokenizer.json")?;
        tokenizers::Tokenizer::from_file(path).map_err(|e| Error::Tokenizer(TokenizerError::new(e)))
    })
}

/// Download GPT-2 weights, acquire a device, and construct a frozen
/// [`Gpt2`] model truncated at `layer_index`.
#[must_use]
pub fn load_gpt2(layer_index: LayerIndex) -> Io<Error, (Gpt2, Device)> {
    acquire_device().flat_map(move |device| {
        download_gpt2_weights().flat_map(move |data| {
            Io::suspend(move || {
                let gpt2 = Gpt2::from_bytes(data, layer_index, &device)?;
                Ok((gpt2, device))
            })
        })
    })
}

/// Load a safetensors file from disk into a [`VarBuilder`].
#[must_use]
pub fn load_safetensors(
    path: std::path::PathBuf,
    device: Device,
) -> Io<Error, VarBuilder<'static>> {
    Io::suspend(move || {
        let data = std::fs::read(&path)?;
        VarBuilder::from_buffered_safetensors(data, DType::F32, &device).map_err(Error::from)
    })
}

/// Save an [`Sae`] checkpoint to a safetensors file.
///
/// # Errors
///
/// Returns [`Error::Candle`] on serialization failure, or
/// [`Error::Io`] if the file cannot be written.
#[must_use]
pub fn save_checkpoint(sae: Sae, path: std::path::PathBuf) -> Io<Error, ()> {
    Io::suspend(move || {
        let tensors = HashMap::from([
            ("w_enc".to_string(), sae.w_enc().clone()),
            ("b_enc".to_string(), sae.b_enc().clone()),
            ("w_dec".to_string(), sae.w_dec().clone()),
            ("b_dec".to_string(), sae.b_dec().clone()),
        ]);
        candle_core::safetensors::save(&tensors, &path)?;
        eprintln!("saved checkpoint to {}", path.display());
        Ok(())
    })
}

/// Save collected activation tensors to a safetensors file.
///
/// Tensors are named `batch_0`, `batch_1`, etc.  Use
/// [`load_activations`] to reload them.
///
/// # Errors
///
/// Returns [`Error::Candle`] on serialization failure, or
/// [`Error::Io`] if the file cannot be written.
#[must_use]
pub fn save_activations(activations: Vec<Tensor>, path: std::path::PathBuf) -> Io<Error, ()> {
    Io::suspend(move || {
        let count = activations.len();
        let tensors: HashMap<String, Tensor> = activations
            .into_iter()
            .enumerate()
            .map(|(i, t)| (format!("batch_{i}"), t))
            .collect();
        candle_core::safetensors::save(&tensors, &path)?;
        eprintln!("saved {count} activation batches to {}", path.display());
        Ok(())
    })
}

/// Load activation tensors previously written by [`save_activations`].
///
/// # Errors
///
/// Returns [`Error::Boundary`] if any expected `batch_N` tensor is
/// missing, or [`Error::Candle`] on tensor loading failure.
#[must_use]
pub fn load_activations(path: std::path::PathBuf, device: Device) -> Io<Error, Vec<Tensor>> {
    Io::suspend(move || {
        let tensors = candle_core::safetensors::load(&path, &device)?;
        let count = tensors.len();
        (0..count)
            .map(|i| {
                let key = format!("batch_{i}");
                tensors
                    .get(&key)
                    .ok_or_else(|| Error::Boundary {
                        reason: format!("missing tensor `{key}` in {}", path.display()),
                    })
                    .and_then(|t| t.to_dtype(DType::F32).map_err(Error::from))
            })
            .collect()
    })
}

/// Metadata sidecar for an SAE checkpoint.  Saved alongside the
/// safetensors file as `{stem}.meta.json` so the checkpoint is
/// self-documenting.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct CheckpointMeta {
    /// GPT-2 hookpoint layer.
    pub layer: usize,
    /// Model residual stream width.
    pub model_dim: usize,
    /// SAE dictionary width.
    pub sae_dim: usize,
    /// L1 sparsity coefficient.
    pub l1_coefficient: f64,
    /// Peak learning rate.
    pub lr_peak: f64,
    /// Minimum learning rate (cosine floor).
    pub lr_min: f64,
    /// Linear warmup steps.
    pub warmup_steps: u64,
    /// Total training steps.
    pub total_steps: u64,
    /// Dead feature resample interval (0 = disabled).
    pub resample_interval: u64,
    /// Final MSE at end of training.
    pub final_mse: f64,
    /// Final L0 sparsity.
    pub final_l0: f64,
    /// Final variance explained.
    pub final_var_explained: f64,
    /// Final dead fraction.
    pub final_dead_fraction: f64,
}

impl CheckpointMeta {
    /// Construct metadata from final [`TrainMetrics`] and raw
    /// hyperparameters.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn from_metrics(
        metrics: &TrainMetrics,
        layer: usize,
        model_dim: usize,
        sae_dim: usize,
        l1_coefficient: f64,
        lr_peak: f64,
        lr_min: f64,
        warmup_steps: u64,
        total_steps: u64,
        resample_interval: u64,
    ) -> Self {
        Self {
            layer,
            model_dim,
            sae_dim,
            l1_coefficient,
            lr_peak,
            lr_min,
            warmup_steps,
            total_steps,
            resample_interval,
            final_mse: metrics.mse().as_f64(),
            final_l0: metrics.l0().as_f64(),
            final_var_explained: metrics.variance_explained().as_f64(),
            final_dead_fraction: metrics.dead_fraction().as_f64(),
        }
    }
}

/// Derive the metadata sidecar path from a checkpoint path.
///
/// `sae_checkpoint.safetensors` becomes `sae_checkpoint.meta.json`.
#[must_use]
pub fn meta_path(checkpoint_path: &std::path::Path) -> std::path::PathBuf {
    checkpoint_path.with_extension("meta.json")
}

/// Save checkpoint metadata as a JSON sidecar file.
///
/// # Errors
///
/// Returns [`Error::Json`] on serialization failure, or
/// [`Error::Io`] if the file cannot be written.
#[must_use]
pub fn save_checkpoint_meta(meta: CheckpointMeta, path: std::path::PathBuf) -> Io<Error, ()> {
    Io::suspend(move || {
        let json = serde_json::to_string_pretty(&meta)?;
        std::fs::write(&path, json)?;
        eprintln!("saved metadata to {}", path.display());
        Ok(())
    })
}
