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
///
/// All fields are private; the serde derives still see them for
/// `(de)serialization`, and downstream code reads them through the
/// accessor methods below.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct CheckpointMeta {
    layer: usize,
    model_dim: usize,
    sae_dim: usize,
    l1_coefficient: f64,
    lr_peak: f64,
    lr_min: f64,
    warmup_steps: u64,
    total_steps: u64,
    resample_interval: u64,
    final_mse: f64,
    final_l0: f64,
    final_var_explained: f64,
    final_dead_fraction: f64,
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

    /// GPT-2 hookpoint layer.
    #[must_use]
    pub fn layer(&self) -> usize {
        self.layer
    }

    /// Model residual stream width.
    #[must_use]
    pub fn model_dim(&self) -> usize {
        self.model_dim
    }

    /// SAE dictionary width.
    #[must_use]
    pub fn sae_dim(&self) -> usize {
        self.sae_dim
    }

    /// L1 sparsity coefficient.
    #[must_use]
    pub fn l1_coefficient(&self) -> f64 {
        self.l1_coefficient
    }

    /// Peak learning rate.
    #[must_use]
    pub fn lr_peak(&self) -> f64 {
        self.lr_peak
    }

    /// Minimum learning rate (cosine floor).
    #[must_use]
    pub fn lr_min(&self) -> f64 {
        self.lr_min
    }

    /// Linear warmup steps.
    #[must_use]
    pub fn warmup_steps(&self) -> u64 {
        self.warmup_steps
    }

    /// Total training steps.
    #[must_use]
    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }

    /// Dead feature resample interval (0 = disabled).
    #[must_use]
    pub fn resample_interval(&self) -> u64 {
        self.resample_interval
    }

    /// Final MSE at end of training.
    #[must_use]
    pub fn final_mse(&self) -> f64 {
        self.final_mse
    }

    /// Final L0 sparsity.
    #[must_use]
    pub fn final_l0(&self) -> f64 {
        self.final_l0
    }

    /// Final variance explained.
    #[must_use]
    pub fn final_var_explained(&self) -> f64 {
        self.final_var_explained
    }

    /// Final dead fraction.
    #[must_use]
    pub fn final_dead_fraction(&self) -> f64 {
        self.final_dead_fraction
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ModelDim, SaeDim};
    use crate::metrics::{DeadFraction, L0, Mse, TrainMetrics, VarianceExplained};
    use candle_nn::{VarBuilder, VarMap};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Each test grabs a unique path in the OS temp dir, avoiding
    /// collisions under parallel execution.
    fn unique_temp_path(stem: &str, ext: &str) -> std::path::PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        std::env::temp_dir().join(format!("gpt2_sae_test_{stem}_{pid}_{n}.{ext}"))
    }

    fn sample_metrics() -> Result<TrainMetrics, Error> {
        let device = Device::Cpu;
        let zeros = Tensor::zeros((2, 4), DType::F32, &device)?;
        let ones = Tensor::ones((2, 4), DType::F32, &device)?;
        let l0 = L0::compute(&zeros)?;
        let mse = Mse::compute(&zeros, &ones)?;
        let var = VarianceExplained::compute(&zeros, &ones)?;
        let counts = Tensor::zeros(4, DType::F32, &device)?;
        let dead = DeadFraction::compute(&counts)?;
        Ok(TrainMetrics::new(100, l0, mse, var, dead, 3e-4, 0))
    }

    fn sample_meta() -> Result<CheckpointMeta, Error> {
        let metrics = sample_metrics()?;
        Ok(CheckpointMeta::from_metrics(
            &metrics, 8, 768, 6144, 5e-4, 3e-4, 3e-5, 500, 10_000, 1_000,
        ))
    }

    // -- meta_path ----------------------------------------------------------

    #[test]
    fn meta_path_from_safetensors() {
        let p = std::path::Path::new("sae_checkpoint.safetensors");
        assert_eq!(
            meta_path(p),
            std::path::PathBuf::from("sae_checkpoint.meta.json"),
        );
    }

    #[test]
    fn meta_path_from_nested_path() {
        let p = std::path::Path::new("runs/l4_16x/sae.safetensors");
        assert_eq!(
            meta_path(p),
            std::path::PathBuf::from("runs/l4_16x/sae.meta.json"),
        );
    }

    // -- CheckpointMeta accessors ------------------------------------------

    #[test]
    fn checkpoint_meta_accessors() -> Result<(), Error> {
        let meta = sample_meta()?;
        assert_eq!(meta.layer(), 8);
        assert_eq!(meta.model_dim(), 768);
        assert_eq!(meta.sae_dim(), 6144);
        assert!((meta.l1_coefficient() - 5e-4).abs() < 1e-12);
        assert!((meta.lr_peak() - 3e-4).abs() < 1e-12);
        assert!((meta.lr_min() - 3e-5).abs() < 1e-12);
        assert_eq!(meta.warmup_steps(), 500);
        assert_eq!(meta.total_steps(), 10_000);
        assert_eq!(meta.resample_interval(), 1_000);
        // Final metrics are copied from TrainMetrics.
        assert!((meta.final_mse() - 1.0).abs() < 1e-6);
        assert!((meta.final_l0() - 0.0).abs() < 1e-6);
        assert!((meta.final_dead_fraction() - 1.0).abs() < 1e-6);
        Ok(())
    }

    // -- CheckpointMeta serde roundtrip ------------------------------------

    #[test]
    fn checkpoint_meta_serde_roundtrip() -> Result<(), Error> {
        let meta = sample_meta()?;
        let json = serde_json::to_string(&meta)?;
        let parsed: CheckpointMeta = serde_json::from_str(&json)?;
        assert_eq!(parsed.layer(), meta.layer());
        assert_eq!(parsed.sae_dim(), meta.sae_dim());
        assert!((parsed.l1_coefficient() - meta.l1_coefficient()).abs() < 1e-12);
        assert!((parsed.final_mse() - meta.final_mse()).abs() < 1e-12);
        Ok(())
    }

    // -- save_checkpoint_meta -----------------------------------------------

    #[test]
    fn save_checkpoint_meta_writes_reloadable_json() -> Result<(), Error> {
        let meta = sample_meta()?;
        let path = unique_temp_path("meta", "json");
        save_checkpoint_meta(meta, path.clone()).run()?;
        let content = std::fs::read_to_string(&path)?;
        let parsed: CheckpointMeta = serde_json::from_str(&content)?;
        assert_eq!(parsed.layer(), 8);
        assert_eq!(parsed.sae_dim(), 6144);
        std::fs::remove_file(&path).ok();
        Ok(())
    }

    // -- save_checkpoint + Sae::from_safetensors roundtrip ------------------

    #[test]
    fn save_checkpoint_roundtrips_through_sae() -> Result<(), Error> {
        let device = Device::Cpu;
        let model_dim = ModelDim::new(16)?;
        let sae_dim = SaeDim::new(32)?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let sae = Sae::new(vb, model_dim, sae_dim)?;

        let path = unique_temp_path("sae", "safetensors");
        save_checkpoint(sae.clone(), path.clone()).run()?;

        let reloaded = Sae::from_safetensors(&path, model_dim, sae_dim, &device)?;
        assert_eq!(reloaded.w_enc().dims(), &[16, 32]);
        assert_eq!(reloaded.w_dec().dims(), &[32, 16]);
        assert_eq!(reloaded.b_enc().dims(), &[32]);
        assert_eq!(reloaded.b_dec().dims(), &[16]);

        // Tensors survive the roundtrip bit-for-bit.
        let orig: Vec<f32> = sae.w_enc().flatten_all()?.to_vec1()?;
        let back: Vec<f32> = reloaded.w_enc().flatten_all()?.to_vec1()?;
        assert_eq!(orig, back);

        std::fs::remove_file(&path).ok();
        Ok(())
    }

    // -- save_activations / load_activations roundtrip ---------------------

    #[test]
    fn save_and_load_activations_roundtrip() -> Result<(), Error> {
        let device = Device::Cpu;
        let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
        let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], (2, 2), &device)?;
        let path = unique_temp_path("acts", "safetensors");
        save_activations(vec![a, b], path.clone()).run()?;

        let loaded = load_activations(path.clone(), device).run()?;
        assert_eq!(loaded.len(), 2);
        let first = loaded.first().ok_or_else(|| Error::Boundary {
            reason: "missing batch_0".into(),
        })?;
        let first_vec: Vec<f32> = first.flatten_all()?.to_vec1()?;
        assert_eq!(first_vec, vec![1.0, 2.0, 3.0, 4.0]);

        std::fs::remove_file(&path).ok();
        Ok(())
    }

    #[test]
    fn load_activations_missing_file_errors() {
        let device = Device::Cpu;
        let path = unique_temp_path("nonexistent", "safetensors");
        assert!(load_activations(path, device).run().is_err());
    }

    // -- load_safetensors ---------------------------------------------------

    #[test]
    fn load_safetensors_roundtrip() -> Result<(), Error> {
        let device = Device::Cpu;
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], 3, &device)?;
        let path = unique_temp_path("solo", "safetensors");
        let tensors: HashMap<String, Tensor> = HashMap::from([("x".to_string(), tensor)]);
        candle_core::safetensors::save(&tensors, &path)?;

        let vb = load_safetensors(path.clone(), device).run()?;
        let loaded = vb.get(3, "x")?;
        let loaded_vec: Vec<f32> = loaded.to_vec1()?;
        assert_eq!(loaded_vec, vec![1.0, 2.0, 3.0]);

        std::fs::remove_file(&path).ok();
        Ok(())
    }
}
