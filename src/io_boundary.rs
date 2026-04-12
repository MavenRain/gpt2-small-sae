//! Effectful boundary primitives wrapped in
//! [`comp_cat_rs::effect::io::Io`].
//!
//! Every side effect in this crate (network download, filesystem read, device
//! acquisition, console output) lives behind an [`Io`] program constructed
//! here.  Binary entry points compose these programs with combinators and
//! call [`Io::run`] exactly once at the top level.

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use comp_cat_rs::effect::io::Io;

use crate::config::LayerIndex;
use crate::error::{Error, TokenizerError};
use crate::gpt2::Gpt2;

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
