//! SAE training binary.
//!
//! Downloads GPT-2 small, collects residual stream activations, trains
//! a sparse autoencoder, saves a checkpoint to
//! `sae_checkpoint.safetensors`, and writes per-step metrics to
//! `metrics.jsonl`.
//!
//! # Usage
//!
//! ```text
//! # Train on a text corpus (recommended):
//! cargo run --release --bin train -- corpus.txt
//!
//! # Train on random token IDs (baseline / smoke test):
//! cargo run --release --bin train
//! ```

use std::sync::Arc;

use candle_core::{DType, Tensor};
use comp_cat_rs::effect::io::Io;

use gpt2_small_sae::activations::activation_stream;
use gpt2_small_sae::config::{
    BatchSize, ContextLength, L1Coefficient, LayerIndex, LearningRate, ModelDim, SaeDim,
};
use gpt2_small_sae::dataset::tokenize_corpus;
use gpt2_small_sae::error::Error;
use gpt2_small_sae::gpt2::Gpt2;
use gpt2_small_sae::io_boundary;
use gpt2_small_sae::metrics::TrainMetrics;
use gpt2_small_sae::sae::Sae;
use gpt2_small_sae::train::{AdamConfig, AdamState, LrSchedule, ResampleConfig, training_stream};

const LAYER: usize = 8;
const GPT2_DEPTH: usize = 12;
const EXPANSION: usize = 8;
const L1_COEFF: f64 = 5e-4;
const LR_PEAK: f64 = 3e-4;
const LR_MIN: f64 = 3e-5;
const WARMUP_STEPS: u64 = 500;
const BATCH_SIZE: usize = 4;
const CTX_LEN: usize = 128;
const NUM_RANDOM_BATCHES: usize = 16;
const NUM_STEPS: u64 = 5000;
const RESAMPLE_INTERVAL: u64 = 500;
const LOG_EVERY: u64 = 100;
const VOCAB_UPPER: f32 = 50256.0;
const METRICS_PATH: &str = "metrics.jsonl";

/// Resolve the corpus path from argv, if any.
fn corpus_path() -> Option<std::path::PathBuf> {
    std::env::args().nth(1).map(std::path::PathBuf::from)
}

// -----------------------------------------------------------------------
// Token batch construction
// -----------------------------------------------------------------------

/// Build token-ID batches from a text corpus file.
fn batches_from_corpus(
    path: &std::path::Path,
    tokenizer: &tokenizers::Tokenizer,
    device: &candle_core::Device,
) -> Result<Vec<Tensor>, Error> {
    let text = std::fs::read_to_string(path).map_err(|e| Error::Boundary {
        reason: format!("failed to read corpus {}: {e}", path.display()),
    })?;
    let batch_size = BatchSize::new(BATCH_SIZE)?;
    let ctx_len = ContextLength::new(CTX_LEN)?;
    let batches = tokenize_corpus(&text, tokenizer, batch_size, ctx_len, device)?;
    if batches.is_empty() {
        Err(Error::Boundary {
            reason: format!(
                "corpus {} is too short to fill even one batch \
                 ({BATCH_SIZE} x {CTX_LEN} = {} tokens needed)",
                path.display(),
                BATCH_SIZE * CTX_LEN
            ),
        })
    } else {
        Ok(batches)
    }
}

/// Generate random token-ID batches as a baseline fallback.
fn batches_random(device: &candle_core::Device) -> Result<Vec<Tensor>, Error> {
    (0..NUM_RANDOM_BATCHES)
        .map(|_| {
            Tensor::rand(0.0f32, VOCAB_UPPER, (BATCH_SIZE, CTX_LEN), device)
                .and_then(|t| t.to_dtype(DType::U32))
                .map_err(Error::from)
        })
        .collect()
}

// -----------------------------------------------------------------------
// Main Io program
// -----------------------------------------------------------------------

fn train_program() -> Io<Error, ()> {
    let maybe_corpus = corpus_path();
    io_boundary::acquire_device().flat_map(move |device| {
        io_boundary::download_gpt2_weights().flat_map(move |weights| {
            let maybe_corpus = maybe_corpus.clone();
            let needs_tokenizer = maybe_corpus.is_some();
            let tokenizer_io: Io<Error, Option<tokenizers::Tokenizer>> = if needs_tokenizer {
                io_boundary::download_tokenizer().map(Some)
            } else {
                Io::pure(None)
            };
            tokenizer_io.flat_map(move |maybe_tokenizer| {
                Io::suspend(move || {
                    let layer_index = LayerIndex::new(LAYER, GPT2_DEPTH)?;
                    let model_dim = ModelDim::GPT2_SMALL;
                    let sae_dim = SaeDim::from_expansion(model_dim, EXPANSION)?;

                    let batches = maybe_corpus.as_deref().map_or_else(
                        || {
                            eprintln!("=== SAE training (random tokens) ===");
                            eprintln!(
                                "  hint: pass a text file as the first argument \
                                 for real-corpus training"
                            );
                            batches_random(&device)
                        },
                        |path| {
                            eprintln!("=== SAE training (corpus: {}) ===", path.display());
                            let tokenizer =
                                maybe_tokenizer.as_ref().ok_or_else(|| Error::Boundary {
                                    reason: "tokenizer not loaded".into(),
                                })?;
                            batches_from_corpus(path, tokenizer, &device)
                        },
                    )?;

                    eprintln!(
                        "layer: {LAYER}, expansion: {EXPANSION}x, \
                         l1: {L1_COEFF}, lr_peak: {LR_PEAK}, lr_min: {LR_MIN}"
                    );
                    eprintln!(
                        "warmup: {WARMUP_STEPS}, resample: every {RESAMPLE_INTERVAL}, \
                         batches: {} x {BATCH_SIZE} x {CTX_LEN}, steps: {NUM_STEPS}",
                        batches.len()
                    );
                    eprintln!();

                    eprintln!("loading GPT-2 small (layers 0..{LAYER})...");
                    let gpt2 = Gpt2::from_bytes(weights, layer_index, &device)?;

                    Ok((gpt2, batches, device, model_dim, sae_dim))
                })
                .flat_map(|(gpt2, batches, device, model_dim, sae_dim)| {
                    eprintln!("collecting activations through GPT-2...");
                    activation_stream(gpt2, batches)
                        .collect()
                        .flat_map(move |activations| {
                            eprintln!("collected {} activation batches", activations.len());
                            init_and_train(activations, device, model_dim, sae_dim)
                        })
                })
            })
        })
    })
}

fn init_and_train(
    activations: Vec<Tensor>,
    device: candle_core::Device,
    model_dim: ModelDim,
    sae_dim: SaeDim,
) -> Io<Error, ()> {
    Io::suspend(move || {
        let l1_coefficient = L1Coefficient::new(L1_COEFF)?;
        let lr_peak = LearningRate::new(LR_PEAK)?;
        let lr_min = LearningRate::new(LR_MIN)?;
        let lr_schedule = LrSchedule::new(lr_peak, lr_min, WARMUP_STEPS, NUM_STEPS)?;
        let varmap = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let sae = Sae::new(vb, model_dim, sae_dim)?;
        let sae_for_save = sae.clone();
        let adam = AdamState::init(&varmap)?;
        let resample_config = Some(ResampleConfig::new(RESAMPLE_INTERVAL)?);
        let stream = training_stream(
            sae,
            adam,
            activations,
            &device,
            l1_coefficient,
            lr_schedule,
            AdamConfig::standard(),
            model_dim,
            sae_dim,
            NUM_STEPS,
            resample_config,
        )?;
        // Truncate any prior metrics file.
        std::fs::write(METRICS_PATH, "")?;
        Ok((stream, sae_for_save))
    })
    .flat_map(|(stream, sae_for_save)| {
        eprintln!("training...\n");
        stream
            .fold(
                (),
                Arc::new(|(), metrics: TrainMetrics| {
                    // Append every step to the JSONL log (best-effort).
                    if let Ok(line) = metrics.to_jsonl() {
                        let _ = std::fs::OpenOptions::new()
                            .create(true)
                            .append(true)
                            .open(METRICS_PATH)
                            .and_then(|mut f| {
                                use std::io::Write;
                                f.write_all(line.as_bytes())
                            });
                    }
                    if metrics.step() % LOG_EVERY == 0 || metrics.step() == NUM_STEPS - 1 {
                        eprintln!(
                            "step {:>5} | mse: {:.6} | l0: {:>7.1} | \
                             var_expl: {:>7.4} | dead: {:.4} | lr: {:.2e} | \
                             resamp: {}",
                            metrics.step(),
                            metrics.mse().as_f64(),
                            metrics.l0().as_f64(),
                            metrics.variance_explained().as_f64(),
                            metrics.dead_fraction().as_f64(),
                            metrics.learning_rate(),
                            metrics.resampled(),
                        );
                    }
                }),
            )
            .flat_map(move |()| {
                io_boundary::save_checkpoint(
                    sae_for_save,
                    std::path::PathBuf::from("sae_checkpoint.safetensors"),
                )
            })
    })
}

fn main() {
    train_program().run().unwrap_or_else(|e| {
        eprintln!("training failed: {e}");
        std::process::exit(1);
    });
}
