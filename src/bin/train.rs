//! SAE training binary.
//!
//! Downloads GPT-2 small, collects residual stream activations, trains
//! a sparse autoencoder, saves a checkpoint to a safetensors file, and
//! writes per-step metrics to a JSONL file.
//!
//! # Usage
//!
//! ```text
//! # Train on a text corpus (recommended):
//! cargo run --release --bin train -- --corpus corpus.txt
//!
//! # Train with custom hyperparameters:
//! cargo run --release --bin train -- --corpus corpus.txt --layer 4 --expansion 16
//!
//! # Train on random token IDs (baseline / smoke test):
//! cargo run --release --bin train
//! ```

use std::sync::Arc;

use candle_core::{DType, Tensor};
use comp_cat_rs::effect::io::Io;

use gpt2_small_sae::activations::activation_stream;
use gpt2_small_sae::cli::Args;
use gpt2_small_sae::config::{
    BatchSize, ContextLength, L1Coefficient, LayerIndex, LearningRate, ModelDim, SaeDim,
};
use gpt2_small_sae::dataset::tokenize_corpus;
use gpt2_small_sae::error::Error;
use gpt2_small_sae::gpt2::Gpt2;
use gpt2_small_sae::io_boundary;
use gpt2_small_sae::metrics::TrainMetrics;
use gpt2_small_sae::sae::Sae;
use gpt2_small_sae::train::{
    AdamConfig, AdamState, LrSchedule, ResampleConfig, resume_from_checkpoint, training_stream,
};

const GPT2_DEPTH: usize = 12;
const VOCAB_UPPER: f32 = 50256.0;

/// All CLI-configurable training options.
#[derive(Clone)]
struct TrainOpts {
    layer: usize,
    expansion: usize,
    l1_coeff: f64,
    lr_peak: f64,
    lr_min: f64,
    warmup_steps: u64,
    batch_size: usize,
    ctx_len: usize,
    num_batches: usize,
    num_steps: u64,
    resample_interval: u64,
    log_every: u64,
    corpus: Option<std::path::PathBuf>,
    checkpoint: String,
    metrics_path: String,
    resume: Option<String>,
    save_acts: Option<String>,
    load_acts: Option<String>,
}

fn parse_train_opts() -> Result<TrainOpts, Error> {
    let args = Args::parse();
    Ok(TrainOpts {
        layer: args.get_or("layer", 8_usize)?,
        expansion: args.get_or("expansion", 8_usize)?,
        l1_coeff: args.get_or("l1", 5e-4_f64)?,
        lr_peak: args.get_or("lr-peak", 3e-4_f64)?,
        lr_min: args.get_or("lr-min", 3e-5_f64)?,
        warmup_steps: args.get_or("warmup", 500_u64)?,
        batch_size: args.get_or("batch-size", 4_usize)?,
        ctx_len: args.get_or("ctx-len", 128_usize)?,
        num_batches: args.get_or("batches", 16_usize)?,
        num_steps: args.get_or("steps", 5000_u64)?,
        resample_interval: args.get_or("resample", 500_u64)?,
        log_every: args.get_or("log-every", 100_u64)?,
        corpus: args
            .get("corpus")
            .or_else(|| args.positional(0))
            .map(std::path::PathBuf::from),
        checkpoint: args.get_or("checkpoint", "sae_checkpoint.safetensors".to_string())?,
        metrics_path: args.get_or("metrics", "metrics.jsonl".to_string())?,
        resume: args.get("resume").map(String::from),
        save_acts: args.get("save-acts").map(String::from),
        load_acts: args.get("load-acts").map(String::from),
    })
}

// -----------------------------------------------------------------------
// Token batch construction
// -----------------------------------------------------------------------

/// Build token-ID batches from a text corpus file.
fn batches_from_corpus(
    path: &std::path::Path,
    tokenizer: &tokenizers::Tokenizer,
    batch_size: usize,
    ctx_len: usize,
    device: &candle_core::Device,
) -> Result<Vec<Tensor>, Error> {
    let text = std::fs::read_to_string(path).map_err(|e| Error::Boundary {
        reason: format!("failed to read corpus {}: {e}", path.display()),
    })?;
    let bs = BatchSize::new(batch_size)?;
    let cl = ContextLength::new(ctx_len)?;
    let batches = tokenize_corpus(&text, tokenizer, bs, cl, device)?;
    if batches.is_empty() {
        Err(Error::Boundary {
            reason: format!(
                "corpus {} is too short to fill even one batch \
                 ({batch_size} x {ctx_len} = {} tokens needed)",
                path.display(),
                batch_size * ctx_len
            ),
        })
    } else {
        Ok(batches)
    }
}

/// Generate random token-ID batches as a baseline fallback.
fn batches_random(
    num_batches: usize,
    batch_size: usize,
    ctx_len: usize,
    device: &candle_core::Device,
) -> Result<Vec<Tensor>, Error> {
    (0..num_batches)
        .map(|_| {
            Tensor::rand(0.0f32, VOCAB_UPPER, (batch_size, ctx_len), device)
                .and_then(|t| t.to_dtype(DType::U32))
                .map_err(Error::from)
        })
        .collect()
}

// -----------------------------------------------------------------------
// Activation production (fresh or cached)
// -----------------------------------------------------------------------

/// Load pre-cached activations from a safetensors file, skipping
/// GPT-2 entirely.
fn load_cached_activations(
    path: String,
    expansion: usize,
    device: candle_core::Device,
) -> Io<Error, (Vec<Tensor>, candle_core::Device, ModelDim, SaeDim)> {
    Io::suspend(move || {
        let model_dim = ModelDim::GPT2_SMALL;
        let sae_dim = SaeDim::from_expansion(model_dim, expansion)?;
        eprintln!("=== SAE training (cached activations) ===");
        eprintln!("loading activations from {path}...");
        let tensors = candle_core::safetensors::load(std::path::Path::new(&path), &device)?;
        let count = tensors.len();
        let activations: Vec<Tensor> = (0..count)
            .map(|i| {
                let key = format!("batch_{i}");
                tensors
                    .get(&key)
                    .ok_or_else(|| Error::Boundary {
                        reason: format!("missing tensor `{key}` in {path}"),
                    })
                    .and_then(|t| t.to_dtype(DType::F32).map_err(Error::from))
            })
            .collect::<Result<_, _>>()?;
        eprintln!("loaded {} activation batches", activations.len());
        eprintln!();
        Ok((activations, device, model_dim, sae_dim))
    })
}

/// Download GPT-2, tokenize input, run forward pass, and optionally
/// save the collected activations to disk.
#[allow(clippy::too_many_lines)]
fn collect_fresh_activations(
    opts: TrainOpts,
    device: candle_core::Device,
) -> Io<Error, (Vec<Tensor>, candle_core::Device, ModelDim, SaeDim)> {
    let needs_tokenizer = opts.corpus.is_some();
    io_boundary::download_gpt2_weights().flat_map(move |weights| {
        let tokenizer_io: Io<Error, Option<tokenizers::Tokenizer>> = if needs_tokenizer {
            io_boundary::download_tokenizer().map(Some)
        } else {
            Io::pure(None)
        };
        tokenizer_io.flat_map(move |maybe_tokenizer| {
            let save_acts = opts.save_acts.clone();
            Io::suspend(move || {
                let layer_index = LayerIndex::new(opts.layer, GPT2_DEPTH)?;
                let model_dim = ModelDim::GPT2_SMALL;
                let sae_dim = SaeDim::from_expansion(model_dim, opts.expansion)?;

                let batches = opts.corpus.as_deref().map_or_else(
                    || {
                        eprintln!("=== SAE training (random tokens) ===");
                        eprintln!("  hint: pass --corpus <file> for real-corpus training");
                        batches_random(opts.num_batches, opts.batch_size, opts.ctx_len, &device)
                    },
                    |path| {
                        eprintln!("=== SAE training (corpus: {}) ===", path.display());
                        let tokenizer =
                            maybe_tokenizer.as_ref().ok_or_else(|| Error::Boundary {
                                reason: "tokenizer not loaded".into(),
                            })?;
                        batches_from_corpus(path, tokenizer, opts.batch_size, opts.ctx_len, &device)
                    },
                )?;

                eprintln!(
                    "layer: {}, expansion: {}x, \
                     l1: {}, lr_peak: {}, lr_min: {}",
                    opts.layer, opts.expansion, opts.l1_coeff, opts.lr_peak, opts.lr_min,
                );
                eprintln!(
                    "warmup: {}, resample: every {}, \
                     batches: {} x {} x {}, steps: {}",
                    opts.warmup_steps,
                    opts.resample_interval,
                    batches.len(),
                    opts.batch_size,
                    opts.ctx_len,
                    opts.num_steps,
                );
                eprintln!();

                eprintln!("loading GPT-2 small (layers 0..{})...", opts.layer);
                let gpt2 = Gpt2::from_bytes(weights, layer_index, &device)?;

                Ok((gpt2, batches, device, model_dim, sae_dim))
            })
            .flat_map(move |(gpt2, batches, device, model_dim, sae_dim)| {
                eprintln!("collecting activations through GPT-2...");
                activation_stream(gpt2, batches)
                    .collect()
                    .flat_map(move |activations| {
                        eprintln!("collected {} activation batches", activations.len());
                        let save_io: Io<Error, ()> = save_acts.map_or_else(
                            || Io::pure(()),
                            |path| {
                                io_boundary::save_activations(
                                    activations.clone(),
                                    std::path::PathBuf::from(path),
                                )
                            },
                        );
                        save_io.map(move |()| (activations, device, model_dim, sae_dim))
                    })
            })
        })
    })
}

/// Branch on `--load-acts` to either load cached activations or run
/// the full GPT-2 pipeline.
fn produce_activations(
    opts: TrainOpts,
    device: candle_core::Device,
) -> Io<Error, (Vec<Tensor>, candle_core::Device, ModelDim, SaeDim)> {
    let expansion = opts.expansion;
    let device_for_cache = device.clone();
    opts.load_acts.clone().map_or_else(
        || collect_fresh_activations(opts, device),
        |path| load_cached_activations(path, expansion, device_for_cache),
    )
}

// -----------------------------------------------------------------------
// Main Io program
// -----------------------------------------------------------------------

fn train_program() -> Io<Error, ()> {
    parse_train_opts().map_or_else(
        |e| Io::suspend(move || Err(e)),
        |opts| {
            let opts_for_train = opts.clone();
            io_boundary::acquire_device().flat_map(move |device| {
                produce_activations(opts, device).flat_map(
                    move |(activations, device, model_dim, sae_dim)| {
                        init_and_train(activations, device, model_dim, sae_dim, opts_for_train)
                    },
                )
            })
        },
    )
}

#[allow(clippy::too_many_lines)]
fn init_and_train(
    activations: Vec<Tensor>,
    device: candle_core::Device,
    model_dim: ModelDim,
    sae_dim: SaeDim,
    opts: TrainOpts,
) -> Io<Error, ()> {
    let checkpoint_path = opts.checkpoint.clone();
    Io::suspend(move || {
        let l1_coefficient = L1Coefficient::new(opts.l1_coeff)?;
        let lr_peak = LearningRate::new(opts.lr_peak)?;
        let lr_min = LearningRate::new(opts.lr_min)?;
        let lr_schedule = LrSchedule::new(lr_peak, lr_min, opts.warmup_steps, opts.num_steps)?;
        let varmap = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let sae = Sae::new(vb, model_dim, sae_dim)?;
        opts.resume.as_deref().map_or(Ok::<_, Error>(()), |path| {
            resume_from_checkpoint(&varmap, std::path::Path::new(path), &device)?;
            eprintln!("resumed from checkpoint: {path}");
            Ok(())
        })?;
        let sae_for_save = sae.clone();
        let adam = AdamState::init(&varmap)?;
        let resample_config = (opts.resample_interval > 0)
            .then(|| ResampleConfig::new(opts.resample_interval))
            .transpose()?;
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
            opts.num_steps,
            resample_config,
        )?;
        // Truncate any prior metrics file.
        std::fs::write(&opts.metrics_path, "")?;
        Ok((stream, sae_for_save, opts))
    })
    .flat_map(move |(stream, sae_for_save, opts)| {
        eprintln!("training...\n");
        let metrics_path = opts.metrics_path.clone();
        let log_every = opts.log_every;
        let num_steps = opts.num_steps;
        stream
            .fold(
                None::<TrainMetrics>,
                Arc::new(move |_, metrics: TrainMetrics| {
                    // Append every step to the JSONL log (best-effort).
                    if let Ok(line) = metrics.to_jsonl() {
                        let _ = std::fs::OpenOptions::new()
                            .create(true)
                            .append(true)
                            .open(&metrics_path)
                            .and_then(|mut f| {
                                use std::io::Write;
                                f.write_all(line.as_bytes())
                            });
                    }
                    if metrics.step() % log_every == 0 || metrics.step() == num_steps - 1 {
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
                    Some(metrics)
                }),
            )
            .flat_map(move |final_metrics| {
                let ckpt = std::path::PathBuf::from(checkpoint_path);
                final_metrics.map_or_else(
                    || {
                        Io::suspend(|| {
                            Err(Error::Train {
                                reason: "no training steps completed".into(),
                            })
                        })
                    },
                    |metrics| {
                        let meta = io_boundary::CheckpointMeta::from_metrics(
                            &metrics,
                            opts.layer,
                            model_dim.as_usize(),
                            sae_dim.as_usize(),
                            opts.l1_coeff,
                            opts.lr_peak,
                            opts.lr_min,
                            opts.warmup_steps,
                            opts.num_steps,
                            opts.resample_interval,
                        );
                        let meta_file = io_boundary::meta_path(&ckpt);
                        io_boundary::save_checkpoint(sae_for_save, ckpt)
                            .flat_map(move |()| io_boundary::save_checkpoint_meta(meta, meta_file))
                    },
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
