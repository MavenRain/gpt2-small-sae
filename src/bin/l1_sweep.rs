//! L1 coefficient sweep binary.
//!
//! Trains sparse autoencoders at a fixed GPT-2 layer with varying L1
//! sparsity coefficients, saves per-coefficient checkpoints and metrics,
//! and prints a comparison table to identify the best sparsity/fidelity
//! tradeoff.
//!
//! Activations are collected once and reused across all L1 values,
//! making repeated sweeps much cheaper than separate training runs.
//!
//! # Usage
//!
//! ```text
//! # Sweep default L1 values with a corpus:
//! cargo run --release --bin l1_sweep -- --corpus corpus.txt
//!
//! # Custom L1 values:
//! cargo run --release --bin l1_sweep -- --corpus corpus.txt \
//!   --l1-values 1e-4,5e-4,1e-3,5e-3
//!
//! # Cache activations for re-sweeping:
//! cargo run --release --bin l1_sweep -- --corpus corpus.txt --save-acts true
//!
//! # Re-sweep from cached activations:
//! cargo run --release --bin l1_sweep -- --load-acts true --l1-values 2e-4,3e-4,4e-4
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
use gpt2_small_sae::train::{AdamConfig, AdamState, LrSchedule, ResampleConfig, training_stream};

const GPT2_DEPTH: usize = 12;
const VOCAB_UPPER: f32 = 50256.0;

/// Shared hyperparameters for all L1 runs (everything except the L1
/// coefficient itself).
#[derive(Clone, Copy)]
struct L1SweepHyperParams {
    expansion: usize,
    lr_peak: f64,
    lr_min: f64,
    warmup_steps: u64,
    num_steps: u64,
    resample_interval: u64,
    log_every: u64,
}

/// Full parsed L1 sweep configuration.
#[derive(Clone)]
struct L1SweepOpts {
    layer: usize,
    l1_values: Vec<f64>,
    hyper: L1SweepHyperParams,
    batch_size: usize,
    ctx_len: usize,
    num_batches: usize,
    corpus: Option<std::path::PathBuf>,
    prefix: String,
    save_acts: bool,
    load_acts: bool,
}

fn parse_l1_sweep_opts() -> Result<L1SweepOpts, Error> {
    let args = Args::parse();
    let l1_str = args.get_or("l1-values", "1e-4,5e-4,1e-3,5e-3".to_string())?;
    let l1_values: Vec<f64> = l1_str
        .split(',')
        .map(|s| {
            s.trim().parse::<f64>().map_err(|e| Error::Config {
                reason: format!("invalid L1 coefficient in --l1-values: {e}"),
            })
        })
        .collect::<Result<_, _>>()?;
    if l1_values.is_empty() {
        Err(Error::Config {
            reason: "--l1-values must specify at least one coefficient".into(),
        })?;
    }
    let layer: usize = args.get_or("layer", 8_usize)?;
    (layer < GPT2_DEPTH)
        .then_some(())
        .ok_or_else(|| Error::Config {
            reason: format!("layer {layer} out of range for GPT-2 depth {GPT2_DEPTH}"),
        })?;
    Ok(L1SweepOpts {
        layer,
        l1_values,
        hyper: L1SweepHyperParams {
            expansion: args.get_or("expansion", 8_usize)?,
            lr_peak: args.get_or("lr-peak", 3e-4_f64)?,
            lr_min: args.get_or("lr-min", 3e-5_f64)?,
            warmup_steps: args.get_or("warmup", 500_u64)?,
            num_steps: args.get_or("steps", 5000_u64)?,
            resample_interval: args.get_or("resample", 500_u64)?,
            log_every: args.get_or("log-every", 100_u64)?,
        },
        batch_size: args.get_or("batch-size", 4_usize)?,
        ctx_len: args.get_or("ctx-len", 128_usize)?,
        num_batches: args.get_or("batches", 16_usize)?,
        corpus: args
            .get("corpus")
            .or_else(|| args.positional(0))
            .map(std::path::PathBuf::from),
        prefix: args.get_or("prefix", String::new())?,
        save_acts: args.get_or("save-acts", false)?,
        load_acts: args.get_or("load-acts", false)?,
    })
}

// -----------------------------------------------------------------------
// Main Io program
// -----------------------------------------------------------------------

fn l1_sweep_program() -> Io<Error, ()> {
    parse_l1_sweep_opts().map_or_else(
        |e| Io::suspend(move || Err(e)),
        |opts| {
            io_boundary::acquire_device().flat_map(move |device| {
                if opts.load_acts {
                    l1_sweep_cached(opts, device)
                } else {
                    l1_sweep_fresh(opts, device)
                }
            })
        },
    )
}

/// Download GPT-2, build token batches, collect activations, and
/// sweep all L1 values.
#[allow(clippy::too_many_lines)]
fn l1_sweep_fresh(opts: L1SweepOpts, device: candle_core::Device) -> Io<Error, ()> {
    let needs_tokenizer = opts.corpus.is_some();
    io_boundary::download_gpt2_weights().flat_map(move |weights| {
        let opts = opts.clone();
        let tokenizer_io: Io<Error, Option<tokenizers::Tokenizer>> = if needs_tokenizer {
            io_boundary::download_tokenizer().map(Some)
        } else {
            Io::pure(None)
        };
        tokenizer_io.flat_map(move |maybe_tokenizer| {
            let opts2 = opts.clone();
            Io::suspend(move || {
                eprintln!("=== L1 coefficient sweep (layer {}) ===", opts.layer);
                eprintln!(
                    "l1 values: {:?}, expansion: {}x, steps: {}",
                    opts.l1_values, opts.hyper.expansion, opts.hyper.num_steps,
                );
                eprintln!();

                let batches = opts.corpus.as_deref().map_or_else(
                    || {
                        eprintln!(
                            "generating {} random token batches ({} x {})...",
                            opts.num_batches, opts.batch_size, opts.ctx_len,
                        );
                        (0..opts.num_batches)
                            .map(|_| {
                                Tensor::rand(
                                    0.0f32,
                                    VOCAB_UPPER,
                                    (opts.batch_size, opts.ctx_len),
                                    &device,
                                )
                                .and_then(|t| t.to_dtype(DType::U32))
                                .map_err(Error::from)
                            })
                            .collect::<Result<Vec<_>, _>>()
                    },
                    |path| {
                        eprintln!("tokenizing corpus: {}...", path.display());
                        let tokenizer =
                            maybe_tokenizer.as_ref().ok_or_else(|| Error::Boundary {
                                reason: "tokenizer not loaded".into(),
                            })?;
                        let text = std::fs::read_to_string(path).map_err(|e| Error::Boundary {
                            reason: format!("failed to read corpus {}: {e}", path.display()),
                        })?;
                        let bs = BatchSize::new(opts.batch_size)?;
                        let cl = ContextLength::new(opts.ctx_len)?;
                        let batches = tokenize_corpus(&text, tokenizer, bs, cl, &device)?;
                        if batches.is_empty() {
                            Err(Error::Boundary {
                                reason: format!(
                                    "corpus too short ({} tokens needed)",
                                    opts.batch_size * opts.ctx_len
                                ),
                            })
                        } else {
                            Ok(batches)
                        }
                    },
                )?;

                let layer_index = LayerIndex::new(opts.layer, GPT2_DEPTH)?;
                eprintln!("loading GPT-2 small (layers 0..{})...", opts.layer);
                let gpt2 = Gpt2::from_bytes(weights, layer_index, &device)?;
                Ok((gpt2, batches, device))
            })
            .flat_map(move |(gpt2, batches, device)| {
                eprintln!("collecting activations for layer {}...", opts2.layer);
                activation_stream(gpt2, batches)
                    .collect()
                    .flat_map(move |activations| {
                        let save_io: Io<Error, ()> = if opts2.save_acts {
                            let acts_path = format!(
                                "{}activations_layer_{}.safetensors",
                                opts2.prefix, opts2.layer,
                            );
                            io_boundary::save_activations(
                                activations.clone(),
                                std::path::PathBuf::from(acts_path),
                            )
                        } else {
                            Io::pure(())
                        };
                        save_io.flat_map(move |()| l1_sweep_runs(activations, device, opts2))
                    })
            })
        })
    })
}

/// Load pre-cached activations and sweep L1 values without running
/// GPT-2 at all.
fn l1_sweep_cached(opts: L1SweepOpts, device: candle_core::Device) -> Io<Error, ()> {
    Io::suspend(move || {
        eprintln!(
            "=== L1 coefficient sweep (layer {}, cached) ===",
            opts.layer,
        );
        eprintln!(
            "l1 values: {:?}, expansion: {}x, steps: {}",
            opts.l1_values, opts.hyper.expansion, opts.hyper.num_steps,
        );
        eprintln!();
        let acts_path = format!(
            "{}activations_layer_{}.safetensors",
            opts.prefix, opts.layer,
        );
        eprintln!("loading activations from {acts_path}...");
        let tensors = candle_core::safetensors::load(std::path::Path::new(&acts_path), &device)?;
        let count = tensors.len();
        let activations: Vec<Tensor> = (0..count)
            .map(|i| {
                let key = format!("batch_{i}");
                tensors
                    .get(&key)
                    .ok_or_else(|| Error::Boundary {
                        reason: format!("missing tensor `{key}` in {acts_path}"),
                    })
                    .and_then(|t| t.to_dtype(DType::F32).map_err(Error::from))
            })
            .collect::<Result<_, _>>()?;
        eprintln!("loaded {} activation batches", activations.len());
        Ok((activations, device, opts))
    })
    .flat_map(|(activations, device, opts)| l1_sweep_runs(activations, device, opts))
}

// -----------------------------------------------------------------------
// L1 sweep fold
// -----------------------------------------------------------------------

/// Sequentially train an SAE for each L1 value and collect results.
fn l1_sweep_runs(
    activations: Vec<Tensor>,
    device: candle_core::Device,
    opts: L1SweepOpts,
) -> Io<Error, ()> {
    Io::suspend(move || {
        let model_dim = ModelDim::GPT2_SMALL;
        let sae_dim = SaeDim::from_expansion(model_dim, opts.hyper.expansion)?;
        Ok((activations, device, opts, model_dim, sae_dim))
    })
    .flat_map(|(activations, device, opts, model_dim, sae_dim)| {
        let L1SweepOpts {
            l1_values,
            hyper,
            prefix,
            layer,
            ..
        } = opts;
        l1_values
            .into_iter()
            .fold(
                Io::pure(Vec::<(f64, TrainMetrics)>::new()),
                move |acc, l1| {
                    let activations = activations.clone();
                    let device = device.clone();
                    let prefix = prefix.clone();
                    acc.flat_map(move |results| {
                        train_l1(
                            activations,
                            device,
                            model_dim,
                            sae_dim,
                            layer,
                            l1,
                            hyper,
                            prefix,
                        )
                        .map(move |metrics| {
                            results
                                .into_iter()
                                .chain(std::iter::once((l1, metrics)))
                                .collect()
                        })
                    })
                },
            )
            .flat_map(|results| {
                Io::suspend(move || {
                    print_comparison(&results);
                    Ok(())
                })
            })
    })
}

/// Initialize and train an SAE for a single L1 coefficient.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn train_l1(
    activations: Vec<Tensor>,
    device: candle_core::Device,
    model_dim: ModelDim,
    sae_dim: SaeDim,
    layer: usize,
    l1: f64,
    hyper: L1SweepHyperParams,
    prefix: String,
) -> Io<Error, TrainMetrics> {
    Io::suspend(move || {
        let l1_tag = format!("{l1:e}");
        eprintln!("\n=== l1 = {l1_tag} ===");
        let l1_coefficient = L1Coefficient::new(l1)?;
        let lr_peak = LearningRate::new(hyper.lr_peak)?;
        let lr_min = LearningRate::new(hyper.lr_min)?;
        let lr_schedule = LrSchedule::new(lr_peak, lr_min, hyper.warmup_steps, hyper.num_steps)?;
        let varmap = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let sae = Sae::new(vb, model_dim, sae_dim)?;
        let sae_for_save = sae.clone();
        let adam = AdamState::init(&varmap)?;
        let resample_config = (hyper.resample_interval > 0)
            .then(|| ResampleConfig::new(hyper.resample_interval))
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
            hyper.num_steps,
            resample_config,
        )?;
        let metrics_path = format!("{prefix}metrics_l1_{l1_tag}.jsonl");
        std::fs::write(&metrics_path, "")?;
        Ok((stream, sae_for_save, l1_tag, prefix, metrics_path))
    })
    .flat_map(
        move |(stream, sae_for_save, l1_tag, prefix, metrics_path)| {
            let l1_tag_for_save = l1_tag.clone();
            eprintln!("training l1={l1_tag}...\n");
            let log_every = hyper.log_every;
            let num_steps = hyper.num_steps;
            stream
                .fold(
                    None::<TrainMetrics>,
                    Arc::new(move |_, metrics: TrainMetrics| {
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
                                "  [l1={l1_tag}] step {:>5} | mse: {:.6} | \
                             l0: {:>7.1} | var_expl: {:>7.4} | dead: {:.4}",
                                metrics.step(),
                                metrics.mse().as_f64(),
                                metrics.l0().as_f64(),
                                metrics.variance_explained().as_f64(),
                                metrics.dead_fraction().as_f64(),
                            );
                        }
                        Some(metrics)
                    }),
                )
                .flat_map(move |final_metrics| {
                    let checkpoint_path = format!("{prefix}sae_l1_{l1_tag_for_save}.safetensors");
                    final_metrics.map_or_else(
                        || {
                            Io::suspend(|| {
                                Err(Error::Train {
                                    reason: "no training steps completed".into(),
                                })
                            })
                        },
                        |metrics| {
                            let ckpt = std::path::PathBuf::from(checkpoint_path);
                            let meta = io_boundary::CheckpointMeta::from_metrics(
                                &metrics,
                                layer,
                                model_dim.as_usize(),
                                sae_dim.as_usize(),
                                l1,
                                hyper.lr_peak,
                                hyper.lr_min,
                                hyper.warmup_steps,
                                hyper.num_steps,
                                hyper.resample_interval,
                            );
                            let meta_file = io_boundary::meta_path(&ckpt);
                            io_boundary::save_checkpoint(sae_for_save, ckpt)
                                .flat_map(move |()| {
                                    io_boundary::save_checkpoint_meta(meta, meta_file)
                                })
                                .map(move |()| metrics)
                        },
                    )
                })
        },
    )
}

// -----------------------------------------------------------------------
// Comparison table
// -----------------------------------------------------------------------

/// Print a side-by-side metrics comparison across L1 coefficients.
#[allow(clippy::cast_precision_loss)]
fn print_comparison(results: &[(f64, TrainMetrics)]) {
    eprintln!("\n=== L1 sweep comparison ===");
    eprintln!(
        "{:>10} | {:>10} | {:>8} | {:>8} | {:>8} | {:>6}",
        "l1", "mse", "l0", "var_expl", "dead", "resamp",
    );
    eprintln!("{}", "-".repeat(62));
    results.iter().fold((), |(), (l1, m)| {
        eprintln!(
            "{l1:>10.2e} | {:>10.6} | {:>8.1} | {:>8.4} | {:>8.4} | {:>6}",
            m.mse().as_f64(),
            m.l0().as_f64(),
            m.variance_explained().as_f64(),
            m.dead_fraction().as_f64(),
            m.resampled(),
        );
    });
}

fn main() {
    l1_sweep_program().run().unwrap_or_else(|e| {
        eprintln!("l1 sweep failed: {e}");
        std::process::exit(1);
    });
}
