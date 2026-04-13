//! Expansion factor sweep binary.
//!
//! Trains sparse autoencoders at a fixed GPT-2 layer with varying
//! dictionary widths (SAE expansion factors), saves per-run checkpoints
//! and metrics, and prints a comparison table to identify the best
//! capacity/sparsity tradeoff.
//!
//! Activations are collected once and reused across all expansion
//! values, making repeated sweeps much cheaper than separate training
//! runs.  Cache files use the same naming convention as
//! [`sweep`](../sweep/index.html) and [`l1_sweep`](../l1_sweep/index.html)
//! so activation files are interchangeable across all sweep binaries.
//!
//! # Usage
//!
//! ```text
//! # Sweep default expansion values with a corpus:
//! cargo run --release --bin expansion_sweep -- --corpus corpus.txt
//!
//! # Custom expansion values:
//! cargo run --release --bin expansion_sweep -- --corpus corpus.txt \
//!   --expansion-values 4,8,16,32,64
//!
//! # Cache activations for re-sweeping:
//! cargo run --release --bin expansion_sweep -- --corpus corpus.txt --save-acts true
//!
//! # Re-sweep from cached activations:
//! cargo run --release --bin expansion_sweep -- --load-acts true --expansion-values 8,16
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

/// Shared hyperparameters for all expansion runs (everything except the
/// expansion factor itself).
#[derive(Clone, Copy)]
struct ExpansionSweepHyperParams {
    l1: f64,
    lr_peak: f64,
    lr_min: f64,
    warmup_steps: u64,
    num_steps: u64,
    resample_interval: u64,
    log_every: u64,
}

/// Full parsed expansion sweep configuration.
#[derive(Clone)]
struct ExpansionSweepOpts {
    layer: usize,
    expansion_values: Vec<usize>,
    hyper: ExpansionSweepHyperParams,
    batch_size: usize,
    ctx_len: usize,
    num_batches: usize,
    corpus: Option<std::path::PathBuf>,
    prefix: String,
    save_acts: bool,
    load_acts: bool,
}

fn parse_expansion_sweep_opts() -> Result<ExpansionSweepOpts, Error> {
    let args = Args::parse();
    let exp_str = args.get_or("expansion-values", "4,8,16,32".to_string())?;
    let expansion_values: Vec<usize> = exp_str
        .split(',')
        .map(|s| {
            s.trim().parse::<usize>().map_err(|e| Error::Config {
                reason: format!("invalid expansion factor in --expansion-values: {e}"),
            })
        })
        .collect::<Result<_, _>>()?;
    if expansion_values.is_empty() {
        Err(Error::Config {
            reason: "--expansion-values must specify at least one factor".into(),
        })?;
    }
    let layer: usize = args.get_or("layer", 8_usize)?;
    (layer < GPT2_DEPTH)
        .then_some(())
        .ok_or_else(|| Error::Config {
            reason: format!("layer {layer} out of range for GPT-2 depth {GPT2_DEPTH}"),
        })?;
    Ok(ExpansionSweepOpts {
        layer,
        expansion_values,
        hyper: ExpansionSweepHyperParams {
            l1: args.get_or("l1", 5e-4_f64)?,
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

fn expansion_sweep_program() -> Io<Error, ()> {
    parse_expansion_sweep_opts().map_or_else(
        |e| Io::suspend(move || Err(e)),
        |opts| {
            io_boundary::acquire_device().flat_map(move |device| {
                if opts.load_acts {
                    expansion_sweep_cached(opts, device)
                } else {
                    expansion_sweep_fresh(opts, device)
                }
            })
        },
    )
}

/// Download GPT-2, build token batches, collect activations, and sweep
/// all expansion values.
#[allow(clippy::too_many_lines)]
fn expansion_sweep_fresh(opts: ExpansionSweepOpts, device: candle_core::Device) -> Io<Error, ()> {
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
                eprintln!("=== expansion factor sweep (layer {}) ===", opts.layer);
                eprintln!(
                    "expansion values: {:?}, l1: {}, steps: {}",
                    opts.expansion_values, opts.hyper.l1, opts.hyper.num_steps,
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
                        save_io.flat_map(move |()| expansion_sweep_runs(activations, device, opts2))
                    })
            })
        })
    })
}

/// Load pre-cached activations and sweep expansion values without
/// running GPT-2 at all.
fn expansion_sweep_cached(opts: ExpansionSweepOpts, device: candle_core::Device) -> Io<Error, ()> {
    Io::suspend(move || {
        eprintln!(
            "=== expansion factor sweep (layer {}, cached) ===",
            opts.layer,
        );
        eprintln!(
            "expansion values: {:?}, l1: {}, steps: {}",
            opts.expansion_values, opts.hyper.l1, opts.hyper.num_steps,
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
    .flat_map(|(activations, device, opts)| expansion_sweep_runs(activations, device, opts))
}

// -----------------------------------------------------------------------
// Expansion sweep fold
// -----------------------------------------------------------------------

/// Sequentially train an SAE for each expansion value and collect
/// results.
fn expansion_sweep_runs(
    activations: Vec<Tensor>,
    device: candle_core::Device,
    opts: ExpansionSweepOpts,
) -> Io<Error, ()> {
    let ExpansionSweepOpts {
        expansion_values,
        hyper,
        prefix,
        layer,
        ..
    } = opts;
    let model_dim = ModelDim::GPT2_SMALL;
    expansion_values
        .into_iter()
        .fold(
            Io::pure(Vec::<(usize, TrainMetrics)>::new()),
            move |acc, expansion| {
                let activations = activations.clone();
                let device = device.clone();
                let prefix = prefix.clone();
                acc.flat_map(move |results| {
                    train_expansion(
                        activations,
                        device,
                        model_dim,
                        layer,
                        expansion,
                        hyper,
                        prefix,
                    )
                    .map(move |metrics| {
                        results
                            .into_iter()
                            .chain(std::iter::once((expansion, metrics)))
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
}

/// Initialize and train an SAE for a single expansion factor.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn train_expansion(
    activations: Vec<Tensor>,
    device: candle_core::Device,
    model_dim: ModelDim,
    layer: usize,
    expansion: usize,
    hyper: ExpansionSweepHyperParams,
    prefix: String,
) -> Io<Error, TrainMetrics> {
    Io::suspend(move || {
        let tag = format!("{expansion}x");
        eprintln!("\n=== expansion = {tag} ===");
        let sae_dim = SaeDim::from_expansion(model_dim, expansion)?;
        let l1_coefficient = L1Coefficient::new(hyper.l1)?;
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
        let metrics_path = format!("{prefix}metrics_expansion_{tag}.jsonl");
        std::fs::write(&metrics_path, "")?;
        Ok((stream, sae_for_save, sae_dim, tag, prefix, metrics_path))
    })
    .flat_map(
        move |(stream, sae_for_save, sae_dim, tag, prefix, metrics_path)| {
            let tag_for_save = tag.clone();
            eprintln!("training expansion={tag}...\n");
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
                                "  [exp={tag}] step {:>5} | mse: {:.6} | \
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
                    let checkpoint_path =
                        format!("{prefix}sae_expansion_{tag_for_save}.safetensors");
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
                                hyper.l1,
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

/// Print a side-by-side metrics comparison across expansion factors.
#[allow(clippy::cast_precision_loss)]
fn print_comparison(results: &[(usize, TrainMetrics)]) {
    eprintln!("\n=== expansion sweep comparison ===");
    eprintln!(
        "{:>10} | {:>10} | {:>8} | {:>8} | {:>8} | {:>6}",
        "expansion", "mse", "l0", "var_expl", "dead", "resamp",
    );
    eprintln!("{}", "-".repeat(62));
    results.iter().fold((), |(), (expansion, m)| {
        let tag = format!("{expansion}x");
        eprintln!(
            "{tag:>10} | {:>10.6} | {:>8.1} | {:>8.4} | {:>8.4} | {:>6}",
            m.mse().as_f64(),
            m.l0().as_f64(),
            m.variance_explained().as_f64(),
            m.dead_fraction().as_f64(),
            m.resampled(),
        );
    });
}

fn main() {
    expansion_sweep_program().run().unwrap_or_else(|e| {
        eprintln!("expansion sweep failed: {e}");
        std::process::exit(1);
    });
}
