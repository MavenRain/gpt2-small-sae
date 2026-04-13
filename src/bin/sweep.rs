//! Multi-layer SAE training sweep.
//!
//! Trains a sparse autoencoder at each specified GPT-2 hookpoint layer,
//! saves per-layer checkpoints and metrics, and prints a comparison
//! table of final training metrics across layers.
//!
//! # Usage
//!
//! ```text
//! # Sweep default layers (0, 4, 8, 11) with a corpus:
//! cargo run --release --bin sweep -- --corpus corpus.txt
//!
//! # Sweep specific layers with custom hyperparameters:
//! cargo run --release --bin sweep -- --layers 2,6,10 --expansion 16 --steps 3000
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

/// Shared hyperparameters used for every layer in the sweep.
#[derive(Clone, Copy)]
struct SweepHyperParams {
    expansion: usize,
    l1_coeff: f64,
    lr_peak: f64,
    lr_min: f64,
    warmup_steps: u64,
    num_steps: u64,
    resample_interval: u64,
    log_every: u64,
}

/// Full parsed sweep configuration.
#[derive(Clone)]
struct SweepOpts {
    layers: Vec<usize>,
    hyper: SweepHyperParams,
    batch_size: usize,
    ctx_len: usize,
    num_batches: usize,
    corpus: Option<std::path::PathBuf>,
    prefix: String,
}

fn parse_sweep_opts() -> Result<SweepOpts, Error> {
    let args = Args::parse();
    let layers_str = args.get_or("layers", "0,4,8,11".to_string())?;
    let layers: Vec<usize> = layers_str
        .split(',')
        .map(|s| {
            s.trim().parse::<usize>().map_err(|e| Error::Config {
                reason: format!("invalid layer in --layers: {e}"),
            })
        })
        .collect::<Result<_, _>>()?;
    if layers.is_empty() {
        Err(Error::Config {
            reason: "--layers must specify at least one layer".into(),
        })?;
    }
    // Validate all layers are within GPT-2 depth.
    layers.iter().try_for_each(|&l| {
        (l < GPT2_DEPTH).then_some(()).ok_or_else(|| Error::Config {
            reason: format!("layer {l} out of range for GPT-2 depth {GPT2_DEPTH}"),
        })
    })?;
    Ok(SweepOpts {
        layers,
        hyper: SweepHyperParams {
            expansion: args.get_or("expansion", 8_usize)?,
            l1_coeff: args.get_or("l1", 5e-4_f64)?,
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
    })
}

// -----------------------------------------------------------------------
// Main Io program
// -----------------------------------------------------------------------

#[allow(clippy::too_many_lines)]
fn sweep_program() -> Io<Error, ()> {
    parse_sweep_opts().map_or_else(
        |e| Io::suspend(move || Err(e)),
        |opts| {
            let needs_tokenizer = opts.corpus.is_some();
            io_boundary::acquire_device().flat_map(move |device| {
                let opts = opts.clone();
                io_boundary::download_gpt2_weights().flat_map(move |weights| {
                    let opts = opts.clone();
                    let tokenizer_io: Io<Error, Option<tokenizers::Tokenizer>> = if needs_tokenizer
                    {
                        io_boundary::download_tokenizer().map(Some)
                    } else {
                        Io::pure(None)
                    };
                    tokenizer_io.flat_map(move |maybe_tokenizer| {
                        let opts2 = opts.clone();
                        Io::suspend(move || {
                            eprintln!("=== multi-layer SAE sweep ===");
                            eprintln!(
                                "layers: {:?}, expansion: {}x, steps: {}",
                                opts.layers, opts.hyper.expansion, opts.hyper.num_steps,
                            );
                            eprintln!();

                            let batches = opts.corpus.as_deref().map_or_else(
                                || {
                                    eprintln!(
                                        "generating {} random token batches \
                                         ({} x {})...",
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
                                    let tokenizer = maybe_tokenizer.as_ref().ok_or_else(|| {
                                        Error::Boundary {
                                            reason: "tokenizer not loaded".into(),
                                        }
                                    })?;
                                    let text = std::fs::read_to_string(path).map_err(|e| {
                                        Error::Boundary {
                                            reason: format!(
                                                "failed to read corpus {}: {e}",
                                                path.display()
                                            ),
                                        }
                                    })?;
                                    let bs = BatchSize::new(opts.batch_size)?;
                                    let cl = ContextLength::new(opts.ctx_len)?;
                                    let batches =
                                        tokenize_corpus(&text, tokenizer, bs, cl, &device)?;
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

                            Ok((Arc::new(weights), batches, device))
                        })
                        .flat_map(move |(weights, batches, device)| {
                            sweep_layers(opts2, weights, batches, device)
                        })
                    })
                })
            })
        },
    )
}

// -----------------------------------------------------------------------
// Layer sweep fold
// -----------------------------------------------------------------------

/// Sequentially train an SAE at each layer and collect results.
fn sweep_layers(
    opts: SweepOpts,
    weights: Arc<Vec<u8>>,
    batches: Vec<Tensor>,
    device: candle_core::Device,
) -> Io<Error, ()> {
    let SweepOpts {
        layers,
        hyper,
        prefix,
        ..
    } = opts;
    layers
        .into_iter()
        .fold(
            Io::pure(Vec::<(usize, TrainMetrics)>::new()),
            move |acc, layer| {
                let weights = Arc::clone(&weights);
                let batches = batches.clone();
                let device = device.clone();
                let prefix = prefix.clone();
                acc.flat_map(move |results| {
                    sweep_one_layer(layer, weights, batches, device, hyper, prefix).map(
                        move |metrics| {
                            results
                                .into_iter()
                                .chain(std::iter::once((layer, metrics)))
                                .collect()
                        },
                    )
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

/// Train a single SAE at the given layer and return final metrics.
fn sweep_one_layer(
    layer: usize,
    weights: Arc<Vec<u8>>,
    batches: Vec<Tensor>,
    device: candle_core::Device,
    hyper: SweepHyperParams,
    prefix: String,
) -> Io<Error, TrainMetrics> {
    Io::suspend(move || {
        eprintln!("\n=== layer {layer} ===");
        let layer_index = LayerIndex::new(layer, GPT2_DEPTH)?;
        let model_dim = ModelDim::GPT2_SMALL;
        let sae_dim = SaeDim::from_expansion(model_dim, hyper.expansion)?;
        eprintln!("loading GPT-2 small (layers 0..{layer})...");
        let gpt2 = Gpt2::from_bytes((*weights).clone(), layer_index, &device)?;
        Ok((gpt2, batches, device, model_dim, sae_dim, layer, prefix))
    })
    .flat_map(
        move |(gpt2, batches, device, model_dim, sae_dim, layer, prefix)| {
            eprintln!("collecting activations for layer {layer}...");
            activation_stream(gpt2, batches)
                .collect()
                .flat_map(move |activations| {
                    train_layer(
                        activations,
                        device,
                        model_dim,
                        sae_dim,
                        layer,
                        hyper,
                        prefix,
                    )
                })
        },
    )
}

/// Initialize and train an SAE for a single layer.
#[allow(clippy::too_many_arguments)]
fn train_layer(
    activations: Vec<Tensor>,
    device: candle_core::Device,
    model_dim: ModelDim,
    sae_dim: SaeDim,
    layer: usize,
    hyper: SweepHyperParams,
    prefix: String,
) -> Io<Error, TrainMetrics> {
    Io::suspend(move || {
        let l1_coefficient = L1Coefficient::new(hyper.l1_coeff)?;
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
        let metrics_path = format!("{prefix}metrics_layer_{layer}.jsonl");
        std::fs::write(&metrics_path, "")?;
        Ok((stream, sae_for_save, layer, prefix, metrics_path))
    })
    .flat_map(move |(stream, sae_for_save, layer, prefix, metrics_path)| {
        eprintln!("training layer {layer}...\n");
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
                            "  [L{layer:>2}] step {:>5} | mse: {:.6} | \
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
                let checkpoint_path = format!("{prefix}sae_layer_{layer}.safetensors");
                final_metrics.map_or_else(
                    || {
                        Io::suspend(|| {
                            Err(Error::Train {
                                reason: "no training steps completed".into(),
                            })
                        })
                    },
                    |metrics| {
                        io_boundary::save_checkpoint(
                            sae_for_save,
                            std::path::PathBuf::from(checkpoint_path),
                        )
                        .map(move |()| metrics)
                    },
                )
            })
    })
}

// -----------------------------------------------------------------------
// Comparison table
// -----------------------------------------------------------------------

/// Print a side-by-side metrics comparison across layers.
fn print_comparison(results: &[(usize, TrainMetrics)]) {
    eprintln!("\n=== sweep comparison ===");
    eprintln!(
        "{:>5} | {:>10} | {:>8} | {:>8} | {:>8} | {:>6}",
        "layer", "mse", "l0", "var_expl", "dead", "resamp",
    );
    eprintln!("{}", "-".repeat(58));
    results.iter().fold((), |(), (layer, m)| {
        eprintln!(
            "{layer:>5} | {:>10.6} | {:>8.1} | {:>8.4} | {:>8.4} | {:>6}",
            m.mse().as_f64(),
            m.l0().as_f64(),
            m.variance_explained().as_f64(),
            m.dead_fraction().as_f64(),
            m.resampled(),
        );
    });
}

fn main() {
    sweep_program().run().unwrap_or_else(|e| {
        eprintln!("sweep failed: {e}");
        std::process::exit(1);
    });
}
