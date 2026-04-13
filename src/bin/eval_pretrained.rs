//! Evaluate a pretrained SAE checkpoint against fresh GPT-2 residual
//! stream activations.
//!
//! Loads an SAE from a safetensors file, runs GPT-2 through the same
//! hookpoint used during training, and reports aggregate metrics (MSE,
//! L0, variance explained, dead fraction) over the evaluation window.
//!
//! # Usage
//!
//! ```text
//! # Evaluate with random tokens (default):
//! cargo run --release --bin eval_pretrained -- --checkpoint sae.safetensors
//!
//! # Evaluate against a real corpus:
//! cargo run --release --bin eval_pretrained -- --checkpoint sae.safetensors --corpus corpus.txt
//!
//! # Write results to JSON:
//! cargo run --release --bin eval_pretrained -- --checkpoint sae.safetensors --output eval.json
//! ```

use candle_core::{DType, Tensor};
use comp_cat_rs::effect::io::Io;

use gpt2_small_sae::activations::activation_stream;
use gpt2_small_sae::cli::Args;
use gpt2_small_sae::config::{BatchSize, ContextLength, LayerIndex, ModelDim, SaeDim};
use gpt2_small_sae::dataset::tokenize_corpus;
use gpt2_small_sae::error::Error;
use gpt2_small_sae::gpt2::Gpt2;
use gpt2_small_sae::io_boundary;
use gpt2_small_sae::metrics::{DeadFraction, L0, Mse, VarianceExplained};
use gpt2_small_sae::sae::Sae;

const GPT2_DEPTH: usize = 12;
const VOCAB_UPPER: f32 = 50256.0;

/// CLI-configurable evaluation options.
#[derive(Clone)]
struct EvalOpts {
    layer: usize,
    expansion: usize,
    batch_size: usize,
    ctx_len: usize,
    num_batches: usize,
    corpus: Option<std::path::PathBuf>,
    checkpoint: String,
    output: Option<String>,
}

fn parse_eval_opts() -> Result<EvalOpts, Error> {
    let args = Args::parse();
    Ok(EvalOpts {
        layer: args.get_or("layer", 8_usize)?,
        expansion: args.get_or("expansion", 8_usize)?,
        batch_size: args.get_or("batch-size", 4_usize)?,
        ctx_len: args.get_or("ctx-len", 128_usize)?,
        num_batches: args.get_or("batches", 8_usize)?,
        corpus: args.get("corpus").map(std::path::PathBuf::from),
        checkpoint: args
            .get("checkpoint")
            .or_else(|| args.positional(0))
            .map_or_else(|| "sae_checkpoint.safetensors".to_string(), String::from),
        output: args.get("output").map(String::from),
    })
}

/// Running accumulators for evaluation metrics.
struct EvalAccum {
    l0_sum: f64,
    mse_sum: f64,
    var_expl_sum: f64,
    feature_counts: Tensor,
    count: u64,
}

#[allow(clippy::too_many_lines)]
fn eval_program() -> Io<Error, ()> {
    parse_eval_opts().map_or_else(
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
                            let layer_index = LayerIndex::new(opts.layer, GPT2_DEPTH)?;
                            let model_dim = ModelDim::GPT2_SMALL;
                            let sae_dim = SaeDim::from_expansion(model_dim, opts.expansion)?;

                            eprintln!("=== SAE evaluation ===");
                            eprintln!("checkpoint: {}", opts.checkpoint);
                            eprintln!("layer: {}, expansion: {}x", opts.layer, opts.expansion);

                            let batches = opts.corpus.as_deref().map_or_else(
                                || {
                                    eprintln!(
                                        "eval batches: {} x {} x {} (random tokens)",
                                        opts.num_batches, opts.batch_size, opts.ctx_len
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
                                    eprintln!("corpus: {}", path.display());
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
                                        eprintln!(
                                            "eval batches: {} x {} x {}",
                                            batches.len(),
                                            opts.batch_size,
                                            opts.ctx_len,
                                        );
                                        Ok(batches)
                                    }
                                },
                            )?;
                            eprintln!();

                            eprintln!("loading SAE from {}...", opts.checkpoint);
                            let sae = Sae::from_safetensors(
                                std::path::Path::new(&opts.checkpoint),
                                model_dim,
                                sae_dim,
                                &device,
                            )?;

                            eprintln!("loading GPT-2 small (layers 0..{})...", opts.layer);
                            let gpt2 = Gpt2::from_bytes(weights, layer_index, &device)?;

                            let feature_counts =
                                Tensor::zeros((sae_dim.as_usize(),), DType::F32, &device)?;

                            Ok((sae, gpt2, batches, sae_dim, feature_counts))
                        })
                        .flat_map(
                            move |(sae, gpt2, batches, sae_dim, feature_counts)| {
                                eprintln!("collecting activations...");
                                activation_stream(gpt2, batches).collect().flat_map(
                                    move |activations| {
                                        Io::suspend(move || {
                                            evaluate(
                                                &sae,
                                                &activations,
                                                feature_counts,
                                                sae_dim,
                                                &opts2,
                                            )
                                        })
                                    },
                                )
                            },
                        )
                    })
                })
            })
        },
    )
}

#[allow(clippy::cast_precision_loss)]
fn evaluate(
    sae: &Sae,
    activations: &[Tensor],
    initial_counts: Tensor,
    sae_dim: SaeDim,
    opts: &EvalOpts,
) -> Result<(), Error> {
    let accum = activations.iter().try_fold(
        EvalAccum {
            l0_sum: 0.0,
            mse_sum: 0.0,
            var_expl_sum: 0.0,
            feature_counts: initial_counts,
            count: 0,
        },
        |acc, batch| {
            let fwd = sae.forward(batch)?;
            let l0 = L0::compute(fwd.features())?;
            let mse = Mse::compute(fwd.reconstruction(), batch)?;
            let var_expl = VarianceExplained::compute(fwd.reconstruction(), batch)?;
            let active = fwd
                .features()
                .gt(0.0_f32)?
                .to_dtype(DType::F32)?
                .transpose(0, 1)?
                .sum(candle_core::D::Minus1)?;
            let feature_counts = acc.feature_counts.add(&active)?;
            Ok::<_, Error>(EvalAccum {
                l0_sum: acc.l0_sum + l0.as_f64(),
                mse_sum: acc.mse_sum + mse.as_f64(),
                var_expl_sum: acc.var_expl_sum + var_expl.as_f64(),
                feature_counts,
                count: acc.count + 1,
            })
        },
    )?;

    let n = accum.count.max(1) as f64;
    let dead = DeadFraction::compute(&accum.feature_counts)?;
    let mse_avg = accum.mse_sum / n;
    let l0_avg = accum.l0_sum / n;
    let var_expl_avg = accum.var_expl_sum / n;
    let dead_frac = dead.as_f64();

    eprintln!();
    eprintln!("=== evaluation results ({} batches) ===", accum.count);
    eprintln!("  mse:           {mse_avg:.6}");
    eprintln!("  l0:            {l0_avg:.1}");
    eprintln!("  var_explained: {var_expl_avg:.4}");
    eprintln!("  dead_fraction: {dead_frac:.4}");
    eprintln!(
        "  sae_dim:       {} ({}x)",
        sae_dim.as_usize(),
        sae_dim.as_usize() / sae.model_dim().as_usize()
    );

    // Optionally write results as JSON.
    opts.output.as_ref().map_or(Ok(()), |path| {
        let record = serde_json::json!({
            "checkpoint": opts.checkpoint,
            "layer": opts.layer,
            "expansion": opts.expansion,
            "eval_batches": accum.count,
            "mse": mse_avg,
            "l0": l0_avg,
            "var_explained": var_expl_avg,
            "dead_fraction": dead_frac,
        });
        let formatted = serde_json::to_string_pretty(&record)?;
        std::fs::write(path, formatted)?;
        eprintln!("\nresults written to {path}");
        Ok(())
    })
}

fn main() {
    eval_program().run().unwrap_or_else(|e| {
        eprintln!("evaluation failed: {e}");
        std::process::exit(1);
    });
}
