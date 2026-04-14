//! Activation patching evaluation binary.
//!
//! Runs the full GPT-2 small model with and without an SAE spliced into
//! the residual stream at a chosen hookpoint, measuring the cross-entropy
//! loss delta to quantify how faithfully the SAE preserves downstream
//! model behavior.
//!
//! The binary computes two forward passes per batch:
//!
//! 1. **Baseline**: full GPT-2 forward pass producing next-token logits.
//! 2. **Patched**: GPT-2 blocks `0..hook_layer`, SAE reconstruction
//!    replacing the residual, then blocks `hook_layer..12` through `ln_f`
//!    and the tied LM head.
//!
//! Reports baseline CE, patched CE, absolute delta, and percentage CE
//! increase.
//!
//! # Usage
//!
//! ```text
//! # Evaluate with a corpus (recommended):
//! cargo run --release --bin patch_eval -- --checkpoint sae.safetensors --corpus corpus.txt
//!
//! # Custom layer and expansion:
//! cargo run --release --bin patch_eval -- --checkpoint sae.safetensors --corpus corpus.txt --layer 4 --expansion 16
//!
//! # Write results to JSON:
//! cargo run --release --bin patch_eval -- --checkpoint sae.safetensors --corpus corpus.txt --output patch.json
//! ```

use candle_core::Tensor;
use comp_cat_rs::effect::io::Io;

use gpt2_small_sae::cli::Args;
use gpt2_small_sae::config::{LayerIndex, ModelDim, SaeDim};
use gpt2_small_sae::error::Error;
use gpt2_small_sae::eval_opts::{SharedEvalOpts, build_batches};
use gpt2_small_sae::gpt2::Gpt2Full;
use gpt2_small_sae::io_boundary;
use gpt2_small_sae::sae::Sae;

const GPT2_DEPTH: usize = 12;

fn parse_patch_opts() -> Result<SharedEvalOpts, Error> {
    SharedEvalOpts::parse(&Args::parse())
}

// -----------------------------------------------------------------------
// Cross-entropy helpers
// -----------------------------------------------------------------------

/// Next-token cross-entropy loss averaged over all predicted positions.
///
/// Shifts logits and labels by one position (logits `[:, :-1, :]` vs
/// targets `[:, 1:]`), flattens, and computes mean CE.
#[allow(clippy::cast_precision_loss)]
fn next_token_ce(logits: &Tensor, input_ids: &Tensor) -> Result<f64, Error> {
    let (batch, seq_len, vocab) = match logits.dims() {
        [b, s, v] => Ok((*b, *s, *v)),
        other => Err(Error::Shape {
            what: "logits for cross-entropy",
            expected: vec![0, 0, 0],
            actual: other.to_vec(),
        }),
    }?;
    (seq_len > 1).then_some(()).ok_or(Error::Boundary {
        reason: "context length must be >= 2 for next-token cross-entropy".into(),
    })?;
    let pred = logits.narrow(1, 0, seq_len - 1)?;
    let tgt = input_ids.narrow(1, 1, seq_len - 1)?;
    let pred_flat = pred.reshape((batch * (seq_len - 1), vocab))?;
    let tgt_flat = tgt.reshape((batch * (seq_len - 1),))?;
    let ce = candle_nn::loss::cross_entropy(&pred_flat, &tgt_flat)?;
    ce.to_scalar::<f32>().map(f64::from).map_err(Error::from)
}

// -----------------------------------------------------------------------
// Evaluation
// -----------------------------------------------------------------------

/// Running accumulators for patching evaluation.
struct CeAccum {
    baseline_sum: f64,
    patched_sum: f64,
    count: u64,
}

#[allow(clippy::cast_precision_loss)]
fn evaluate(
    gpt2: &Gpt2Full,
    sae: &Sae,
    batches: &[Tensor],
    hook_layer: LayerIndex,
    opts: &SharedEvalOpts,
) -> Result<(), Error> {
    let accum = batches.iter().enumerate().try_fold(
        CeAccum {
            baseline_sum: 0.0,
            patched_sum: 0.0,
            count: 0,
        },
        |acc, (i, batch)| {
            let baseline_logits = gpt2.logits(batch)?;
            let patched_logits = gpt2.patched_logits(batch, sae, hook_layer)?;
            let baseline_ce = next_token_ce(&baseline_logits, batch)?;
            let patched_ce = next_token_ce(&patched_logits, batch)?;
            eprintln!(
                "  batch {:>3} | baseline: {:.4} | patched: {:.4} | delta: {:.4}",
                i,
                baseline_ce,
                patched_ce,
                patched_ce - baseline_ce,
            );
            Ok::<_, Error>(CeAccum {
                baseline_sum: acc.baseline_sum + baseline_ce,
                patched_sum: acc.patched_sum + patched_ce,
                count: acc.count + 1,
            })
        },
    )?;

    let n = (accum.count.max(1)) as f64;
    let baseline_avg = accum.baseline_sum / n;
    let patched_avg = accum.patched_sum / n;
    let delta = patched_avg - baseline_avg;
    let pct_increase = if baseline_avg > 0.0 {
        delta / baseline_avg * 100.0
    } else {
        0.0
    };

    eprintln!();
    eprintln!(
        "=== activation patching results ({} batches) ===",
        accum.count
    );
    eprintln!("  baseline CE:    {baseline_avg:.6}");
    eprintln!("  patched CE:     {patched_avg:.6}");
    eprintln!("  CE increase:    {delta:.6}");
    eprintln!("  % CE increase:  {pct_increase:.2}%");

    opts.output().map_or(Ok(()), |path| {
        let record = serde_json::json!({
            "checkpoint": opts.checkpoint(),
            "layer": opts.layer(),
            "expansion": opts.expansion(),
            "eval_batches": accum.count,
            "baseline_ce": baseline_avg,
            "patched_ce": patched_avg,
            "ce_increase": delta,
            "pct_ce_increase": pct_increase,
        });
        let formatted = serde_json::to_string_pretty(&record)?;
        std::fs::write(path, formatted)?;
        eprintln!("\nresults written to {path}");
        Ok(())
    })
}

// -----------------------------------------------------------------------
// Main Io program
// -----------------------------------------------------------------------

fn patch_eval_program() -> Io<Error, ()> {
    parse_patch_opts().map_or_else(
        |e| Io::suspend(move || Err(e)),
        |opts| {
            let needs_tokenizer = opts.needs_tokenizer();
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
                        Io::suspend(move || {
                            let layer_index = LayerIndex::new(opts.layer(), GPT2_DEPTH)?;
                            let model_dim = ModelDim::GPT2_SMALL;
                            let sae_dim = SaeDim::from_expansion(model_dim, opts.expansion())?;

                            eprintln!("=== activation patching evaluation ===");
                            eprintln!("checkpoint: {}", opts.checkpoint());
                            eprintln!("layer: {}, expansion: {}x", opts.layer(), opts.expansion());

                            let batches = build_batches(&opts, maybe_tokenizer.as_ref(), &device)?;
                            eprintln!();

                            eprintln!("loading SAE from {}...", opts.checkpoint());
                            let sae = Sae::from_safetensors(
                                std::path::Path::new(opts.checkpoint()),
                                model_dim,
                                sae_dim,
                                &device,
                            )?;

                            eprintln!("loading full GPT-2 small (all 12 layers)...");
                            let gpt2 = Gpt2Full::from_bytes(weights, &device)?;
                            eprintln!();

                            evaluate(&gpt2, &sae, &batches, layer_index, &opts)
                        })
                    })
                })
            })
        },
    )
}

fn main() {
    patch_eval_program().run().unwrap_or_else(|e| {
        eprintln!("activation patching evaluation failed: {e}");
        std::process::exit(1);
    });
}
