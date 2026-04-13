//! Evaluate a pretrained SAE checkpoint against fresh GPT-2 residual
//! stream activations.
//!
//! Loads an SAE from a safetensors file, runs GPT-2 through the same
//! hookpoint used during training, and reports aggregate metrics (MSE,
//! L0, variance explained, dead fraction) over the evaluation window.

use candle_core::{DType, Tensor};
use comp_cat_rs::effect::io::Io;

use gpt2_small_sae::activations::activation_stream;
use gpt2_small_sae::config::{LayerIndex, ModelDim, SaeDim};
use gpt2_small_sae::error::Error;
use gpt2_small_sae::gpt2::Gpt2;
use gpt2_small_sae::io_boundary;
use gpt2_small_sae::metrics::{DeadFraction, L0, Mse, VarianceExplained};
use gpt2_small_sae::sae::Sae;

const LAYER: usize = 8;
const GPT2_DEPTH: usize = 12;
const EXPANSION: usize = 8;
const BATCH_SIZE: usize = 4;
const CTX_LEN: usize = 128;
const NUM_EVAL_BATCHES: usize = 8;
const VOCAB_UPPER: f32 = 50256.0;

/// Running accumulators for evaluation metrics.
struct EvalAccum {
    l0_sum: f64,
    mse_sum: f64,
    var_expl_sum: f64,
    feature_counts: Tensor,
    count: u64,
}

fn eval_program() -> Io<Error, ()> {
    let checkpoint_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "sae_checkpoint.safetensors".to_string());
    io_boundary::acquire_device().flat_map(move |device| {
        io_boundary::download_gpt2_weights().flat_map(move |weights| {
            Io::suspend(move || {
                let layer_index = LayerIndex::new(LAYER, GPT2_DEPTH)?;
                let model_dim = ModelDim::GPT2_SMALL;
                let sae_dim = SaeDim::from_expansion(model_dim, EXPANSION)?;

                eprintln!("=== SAE evaluation ===");
                eprintln!("checkpoint: {checkpoint_path}");
                eprintln!("layer: {LAYER}, expansion: {EXPANSION}x");
                eprintln!("eval batches: {NUM_EVAL_BATCHES} x {BATCH_SIZE} x {CTX_LEN}");
                eprintln!();

                eprintln!("loading SAE from {checkpoint_path}...");
                let sae = Sae::from_safetensors(
                    std::path::Path::new(&checkpoint_path),
                    model_dim,
                    sae_dim,
                    &device,
                )?;

                eprintln!("loading GPT-2 small (layers 0..{LAYER})...");
                let gpt2 = Gpt2::from_bytes(weights, layer_index, &device)?;

                eprintln!("generating {NUM_EVAL_BATCHES} random token batches...");
                let batches = (0..NUM_EVAL_BATCHES)
                    .map(|_| {
                        Tensor::rand(0.0f32, VOCAB_UPPER, (BATCH_SIZE, CTX_LEN), &device)
                            .and_then(|t| t.to_dtype(DType::U32))
                            .map_err(Error::from)
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                let feature_counts = Tensor::zeros((sae_dim.as_usize(),), DType::F32, &device)?;

                Ok((sae, gpt2, batches, device, sae_dim, feature_counts))
            })
            .flat_map(|(sae, gpt2, batches, device, sae_dim, feature_counts)| {
                eprintln!("collecting activations...");
                activation_stream(gpt2, batches)
                    .collect()
                    .flat_map(move |activations| {
                        Io::suspend(move || {
                            evaluate(&sae, &activations, feature_counts, sae_dim, &device)
                        })
                    })
            })
        })
    })
}

#[allow(clippy::cast_precision_loss)]
fn evaluate(
    sae: &Sae,
    activations: &[Tensor],
    initial_counts: Tensor,
    sae_dim: SaeDim,
    _device: &candle_core::Device,
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

    eprintln!();
    eprintln!("=== evaluation results ({} batches) ===", accum.count);
    eprintln!("  mse:           {:.6}", accum.mse_sum / n);
    eprintln!("  l0:            {:.1}", accum.l0_sum / n);
    eprintln!("  var_explained: {:.4}", accum.var_expl_sum / n);
    eprintln!("  dead_fraction: {:.4}", dead.as_f64());
    eprintln!(
        "  sae_dim:       {} ({}x)",
        sae_dim.as_usize(),
        sae_dim.as_usize() / sae.model_dim().as_usize()
    );

    Ok(())
}

fn main() {
    eval_program().run().unwrap_or_else(|e| {
        eprintln!("evaluation failed: {e}");
        std::process::exit(1);
    });
}
