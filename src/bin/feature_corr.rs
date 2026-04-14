//! Feature correlation analysis: pairwise cosine similarity between the
//! decoder rows of a trained SAE.
//!
//! Loads a checkpoint, normalizes each decoder row to unit L2 length,
//! computes the full `(sae_dim x sae_dim)` cosine similarity matrix, and
//! reports:
//!
//! 1. **Summary statistics**: mean and max of |off-diagonal| cosine
//!    similarity, an aggregate measure of how orthogonal the learned
//!    dictionary is.
//! 2. **Top-K most similar pairs**: ranked by each feature's single
//!    nearest non-self neighbor.  Mutual nearest neighbors (the usual
//!    redundancy signal) both appear at high rank.
//!
//! The dictionary for a clean, well-trained SAE should be close to
//! orthogonal: mean off-diagonal near 0 and max below roughly 0.3.
//! Persistent high-similarity pairs typically indicate feature splitting
//! or duplication and suggest the L1 coefficient should be raised (or
//! the dictionary width lowered).
//!
//! Passing `--heatmap path.png` also renders a diverging red-white-blue
//! heatmap of the (optionally downsampled) correlation matrix for a
//! quick visual read of off-diagonal structure.
//!
//! # Usage
//!
//! ```text
//! # Default layer/expansion (8 / 8x):
//! cargo run --release --bin feature_corr -- --checkpoint sae_checkpoint.safetensors
//!
//! # Custom configuration with JSON output:
//! cargo run --release --bin feature_corr -- \
//!   --checkpoint sae_l4_16x.safetensors --layer 4 --expansion 16 \
//!   --top-k 30 --output corr.json
//!
//! # Render a heatmap PNG too:
//! cargo run --release --bin feature_corr -- \
//!   --checkpoint sae_l4_16x.safetensors --expansion 16 \
//!   --heatmap corr_l4_16x.png --heatmap-size 256
//! ```

use std::collections::BinaryHeap;

use candle_core::{D, DType, Device, Tensor};
use comp_cat_rs::effect::io::Io;
use plotters::prelude::{
    BitMapBackend, ChartBuilder, IntoDrawingArea, RGBColor, Rectangle, ShapeStyle, WHITE,
};

use gpt2_small_sae::cli::Args;
use gpt2_small_sae::config::{ModelDim, SaeDim};
use gpt2_small_sae::error::Error;
use gpt2_small_sae::io_boundary;
use gpt2_small_sae::sae::Sae;

/// Mask scalar added to the diagonal of the similarity matrix so that
/// self-similarity cannot win an argmax.  Any value safely above the
/// cosine-similarity range of [-1, 1] works; 1000 is comfortably large.
const DIAG_MASK: f64 = 1000.0;

/// CLI-configurable options for the feature correlation binary.
#[derive(Clone)]
struct CorrOpts {
    layer: usize,
    expansion: usize,
    checkpoint: String,
    top_k: usize,
    output: Option<String>,
    heatmap: Option<String>,
    heatmap_size: usize,
    heatmap_width: u32,
    heatmap_height: u32,
}

fn parse_corr_opts() -> Result<CorrOpts, Error> {
    let args = Args::parse();
    Ok(CorrOpts {
        layer: args.get_or("layer", 8_usize)?,
        expansion: args.get_or("expansion", 8_usize)?,
        checkpoint: args
            .get("checkpoint")
            .or_else(|| args.positional(0))
            .map_or_else(|| "sae_checkpoint.safetensors".to_string(), String::from),
        top_k: args.get_or("top-k", 20_usize)?,
        output: args.get("output").map(String::from),
        heatmap: args.get("heatmap").map(String::from),
        heatmap_size: args.get_or("heatmap-size", 256_usize)?,
        heatmap_width: args.get_or("heatmap-width", 900_u32)?,
        heatmap_height: args.get_or("heatmap-height", 900_u32)?,
    })
}

// -----------------------------------------------------------------------
// Similarity matrix construction
// -----------------------------------------------------------------------

/// Normalize every row of the decoder to unit L2 length.  A tiny
/// constant is added to the norms to keep dead features (which may have
/// near-zero decoder rows) from producing NaNs.
fn row_normalize(w_dec: &Tensor) -> Result<Tensor, Error> {
    let norms = w_dec.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    let safe = norms.affine(1.0, 1e-12)?;
    w_dec.broadcast_div(&safe).map_err(Error::from)
}

/// Cosine similarity matrix `(sae_dim, sae_dim)` whose diagonal is 1.
fn cosine_similarity(w_dec: &Tensor) -> Result<Tensor, Error> {
    let normalized = row_normalize(w_dec)?;
    normalized.matmul(&normalized.t()?).map_err(Error::from)
}

/// Extract the square side length of a 2-D tensor, erroring if the
/// tensor is not square.
fn square_side(t: &Tensor, what: &'static str) -> Result<usize, Error> {
    match t.dims() {
        [a, b] if a == b => Ok(*a),
        other => Err(Error::Shape {
            what,
            expected: vec![0, 0],
            actual: other.to_vec(),
        }),
    }
}

// -----------------------------------------------------------------------
// Summary statistics
// -----------------------------------------------------------------------

struct CorrStats {
    mean_off_diag: f64,
    max_abs_off_diag: f64,
}

#[allow(clippy::cast_precision_loss)]
fn compute_stats(corr: &Tensor, device: &Device) -> Result<CorrStats, Error> {
    let n = square_side(corr, "correlation matrix")?;
    let eye = Tensor::eye(n, DType::F32, device)?;
    let off_diag = corr.sub(&eye)?;
    let sum_off_diag = off_diag.sum_all()?.to_scalar::<f32>()?;
    let pair_count = (n * n - n) as f32;
    let mean_off_diag = f64::from(sum_off_diag / pair_count);
    let max_abs_off_diag = f64::from(off_diag.abs()?.max_all()?.to_scalar::<f32>()?);
    Ok(CorrStats {
        mean_off_diag,
        max_abs_off_diag,
    })
}

// -----------------------------------------------------------------------
// Top-K nearest-neighbor ranking
// -----------------------------------------------------------------------

struct NearestPair {
    feature: usize,
    partner: usize,
    similarity: f32,
}

/// For every feature, find its single most-similar non-self partner by
/// magnitude, then rank features by that partner similarity and return
/// the top K.  Mutual nearest neighbors both surface at high rank.
fn top_nearest_pairs(corr: &Tensor, k: usize, device: &Device) -> Result<Vec<NearestPair>, Error> {
    let n = square_side(corr, "correlation matrix")?;
    let eye = Tensor::eye(n, DType::F32, device)?;
    let off_diag = corr.sub(&eye)?;
    let abs_off = off_diag.abs()?;

    // Mask diagonal with a large value so magnitude-based argmax ignores
    // self-similarity.  Using `abs_off` already zeroes the diagonal, but
    // we still need a guarded argmax below.
    let mask = eye.affine(DIAG_MASK, 0.0)?;
    let guarded_abs = abs_off.broadcast_sub(&mask)?;

    let per_row_abs_max = guarded_abs.max(D::Minus1)?.to_vec1::<f32>()?;
    let per_row_argmax: Vec<u32> = guarded_abs.argmax(D::Minus1)?.to_vec1::<u32>()?;
    let signed_corr: Vec<f32> = off_diag.flatten_all()?.to_vec1::<f32>()?;

    let ranking: Vec<(u32, usize)> = per_row_abs_max
        .iter()
        .enumerate()
        .map(|(i, &s)| (s.max(0.0).to_bits(), i))
        .collect();

    let top: Vec<NearestPair> = BinaryHeap::from(ranking)
        .into_sorted_vec()
        .into_iter()
        .rev()
        .take(k)
        .filter_map(|(_, i)| {
            let partner = per_row_argmax.get(i).map(|&j| j as usize)?;
            let similarity = signed_corr.get(i * n + partner).copied()?;
            Some(NearestPair {
                feature: i,
                partner,
                similarity,
            })
        })
        .collect();
    Ok(top)
}

// -----------------------------------------------------------------------
// Heatmap rendering
// -----------------------------------------------------------------------

/// Average-pool the correlation matrix down to a roughly `target`-sided
/// square.  When `n <= target` the matrix is returned unchanged;
/// otherwise the pooling block size is `n / target` and the bottom-right
/// `n mod target` rows/columns are trimmed to keep the pooling divisible.
fn downsample(corr: &Tensor, target: usize) -> Result<(Tensor, usize), Error> {
    let n = square_side(corr, "correlation matrix")?;
    if n <= target {
        Ok((corr.clone(), n))
    } else {
        let block = n / target;
        let trimmed_side = target * block;
        let trimmed = corr
            .narrow(0, 0, trimmed_side)?
            .narrow(1, 0, trimmed_side)?;
        let pooled = trimmed
            .unsqueeze(0)?
            .unsqueeze(0)?
            .avg_pool2d(block)?
            .squeeze(0)?
            .squeeze(0)?;
        Ok((pooled, target))
    }
}

/// Diverging red-white-blue colormap.  `v` is clamped into `[-1, 1]`;
/// positive values interpolate from white to the project's canonical
/// red `(228, 26, 28)` and negative values from white to blue
/// `(55, 126, 184)`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]
fn value_to_color(value: f32) -> RGBColor {
    let clamped = value.clamp(-1.0, 1.0);
    if clamped >= 0.0 {
        let t = clamped;
        let red = (255.0 - t * 27.0) as u8;
        let green = (255.0 - t * 229.0) as u8;
        let blue = (255.0 - t * 227.0) as u8;
        RGBColor(red, green, blue)
    } else {
        let t = -clamped;
        let red = (255.0 - t * 200.0) as u8;
        let green = (255.0 - t * 129.0) as u8;
        let blue = (255.0 - t * 71.0) as u8;
        RGBColor(red, green, blue)
    }
}

fn plot_err<E: std::fmt::Debug>(e: E) -> Error {
    Error::Boundary {
        reason: format!("plot error: {e:?}"),
    }
}

#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
fn draw_heatmap(corr: &Tensor, opts: &CorrOpts, path: &str) -> Result<(), Error> {
    let (downsampled, side) = downsample(corr, opts.heatmap_size)?;
    let data: Vec<f32> = downsampled.flatten_all()?.to_vec1::<f32>()?;
    let side_i = side as i32;

    let root =
        BitMapBackend::new(path, (opts.heatmap_width, opts.heatmap_height)).into_drawing_area();
    root.fill(&WHITE).map_err(plot_err)?;

    ChartBuilder::on(&root)
        .caption(
            format!("Feature correlation heatmap ({side}x{side})"),
            ("sans-serif", 20),
        )
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0_i32..side_i, side_i..0_i32)
        .map_err(plot_err)
        .and_then(|mut chart| {
            chart
                .configure_mesh()
                .disable_mesh()
                .x_desc("feature j")
                .y_desc("feature i")
                .draw()
                .map_err(plot_err)?;
            let data_ref = &data;
            let rects: Vec<_> = (0..side)
                .flat_map(|y| {
                    (0..side).filter_map(move |x| {
                        data_ref.get(y * side + x).map(|&v| {
                            Rectangle::new(
                                [(x as i32, y as i32), ((x + 1) as i32, (y + 1) as i32)],
                                ShapeStyle::from(value_to_color(v)).filled(),
                            )
                        })
                    })
                })
                .collect();
            chart.draw_series(rects).map_err(plot_err)?;
            Ok::<_, Error>(())
        })?;

    root.present().map_err(plot_err)?;
    Ok(())
}

// -----------------------------------------------------------------------
// Reporting
// -----------------------------------------------------------------------

fn report(
    stats: &CorrStats,
    top: &[NearestPair],
    sae_dim: usize,
    opts: &CorrOpts,
) -> Result<(), Error> {
    eprintln!("=== feature correlation analysis ===");
    eprintln!("checkpoint: {}", opts.checkpoint);
    eprintln!("sae_dim: {sae_dim}");
    eprintln!();
    eprintln!("  mean off-diagonal cosine:     {:.6}", stats.mean_off_diag);
    eprintln!(
        "  max  |off-diagonal| cosine:   {:.6}",
        stats.max_abs_off_diag
    );
    eprintln!();
    eprintln!("top {} features by nearest-neighbor similarity:", top.len());
    top.iter().try_for_each(|p| {
        eprintln!(
            "  feature {:>6}  <-> {:>6}   sim = {:+.4}",
            p.feature, p.partner, p.similarity,
        );
        Ok::<_, Error>(())
    })?;

    opts.output.as_ref().map_or(Ok(()), |path| {
        let pairs_json: Vec<_> = top
            .iter()
            .map(|p| {
                serde_json::json!({
                    "feature": p.feature,
                    "partner": p.partner,
                    "similarity": p.similarity,
                })
            })
            .collect();
        let record = serde_json::json!({
            "checkpoint": opts.checkpoint,
            "layer": opts.layer,
            "expansion": opts.expansion,
            "sae_dim": sae_dim,
            "mean_off_diag": stats.mean_off_diag,
            "max_abs_off_diag": stats.max_abs_off_diag,
            "top_pairs": pairs_json,
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

fn feature_corr_program() -> Io<Error, ()> {
    parse_corr_opts().map_or_else(
        |e| Io::suspend(move || Err(e)),
        |opts| {
            io_boundary::acquire_device().flat_map(move |device| {
                Io::suspend(move || {
                    let model_dim = ModelDim::GPT2_SMALL;
                    let sae_dim = SaeDim::from_expansion(model_dim, opts.expansion)?;
                    let sae = Sae::from_safetensors(
                        std::path::Path::new(&opts.checkpoint),
                        model_dim,
                        sae_dim,
                        &device,
                    )?;
                    let corr = cosine_similarity(sae.w_dec())?;
                    let stats = compute_stats(&corr, &device)?;
                    let top = top_nearest_pairs(&corr, opts.top_k, &device)?;
                    report(&stats, &top, sae_dim.as_usize(), &opts)?;
                    opts.heatmap.as_ref().map_or(Ok(()), |path| {
                        eprintln!("\nrendering heatmap to {path}...");
                        draw_heatmap(&corr, &opts, path)?;
                        eprintln!("heatmap written to {path}");
                        Ok(())
                    })
                })
            })
        },
    )
}

fn main() {
    feature_corr_program().run().unwrap_or_else(|e| {
        eprintln!("feature correlation analysis failed: {e}");
        std::process::exit(1);
    });
}
