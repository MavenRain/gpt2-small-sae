//! Feature activation histogram: distribution of per-feature firing
//! rates across an evaluation corpus.
//!
//! Loads a trained SAE, runs GPT-2 residual activations through it,
//! and tallies how often each feature fires (`feature > 0`) per token.
//! The per-feature firing rate (active tokens / total tokens) is then
//! bucketed into a log-spaced histogram, with dead features (firing
//! rate exactly zero) reported separately.
//!
//! The resulting distribution is a quick diagnostic for L1 tuning:
//!
//! - A large **dead** mass means L1 is too strong or dictionary too
//!   wide; bump down L1 or run the resampler more aggressively.
//! - A large **saturated** mass (features firing on > ~10% of tokens)
//!   means L1 is too weak and features are polysemantic / unused.
//! - A healthy dictionary concentrates mass in the 1e-4..1e-2 range.
//!
//! # Usage
//!
//! ```text
//! # Default layer/expansion, random tokens:
//! cargo run --release --bin feature_hist -- --checkpoint sae_checkpoint.safetensors
//!
//! # Real corpus, JSON output:
//! cargo run --release --bin feature_hist -- \
//!   --checkpoint sae_l4_16x.safetensors --layer 4 --expansion 16 \
//!   --corpus corpus.txt --output hist.json
//!
//! # PNG histogram:
//! cargo run --release --bin feature_hist -- \
//!   --checkpoint sae_l4_16x.safetensors --expansion 16 \
//!   --histogram feature_hist.png
//! ```

use candle_core::{D, DType, Tensor};
use comp_cat_rs::effect::io::Io;
use plotters::prelude::{
    BitMapBackend, ChartBuilder, IntoDrawingArea, RGBColor, Rectangle, ShapeStyle, WHITE,
};

use gpt2_small_sae::activations::activation_stream;
use gpt2_small_sae::cli::Args;
use gpt2_small_sae::config::{LayerIndex, ModelDim, SaeDim};
use gpt2_small_sae::error::Error;
use gpt2_small_sae::eval_opts::{SharedEvalOpts, build_batches};
use gpt2_small_sae::gpt2::Gpt2;
use gpt2_small_sae::io_boundary;
use gpt2_small_sae::sae::Sae;

const GPT2_DEPTH: usize = 12;
/// Log-space floor for the histogram: any nonzero rate below this is
/// clamped into the first bin so no firing data is dropped silently.
const LOG_FLOOR: f64 = 1e-6;
/// Threshold above which a feature is considered "saturated" and a
/// candidate for polysemanticity / underpenalized-L1 diagnosis.
const SATURATED_RATE: f64 = 0.1;
/// Bar color: the project's canonical red `(228, 26, 28)`.
const BAR_COLOR: RGBColor = RGBColor(228, 26, 28);
/// Dead-bucket color: a neutral gray to distinguish it from the
/// log-binned mass.
const DEAD_COLOR: RGBColor = RGBColor(120, 120, 120);

/// CLI-configurable options for the histogram binary.  Combines the
/// shared eval flags with the histogram-specific rendering options.
#[derive(Clone)]
struct HistOpts {
    shared: SharedEvalOpts,
    num_bins: usize,
    histogram: Option<String>,
    histogram_width: u32,
    histogram_height: u32,
}

fn parse_hist_opts() -> Result<HistOpts, Error> {
    let args = Args::parse();
    Ok(HistOpts {
        shared: SharedEvalOpts::parse(&args)?,
        num_bins: args.get_or("num-bins", 20_usize)?,
        histogram: args.get("histogram").map(String::from),
        histogram_width: args.get_or("histogram-width", 900_u32)?,
        histogram_height: args.get_or("histogram-height", 600_u32)?,
    })
}

// -----------------------------------------------------------------------
// Firing-rate accumulation
// -----------------------------------------------------------------------

struct Accum {
    counts: Tensor,
    tokens: u64,
}

/// Fold an activation stream into (per-feature firing counts, total
/// tokens seen).  Counts are a `(sae_dim,)` `f32` tensor; each active
/// entry (`feature > 0`) contributes `+1`.
fn accumulate_counts(sae: &Sae, activations: &[Tensor], init: Tensor) -> Result<Accum, Error> {
    activations.iter().try_fold(
        Accum {
            counts: init,
            tokens: 0,
        },
        |acc, batch| {
            let tokens = match batch.dims() {
                [t, _] => *t,
                other => {
                    return Err(Error::Shape {
                        what: "activation batch",
                        expected: vec![0, 0],
                        actual: other.to_vec(),
                    });
                }
            };
            let fwd = sae.forward(batch)?;
            let active = fwd
                .features()
                .gt(0.0_f32)?
                .to_dtype(DType::F32)?
                .transpose(0, 1)?
                .sum(D::Minus1)?;
            let counts = acc.counts.add(&active)?;
            Ok::<_, Error>(Accum {
                counts,
                tokens: acc.tokens + tokens as u64,
            })
        },
    )
}

// -----------------------------------------------------------------------
// Histogram binning
// -----------------------------------------------------------------------

/// A log-spaced histogram of per-feature firing rates, plus the count
/// of dead features (firing rate exactly zero).
struct Histogram {
    edges: Vec<f64>,
    counts: Vec<u64>,
    dead: u64,
    saturated: u64,
    total: u64,
}

/// Bin `rates` into `num_bins` log-spaced buckets between
/// [`LOG_FLOOR`] and `1.0`.  Zero rates land in the separate `dead`
/// counter.  Any nonzero rate below `LOG_FLOOR` is pinned to the first
/// bin so the plot never silently drops data.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]
fn bucket_rates(rates: &[f64], num_bins: usize) -> Histogram {
    let n_bins = num_bins.max(1);
    let log_lo = LOG_FLOOR.log10();
    let log_hi: f64 = 0.0; // log10(1.0)
    let step = (log_hi - log_lo) / (n_bins as f64);
    let edges: Vec<f64> = (0..=n_bins).map(|i| log_lo + step * (i as f64)).collect();

    let (counts, dead, saturated) = rates.iter().fold(
        (vec![0_u64; n_bins], 0_u64, 0_u64),
        |(counts, dead, saturated), &r| {
            if r <= 0.0 {
                (counts, dead + 1, saturated)
            } else {
                let log_r = r.max(LOG_FLOOR).log10();
                let raw = ((log_r - log_lo) / step).floor();
                let raw_nn = raw.max(0.0) as usize;
                let idx = raw_nn.min(n_bins - 1);
                let next = counts
                    .iter()
                    .enumerate()
                    .map(|(i, c)| if i == idx { c + 1 } else { *c })
                    .collect();
                let sat_inc = u64::from(r >= SATURATED_RATE);
                (next, dead, saturated + sat_inc)
            }
        },
    );

    Histogram {
        edges,
        counts,
        dead,
        saturated,
        total: rates.len() as u64,
    }
}

// -----------------------------------------------------------------------
// Reporting
// -----------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
fn report(hist: &Histogram, opts: &HistOpts, sae_dim: usize) -> Result<(), Error> {
    let total = hist.total.max(1) as f64;
    let dead_pct = (hist.dead as f64) / total * 100.0;
    let sat_pct = (hist.saturated as f64) / total * 100.0;

    eprintln!("=== feature activation histogram ===");
    eprintln!("checkpoint: {}", opts.shared.checkpoint());
    eprintln!("sae_dim: {sae_dim}");
    eprintln!();
    eprintln!(
        "  dead features:      {:>6} / {sae_dim}   ({dead_pct:>6.2}%)",
        hist.dead
    );
    eprintln!(
        "  saturated (>= {:.0}%): {:>6} / {sae_dim}   ({sat_pct:>6.2}%)",
        SATURATED_RATE * 100.0,
        hist.saturated
    );
    eprintln!();
    eprintln!("log-firing-rate distribution:");
    hist.edges
        .windows(2)
        .zip(hist.counts.iter())
        .try_for_each(|(window, count)| {
            let (lo, hi) = match window {
                [a, b] => (*a, *b),
                _ => (0.0, 0.0),
            };
            eprintln!(
                "  [10^{lo:>5.2}, 10^{hi:>5.2}):  {count:>6}   ({:>5.2}%)",
                (*count as f64) / total * 100.0
            );
            Ok::<_, Error>(())
        })?;

    opts.shared.output().map_or(Ok(()), |path| {
        let bins_json: Vec<_> = hist
            .edges
            .windows(2)
            .zip(hist.counts.iter())
            .map(|(window, count)| {
                let (lo, hi) = match window {
                    [a, b] => (*a, *b),
                    _ => (0.0, 0.0),
                };
                serde_json::json!({
                    "log_lo": lo,
                    "log_hi": hi,
                    "count": *count,
                })
            })
            .collect();
        let record = serde_json::json!({
            "checkpoint": opts.shared.checkpoint(),
            "layer": opts.shared.layer(),
            "expansion": opts.shared.expansion(),
            "sae_dim": sae_dim,
            "dead_count": hist.dead,
            "saturated_count": hist.saturated,
            "bins": bins_json,
        });
        let formatted = serde_json::to_string_pretty(&record)?;
        std::fs::write(path, formatted)?;
        eprintln!("\nresults written to {path}");
        Ok(())
    })
}

// -----------------------------------------------------------------------
// Histogram rendering
// -----------------------------------------------------------------------

fn plot_err<E: std::fmt::Debug>(e: E) -> Error {
    Error::Boundary {
        reason: format!("plot error: {e:?}"),
    }
}

/// Render a PNG histogram.  The main panel is a log-binned bar chart
/// of nonzero firing rates; the first bar is a neutral-gray dead
/// bucket placed to the left of the log range for visual separation.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]
fn draw_histogram(hist: &Histogram, opts: &HistOpts, path: &str) -> Result<(), Error> {
    let n_bins = hist.counts.len();
    let max_count = hist
        .counts
        .iter()
        .copied()
        .chain(std::iter::once(hist.dead))
        .max()
        .unwrap_or(1)
        .max(1);

    let root =
        BitMapBackend::new(path, (opts.histogram_width, opts.histogram_height)).into_drawing_area();
    root.fill(&WHITE).map_err(plot_err)?;

    // Reserve bin 0 for the "dead" bucket, bins 1..=n_bins for the
    // log-spaced firing-rate mass.  Using an integer x axis keeps
    // rectangle drawing trivial.
    let x_upper = (n_bins + 1) as i32;
    let y_upper = (max_count as f64 * 1.10).ceil() as i64;

    ChartBuilder::on(&root)
        .caption(
            format!("Feature firing-rate distribution ({} features)", hist.total),
            ("sans-serif", 20),
        )
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0_i32..x_upper, 0_i64..y_upper)
        .map_err(plot_err)
        .and_then(|mut chart| {
            chart
                .configure_mesh()
                .x_desc("bin (0 = dead, 1..N = log10 firing rate)")
                .y_desc("feature count")
                .draw()
                .map_err(plot_err)?;

            let dead_style = ShapeStyle::from(&DEAD_COLOR).filled();
            chart
                .draw_series(std::iter::once(Rectangle::new(
                    [(0_i32, 0_i64), (1_i32, hist.dead as i64)],
                    dead_style,
                )))
                .map_err(plot_err)?;

            let bar_style = ShapeStyle::from(&BAR_COLOR).filled();
            chart
                .draw_series(hist.counts.iter().enumerate().map(|(i, &c)| {
                    let x0 = (i + 1) as i32;
                    let x1 = x0 + 1;
                    Rectangle::new([(x0, 0_i64), (x1, c as i64)], bar_style)
                }))
                .map_err(plot_err)?;
            Ok::<_, Error>(())
        })?;

    root.present().map_err(plot_err)?;
    Ok(())
}

// -----------------------------------------------------------------------
// Main Io program
// -----------------------------------------------------------------------

fn feature_hist_program() -> Io<Error, ()> {
    parse_hist_opts().map_or_else(
        |e| Io::suspend(move || Err(e)),
        |opts| {
            let needs_tokenizer = opts.shared.needs_tokenizer();
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
                            let layer_index = LayerIndex::new(opts.shared.layer(), GPT2_DEPTH)?;
                            let model_dim = ModelDim::GPT2_SMALL;
                            let sae_dim =
                                SaeDim::from_expansion(model_dim, opts.shared.expansion())?;

                            eprintln!("=== feature activation histogram ===");
                            eprintln!("checkpoint: {}", opts.shared.checkpoint());
                            eprintln!(
                                "layer: {}, expansion: {}x",
                                opts.shared.layer(),
                                opts.shared.expansion()
                            );

                            let batches =
                                build_batches(&opts.shared, maybe_tokenizer.as_ref(), &device)?;
                            eprintln!();

                            eprintln!("loading SAE from {}...", opts.shared.checkpoint());
                            let sae = Sae::from_safetensors(
                                std::path::Path::new(opts.shared.checkpoint()),
                                model_dim,
                                sae_dim,
                                &device,
                            )?;

                            eprintln!("loading GPT-2 small (layers 0..{})...", opts.shared.layer());
                            let gpt2 = Gpt2::from_bytes(weights, layer_index, &device)?;

                            let init_counts =
                                Tensor::zeros((sae_dim.as_usize(),), DType::F32, &device)?;
                            Ok((sae, gpt2, batches, sae_dim, init_counts))
                        })
                        .flat_map(
                            move |(sae, gpt2, batches, sae_dim, init_counts)| {
                                eprintln!("collecting activations...");
                                activation_stream(gpt2, batches).collect().flat_map(
                                    move |activations| {
                                        Io::suspend(move || {
                                            finalize(
                                                &sae,
                                                &activations,
                                                init_counts,
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
fn finalize(
    sae: &Sae,
    activations: &[Tensor],
    init_counts: Tensor,
    sae_dim: SaeDim,
    opts: &HistOpts,
) -> Result<(), Error> {
    let accum = accumulate_counts(sae, activations, init_counts)?;
    let token_count = accum.tokens.max(1) as f64;
    let counts_vec: Vec<f32> = accum.counts.to_vec1()?;
    let rates: Vec<f64> = counts_vec
        .iter()
        .map(|&c| f64::from(c) / token_count)
        .collect();

    let hist = bucket_rates(&rates, opts.num_bins);
    report(&hist, opts, sae_dim.as_usize())?;

    opts.histogram.as_ref().map_or(Ok(()), |path| {
        eprintln!("\nrendering histogram to {path}...");
        draw_histogram(&hist, opts, path)?;
        eprintln!("histogram written to {path}");
        Ok(())
    })
}

fn main() {
    feature_hist_program().run().unwrap_or_else(|e| {
        eprintln!("feature histogram failed: {e}");
        std::process::exit(1);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_empty_rates() {
        let hist = bucket_rates(&[], 10);
        assert_eq!(hist.total, 0);
        assert_eq!(hist.dead, 0);
        assert_eq!(hist.saturated, 0);
        assert_eq!(hist.counts.iter().sum::<u64>(), 0);
    }

    #[test]
    fn bucket_dead_features() {
        let rates = vec![0.0, 0.0, 0.0];
        let hist = bucket_rates(&rates, 10);
        assert_eq!(hist.dead, 3);
        assert_eq!(hist.counts.iter().sum::<u64>(), 0);
    }

    #[test]
    fn bucket_saturated_features() {
        let rates = vec![0.5, 0.2, 0.15];
        let hist = bucket_rates(&rates, 10);
        assert_eq!(hist.saturated, 3);
        assert_eq!(hist.dead, 0);
        assert_eq!(hist.counts.iter().sum::<u64>(), 3);
    }

    #[test]
    fn bucket_conserves_mass() {
        let rates = vec![0.0, 1e-5, 1e-3, 0.01, 0.5, 0.0, 1e-2];
        let hist = bucket_rates(&rates, 20);
        let binned = hist.counts.iter().sum::<u64>();
        assert_eq!(binned + hist.dead, rates.len() as u64);
    }

    #[test]
    fn bucket_clamps_below_floor() {
        // Rate well below LOG_FLOOR should still land in bin 0, not
        // vanish or panic.
        let rates = vec![1e-12];
        let hist = bucket_rates(&rates, 10);
        assert_eq!(hist.dead, 0);
        assert_eq!(hist.counts.first().copied().unwrap_or(0), 1);
    }
}
