//! Pareto frontier plotter for SAE sweep results.
//!
//! Reads checkpoint metadata sidecar files (`.meta.json`) produced by the
//! training binaries and draws a scatter of final MSE vs final L0 with
//! the Pareto-optimal runs highlighted.  A run is Pareto-optimal when no
//! other run strictly dominates it on both axes (lower MSE and lower
//! L0).  Connecting the frontier in sorted order yields the classic
//! fidelity/sparsity tradeoff curve.
//!
//! The `--checkpoints` flag accepts a comma-separated list of paths.
//! Each entry may point to either a `.safetensors` checkpoint (the
//! sidecar is derived via [`io_boundary::meta_path`]) or the `.meta.json`
//! file directly.
//!
//! # Usage
//!
//! ```text
//! # Plot a single sweep (expansion factors):
//! cargo run --release --bin pareto -- \
//!   --checkpoints sae_expansion_4x.safetensors,sae_expansion_8x.safetensors,sae_expansion_16x.safetensors
//!
//! # Mix different sweeps and name the output:
//! cargo run --release --bin pareto -- \
//!   --checkpoints l1_1e-4.meta.json,l1_5e-4.meta.json,l1_1e-3.meta.json \
//!   --output l1_pareto.png
//! ```

use comp_cat_rs::effect::io::Io;
use plotters::prelude::{
    BitMapBackend, ChartBuilder, Circle, IntoDrawingArea, LineSeries, RGBColor, ShapeStyle, WHITE,
};

use gpt2_small_sae::cli::Args;
use gpt2_small_sae::error::Error;
use gpt2_small_sae::io_boundary::{self, CheckpointMeta};

/// A single point on the Pareto scatter plot.
struct PointRecord {
    label: String,
    mse: f64,
    l0: f64,
}

/// CLI options for the pareto plotter.
struct ParetoOpts {
    checkpoints: Vec<String>,
    output_path: String,
    width: u32,
    height: u32,
}

fn parse_pareto_opts() -> Result<ParetoOpts, Error> {
    let args = Args::parse();
    let raw = args
        .get("checkpoints")
        .or_else(|| args.positional(0))
        .ok_or_else(|| Error::Config {
            reason: "missing --checkpoints (comma-separated list of .safetensors or .meta.json)"
                .into(),
        })?;
    let checkpoints: Vec<String> = raw
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    (!checkpoints.is_empty())
        .then_some(())
        .ok_or_else(|| Error::Config {
            reason: "--checkpoints was empty after trimming".into(),
        })?;
    Ok(ParetoOpts {
        checkpoints,
        output_path: args.get_or("output", "pareto.png".to_string())?,
        width: args.get_or("width", 1200_u32)?,
        height: args.get_or("height", 900_u32)?,
    })
}

/// Resolve a user-supplied path to its `.meta.json` sidecar.  A path
/// already ending in `.meta.json` is returned as-is; any other extension
/// is rewritten via [`io_boundary::meta_path`].
fn resolve_meta_path(raw: &str) -> std::path::PathBuf {
    let path = std::path::Path::new(raw);
    let is_meta = path
        .file_name()
        .and_then(|s| s.to_str())
        .is_some_and(|s| s.ends_with(".meta.json"));
    if is_meta {
        path.to_path_buf()
    } else {
        io_boundary::meta_path(path)
    }
}

/// Load and parse a single metadata sidecar file into a [`PointRecord`].
fn load_point(raw: &str) -> Result<PointRecord, Error> {
    let meta_file = resolve_meta_path(raw);
    let content = std::fs::read_to_string(&meta_file).map_err(|e| Error::Boundary {
        reason: format!("failed to read metadata {}: {e}", meta_file.display()),
    })?;
    let meta: CheckpointMeta = serde_json::from_str(&content)?;
    let expansion = meta.sae_dim().checked_div(meta.model_dim()).unwrap_or(0);
    let label = format!(
        "L{}/{}x/l1={:.1e}",
        meta.layer(),
        expansion,
        meta.l1_coefficient()
    );
    Ok(PointRecord {
        label,
        mse: meta.final_mse(),
        l0: meta.final_l0(),
    })
}

// -----------------------------------------------------------------------
// Pareto frontier computation
// -----------------------------------------------------------------------

/// A point is Pareto-optimal when no other point has both a smaller or
/// equal MSE AND a smaller or equal L0, with at least one strictly
/// smaller.
fn is_pareto_optimal(points: &[PointRecord], idx: usize) -> bool {
    points.get(idx).is_some_and(|me| {
        points.iter().enumerate().all(|(j, other)| {
            j == idx
                || !(other.mse <= me.mse
                    && other.l0 <= me.l0
                    && (other.mse < me.mse || other.l0 < me.l0))
        })
    })
}

/// Compute the Pareto frontier indices, sorted by ascending L0 so the
/// returned order can be drawn as a connected line.
fn pareto_frontier_sorted(points: &[PointRecord]) -> Vec<usize> {
    (0..points.len())
        .filter(|&i| is_pareto_optimal(points, i))
        .fold(Vec::<usize>::new(), |acc, i| {
            let target_l0 = points.get(i).map_or(0.0, |p| p.l0);
            let pos = acc
                .iter()
                .position(|&j| points.get(j).map_or(f64::INFINITY, |p| p.l0) > target_l0)
                .unwrap_or(acc.len());
            let (before, after) = acc.split_at(pos);
            before
                .iter()
                .copied()
                .chain(std::iter::once(i))
                .chain(after.iter().copied())
                .collect()
        })
}

// -----------------------------------------------------------------------
// Drawing
// -----------------------------------------------------------------------

const GRAY: RGBColor = RGBColor(150, 150, 150);
const RED: RGBColor = RGBColor(228, 26, 28);

/// Convert a plotters error into our domain error.
fn plot_err<E: std::fmt::Debug>(e: E) -> Error {
    Error::Boundary {
        reason: format!("plot error: {e:?}"),
    }
}

fn axis_bounds(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let (lo, hi) = values.fold((f64::MAX, f64::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));
    let spread = (hi - lo).abs().max(1e-6);
    let pad = spread * 0.1;
    (lo - pad, hi + pad)
}

fn draw_pareto(points: &[PointRecord], frontier: &[usize], opts: &ParetoOpts) -> Result<(), Error> {
    let (x_min, x_max) = axis_bounds(points.iter().map(|p| p.l0));
    let (y_min, y_max) = axis_bounds(points.iter().map(|p| p.mse));

    let root = BitMapBackend::new(&opts.output_path, (opts.width, opts.height)).into_drawing_area();
    root.fill(&WHITE).map_err(plot_err)?;

    ChartBuilder::on(&root)
        .caption("SAE Pareto frontier: MSE vs L0", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .map_err(plot_err)
        .and_then(|mut chart| {
            chart
                .configure_mesh()
                .x_desc("L0 (active features)")
                .y_desc("MSE")
                .draw()
                .map_err(plot_err)?;
            let gray_style = ShapeStyle::from(&GRAY).filled();
            chart
                .draw_series(
                    points
                        .iter()
                        .map(|p| Circle::new((p.l0, p.mse), 5_i32, gray_style)),
                )
                .map_err(plot_err)?;
            let red_style = ShapeStyle::from(&RED).filled();
            chart
                .draw_series(frontier.iter().filter_map(|&i| {
                    points
                        .get(i)
                        .map(|p| Circle::new((p.l0, p.mse), 7_i32, red_style))
                }))
                .map_err(plot_err)?;
            chart
                .draw_series(LineSeries::new(
                    frontier
                        .iter()
                        .filter_map(|&i| points.get(i).map(|p| (p.l0, p.mse))),
                    &RED,
                ))
                .map_err(plot_err)?;
            Ok::<_, Error>(())
        })?;

    root.present().map_err(plot_err)?;
    Ok(())
}

// -----------------------------------------------------------------------
// Main Io program
// -----------------------------------------------------------------------

fn pareto_program() -> Io<Error, ()> {
    parse_pareto_opts().map_or_else(
        |e| Io::suspend(move || Err(e)),
        |opts| {
            Io::suspend(move || {
                let points: Vec<PointRecord> = opts
                    .checkpoints
                    .iter()
                    .map(|p| load_point(p))
                    .collect::<Result<_, _>>()?;
                eprintln!("=== Pareto frontier plot ===");
                eprintln!("loaded {} checkpoints:", points.len());
                points.iter().try_for_each(|p| {
                    eprintln!("  {:30}  mse={:.6}  l0={:.3}", p.label, p.mse, p.l0);
                    Ok::<_, Error>(())
                })?;

                let frontier = pareto_frontier_sorted(&points);
                eprintln!("\nPareto-optimal runs ({}):", frontier.len());
                frontier.iter().try_for_each(|&i| {
                    points.get(i).map_or(Ok(()), |p| {
                        eprintln!("  * {:28}  mse={:.6}  l0={:.3}", p.label, p.mse, p.l0);
                        Ok::<_, Error>(())
                    })
                })?;

                draw_pareto(&points, &frontier, &opts)?;
                eprintln!("\nsaved plot to {}", opts.output_path);
                Ok(())
            })
        },
    )
}

fn main() {
    pareto_program().run().unwrap_or_else(|e| {
        eprintln!("pareto plot failed: {e}");
        std::process::exit(1);
    });
}
