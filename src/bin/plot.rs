//! Loss curve plotter: reads one or more metrics JSONL files and
//! produces a 2x2 panel PNG with MSE, L0, variance explained, and
//! dead fraction curves.
//!
//! When multiple files are given (comma-separated), each file is drawn
//! as a separate line with a distinct color for easy comparison across
//! training runs or sweep layers.
//!
//! # Usage
//!
//! ```text
//! # Plot a single training run:
//! cargo run --release --bin plot -- --metrics metrics.jsonl
//!
//! # Compare sweep layers:
//! cargo run --release --bin plot -- \
//!   --metrics metrics_layer_0.jsonl,metrics_layer_8.jsonl
//!
//! # Custom output and dimensions:
//! cargo run --release --bin plot -- --output curves.png --width 1600 --height 1200
//! ```

use comp_cat_rs::effect::io::Io;
use plotters::prelude::{
    BitMapBackend, ChartBuilder, IntoDrawingArea, LineSeries, RGBColor, WHITE,
};

use gpt2_small_sae::cli::Args;
use gpt2_small_sae::error::Error;

/// Color palette for distinguishing multiple overlaid runs.
const PALETTE: [RGBColor; 8] = [
    RGBColor(228, 26, 28),
    RGBColor(55, 126, 184),
    RGBColor(77, 175, 74),
    RGBColor(152, 78, 163),
    RGBColor(255, 127, 0),
    RGBColor(166, 86, 40),
    RGBColor(247, 129, 191),
    RGBColor(153, 153, 153),
];

/// Per-metric colors used when plotting a single file.
const SINGLE_COLORS: [RGBColor; 4] = [
    RGBColor(228, 26, 28),
    RGBColor(55, 126, 184),
    RGBColor(77, 175, 74),
    RGBColor(152, 78, 163),
];

/// A single deserialized metrics record from the JSONL file.
#[derive(serde::Deserialize)]
struct MetricsRecord {
    step: u64,
    l0: f64,
    mse: f64,
    var_explained: f64,
    dead_fraction: f64,
}

/// One loaded metrics file with its display label.
struct MetricsFile {
    label: String,
    records: Vec<MetricsRecord>,
}

/// CLI-configurable plot options.
struct PlotOpts {
    metrics_paths: Vec<String>,
    output_path: String,
    width: u32,
    height: u32,
}

fn parse_plot_opts() -> Result<PlotOpts, Error> {
    let args = Args::parse();
    let raw = args
        .get("metrics")
        .or_else(|| args.positional(0))
        .map_or_else(|| "metrics.jsonl".to_string(), String::from);
    let metrics_paths: Vec<String> = raw.split(',').map(|s| s.trim().to_string()).collect();
    Ok(PlotOpts {
        metrics_paths,
        output_path: args.get_or("output", "curves.png".to_string())?,
        width: args.get_or("width", 1200_u32)?,
        height: args.get_or("height", 900_u32)?,
    })
}

/// Convert a plotters error into our domain error.
fn plot_err<E: std::fmt::Debug>(e: E) -> Error {
    Error::Boundary {
        reason: format!("plot error: {e:?}"),
    }
}

/// Configuration for a single chart panel.
struct PanelConfig {
    title: &'static str,
    extract: fn(&MetricsRecord) -> f64,
}

/// Load a metrics JSONL file and return its records.
fn load_metrics_file(path: &str) -> Result<MetricsFile, Error> {
    let content = std::fs::read_to_string(path).map_err(|e| Error::Boundary {
        reason: format!("failed to read metrics file {path}: {e}"),
    })?;
    let records: Vec<MetricsRecord> = content
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_str(line).map_err(Error::from))
        .collect::<Result<_, _>>()?;
    if records.is_empty() {
        Err(Error::Boundary {
            reason: format!("no metrics records found in {path}"),
        })
    } else {
        // Derive a short label from the filename.
        let label = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .map_or_else(|| path.to_string(), String::from);
        Ok(MetricsFile { label, records })
    }
}

/// Draw a 2x2 panel grid of training curves to a PNG file.
#[allow(clippy::cast_precision_loss)]
fn draw_panels(files: &[MetricsFile], opts: &PlotOpts) -> Result<(), Error> {
    let panels = [
        PanelConfig {
            title: "MSE",
            extract: (|r| r.mse) as fn(&MetricsRecord) -> f64,
        },
        PanelConfig {
            title: "L0",
            extract: (|r| r.l0) as fn(&MetricsRecord) -> f64,
        },
        PanelConfig {
            title: "Variance Explained",
            extract: (|r| r.var_explained) as fn(&MetricsRecord) -> f64,
        },
        PanelConfig {
            title: "Dead Fraction",
            extract: (|r| r.dead_fraction) as fn(&MetricsRecord) -> f64,
        },
    ];

    let x_max = files
        .iter()
        .filter_map(|f| f.records.last())
        .map(|r| r.step as f64)
        .fold(1.0_f64, f64::max);

    let single_file = files.len() == 1;

    let root = BitMapBackend::new(&opts.output_path, (opts.width, opts.height)).into_drawing_area();
    root.fill(&WHITE).map_err(plot_err)?;

    let areas = root.split_evenly((2, 2));

    areas
        .iter()
        .zip(panels.iter())
        .enumerate()
        .try_for_each(|(panel_idx, (area, panel))| {
            let (y_min, y_max) = files.iter().flat_map(|f| f.records.iter()).fold(
                (f64::MAX, f64::MIN),
                |(lo, hi), r| {
                    let v = (panel.extract)(r);
                    (lo.min(v), hi.max(v))
                },
            );
            let spread = (y_max - y_min).abs().max(1e-6);
            let padding = spread * 0.1;

            ChartBuilder::on(area)
                .caption(panel.title, ("sans-serif", 18))
                .margin(10)
                .x_label_area_size(30)
                .y_label_area_size(50)
                .build_cartesian_2d(0.0..x_max, (y_min - padding)..(y_max + padding))
                .map_err(plot_err)
                .and_then(|mut chart| {
                    chart.configure_mesh().draw().map_err(plot_err)?;
                    files.iter().enumerate().try_for_each(|(file_idx, file)| {
                        let color = if single_file {
                            SINGLE_COLORS
                                .get(panel_idx % SINGLE_COLORS.len())
                                .copied()
                                .unwrap_or(SINGLE_COLORS[0])
                        } else {
                            PALETTE
                                .get(file_idx % PALETTE.len())
                                .copied()
                                .unwrap_or(PALETTE[0])
                        };
                        chart
                            .draw_series(LineSeries::new(
                                file.records
                                    .iter()
                                    .map(|r| (r.step as f64, (panel.extract)(r))),
                                &color,
                            ))
                            .map_err(plot_err)?;
                        Ok::<_, Error>(())
                    })
                })
        })?;

    root.present().map_err(plot_err)?;
    Ok(())
}

fn plot_program() -> Io<Error, ()> {
    parse_plot_opts().map_or_else(
        |e| Io::suspend(move || Err(e)),
        |opts| {
            Io::suspend(move || {
                let files: Vec<MetricsFile> = opts
                    .metrics_paths
                    .iter()
                    .map(|p| load_metrics_file(p))
                    .collect::<Result<_, _>>()?;

                if files.len() == 1 {
                    eprintln!(
                        "plotting {} steps from {}...",
                        files.first().map_or(0, |f| f.records.len()),
                        opts.metrics_paths.first().map_or("?", String::as_str),
                    );
                } else {
                    eprintln!(
                        "overlaying {} files: {}",
                        files.len(),
                        files
                            .iter()
                            .map(|f| f.label.as_str())
                            .collect::<Vec<_>>()
                            .join(", "),
                    );
                }

                draw_panels(&files, &opts)?;
                eprintln!("saved plot to {}", opts.output_path);
                Ok(())
            })
        },
    )
}

fn main() {
    plot_program().run().unwrap_or_else(|e| {
        eprintln!("plot failed: {e}");
        std::process::exit(1);
    });
}
