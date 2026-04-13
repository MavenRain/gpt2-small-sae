//! Loss curve plotter: reads a metrics JSONL file and produces a 2x2
//! panel PNG with MSE, L0, variance explained, and dead fraction curves.
//!
//! # Usage
//!
//! ```text
//! # Plot training curves from the default metrics file:
//! cargo run --release --bin plot
//!
//! # Plot with custom input/output:
//! cargo run --release --bin plot -- --metrics metrics.jsonl --output curves.png
//!
//! # Custom image dimensions:
//! cargo run --release --bin plot -- --width 1600 --height 1200
//! ```

use comp_cat_rs::effect::io::Io;
use plotters::prelude::{
    BLUE, BitMapBackend, ChartBuilder, GREEN, IntoDrawingArea, LineSeries, MAGENTA, RED, RGBColor,
    WHITE,
};

use gpt2_small_sae::cli::Args;
use gpt2_small_sae::error::Error;

/// A single deserialized metrics record from the JSONL file.
#[derive(serde::Deserialize)]
struct MetricsRecord {
    step: u64,
    l0: f64,
    mse: f64,
    var_explained: f64,
    dead_fraction: f64,
}

/// CLI-configurable plot options.
struct PlotOpts {
    metrics_path: String,
    output_path: String,
    width: u32,
    height: u32,
}

fn parse_plot_opts() -> Result<PlotOpts, Error> {
    let args = Args::parse();
    Ok(PlotOpts {
        metrics_path: args
            .get("metrics")
            .or_else(|| args.positional(0))
            .map_or_else(|| "metrics.jsonl".to_string(), String::from),
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
    color: RGBColor,
}

/// Draw a 2x2 panel grid of training curves to a PNG file.
#[allow(clippy::cast_precision_loss)]
fn draw_panels(records: &[MetricsRecord], opts: &PlotOpts) -> Result<(), Error> {
    let panels = [
        PanelConfig {
            title: "MSE",
            extract: (|r| r.mse) as fn(&MetricsRecord) -> f64,
            color: RED,
        },
        PanelConfig {
            title: "L0",
            extract: (|r| r.l0) as fn(&MetricsRecord) -> f64,
            color: BLUE,
        },
        PanelConfig {
            title: "Variance Explained",
            extract: (|r| r.var_explained) as fn(&MetricsRecord) -> f64,
            color: GREEN,
        },
        PanelConfig {
            title: "Dead Fraction",
            extract: (|r| r.dead_fraction) as fn(&MetricsRecord) -> f64,
            color: MAGENTA,
        },
    ];

    let x_max = records.last().map_or(1.0, |r| (r.step as f64).max(1.0));

    let root = BitMapBackend::new(&opts.output_path, (opts.width, opts.height)).into_drawing_area();
    root.fill(&WHITE).map_err(plot_err)?;

    let areas = root.split_evenly((2, 2));

    areas
        .iter()
        .zip(panels.iter())
        .try_for_each(|(area, panel)| {
            let (y_min, y_max) = records.iter().fold((f64::MAX, f64::MIN), |(lo, hi), r| {
                let v = (panel.extract)(r);
                (lo.min(v), hi.max(v))
            });
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
                    chart
                        .draw_series(LineSeries::new(
                            records.iter().map(|r| (r.step as f64, (panel.extract)(r))),
                            &panel.color,
                        ))
                        .map_err(plot_err)?;
                    Ok(())
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
                let content =
                    std::fs::read_to_string(&opts.metrics_path).map_err(|e| Error::Boundary {
                        reason: format!("failed to read metrics file {}: {e}", opts.metrics_path),
                    })?;
                let records: Vec<MetricsRecord> = content
                    .lines()
                    .filter(|line| !line.is_empty())
                    .map(|line| serde_json::from_str(line).map_err(Error::from))
                    .collect::<Result<_, _>>()?;

                if records.is_empty() {
                    Err(Error::Boundary {
                        reason: format!("no metrics records found in {}", opts.metrics_path),
                    })
                } else {
                    eprintln!(
                        "plotting {} steps from {}...",
                        records.len(),
                        opts.metrics_path,
                    );
                    draw_panels(&records, &opts)?;
                    eprintln!("saved plot to {}", opts.output_path);
                    Ok(())
                }
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
