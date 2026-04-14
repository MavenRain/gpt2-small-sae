//! Feature co-activation analysis: pairwise firing-overlap statistics.
//!
//! Complements [`feature_corr`], which measures similarity in the SAE's
//! decoder space.  Two decoder directions can be near-orthogonal yet
//! still fire on the same tokens, in which case the SAE is effectively
//! tracking one concept with two features.  `feature_corr` cannot see
//! that; this binary can.
//!
//! For every pair `(i, j)` of features we accumulate:
//!
//! - `active[i]`     -- number of tokens where feature `i` fires
//!   (`feature > 0`).
//! - `co_active[i, j]` -- number of tokens where `i` and `j` both fire.
//!
//! From those we derive the Jaccard overlap
//! `J[i, j] = co_active[i, j] / (active[i] + active[j] - co_active[i, j])`
//! and mask the diagonal.  For each feature we then find its single
//! nearest co-activation partner (mirroring [`feature_corr`]'s per-row
//! ranking) and report the top-K features by partner Jaccard.  Each
//! row also includes the decoder-cosine similarity between the pair,
//! so a reader can scan for the interesting diagnostic: **high
//! co-activation alongside near-zero decoder cosine** indicates an
//! orthogonal feature split the decoder matrix alone cannot reveal.
//!
//! # Usage
//!
//! ```text
//! # Default layer/expansion, random tokens:
//! cargo run --release --bin feature_coact -- --checkpoint sae_checkpoint.safetensors
//!
//! # Real corpus, JSON output:
//! cargo run --release --bin feature_coact -- \
//!   --checkpoint sae_l4_16x.safetensors --layer 4 --expansion 16 \
//!   --corpus corpus.txt --top-k 30 --output coact.json
//! ```

use std::collections::BinaryHeap;

use candle_core::{D, DType, Device, Tensor};
use comp_cat_rs::effect::io::Io;

use gpt2_small_sae::activations::activation_stream;
use gpt2_small_sae::cli::Args;
use gpt2_small_sae::config::{LayerIndex, ModelDim, SaeDim};
use gpt2_small_sae::error::Error;
use gpt2_small_sae::eval_opts::{SharedEvalOpts, build_batches};
use gpt2_small_sae::gpt2::Gpt2;
use gpt2_small_sae::io_boundary;
use gpt2_small_sae::sae::Sae;

const GPT2_DEPTH: usize = 12;
/// Mask scalar subtracted from the Jaccard diagonal so that per-row
/// argmax always picks a non-self partner.  Any value well above 1.0
/// works; 1000 matches the convention in `feature_corr`.
const DIAG_MASK: f64 = 1000.0;

// -----------------------------------------------------------------------
// CLI parsing
// -----------------------------------------------------------------------

/// CLI-configurable options for the co-activation binary.
#[derive(Clone)]
struct CoactOpts {
    shared: SharedEvalOpts,
    top_k: usize,
}

fn parse_coact_opts() -> Result<CoactOpts, Error> {
    let args = Args::parse();
    Ok(CoactOpts {
        shared: SharedEvalOpts::parse(&args)?,
        top_k: args.get_or("top-k", 20_usize)?,
    })
}

// -----------------------------------------------------------------------
// Accumulation
// -----------------------------------------------------------------------

/// Running totals of firing counts and pairwise co-activation over an
/// activation batch stream.
struct CoActAccum {
    active: Tensor,
    co_active: Tensor,
    tokens: u64,
}

/// Fold an activation stream into `(active, co_active, tokens)`.
///
/// - `active` is a `(sae_dim,)` `f32` tensor of per-feature firing
///   counts.
/// - `co_active` is a `(sae_dim, sae_dim)` `f32` tensor whose `(i, j)`
///   entry counts tokens where features `i` and `j` both fired.
fn accumulate_coact(
    sae: &Sae,
    activations: &[Tensor],
    init_active: Tensor,
    init_co: Tensor,
) -> Result<CoActAccum, Error> {
    activations.iter().try_fold(
        CoActAccum {
            active: init_active,
            co_active: init_co,
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
            let fires = fwd.features().gt(0.0_f32)?.to_dtype(DType::F32)?;
            let active_batch = fires.sum(0)?;
            let co_batch = fires.t()?.matmul(&fires)?;
            Ok::<_, Error>(CoActAccum {
                active: acc.active.add(&active_batch)?,
                co_active: acc.co_active.add(&co_batch)?,
                tokens: acc.tokens + tokens as u64,
            })
        },
    )
}

// -----------------------------------------------------------------------
// Jaccard matrix and summary statistics
// -----------------------------------------------------------------------

/// Build the `(sae_dim, sae_dim)` Jaccard overlap matrix from the
/// accumulated firing counts.  The diagonal is zeroed so summary
/// statistics only see cross-feature pairs.
fn jaccard_matrix(accum: &CoActAccum, device: &Device, sae_dim: usize) -> Result<Tensor, Error> {
    let active_col = accum.active.reshape((sae_dim, 1))?;
    let active_row = accum.active.reshape((1, sae_dim))?;
    let sum_ab = active_col.broadcast_add(&active_row)?;
    let union = sum_ab.sub(&accum.co_active)?;
    // Epsilon added to the denominator, not the numerator, so that
    // pairs with union=0 (both features dead) give J=0 rather than NaN.
    let union_safe = union.affine(1.0, 1e-9)?;
    let jaccard = accum.co_active.div(&union_safe)?;
    let eye = Tensor::eye(sae_dim, DType::F32, device)?;
    let one_minus_eye = eye.affine(-1.0, 1.0)?;
    jaccard.mul(&one_minus_eye).map_err(Error::from)
}

/// Summary statistics over the off-diagonal Jaccard matrix.
struct CoActStats {
    mean_off_diag: f64,
    max_off_diag: f64,
    overlapping_pairs: u64,
}

const OVERLAP_THRESHOLD: f32 = 0.5;

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn compute_stats(jaccard: &Tensor) -> Result<CoActStats, Error> {
    let n = jaccard.dim(0)?;
    let pair_count = (n * n - n) as f64;
    let sum_off = f64::from(jaccard.sum_all()?.to_scalar::<f32>()?);
    let mean_off_diag = sum_off / pair_count;
    let max_off_diag = f64::from(jaccard.max_all()?.to_scalar::<f32>()?);
    let overlaps = jaccard
        .ge(OVERLAP_THRESHOLD)?
        .to_dtype(DType::F32)?
        .sum_all()?
        .to_scalar::<f32>()?;
    // Each strongly overlapping pair is counted twice (i,j) and (j,i);
    // halve to get the distinct-pair count.  ge(...) emits only 0/1
    // values so the sign-loss and truncation casts below are safe.
    let overlapping_pairs = (f64::from(overlaps) / 2.0).round() as u64;
    Ok(CoActStats {
        mean_off_diag,
        max_off_diag,
        overlapping_pairs,
    })
}

// -----------------------------------------------------------------------
// Top-K nearest co-activation partner
// -----------------------------------------------------------------------

/// A single (feature, partner) co-activation entry for reporting.
struct CoActPair {
    feature: usize,
    partner: usize,
    jaccard: f64,
    co_count: f64,
    active_i: f64,
    active_j: f64,
    decoder_cosine: f64,
}

/// Compute each row's unit-normalized decoder direction for per-pair
/// cosine lookups.  Avoids materializing the full `(sae_dim, sae_dim)`
/// cosine matrix, which is prohibitive for large dictionaries.
fn row_normalize(w_dec: &Tensor) -> Result<Tensor, Error> {
    let norms = w_dec.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    let safe = norms.affine(1.0, 1e-8)?;
    w_dec.broadcast_div(&safe).map_err(Error::from)
}

/// Per-feature nearest co-activation partner, ranked globally by
/// partner Jaccard.  Returns the top `k` entries, each annotated with
/// raw firing counts and the decoder cosine of the pair.
#[allow(clippy::cast_precision_loss)]
fn top_coact_pairs(
    jaccard: &Tensor,
    accum: &CoActAccum,
    w_dec_norm: &Tensor,
    k: usize,
    device: &Device,
) -> Result<Vec<CoActPair>, Error> {
    let n = jaccard.dim(0)?;
    let eye = Tensor::eye(n, DType::F32, device)?;
    let mask = eye.affine(DIAG_MASK, 0.0)?;
    let guarded = jaccard.sub(&mask)?;

    let per_row_max: Vec<f32> = guarded.max(D::Minus1)?.to_vec1()?;
    let per_row_argmax: Vec<u32> = guarded.argmax(D::Minus1)?.to_vec1()?;
    let jaccard_flat: Vec<f32> = jaccard.flatten_all()?.to_vec1()?;
    let active: Vec<f32> = accum.active.to_vec1()?;
    let co_flat: Vec<f32> = accum.co_active.flatten_all()?.to_vec1()?;
    let w_dec_flat: Vec<f32> = w_dec_norm.flatten_all()?.to_vec1()?;
    let model_dim = w_dec_norm.dim(1)?;

    let ranking: Vec<(u32, usize)> = per_row_max
        .iter()
        .enumerate()
        .map(|(i, &s)| (s.max(0.0).to_bits(), i))
        .collect();

    let top: Vec<CoActPair> = BinaryHeap::from(ranking)
        .into_sorted_vec()
        .into_iter()
        .rev()
        .take(k)
        .filter_map(|(_, i)| {
            let j = per_row_argmax.get(i).map(|&x| x as usize)?;
            let jacc = f64::from(jaccard_flat.get(i * n + j).copied()?);
            let co_count = f64::from(co_flat.get(i * n + j).copied()?);
            let active_i = f64::from(active.get(i).copied()?);
            let active_j = f64::from(active.get(j).copied()?);
            let row_i = w_dec_flat.get(i * model_dim..(i + 1) * model_dim)?;
            let row_j = w_dec_flat.get(j * model_dim..(j + 1) * model_dim)?;
            let dec_cos: f32 = row_i.iter().zip(row_j).map(|(a, b)| a * b).sum();
            Some(CoActPair {
                feature: i,
                partner: j,
                jaccard: jacc,
                co_count,
                active_i,
                active_j,
                decoder_cosine: f64::from(dec_cos),
            })
        })
        .collect();
    Ok(top)
}

// -----------------------------------------------------------------------
// Reporting
// -----------------------------------------------------------------------

#[allow(clippy::cast_precision_loss)]
fn report(
    stats: &CoActStats,
    pairs: &[CoActPair],
    opts: &CoactOpts,
    sae_dim: usize,
    token_count: u64,
) -> Result<(), Error> {
    eprintln!("=== feature co-activation analysis ===");
    eprintln!("checkpoint: {}", opts.shared.checkpoint());
    eprintln!("sae_dim: {sae_dim}");
    eprintln!("tokens: {token_count}");
    eprintln!();
    eprintln!("  mean off-diagonal Jaccard:   {:.6}", stats.mean_off_diag);
    eprintln!("  max  off-diagonal Jaccard:   {:.6}", stats.max_off_diag);
    eprintln!(
        "  pairs with Jaccard >= {:.2}: {}",
        OVERLAP_THRESHOLD, stats.overlapping_pairs
    );
    eprintln!();
    eprintln!("top {} nearest co-activation partners:", pairs.len());
    eprintln!(
        "  {:>6}  {:>7}  {:>8}  {:>10}  {:>10}  {:>8}",
        "feat", "partner", "jaccard", "co_count", "min_active", "dec_cos"
    );
    pairs.iter().try_for_each(|p| {
        let min_active = p.active_i.min(p.active_j);
        eprintln!(
            "  {:>6}  {:>7}  {:>8.4}  {:>10.0}  {:>10.0}  {:>+8.4}",
            p.feature, p.partner, p.jaccard, p.co_count, min_active, p.decoder_cosine
        );
        Ok::<_, Error>(())
    })?;

    opts.shared.output().map_or(Ok(()), |path| {
        let pairs_json: Vec<_> = pairs
            .iter()
            .map(|p| {
                serde_json::json!({
                    "feature": p.feature,
                    "partner": p.partner,
                    "jaccard": p.jaccard,
                    "co_count": p.co_count,
                    "active_feature": p.active_i,
                    "active_partner": p.active_j,
                    "decoder_cosine": p.decoder_cosine,
                })
            })
            .collect();
        let record = serde_json::json!({
            "checkpoint": opts.shared.checkpoint(),
            "layer": opts.shared.layer(),
            "expansion": opts.shared.expansion(),
            "sae_dim": sae_dim,
            "tokens": token_count,
            "mean_off_diag_jaccard": stats.mean_off_diag,
            "max_off_diag_jaccard": stats.max_off_diag,
            "overlap_threshold": OVERLAP_THRESHOLD,
            "overlapping_pairs": stats.overlapping_pairs,
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

fn finalize(
    sae: &Sae,
    activations: &[Tensor],
    init_active: Tensor,
    init_co: Tensor,
    sae_dim: SaeDim,
    device: &Device,
    opts: &CoactOpts,
) -> Result<(), Error> {
    let accum = accumulate_coact(sae, activations, init_active, init_co)?;
    let jaccard = jaccard_matrix(&accum, device, sae_dim.as_usize())?;
    let stats = compute_stats(&jaccard)?;
    let w_dec_norm = row_normalize(sae.w_dec())?;
    let pairs = top_coact_pairs(&jaccard, &accum, &w_dec_norm, opts.top_k, device)?;
    report(&stats, &pairs, opts, sae_dim.as_usize(), accum.tokens)
}

fn feature_coact_program() -> Io<Error, ()> {
    parse_coact_opts().map_or_else(
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
                        let opts_setup = opts.clone();
                        let opts_finalize = opts.clone();
                        let device_for_setup = device.clone();
                        let device_for_finalize = device.clone();
                        Io::suspend(move || {
                            let layer_index =
                                LayerIndex::new(opts_setup.shared.layer(), GPT2_DEPTH)?;
                            let model_dim = ModelDim::GPT2_SMALL;
                            let sae_dim =
                                SaeDim::from_expansion(model_dim, opts_setup.shared.expansion())?;

                            eprintln!("=== feature co-activation analysis ===");
                            eprintln!("checkpoint: {}", opts_setup.shared.checkpoint());
                            eprintln!(
                                "layer: {}, expansion: {}x",
                                opts_setup.shared.layer(),
                                opts_setup.shared.expansion()
                            );

                            let batches = build_batches(
                                &opts_setup.shared,
                                maybe_tokenizer.as_ref(),
                                &device_for_setup,
                            )?;
                            eprintln!();

                            eprintln!("loading SAE from {}...", opts_setup.shared.checkpoint());
                            let sae = Sae::from_safetensors(
                                std::path::Path::new(opts_setup.shared.checkpoint()),
                                model_dim,
                                sae_dim,
                                &device_for_setup,
                            )?;

                            eprintln!(
                                "loading GPT-2 small (layers 0..{})...",
                                opts_setup.shared.layer()
                            );
                            let gpt2 = Gpt2::from_bytes(weights, layer_index, &device_for_setup)?;

                            let init_active = Tensor::zeros(
                                (sae_dim.as_usize(),),
                                DType::F32,
                                &device_for_setup,
                            )?;
                            let init_co = Tensor::zeros(
                                (sae_dim.as_usize(), sae_dim.as_usize()),
                                DType::F32,
                                &device_for_setup,
                            )?;
                            Ok((sae, gpt2, batches, sae_dim, init_active, init_co))
                        })
                        .flat_map(
                            move |(sae, gpt2, batches, sae_dim, init_active, init_co)| {
                                eprintln!("collecting activations...");
                                activation_stream(gpt2, batches).collect().flat_map(
                                    move |activations| {
                                        Io::suspend(move || {
                                            finalize(
                                                &sae,
                                                &activations,
                                                init_active,
                                                init_co,
                                                sae_dim,
                                                &device_for_finalize,
                                                &opts_finalize,
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

fn main() {
    feature_coact_program().run().unwrap_or_else(|e| {
        eprintln!("feature co-activation analysis failed: {e}");
        std::process::exit(1);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{VarBuilder, VarMap};
    use gpt2_small_sae::config::{ModelDim, SaeDim};

    fn cpu() -> Device {
        Device::Cpu
    }

    /// Hand-built 3-feature, 4-token firing fixture used by the Jaccard
    /// and stats tests.
    ///
    ///  tok0: f0, f1 fire
    ///  tok1: f0, f1 fire
    ///  tok2: f0      fires
    ///  tok3: f2      fires
    ///
    /// `active   = [3, 2, 1]`
    /// `co_active = [[3,2,0], [2,2,0], [0,0,1]]`
    ///
    /// Derived Jaccards: `J[0,1] = 2/3`, `J[0,2] = J[1,2] = 0`.
    fn hand_built_accum(device: &Device) -> Result<CoActAccum, Error> {
        let active = Tensor::from_slice(&[3.0_f32, 2.0, 1.0], (3,), device)?;
        let co_active = Tensor::from_slice(
            &[3.0_f32, 2.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0],
            (3, 3),
            device,
        )?;
        Ok(CoActAccum {
            active,
            co_active,
            tokens: 4,
        })
    }

    #[test]
    fn jaccard_matches_hand_computed_values() -> Result<(), Error> {
        let device = cpu();
        let accum = hand_built_accum(&device)?;
        let jaccard = jaccard_matrix(&accum, &device, 3)?;
        let rows: Vec<Vec<f32>> = jaccard.to_vec2()?;
        let get = |i: usize, j: usize| {
            rows.get(i)
                .and_then(|r| r.get(j).copied())
                .unwrap_or(f32::NAN)
        };
        // Diagonal zeroed.
        assert!(get(0, 0).abs() < 1e-5);
        assert!(get(1, 1).abs() < 1e-5);
        assert!(get(2, 2).abs() < 1e-5);
        // Symmetric J[0,1] = J[1,0] ~= 2/3.
        assert!((get(0, 1) - 2.0 / 3.0).abs() < 1e-4);
        assert!((get(1, 0) - 2.0 / 3.0).abs() < 1e-4);
        // Disjoint pairs zero.
        assert!(get(0, 2).abs() < 1e-5);
        assert!(get(1, 2).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn stats_report_expected_mean_and_max() -> Result<(), Error> {
        let device = cpu();
        let accum = hand_built_accum(&device)?;
        let jaccard = jaccard_matrix(&accum, &device, 3)?;
        let stats = compute_stats(&jaccard)?;
        assert!((stats.max_off_diag - 2.0 / 3.0).abs() < 1e-4);
        // Mean over 6 off-diagonal entries: (2 * (2/3) + 4 * 0) / 6 = 2/9.
        assert!((stats.mean_off_diag - 2.0 / 9.0).abs() < 1e-4);
        Ok(())
    }

    #[test]
    fn accumulate_conserves_feature_counts() -> Result<(), Error> {
        // Fold a batch through a fresh tiny SAE and check that the
        // diagonal of `co_active` matches `active` (a feature always
        // co-activates with itself).
        let device = cpu();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model_dim = ModelDim::new(4)?;
        let sae_dim = SaeDim::new(6)?;
        let sae = Sae::new(vb, model_dim, sae_dim)?;
        let batch = Tensor::randn(0.0_f32, 1.0_f32, (4, 4), &device)?;
        let sae_dim_usize = sae_dim.as_usize();
        let init_active = Tensor::zeros((sae_dim_usize,), DType::F32, &device)?;
        let init_co = Tensor::zeros((sae_dim_usize, sae_dim_usize), DType::F32, &device)?;
        let accum = accumulate_coact(&sae, &[batch], init_active, init_co)?;
        let active: Vec<f32> = accum.active.to_vec1()?;
        let co_rows: Vec<Vec<f32>> = accum.co_active.to_vec2()?;
        active.iter().enumerate().try_for_each(|(i, a)| {
            let d = co_rows
                .get(i)
                .and_then(|r| r.get(i).copied())
                .unwrap_or(f32::NAN);
            assert!((a - d).abs() < 1e-4);
            Ok::<_, Error>(())
        })?;
        assert_eq!(accum.tokens, 4);
        Ok(())
    }
}
