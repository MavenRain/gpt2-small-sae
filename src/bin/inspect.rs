//! Top-k feature inspection: for each SAE feature, displays the tokens
//! that produce the highest activations alongside surrounding context.
//!
//! # Usage
//!
//! ```text
//! # Inspect with a text corpus (recommended):
//! cargo run --release --bin inspect -- --checkpoint sae.safetensors --corpus corpus.txt
//!
//! # Inspect with random tokens:
//! cargo run --release --bin inspect
//!
//! # Write structured JSON dashboard:
//! cargo run --release --bin inspect -- --corpus corpus.txt --json dashboard.json
//! ```

use std::collections::BinaryHeap;

use candle_core::{DType, Tensor};
use comp_cat_rs::effect::io::Io;

use gpt2_small_sae::activations::activation_stream;
use gpt2_small_sae::cli::Args;
use gpt2_small_sae::config::{BatchSize, ContextLength, LayerIndex, ModelDim, SaeDim};
use gpt2_small_sae::dataset::tokenize_corpus;
use gpt2_small_sae::error::{Error, TokenizerError};
use gpt2_small_sae::gpt2::Gpt2;
use gpt2_small_sae::io_boundary;
use gpt2_small_sae::sae::Sae;

const GPT2_DEPTH: usize = 12;
const VOCAB_UPPER: f32 = 50256.0;

/// CLI-configurable inspection options.
#[derive(Clone)]
struct InspectOpts {
    layer: usize,
    expansion: usize,
    batch_size: usize,
    ctx_len: usize,
    num_batches: usize,
    top_features: usize,
    top_k: usize,
    context_window: usize,
    corpus: Option<std::path::PathBuf>,
    checkpoint: String,
    json: Option<String>,
}

fn parse_inspect_opts() -> Result<InspectOpts, Error> {
    let args = Args::parse();
    Ok(InspectOpts {
        layer: args.get_or("layer", 8_usize)?,
        expansion: args.get_or("expansion", 8_usize)?,
        batch_size: args.get_or("batch-size", 4_usize)?,
        ctx_len: args.get_or("ctx-len", 128_usize)?,
        num_batches: args.get_or("batches", 8_usize)?,
        top_features: args.get_or("top-features", 20_usize)?,
        top_k: args.get_or("top-k", 10_usize)?,
        context_window: args.get_or("context-window", 5_usize)?,
        corpus: args
            .get("corpus")
            .or_else(|| args.positional(1))
            .map(std::path::PathBuf::from),
        checkpoint: args
            .get("checkpoint")
            .or_else(|| args.positional(0))
            .map_or_else(|| "sae_checkpoint.safetensors".to_string(), String::from),
        json: args.get("json").map(String::from),
    })
}

// -----------------------------------------------------------------------
// Main Io program
// -----------------------------------------------------------------------

#[allow(clippy::too_many_lines)]
fn inspect_program() -> Io<Error, ()> {
    parse_inspect_opts().map_or_else(
        |e| Io::suspend(move || Err(e)),
        |opts| {
            io_boundary::acquire_device().flat_map(move |device| {
                io_boundary::download_gpt2_weights().flat_map(move |weights| {
                    let opts = opts.clone();
                    io_boundary::download_tokenizer().flat_map(move |tokenizer| {
                        let opts2 = opts.clone();
                        Io::suspend(move || {
                            let layer_index = LayerIndex::new(opts.layer, GPT2_DEPTH)?;
                            let model_dim = ModelDim::GPT2_SMALL;
                            let sae_dim = SaeDim::from_expansion(model_dim, opts.expansion)?;

                            eprintln!("=== SAE feature inspection ===");
                            eprintln!("checkpoint: {}", opts.checkpoint);
                            eprintln!(
                                "layer: {}, expansion: {}x, \
                                 top {} features x {} tokens",
                                opts.layer, opts.expansion, opts.top_features, opts.top_k,
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
                                    let result =
                                        tokenize_corpus(&text, &tokenizer, bs, cl, &device)?;
                                    if result.is_empty() {
                                        Err(Error::Boundary {
                                            reason: format!(
                                                "corpus too short ({} tokens needed)",
                                                opts.batch_size * opts.ctx_len
                                            ),
                                        })
                                    } else {
                                        Ok(result)
                                    }
                                },
                            )?;

                            // Extract token IDs to CPU before batches are consumed.
                            let n_per_batch = opts.batch_size * opts.ctx_len;
                            let all_token_ids: Vec<u32> = batches
                                .iter()
                                .map(|b| {
                                    b.reshape(n_per_batch)
                                        .and_then(|t| t.to_vec1::<u32>())
                                        .map_err(Error::from)
                                })
                                .collect::<Result<Vec<Vec<u32>>, Error>>()?
                                .into_iter()
                                .flatten()
                                .collect();

                            eprintln!("loading SAE from {}...", opts.checkpoint);
                            let sae = Sae::from_safetensors(
                                std::path::Path::new(&opts.checkpoint),
                                model_dim,
                                sae_dim,
                                &device,
                            )?;

                            eprintln!("loading GPT-2 small (layers 0..{})...", opts.layer);
                            let gpt2 = Gpt2::from_bytes(weights, layer_index, &device)?;

                            Ok((sae, gpt2, batches, all_token_ids, tokenizer, sae_dim))
                        })
                        .flat_map(
                            move |(sae, gpt2, batches, all_token_ids, tokenizer, sae_dim)| {
                                eprintln!("collecting activations...");
                                activation_stream(gpt2, batches).collect().flat_map(
                                    move |activations| {
                                        Io::suspend(move || {
                                            inspect_features(
                                                &sae,
                                                &activations,
                                                &all_token_ids,
                                                &tokenizer,
                                                sae_dim.as_usize(),
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

// -----------------------------------------------------------------------
// Feature inspection
// -----------------------------------------------------------------------

/// Per-feature summary: (index, max activation, fire count, top-k hits).
type FeatureSummary = (usize, f32, usize, Vec<(f32, usize)>);

#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
fn inspect_features(
    sae: &Sae,
    activations: &[Tensor],
    all_token_ids: &[u32],
    tokenizer: &tokenizers::Tokenizer,
    d_sae: usize,
    opts: &InspectOpts,
) -> Result<(), Error> {
    let total_tokens = all_token_ids.len();
    eprintln!(
        "scanning {} tokens across {} activation batches...\n",
        total_tokens,
        activations.len()
    );

    // Single pass: collect all nonzero (activation, global_pos) per feature.
    let (per_feature, _): (Vec<Vec<(f32, usize)>>, usize) = activations.iter().try_fold(
        ((0..d_sae).map(|_| Vec::new()).collect::<Vec<_>>(), 0_usize),
        |(per_feat, offset), batch| {
            let fwd = sae.forward(batch)?;
            let feat = fwd.features();
            let n_tok: usize = feat.dims().iter().rev().skip(1).product();
            let data: Vec<f32> = feat.reshape(n_tok * d_sae)?.to_vec1()?;

            let updated: Vec<Vec<(f32, usize)>> = per_feat
                .into_iter()
                .enumerate()
                .map(|(j, existing)| {
                    let new_hits: Vec<(f32, usize)> = (0..n_tok)
                        .filter_map(|t| {
                            data.get(t * d_sae + j)
                                .copied()
                                .filter(|&a| a > 0.0)
                                .map(|a| (a, offset + t))
                        })
                        .collect();
                    existing.into_iter().chain(new_hits).collect()
                })
                .collect();

            Ok::<_, Error>((updated, offset + n_tok))
        },
    )?;

    // Per-feature summaries: sort hits descending, take top-K, record stats.
    let summaries: Vec<FeatureSummary> = per_feature
        .into_iter()
        .enumerate()
        .filter_map(|(j, hits)| {
            if hits.is_empty() {
                None
            } else {
                let fire_count = hits.len();
                let top: Vec<(f32, usize)> = BinaryHeap::from(
                    hits.into_iter()
                        .map(|(a, pos)| (a.to_bits(), pos))
                        .collect::<Vec<_>>(),
                )
                .into_sorted_vec()
                .into_iter()
                .rev()
                .take(opts.top_k)
                .map(|(bits, pos)| (f32::from_bits(bits), pos))
                .collect();
                let max_act = top.first().map_or(0.0, |&(a, _)| a);
                Some((j, max_act, fire_count, top))
            }
        })
        .collect();

    let n_active = summaries.len();

    // Rank features by max activation, select top N.
    let ranking: Vec<(u32, usize)> = summaries
        .iter()
        .enumerate()
        .map(|(i, &(_, max_act, _, _))| (max_act.to_bits(), i))
        .collect();
    let top_indices: Vec<usize> = BinaryHeap::from(ranking)
        .into_sorted_vec()
        .into_iter()
        .rev()
        .take(opts.top_features)
        .map(|(_, i)| i)
        .collect();

    eprintln!(
        "{n_active}/{d_sae} features fired at least once ({} dead)\n",
        d_sae - n_active
    );

    // Display each selected feature's top-K token hits with context.
    top_indices.iter().try_for_each(|&summary_idx| {
        summaries.get(summary_idx).map_or(
            Ok::<_, Error>(()),
            |(feat_idx, max_act, fire_count, top_hits)| {
                eprintln!(
                    "=== feature {feat_idx} (max: {max_act:.4}, \
                     fired: {fire_count}/{total_tokens} tokens) ==="
                );
                top_hits.iter().try_for_each(|&(act, global_pos)| {
                    let context = decode_context(
                        all_token_ids,
                        global_pos,
                        tokenizer,
                        opts.ctx_len,
                        opts.context_window,
                    )?;
                    eprintln!("  [{act:.4}] {context}");
                    Ok::<_, Error>(())
                })?;
                eprintln!();
                Ok(())
            },
        )
    })?;

    // Optionally write a structured JSON dashboard.
    opts.json.as_ref().map_or(Ok(()), |path| {
        write_dashboard(
            path,
            &summaries,
            &top_indices,
            all_token_ids,
            tokenizer,
            total_tokens,
            d_sae,
            n_active,
            opts,
        )
    })
}

// -----------------------------------------------------------------------
// JSON dashboard
// -----------------------------------------------------------------------

/// Write per-feature summaries as structured JSON.
#[allow(clippy::cast_precision_loss, clippy::too_many_arguments)]
fn write_dashboard(
    path: &str,
    summaries: &[FeatureSummary],
    top_indices: &[usize],
    all_token_ids: &[u32],
    tokenizer: &tokenizers::Tokenizer,
    total_tokens: usize,
    d_sae: usize,
    n_active: usize,
    opts: &InspectOpts,
) -> Result<(), Error> {
    let features_json: Vec<serde_json::Value> = top_indices
        .iter()
        .filter_map(|&idx| {
            summaries
                .get(idx)
                .map(|(feat_idx, max_act, fire_count, top_hits)| {
                    let tokens_json: Vec<serde_json::Value> = top_hits
                        .iter()
                        .filter_map(|&(act, pos)| {
                            decode_context(
                                all_token_ids,
                                pos,
                                tokenizer,
                                opts.ctx_len,
                                opts.context_window,
                            )
                            .ok()
                            .map(|ctx| {
                                serde_json::json!({
                                    "activation": act,
                                    "position": pos,
                                    "context": ctx,
                                })
                            })
                        })
                        .collect();
                    serde_json::json!({
                        "index": feat_idx,
                        "max_activation": max_act,
                        "fire_count": fire_count,
                        "fire_rate": *fire_count as f64 / total_tokens.max(1) as f64,
                        "top_tokens": tokens_json,
                    })
                })
        })
        .collect();

    let dashboard = serde_json::json!({
        "checkpoint": opts.checkpoint,
        "layer": opts.layer,
        "expansion": opts.expansion,
        "total_tokens": total_tokens,
        "active_features": n_active,
        "total_features": d_sae,
        "dead_features": d_sae - n_active,
        "features": features_json,
    });

    let formatted = serde_json::to_string_pretty(&dashboard)?;
    std::fs::write(path, formatted)?;
    eprintln!("dashboard written to {path}");
    Ok(())
}

// -----------------------------------------------------------------------
// Context decoding
// -----------------------------------------------------------------------

/// Decode a token at `center` with surrounding context, keeping within
/// the same sequence boundary.  The center token is shown in brackets:
/// `...before[center]after...`.
fn decode_context(
    all_ids: &[u32],
    center: usize,
    tokenizer: &tokenizers::Tokenizer,
    ctx_len: usize,
    context_window: usize,
) -> Result<String, Error> {
    let seq_start = (center / ctx_len) * ctx_len;
    let seq_end = (seq_start + ctx_len).min(all_ids.len());
    let start = center.saturating_sub(context_window).max(seq_start);
    let end = (center + context_window + 1).min(seq_end);

    let before = all_ids.get(start..center).ok_or_else(|| Error::Train {
        reason: "context before range out of bounds".into(),
    })?;
    let center_id = all_ids.get(center).copied().ok_or_else(|| Error::Train {
        reason: "center token index out of bounds".into(),
    })?;
    let after_start = center + 1;
    let after = all_ids.get(after_start..end).unwrap_or(&[]);

    let decode = |ids: &[u32]| -> Result<String, Error> {
        tokenizer
            .decode(ids, false)
            .map_err(|e| Error::Tokenizer(TokenizerError::new(e)))
    };

    let before_text = decode(before)?;
    let center_text = decode(&[center_id])?;
    let after_text = decode(after)?;

    let prefix = if start > seq_start { "..." } else { "" };
    let suffix = if end < seq_end { "..." } else { "" };

    Ok(format!(
        "{prefix}{before_text}[{center_text}]{after_text}{suffix}"
    ))
}

fn main() {
    inspect_program().run().unwrap_or_else(|e| {
        eprintln!("inspection failed: {e}");
        std::process::exit(1);
    });
}
