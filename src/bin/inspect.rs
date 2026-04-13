//! Top-k feature inspection: for each SAE feature, displays the tokens
//! that produce the highest activations alongside surrounding context.
//!
//! # Usage
//!
//! ```text
//! # Inspect with a text corpus (recommended):
//! cargo run --release --bin inspect -- sae_checkpoint.safetensors corpus.txt
//!
//! # Inspect with random tokens (default checkpoint):
//! cargo run --release --bin inspect
//! ```

use std::collections::BinaryHeap;

use candle_core::{DType, Tensor};
use comp_cat_rs::effect::io::Io;

use gpt2_small_sae::activations::activation_stream;
use gpt2_small_sae::config::{BatchSize, ContextLength, LayerIndex, ModelDim, SaeDim};
use gpt2_small_sae::dataset::tokenize_corpus;
use gpt2_small_sae::error::{Error, TokenizerError};
use gpt2_small_sae::gpt2::Gpt2;
use gpt2_small_sae::io_boundary;
use gpt2_small_sae::sae::Sae;

const LAYER: usize = 8;
const GPT2_DEPTH: usize = 12;
const EXPANSION: usize = 8;
const BATCH_SIZE: usize = 4;
const CTX_LEN: usize = 128;
const NUM_RANDOM_BATCHES: usize = 8;
const TOP_FEATURES: usize = 20;
const TOP_K: usize = 10;
const CONTEXT_WINDOW: usize = 5;
const VOCAB_UPPER: f32 = 50256.0;

// -----------------------------------------------------------------------
// Main Io program
// -----------------------------------------------------------------------

#[allow(clippy::too_many_lines)]
fn inspect_program() -> Io<Error, ()> {
    let checkpoint_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "sae_checkpoint.safetensors".to_string());
    let maybe_corpus = std::env::args().nth(2).map(std::path::PathBuf::from);

    io_boundary::acquire_device().flat_map(move |device| {
        io_boundary::download_gpt2_weights().flat_map(move |weights| {
            io_boundary::download_tokenizer().flat_map(move |tokenizer| {
                Io::suspend(move || {
                    let layer_index = LayerIndex::new(LAYER, GPT2_DEPTH)?;
                    let model_dim = ModelDim::GPT2_SMALL;
                    let sae_dim = SaeDim::from_expansion(model_dim, EXPANSION)?;

                    eprintln!("=== SAE feature inspection ===");
                    eprintln!("checkpoint: {checkpoint_path}");
                    eprintln!(
                        "layer: {LAYER}, expansion: {EXPANSION}x, \
                         top {TOP_FEATURES} features x {TOP_K} tokens"
                    );
                    eprintln!();

                    let batches = maybe_corpus.as_deref().map_or_else(
                        || {
                            eprintln!(
                                "generating {NUM_RANDOM_BATCHES} random token batches \
                                 ({BATCH_SIZE} x {CTX_LEN})..."
                            );
                            (0..NUM_RANDOM_BATCHES)
                                .map(|_| {
                                    Tensor::rand(
                                        0.0f32,
                                        VOCAB_UPPER,
                                        (BATCH_SIZE, CTX_LEN),
                                        &device,
                                    )
                                    .and_then(|t| t.to_dtype(DType::U32))
                                    .map_err(Error::from)
                                })
                                .collect::<Result<Vec<_>, _>>()
                        },
                        |path| {
                            eprintln!("tokenizing corpus: {}...", path.display());
                            let text =
                                std::fs::read_to_string(path).map_err(|e| Error::Boundary {
                                    reason: format!(
                                        "failed to read corpus {}: {e}",
                                        path.display()
                                    ),
                                })?;
                            let batch_size = BatchSize::new(BATCH_SIZE)?;
                            let ctx_len = ContextLength::new(CTX_LEN)?;
                            let result =
                                tokenize_corpus(&text, &tokenizer, batch_size, ctx_len, &device)?;
                            if result.is_empty() {
                                Err(Error::Boundary {
                                    reason: format!(
                                        "corpus too short ({} tokens needed)",
                                        BATCH_SIZE * CTX_LEN
                                    ),
                                })
                            } else {
                                Ok(result)
                            }
                        },
                    )?;

                    // Extract token IDs to CPU before batches are consumed.
                    let n_per_batch = BATCH_SIZE * CTX_LEN;
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

                    eprintln!("loading SAE from {checkpoint_path}...");
                    let sae = Sae::from_safetensors(
                        std::path::Path::new(&checkpoint_path),
                        model_dim,
                        sae_dim,
                        &device,
                    )?;

                    eprintln!("loading GPT-2 small (layers 0..{LAYER})...");
                    let gpt2 = Gpt2::from_bytes(weights, layer_index, &device)?;

                    Ok((sae, gpt2, batches, all_token_ids, tokenizer, sae_dim))
                })
                .flat_map(
                    |(sae, gpt2, batches, all_token_ids, tokenizer, sae_dim)| {
                        eprintln!("collecting activations...");
                        activation_stream(gpt2, batches)
                            .collect()
                            .flat_map(move |activations| {
                                Io::suspend(move || {
                                    inspect_features(
                                        &sae,
                                        &activations,
                                        &all_token_ids,
                                        &tokenizer,
                                        sae_dim.as_usize(),
                                    )
                                })
                            })
                    },
                )
            })
        })
    })
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
                .take(TOP_K)
                .map(|(bits, pos)| (f32::from_bits(bits), pos))
                .collect();
                let max_act = top.first().map_or(0.0, |&(a, _)| a);
                Some((j, max_act, fire_count, top))
            }
        })
        .collect();

    let n_active = summaries.len();

    // Rank features by max activation, select top TOP_FEATURES.
    let ranking: Vec<(u32, usize)> = summaries
        .iter()
        .enumerate()
        .map(|(i, &(_, max_act, _, _))| (max_act.to_bits(), i))
        .collect();
    let top_indices: Vec<usize> = BinaryHeap::from(ranking)
        .into_sorted_vec()
        .into_iter()
        .rev()
        .take(TOP_FEATURES)
        .map(|(_, i)| i)
        .collect();

    eprintln!(
        "{n_active}/{d_sae} features fired at least once ({} dead)\n",
        d_sae - n_active
    );

    // Display each selected feature's top-K token hits with context.
    top_indices.iter().try_for_each(|&summary_idx| {
        summaries
            .get(summary_idx)
            .map_or(Ok(()), |(feat_idx, max_act, fire_count, top_hits)| {
                eprintln!(
                    "=== feature {feat_idx} (max: {max_act:.4}, \
                     fired: {fire_count}/{total_tokens} tokens) ==="
                );
                top_hits.iter().try_for_each(|&(act, global_pos)| {
                    let context = decode_context(all_token_ids, global_pos, tokenizer)?;
                    eprintln!("  [{act:.4}] {context}");
                    Ok::<_, Error>(())
                })?;
                eprintln!();
                Ok(())
            })
    })
}

// -----------------------------------------------------------------------
// Context decoding
// -----------------------------------------------------------------------

/// Decode a token at `center` with surrounding context, keeping within
/// the same sequence (`CTX_LEN` boundary).  The center token is shown
/// in brackets: `...before[center]after...`.
fn decode_context(
    all_ids: &[u32],
    center: usize,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<String, Error> {
    let seq_start = (center / CTX_LEN) * CTX_LEN;
    let seq_end = (seq_start + CTX_LEN).min(all_ids.len());
    let start = center.saturating_sub(CONTEXT_WINDOW).max(seq_start);
    let end = (center + CONTEXT_WINDOW + 1).min(seq_end);

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
