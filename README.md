# gpt2-small-sae

Sparse autoencoders on the GPT-2 small residual stream, implemented in Rust with [candle](https://github.com/huggingface/candle) on the Metal backend.

## Overview

This project trains and evaluates sparse autoencoders (SAEs) against the residual stream of the pretrained GPT-2 small model (124M parameters, 768-dimensional residual, 12 layers).  The SAE learns an overcomplete feature dictionary that reconstructs the residual with high fidelity under tight L1 sparsity constraints, enabling mechanistic interpretability of the model's internal representations.

The entire pipeline is pure Rust.  No Python.

The SAE uses the standard encoder-decoder architecture with a tied pre-encoder bias:

```
features = relu((x - b_dec) @ w_enc + b_enc)
recon    = features @ w_dec + b_dec
loss     = mse(recon, x) + l1_coeff * mean(sum(|features|))
```

The GPT-2 forward pass is hand-rolled (no vendored Python model) and loads pretrained weights directly from HuggingFace safetensors.  Only the transformer blocks up to the chosen hookpoint are materialized for training; the full 12-block model is loaded separately for activation-patching evaluation.

## Stack

| Layer | Crate | Role |
|-------|-------|------|
| Tensors + autograd | `candle-core`, `candle-nn` | Metal-accelerated tensor operations |
| Model weights | `safetensors`, `hf-hub` | Weight serialization and download |
| Tokenization | `tokenizers` | GPT-2 BPE tokenizer |
| Effects | `comp-cat-rs` | `Io` at the boundary, `Stream` for activations |
| Plotting | `plotters` | Training curves, Pareto frontiers, heatmaps, histograms |

## Building

Edition 2024, `rust-version = 1.85`.  `comp-cat-rs` 0.5 is pulled from crates.io automatically.

```sh
# Metal (default, M-series Mac):
cargo build --release

# CPU fallback (CI, non-Mac):
cargo build --release --no-default-features --features cpu
```

Both `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check` must pass with zero diagnostics.

## Usage

All binaries accept `--key value` or `--key=value` flags.  Defaults are tuned for GPT-2 small layer 8 with an 8x dictionary.  The most common flags, shared across every evaluation binary, are:

| Flag           | Default                      | Description                    |
|----------------|------------------------------|--------------------------------|
| `--layer`      | `8`                          | GPT-2 hookpoint layer          |
| `--expansion`  | `8`                          | SAE expansion factor           |
| `--batch-size` | `4`                          | Batch size (sequences)         |
| `--ctx-len`    | `128`                        | Context length (tokens)        |
| `--batches`    | `8`                          | Number of eval batches         |
| `--corpus`     | *(none; random tokens)*      | Path to a text corpus file     |
| `--checkpoint` | `sae_checkpoint.safetensors` | Checkpoint path                |
| `--output`     | *(none)*                     | Optional JSON results path     |

### Smoke test

Downloads GPT-2 small from HuggingFace Hub (cached after first run), tokenizes a sentence, runs the forward pass through 8 transformer blocks, and prints the residual stream shape.

```sh
cargo run --release --bin smoke
```

### Training

```sh
# Train on a text corpus (recommended):
cargo run --release --bin train -- --corpus corpus.txt

# Custom hyperparameters:
cargo run --release --bin train -- --corpus corpus.txt --layer 4 --expansion 16 --l1 5e-4

# Resume from a prior checkpoint:
cargo run --release --bin train -- --corpus corpus.txt --resume sae_checkpoint.safetensors
```

Training writes a `metrics.jsonl` log and saves both the `.safetensors` checkpoint and a `.meta.json` sidecar containing final metrics and hyperparameters.  `--save-acts` / `--load-acts` cache the collected activations to disk so subsequent runs skip the GPT-2 forward pass.

### Evaluation

```sh
# Aggregate metrics (MSE, L0, variance explained, dead fraction):
cargo run --release --bin eval_pretrained -- --checkpoint sae.safetensors --corpus corpus.txt

# Activation patching: cross-entropy delta when the SAE replaces the residual:
cargo run --release --bin patch_eval -- --checkpoint sae.safetensors --corpus corpus.txt
```

### Feature inspection and analysis

```sh
# Top-K features by max activation, with decoded token context:
cargo run --release --bin inspect -- --checkpoint sae.safetensors --corpus corpus.txt --json dash.json

# Decoder-space cosine similarity: finds redundant or split features:
cargo run --release --bin feature_corr -- --checkpoint sae.safetensors --heatmap corr.png

# Firing-rate histogram: diagnoses dead and saturated features for L1 tuning:
cargo run --release --bin feature_hist -- --checkpoint sae.safetensors --histogram hist.png

# Co-activation analysis: flags orthogonal features tracking the same concept
# (high Jaccard overlap alongside near-zero decoder cosine):
cargo run --release --bin feature_coact -- --checkpoint sae.safetensors --top-k 30 --output coact.json
```

`feature_corr` measures similarity in decoder space.  `feature_coact` measures overlap in activation space.  A pair that fires together on the same tokens while pointing in orthogonal decoder directions is a split that neither analysis can catch alone, which is why both exist.

### Hyperparameter sweeps

```sh
# Train across multiple GPT-2 layers, with shared activation caching:
cargo run --release --bin sweep -- --corpus corpus.txt --layers 4,6,8,10 --save-acts true

# Sweep L1 coefficients at a fixed layer:
cargo run --release --bin l1_sweep -- --corpus corpus.txt --layer 8 --l1-values 1e-4,5e-4,1e-3,5e-3

# Sweep dictionary expansion factors:
cargo run --release --bin expansion_sweep -- --corpus corpus.txt --layer 8 --expansions 4,8,16,32
```

All three sweeps emit per-run `.safetensors` + `.meta.json` pairs and reuse cached activations across runs.

### Plotting

```sh
# Loss curves (overlays multiple metrics.jsonl files for run comparison):
cargo run --release --bin plot -- --inputs run1.jsonl,run2.jsonl --output loss.png

# MSE-vs-L0 Pareto frontier across a set of checkpoints:
cargo run --release --bin pareto -- --checkpoints l1_1e-4.safetensors,l1_5e-4.safetensors,l1_1e-3.safetensors
```

## Project structure

```
src/
  lib.rs              Module declarations
  error.rs            Crate-wide hand-rolled Error enum
  config.rs           Newtype wrappers for hyperparameters and dimensions
  metrics.rs          Training metrics (L0, MSE, variance explained, dead fraction)
  sae.rs              SAE model: encoder, decoder, forward, loss
  gpt2.rs             Hand-rolled GPT-2 small forward with hookpoint support
  dataset.rs          Text corpus tokenization and chunking
  activations.rs      Lazy activation Stream via comp-cat-rs
  io_boundary.rs      Io programs for downloads, safetensors I/O, device acquisition
  train.rs            Training loop: Adam, cosine LR schedule, dead-feature resampling
  cli.rs              Lightweight --key value argument parser
  eval_opts.rs        Shared eval CLI options and batch-construction helper
  bin/
    smoke.rs          Smoke test
    train.rs          SAE training binary
    eval_pretrained.rs  Aggregate evaluation metrics
    patch_eval.rs     Activation patching cross-entropy delta
    inspect.rs        Top-K feature inspection with auto-generated labels
    feature_corr.rs   Decoder-space cosine similarity + heatmap
    feature_hist.rs   Per-feature firing-rate histogram
    feature_coact.rs  Activation-space Jaccard co-activation analysis
    sweep.rs          Multi-layer training sweep
    l1_sweep.rs       L1 coefficient sweep
    expansion_sweep.rs  Dictionary width sweep
    plot.rs           Loss curve plotter
    pareto.rs         MSE-vs-L0 Pareto frontier plotter
```

## Design decisions

- **Newtypes everywhere.**  `ModelDim`, `SaeDim`, `L1Coefficient`, `LearningRate`, `BatchSize`, `ContextLength`, etc. prevent dimension and coefficient confusion at compile time.
- **Hand-rolled GPT-2.**  `candle-transformers` 0.10 does not ship a GPT-2 module, so the forward pass is implemented directly against `VarBuilder`.  This also gives precise control over the hookpoint and avoids loading unused layers during training.
- **comp-cat-rs effects.**  Side effects (downloads, I/O, device acquisition, training-loop bookkeeping) are wrapped in `Io` programs composed with `flat_map`, and activations flow through a lazy `Stream`.  The `run` catamorphism is called exactly once in each binary's `main`.
- **Private struct fields + accessor methods.**  No `pub` fields cross module boundaries; callers go through accessors.  Serde derives still work on private fields.
- **Metal first.**  Targeting M-series MacBooks; the `cpu` feature flag provides a fallback for CI.
- **Shared eval setup.**  Every binary that runs GPT-2 activations through an SAE parses its flags via `SharedEvalOpts` and builds its batches via `eval_opts::build_batches`, so adding a new shared flag is a one-touch change.

The full Rust style conventions followed by this crate (no `return`, no `mut`, no `for`/`loop`, no `unwrap`/`expect`, combinators over pattern matching, exhaustive enum matches, hand-rolled errors) are documented in `CLAUDE.md`.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Author

Onyeka Obi
