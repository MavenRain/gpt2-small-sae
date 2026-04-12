# gpt2-small-sae

Sparse autoencoders on the GPT-2 small residual stream, implemented in Rust with [candle](https://github.com/huggingface/candle) on the Metal backend.

## Overview

This project trains sparse autoencoders (SAEs) against the residual stream of the pretrained GPT-2 small model (124M parameters, 768-dimensional residual stream, 12 layers).  The SAE learns an overcomplete feature dictionary that reconstructs the residual with high fidelity under tight L1 sparsity constraints, enabling mechanistic interpretability of the model's internal representations.

The entire pipeline is pure Rust.  No Python.

## Architecture

The SAE uses the standard encoder-decoder architecture with a tied pre-encoder bias:

```
features = relu((x - b_dec) @ w_enc + b_enc)
recon    = features @ w_dec + b_dec
loss     = mse(recon, x) + l1_coeff * mean(sum(|features|))
```

The GPT-2 forward pass is hand-rolled (no vendored Python model) and loads pretrained weights directly from `HuggingFace` safetensors.  Only the transformer blocks up to the chosen hookpoint are materialized, keeping memory usage proportional to the layer depth.

## Stack

| Layer | Crate | Role |
|-------|-------|------|
| Tensors + autograd | `candle-core`, `candle-nn` | Metal-accelerated tensor operations |
| Model weights | `safetensors`, `hf-hub` | Weight serialization and download |
| Tokenization | `tokenizers` | GPT-2 BPE tokenizer |
| Effects | `comp-cat-rs` | `Io` at the boundary, `Stream` for activations |
| Plotting | `plotters` | Training curve visualization |
| Matching | `pathfinding` | Hungarian algorithm for feature stability |

## Building

Requires Rust nightly (edition 2024, rust-version 1.85+).  [comp-cat-rs](https://crates.io/crates/comp-cat-rs) 0.5 is pulled from crates.io automatically.

```sh
# Metal (default, M-series Mac):
cargo build --release

# CPU fallback (CI, non-Mac):
cargo build --release --no-default-features --features cpu
```

## Usage

### Smoke test

Downloads GPT-2 small from `HuggingFace` Hub (cached after first run), tokenizes a sentence, runs the forward pass through 8 transformer blocks, and prints the residual stream shape:

```sh
cargo run --bin smoke --release
```

Expected output:

```
tokens:         7
residual shape: [1, 7, 768]
smoke test passed
```

### Training (coming soon)

```sh
cargo run --bin train --release
```

### Evaluating a pretrained SAE (coming soon)

```sh
cargo run --bin eval_pretrained --release
```

## Project structure

```
src/
  lib.rs              Module declarations
  error.rs            Crate-wide Error enum
  config.rs           Newtype wrappers for hyperparameters
  metrics.rs          Training metrics as newtypes over f64
  sae.rs              SAE model, forward pass, loss
  gpt2.rs             Hand-rolled GPT-2 small forward
  activations.rs      Lazy activation Stream
  io_boundary.rs      Io programs for side effects
  bin/
    smoke.rs          Smoke test binary
    train.rs          Training loop (stub)
    eval_pretrained.rs  Pretrained evaluation (stub)
```

## Design decisions

- **Newtypes everywhere.**  `ModelDim`, `SaeDim`, `L1Coefficient`, etc. prevent dimension and coefficient confusion at compile time.
- **Hand-rolled GPT-2.**  `candle-transformers` 0.10 does not ship a GPT-2 module, so the forward pass is implemented directly against `VarBuilder`.  This also gives precise control over the hookpoint and avoids loading unused layers.
- **`Conv1D` weight layout.**  GPT-2 stores linear weights as `(in_features, out_features)`.  The forward pass uses `x @ W + b` directly, with no transpose.
- **comp-cat-rs effects.**  Side effects (downloads, I/O, device acquisition) are wrapped in `Io` programs composed with `flat_map`.  The `run` catamorphism is called exactly once in each binary's `main`.
- **Metal first.**  Targeting M-series MacBooks; the `cpu` feature flag provides a fallback for CI.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Author

Onyeka Obi
