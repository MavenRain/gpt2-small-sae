//! Sparse autoencoders on the GPT-2 small residual stream.
//!
//! This crate trains sparse autoencoders (SAEs) against the residual stream of
//! the pretrained GPT-2 small model, with the goal of producing an
//! interpretable feature dictionary that reconstructs the residual with high
//! fidelity under tight sparsity constraints.
//!
//! # Stack
//!
//! - [`candle_core`] / [`candle_nn`] for tensors and autograd on the Metal
//!   backend.
//! - [`candle_transformers`] for the pretrained GPT-2 weights.
//! - [`comp_cat_rs`] for effectful orchestration at the boundary; the training
//!   loop is expressed as a [`comp_cat_rs::effect::io::Io`] program built from
//!   combinators and run once in `main`.
//! - [`safetensors`] and [`hf_hub`] for weight I/O.
//!
//! # Module map
//!
//! - [`error`] defines the crate-wide [`error::Error`] enum. Every fallible
//!   function in this crate returns `Result<_, error::Error>`.
//! - [`config`] defines newtype wrappers over the raw numeric hyperparameters
//!   so that dimensions and coefficients cannot be silently confused.
//! - [`metrics`] computes training metrics (L0, reconstruction MSE, variance
//!   explained, dead feature count) from candle [`candle_core::Tensor`]s.
//! - [`sae`] defines the [`sae::Sae`] model, its forward pass, and its loss.
//! - [`gpt2`] wraps the pretrained GPT-2 small model and exposes the residual
//!   stream at a chosen layer.
//! - [`dataset`] tokenizes a text corpus and chunks it into fixed-size
//!   token-ID batches for the GPT-2 forward pass.
//! - [`activations`] exposes a residual stream as a
//!   [`comp_cat_rs::effect::stream::Stream`] for the training loop to fold
//!   over.
//! - [`io_boundary`] contains the effectful primitives (hub download,
//!   safetensors load, device acquisition) wrapped in `Io`.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod activations;
pub mod config;
pub mod dataset;
pub mod error;
pub mod gpt2;
pub mod io_boundary;
pub mod metrics;
pub mod sae;
pub mod train;
