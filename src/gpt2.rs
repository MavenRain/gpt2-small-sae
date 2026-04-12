//! Hand-rolled GPT-2 small forward pass with a residual stream hookpoint.
//!
//! The model loads pretrained weights from a `HuggingFace` safetensors file
//! and runs the forward pass through blocks `0..layer_index`, returning the
//! residual stream at that point (equivalent to `TransformerLens`'s
//! `hook_resid_pre` at the target layer).
//!
//! Weight layout follows GPT-2's `Conv1D` convention: linear weight matrices
//! are stored as `(in_features, out_features)` so the forward operation is
//! `x @ W + b` with no transpose.
//!
//! Only the blocks up to the hookpoint are loaded into memory; later blocks
//! and the LM head are never materialized.

use candle_core::{DType, Device, Tensor};
use candle_nn::{self, LayerNorm, LayerNormConfig, Module, VarBuilder};

use crate::config::LayerIndex;
use crate::error::Error;

const VOCAB_SIZE: usize = 50257;
const MAX_POSITION: usize = 1024;
const N_HEAD: usize = 12;
const D_MODEL: usize = 768;
const D_HEAD: usize = D_MODEL / N_HEAD;
const D_MLP: usize = D_MODEL * 4;

/// Frozen GPT-2 small model truncated at a chosen layer hookpoint.
#[derive(Debug, Clone)]
#[must_use]
pub struct Gpt2 {
    wte: candle_nn::Embedding,
    wpe: candle_nn::Embedding,
    blocks: Vec<Block>,
}

#[derive(Debug, Clone)]
struct Block {
    ln_1: LayerNorm,
    attn: CausalSelfAttention,
    ln_2: LayerNorm,
    mlp: Mlp,
}

#[derive(Debug, Clone)]
struct CausalSelfAttention {
    attn_w: Tensor,
    attn_b: Tensor,
    proj_w: Tensor,
    proj_b: Tensor,
}

#[derive(Debug, Clone)]
struct Mlp {
    fc_w: Tensor,
    fc_b: Tensor,
    proj_w: Tensor,
    proj_b: Tensor,
}

impl Gpt2 {
    /// Load a frozen GPT-2 small model from a [`VarBuilder`] backed by
    /// pretrained safetensors data.  Only blocks `0..layer_index` are
    /// materialized.
    ///
    /// The builder should be rooted at the top level of the safetensors
    /// namespace (keys like `wte.weight`, `h.0.attn.c_attn.weight`, etc.).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Candle`] if expected tensors are missing or have
    /// wrong shapes.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(vb: VarBuilder, layer_index: LayerIndex) -> Result<Self, Error> {
        let wte = candle_nn::embedding(VOCAB_SIZE, D_MODEL, vb.pp("wte"))?;
        let wpe = candle_nn::embedding(MAX_POSITION, D_MODEL, vb.pp("wpe"))?;
        let h = vb.pp("h");
        let blocks = (0..layer_index.as_usize())
            .map(|i| Block::new(h.pp(i.to_string())))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { wte, wpe, blocks })
    }

    /// Load from raw safetensors bytes on `device`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Candle`] if the bytes are not valid safetensors or
    /// contain unexpected shapes.
    pub fn from_bytes(
        data: Vec<u8>,
        layer_index: LayerIndex,
        device: &Device,
    ) -> Result<Self, Error> {
        let vb = VarBuilder::from_buffered_safetensors(data, DType::F32, device)?;
        Self::new(vb, layer_index)
    }

    /// Run the forward pass and return the residual stream at the hookpoint.
    ///
    /// `input_ids` must have shape `(batch, seq_len)` with `u32` dtype.
    /// The returned tensor has shape `(batch, seq_len, 768)`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Shape`] if `input_ids` is not 2-D, or
    /// [`Error::Candle`] on tensor operation failure.
    #[allow(clippy::cast_possible_truncation)]
    pub fn residual_stream(&self, input_ids: &Tensor) -> Result<Tensor, Error> {
        let (batch, seq_len) = match input_ids.dims() {
            [b, s] => Ok((*b, *s)),
            other => Err(Error::Shape {
                what: "gpt2 input_ids",
                expected: vec![0, 0],
                actual: other.to_vec(),
            }),
        }?;
        let device = input_ids.device();
        let position_ids = Tensor::arange(0u32, seq_len as u32, device)?;
        let token_emb = self.wte.forward(input_ids)?;
        let pos_emb = self.wpe.forward(&position_ids)?;
        let hidden = token_emb.broadcast_add(&pos_emb)?;
        self.blocks
            .iter()
            .try_fold(hidden, |h, block| block.forward(&h, batch, seq_len, device))
    }
}

impl Block {
    #[allow(clippy::needless_pass_by_value)]
    fn new(vb: VarBuilder) -> Result<Self, Error> {
        let ln_config = LayerNormConfig::default();
        let ln_1 = candle_nn::layer_norm(D_MODEL, ln_config, vb.pp("ln_1"))?;
        let ln_2 = candle_nn::layer_norm(D_MODEL, ln_config, vb.pp("ln_2"))?;
        let attn = CausalSelfAttention::new(vb.pp("attn"))?;
        let mlp = Mlp::new(vb.pp("mlp"))?;
        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        batch: usize,
        seq_len: usize,
        device: &Device,
    ) -> Result<Tensor, Error> {
        let normed_1 = self.ln_1.forward(x)?;
        let attn_out = self.attn.forward(&normed_1, batch, seq_len, device)?;
        let residual = x.add(&attn_out)?;
        let normed_2 = self.ln_2.forward(&residual)?;
        let mlp_out = self.mlp.forward(&normed_2, batch, seq_len)?;
        residual.add(&mlp_out).map_err(Error::from)
    }
}

impl CausalSelfAttention {
    #[allow(clippy::needless_pass_by_value)]
    fn new(vb: VarBuilder) -> Result<Self, Error> {
        let attn_w = vb.get((D_MODEL, 3 * D_MODEL), "c_attn.weight")?;
        let attn_b = vb.get(3 * D_MODEL, "c_attn.bias")?;
        let proj_w = vb.get((D_MODEL, D_MODEL), "c_proj.weight")?;
        let proj_b = vb.get(D_MODEL, "c_proj.bias")?;
        Ok(Self {
            attn_w,
            attn_b,
            proj_w,
            proj_b,
        })
    }

    #[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
    fn forward(
        &self,
        x: &Tensor,
        batch: usize,
        seq_len: usize,
        device: &Device,
    ) -> Result<Tensor, Error> {
        let n = batch * seq_len;
        let flat = x.reshape((n, D_MODEL))?;
        let qkv = flat
            .matmul(&self.attn_w)?
            .broadcast_add(&self.attn_b)?
            .reshape((batch, seq_len, 3 * D_MODEL))?;
        let q = qkv.narrow(2, 0, D_MODEL)?;
        let k = qkv.narrow(2, D_MODEL, D_MODEL)?;
        let v = qkv.narrow(2, 2 * D_MODEL, D_MODEL)?;
        let q = q
            .reshape((batch, seq_len, N_HEAD, D_HEAD))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch, seq_len, N_HEAD, D_HEAD))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch, seq_len, N_HEAD, D_HEAD))?
            .transpose(1, 2)?
            .contiguous()?;
        let scale = (D_HEAD as f64).sqrt().recip();
        let scores = q
            .matmul(&k.transpose(2, 3)?.contiguous()?)?
            .affine(scale, 0.0)?;
        let mask_data: Vec<f32> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j <= i { 0.0f32 } else { -1e10f32 }))
            .collect();
        let causal = Tensor::from_vec(mask_data, (1, 1, seq_len, seq_len), device)?;
        let masked = scores.broadcast_add(&causal)?;
        let weights = candle_nn::ops::softmax(&masked, candle_core::D::Minus1)?;
        let attn = weights.matmul(&v)?;
        let attn = attn.transpose(1, 2)?.reshape((n, D_MODEL))?;
        attn.matmul(&self.proj_w)?
            .broadcast_add(&self.proj_b)?
            .reshape((batch, seq_len, D_MODEL))
            .map_err(Error::from)
    }
}

impl Mlp {
    #[allow(clippy::needless_pass_by_value)]
    fn new(vb: VarBuilder) -> Result<Self, Error> {
        let fc_w = vb.get((D_MODEL, D_MLP), "c_fc.weight")?;
        let fc_b = vb.get(D_MLP, "c_fc.bias")?;
        let proj_w = vb.get((D_MLP, D_MODEL), "c_proj.weight")?;
        let proj_b = vb.get(D_MODEL, "c_proj.bias")?;
        Ok(Self {
            fc_w,
            fc_b,
            proj_w,
            proj_b,
        })
    }

    fn forward(&self, x: &Tensor, batch: usize, seq_len: usize) -> Result<Tensor, Error> {
        let n = batch * seq_len;
        let flat = x.reshape((n, D_MODEL))?;
        let h = flat.matmul(&self.fc_w)?.broadcast_add(&self.fc_b)?.gelu()?;
        h.matmul(&self.proj_w)?
            .broadcast_add(&self.proj_b)?
            .reshape((batch, seq_len, D_MODEL))
            .map_err(Error::from)
    }
}
