//! Smoke test: download GPT-2, tokenize a sentence, print residual shape.

use candle_core::Tensor;
use comp_cat_rs::effect::io::Io;

use gpt2_small_sae::config::LayerIndex;
use gpt2_small_sae::error::{Error, TokenizerError};
use gpt2_small_sae::io_boundary;

fn smoke_program() -> Io<Error, ()> {
    io_boundary::acquire_device().flat_map(|device| {
        io_boundary::download_gpt2_weights().flat_map(move |weights| {
            io_boundary::download_tokenizer().flat_map(move |tokenizer| {
                Io::suspend(move || {
                    let layer_index = LayerIndex::new(8, 12)?;
                    let gpt2 =
                        gpt2_small_sae::gpt2::Gpt2::from_bytes(weights, layer_index, &device)?;
                    let encoding = tokenizer
                        .encode("The cat sat on the mat.", false)
                        .map_err(|e| Error::Tokenizer(TokenizerError::new(e)))?;
                    let ids: Vec<u32> = encoding.get_ids().to_vec();
                    let n_tokens = ids.len();
                    let input_ids = Tensor::from_vec(ids, (1, n_tokens), &device)?;
                    let resid = gpt2.residual_stream(&input_ids)?;
                    println!("tokens:         {n_tokens}");
                    println!("residual shape: {:?}", resid.dims());
                    println!("smoke test passed");
                    Ok(())
                })
            })
        })
    })
}

fn main() {
    smoke_program().run().unwrap_or_else(|e| {
        eprintln!("smoke test failed: {e}");
        std::process::exit(1);
    });
}
