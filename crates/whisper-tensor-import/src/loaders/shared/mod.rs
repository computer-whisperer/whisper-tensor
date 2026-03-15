mod onnx_model;
mod rnn_supergraph;

use crate::onnx_graph::WeightStorageStrategy;
use std::path::Path;
use std::sync::Arc;
use whisper_tensor::interfaces::TextInferenceTokensInLogitOutInterface;
use whisper_tensor::loader::{LoaderError, LoaderOutput};
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::model::Model;

pub(super) fn onnx_bytes_to_model(
    onnx_data: &[u8],
    model_name: &str,
    base_dir: Option<&Path>,
) -> Result<(Arc<Model>, LoaderOutput), LoaderError> {
    onnx_model::onnx_bytes_to_model(onnx_data, model_name, base_dir)
}

pub(super) fn default_storage() -> WeightStorageStrategy {
    rnn_supergraph::default_storage()
}

pub(super) fn build_rnn_supergraph(
    tokenizer: TokenizerInfo,
    token_input_name: &str,
    logit_output_name: &str,
    state_pairs: &[(String, String)],
    symbolic_graph: &whisper_tensor::symbolic_graph::SymbolicGraph,
    rng: &mut impl rand::Rng,
) -> TextInferenceTokensInLogitOutInterface {
    rnn_supergraph::build_rnn_supergraph(
        tokenizer,
        token_input_name,
        logit_output_name,
        state_pairs,
        symbolic_graph,
        rng,
    )
}
