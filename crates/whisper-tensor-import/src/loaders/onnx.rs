use super::onnx_bytes_to_model;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::dtype::DType;
use whisper_tensor::interfaces::TextInferenceTokensInLogitOutInterface;
use whisper_tensor::loader::*;
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::milli_graph::MilliOpGraph;
use whisper_tensor::milli_graph::ops::{Cast, Constant, Squeeze, Unsqueeze};
use whisper_tensor::super_graph::SuperGraphBuilder;
use whisper_tensor::super_graph::nodes::{
    SuperGraphNode, SuperGraphNodeMilliOpGraph, SuperGraphNodeModelExecution,
};

/// Loader for raw ONNX model files.
///
/// For simple transformer models (e.g. GPT-2), set `tokenizer` to
/// the HuggingFace tokenizer name. The loader will build a
/// `TextInferenceTokensInLogitOutInterface` assuming the first
/// graph input is token IDs and the first output is logits.
pub struct OnnxLoader;

impl Loader for OnnxLoader {
    fn name(&self) -> &str {
        "ONNX"
    }

    fn description(&self) -> &str {
        "Load a raw ONNX model file"
    }

    fn config_schema(&self) -> Vec<ConfigField> {
        vec![
            ConfigField {
                key: "path".to_string(),
                label: "Model Path".to_string(),
                description: "Path to the .onnx file".to_string(),
                field_type: ConfigFieldType::FilePath,
                required: true,
                default: None,
            },
            ConfigField {
                key: "tokenizer".to_string(),
                label: "Tokenizer".to_string(),
                description: "HuggingFace tokenizer name (e.g. 'gpt2'). If set, builds a text inference interface assuming first input=tokens, first output=logits.".to_string(),
                field_type: ConfigFieldType::String,
                required: false,
                default: None,
            },
        ]
    }

    fn load(&self, config: ConfigValues) -> Result<LoaderOutput, LoaderError> {
        let path = require_path(&config, "path")?;
        let tokenizer_name = get_string(&config, "tokenizer")?;

        let onnx_data =
            crate::load_onnx_file(&path).map_err(|e| LoaderError::LoadFailed(e.into()))?;

        let model_name = path
            .file_stem()
            .unwrap_or_default()
            .to_str()
            .unwrap_or("model")
            .to_string();

        let (model, mut output) = onnx_bytes_to_model(&onnx_data, &model_name, path.parent())?;

        // If a tokenizer is specified, build a simple transformer interface
        if let Some(tok_name) = tokenizer_name {
            let graph = model.get_symbolic_graph();
            let input_ids = graph.get_inputs();
            let output_ids = graph.get_outputs();

            let first_input = input_ids.first().and_then(|id| {
                let info = graph.get_tensor_info(*id)?;
                Some((*id, info))
            });
            let first_output = output_ids.first().and_then(|id| {
                let info = graph.get_tensor_info(*id)?;
                Some((*id, info))
            });

            if let (Some((_, input_info)), Some((_, output_info))) = (first_input, first_output) {
                let token_input_name = input_info.name().unwrap();
                let logit_output_name = output_info.name().unwrap();
                let input_dtype = input_info.dtype.unwrap_or(DType::I32);
                let input_rank = input_info.shape.as_ref().map_or(2, |s| s.len());
                let output_rank = output_info.shape.as_ref().map_or(3, |s| s.len());

                let mut rng = rand::rng();
                let interface = build_simple_transformer_supergraph(
                    TokenizerInfo::HFTokenizer(tok_name),
                    &token_input_name,
                    &logit_output_name,
                    input_dtype,
                    input_rank,
                    output_rank,
                    &mut rng,
                );
                output.interfaces.push(LoadedInterface {
                    name: format!("{model_name}-TextInference"),
                    interface: interface.to_any(),
                });
            }
        }

        Ok(output)
    }
}

/// Build a SuperGraph for a simple (non-RNN) transformer model.
///
/// The model takes a 1D token sequence, produces 2D logits (seq_len, vocab).
/// The SuperGraph handles dtype casting and rank adjustments.
fn build_simple_transformer_supergraph(
    tokenizer: TokenizerInfo,
    token_input_name: &str,
    logit_output_name: &str,
    input_dtype: DType,
    input_rank: usize,
    output_rank: usize,
    rng: &mut impl rand::Rng,
) -> TextInferenceTokensInLogitOutInterface {
    let mut super_graph_builder = SuperGraphBuilder::new();
    let token_context_input_link = super_graph_builder.new_tensor_link(rng);
    let model_input_link = super_graph_builder.new_model_link(rng);
    let raw_logit_output_link = super_graph_builder.new_tensor_link(rng);
    let cache_key = super_graph_builder.new_hash_link(rng);

    // Input processing: cast dtype + unsqueeze to match model rank
    let adjusted_token_context = {
        let (mut milli_graph, input_map) =
            MilliOpGraph::new(std::iter::once(token_context_input_link.global_id()), rng);
        let milli_op_graph_input = *input_map
            .get(&token_context_input_link.global_id())
            .unwrap();
        let mut x = Cast::push_new(&mut milli_graph, milli_op_graph_input, input_dtype, rng);
        let zero_tid = Constant::push_new(
            &mut milli_graph,
            NDArrayNumericTensor::from_vec_shape(vec![0i64], &vec![1]).unwrap(),
            rng,
        );
        for _ in 0..(input_rank - 1) {
            x = Unsqueeze::push_new(&mut milli_graph, x, zero_tid, rng);
        }

        let processed_input_link = super_graph_builder.new_tensor_link(rng);
        milli_graph.set_output_map(std::iter::once((x, processed_input_link.global_id())));

        let node = SuperGraphNodeMilliOpGraph::new(milli_graph, rng);
        super_graph_builder.add_node(node.to_any());
        processed_input_link
    };

    let tensor_inputs = vec![(adjusted_token_context, token_input_name.to_string())];
    let tensor_outputs = vec![(logit_output_name.to_string(), raw_logit_output_link)];

    super_graph_builder.add_node(
        SuperGraphNodeModelExecution::new(rng, model_input_link, 0, tensor_inputs, tensor_outputs)
            .to_any(),
    );

    // Output processing: squeeze extra dims + cast to F32
    let processed_logit_output_link = {
        let (mut milli_graph, input_map) =
            MilliOpGraph::new(std::iter::once(raw_logit_output_link.global_id()), rng);
        let milli_op_graph_input = *input_map.get(&raw_logit_output_link.global_id()).unwrap();
        let mut x = milli_op_graph_input;
        let zero_tid = Constant::push_new(
            &mut milli_graph,
            NDArrayNumericTensor::from_vec_shape(vec![0i64], &vec![1]).unwrap(),
            rng,
        );
        for _ in 0..(output_rank - 2) {
            x = Squeeze::push_new(&mut milli_graph, x, zero_tid, rng);
        }
        x = Cast::push_new(&mut milli_graph, x, DType::F32, rng);

        let processed_logit_output_link = super_graph_builder.new_tensor_link(rng);
        milli_graph.set_output_map(std::iter::once((
            x,
            processed_logit_output_link.global_id(),
        )));

        let node = SuperGraphNodeMilliOpGraph::new(milli_graph, rng);
        super_graph_builder.add_node(node.to_any());
        processed_logit_output_link
    };

    let super_graph_inputs = vec![
        cache_key.to_any(),
        model_input_link.to_any(),
        token_context_input_link.to_any(),
    ];
    let super_graph_outputs = vec![processed_logit_output_link.to_any()];
    let super_graph = super_graph_builder.build(
        rng,
        super_graph_inputs.as_slice(),
        super_graph_outputs.as_slice(),
    );

    TextInferenceTokensInLogitOutInterface {
        tokenizer,
        model_input_link,
        token_context_input_link,
        logit_output_link: processed_logit_output_link,
        super_graph,
        cache_key_input_link: cache_key,
    }
}
