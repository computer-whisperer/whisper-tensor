use super::{build_rnn_supergraph, default_storage, onnx_bytes_to_model};
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

/// Loader for HuggingFace Transformers format (directory with config.json + safetensors).
pub struct TransformersLoader;

impl Loader for TransformersLoader {
    fn name(&self) -> &str {
        "HuggingFace Transformers"
    }

    fn description(&self) -> &str {
        "Load a model from a HuggingFace Transformers directory (config.json + safetensors)"
    }

    fn config_schema(&self) -> Vec<ConfigField> {
        vec![ConfigField {
            key: "path".to_string(),
            label: "Model Directory".to_string(),
            description: "Path to directory containing config.json and .safetensors files"
                .to_string(),
            field_type: ConfigFieldType::FilePath,
            required: true,
            default: None,
        }]
    }

    fn load(&self, config: ConfigValues) -> Result<LoaderOutput, LoaderError> {
        let path = require_path(&config, "path")?;
        let storage = default_storage();

        let onnx_data = crate::load_transformers_format(&path, storage)
            .map_err(|e| LoaderError::LoadFailed(e.into()))?;

        let model_name = path
            .file_name()
            .unwrap_or_default()
            .to_str()
            .unwrap_or("model")
            .to_string();

        let (model, mut output) = onnx_bytes_to_model(&onnx_data, &model_name, Some(&path))?;

        // Resolve tokenizer: prefer local tokenizer.json, fall back to HF repo ID
        let tokenizer_info = {
            let local_tokenizer = path.join("tokenizer.json");
            if local_tokenizer.exists() {
                let abs = std::fs::canonicalize(&local_tokenizer).unwrap_or(local_tokenizer);
                TokenizerInfo::HFTokenizerLocal(abs.to_string_lossy().to_string())
            } else {
                TokenizerInfo::HFTokenizer(model_name.clone())
            }
        };

        // Inspect model for state pairs (KV cache)
        let graph = model.get_symbolic_graph();
        let names_by_id = graph.get_tensors_by_name();

        // Llama-style KV cache: kv_cache_input_k_{i}/kv_cache_output_k_{i},
        // kv_cache_input_v_{i}/kv_cache_output_v_{i}
        let mut state_pairs = Vec::new();
        let mut layer = 0;
        loop {
            let mut found_any = false;
            for kind in ["k", "v"] {
                let in_name = format!("kv_cache_input_{kind}_{layer}");
                let out_name = format!("kv_cache_output_{kind}_{layer}");
                if names_by_id.contains_key(&in_name) && names_by_id.contains_key(&out_name) {
                    state_pairs.push((in_name, out_name));
                    found_any = true;
                }
            }
            if !found_any {
                break;
            }
            layer += 1;
        }

        // Check for standard input/output names
        let token_input = if names_by_id.contains_key("input_ids") {
            "input_ids"
        } else {
            // Fall back to first input
            return Ok(output);
        };
        let logit_output = if names_by_id.contains_key("logits") {
            "logits"
        } else {
            return Ok(output);
        };

        let mut rng = rand::rng();
        let interface = if state_pairs.is_empty() {
            // Simple transformer (no KV cache state)
            let input_id = *names_by_id.get(token_input).unwrap();
            let input_info = graph.get_tensor_info(input_id).unwrap();
            let input_dtype = input_info.dtype.unwrap_or(DType::I32);
            let input_rank = input_info.shape.as_ref().map_or(2, |s| s.len());

            let output_id = *names_by_id.get(logit_output).unwrap();
            let output_info = graph.get_tensor_info(output_id).unwrap();
            let output_rank = output_info.shape.as_ref().map_or(3, |s| s.len());

            build_simple_transformer_supergraph(
                tokenizer_info.clone(),
                token_input,
                logit_output,
                input_dtype,
                input_rank,
                output_rank,
                &mut rng,
            )
        } else {
            // RNN-style with KV cache
            build_rnn_supergraph(
                tokenizer_info.clone(),
                token_input,
                logit_output,
                &state_pairs,
                graph,
                &mut rng,
            )
        };

        output.interfaces.push(LoadedInterface {
            name: format!("{model_name}-TextInference"),
            interface: interface.to_any(),
        });

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
