use std::sync::Arc;
use whisper_tensor::interfaces::SpeechToTextInterface;
use whisper_tensor::loader::*;
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::model::Model;
use whisper_tensor::super_graph::SuperGraphBuilder;
use whisper_tensor::super_graph::links::SuperGraphLink;
use whisper_tensor::super_graph::nodes::{SuperGraphNode, SuperGraphNodeModelExecution};

use crate::whisper::WhisperConfig;

/// Loader for Whisper speech-to-text models (HuggingFace safetensors format).
///
/// Expects a directory containing:
/// - `model.safetensors` (or sharded safetensors)
/// - `config.json`
/// - `tokenizer.json`
pub struct WhisperLoader;

impl Loader for WhisperLoader {
    fn name(&self) -> &str {
        "Whisper"
    }

    fn description(&self) -> &str {
        "Load a Whisper speech-to-text model (HuggingFace format)"
    }

    fn config_schema(&self) -> Vec<ConfigField> {
        vec![ConfigField {
            key: "path".to_string(),
            label: "Model Directory".to_string(),
            description: "Path to the Whisper model directory (containing model.safetensors, config.json, tokenizer.json)".to_string(),
            field_type: ConfigFieldType::FilePath,
            required: true,
            default: None,
        }]
    }

    fn load(&self, config: ConfigValues) -> Result<LoaderOutput, LoaderError> {
        let dir = require_path(&config, "path")?;

        // Parse config.json
        let config_path = dir.join("config.json");
        let config_json: serde_json::Value = {
            let data = std::fs::read_to_string(&config_path)
                .map_err(|e| LoaderError::LoadFailed(e.into()))?;
            serde_json::from_str(&data).map_err(|e| LoaderError::LoadFailed(e.into()))?
        };

        let whisper_config = WhisperConfig {
            d_model: config_json["d_model"].as_u64().unwrap_or(768) as usize,
            encoder_layers: config_json["encoder_layers"].as_u64().unwrap_or(12) as usize,
            encoder_attention_heads: config_json["encoder_attention_heads"]
                .as_u64()
                .unwrap_or(12) as usize,
            encoder_ffn_dim: config_json["encoder_ffn_dim"].as_u64().unwrap_or(3072) as usize,
            decoder_layers: config_json["decoder_layers"].as_u64().unwrap_or(12) as usize,
            decoder_attention_heads: config_json["decoder_attention_heads"]
                .as_u64()
                .unwrap_or(12) as usize,
            decoder_ffn_dim: config_json["decoder_ffn_dim"].as_u64().unwrap_or(3072) as usize,
            num_mel_bins: config_json["num_mel_bins"].as_u64().unwrap_or(80) as usize,
            max_source_positions: config_json["max_source_positions"].as_u64().unwrap_or(1500)
                as usize,
            max_target_positions: config_json["max_target_positions"].as_u64().unwrap_or(448)
                as usize,
            vocab_size: config_json["vocab_size"].as_u64().unwrap_or(51864) as usize,
        };

        let decoder_start_token_id = config_json["decoder_start_token_id"]
            .as_u64()
            .unwrap_or(50257) as u32;
        let eos_token_id = config_json["eos_token_id"].as_u64().unwrap_or(50256) as u32;

        // Load safetensors weights
        let safetensors_path = dir.join("model.safetensors");
        if !safetensors_path.exists() {
            return Err(LoaderError::LoadFailed(anyhow::anyhow!(
                "model.safetensors not found in {}",
                dir.display()
            )));
        }

        let origin_path =
            std::fs::canonicalize(&safetensors_path).unwrap_or_else(|_| safetensors_path.clone());

        use crate::onnx_graph::WeightStorageStrategy;
        use crate::onnx_graph::weights::SafetensorsWeightManager;
        use memmap2::Mmap;

        let file = std::fs::File::open(&safetensors_path)
            .map_err(|e| LoaderError::LoadFailed(e.into()))?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| LoaderError::LoadFailed(e.into()))?;
        let wm = SafetensorsWeightManager::new(vec![Arc::new(mmap)])
            .map_err(|e| LoaderError::LoadFailed(e.into()))?;

        let storage = WeightStorageStrategy::OriginReference;

        // Build encoder ONNX model
        eprintln!("Building Whisper encoder...");
        let encoder_onnx =
            crate::whisper::build_encoder(&wm, &whisper_config, storage.clone(), &origin_path)
                .map_err(LoaderError::LoadFailed)?;

        // Build decoder ONNX model
        eprintln!("Building Whisper decoder...");
        let decoder_onnx =
            crate::whisper::build_decoder(&wm, &whisper_config, storage, &origin_path)
                .map_err(LoaderError::LoadFailed)?;

        let base_dir = dir.as_path();

        let mut rng = rand::rng();
        let encoder_model = Model::new_from_onnx(&encoder_onnx, &mut rng, Some(base_dir))
            .map_err(|e| LoaderError::LoadFailed(e.into()))?;
        let encoder_model = Arc::new(encoder_model);

        let decoder_model = Model::new_from_onnx(&decoder_onnx, &mut rng, Some(base_dir))
            .map_err(|e| LoaderError::LoadFailed(e.into()))?;
        let decoder_model = Arc::new(decoder_model);

        // Build SuperGraphs
        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            TokenizerInfo::HFTokenizerLocal(tokenizer_path.to_string_lossy().to_string())
        } else {
            return Err(LoaderError::LoadFailed(anyhow::anyhow!(
                "tokenizer.json not found in {}",
                dir.display()
            )));
        };

        // Build encoder SuperGraph (simple: mel input → model execution → encoder output)
        let (encoder_sg, mel_link, enc_weights, enc_output) = build_encoder_supergraph(&mut rng);

        // Build decoder SuperGraph with RNN cache
        let decoder_sg_result = build_decoder_supergraph(&whisper_config, &mut rng);

        let interface = SpeechToTextInterface {
            encoder_super_graph: encoder_sg,
            mel_input_link: mel_link,
            encoder_weights_link: enc_weights,
            encoder_output_link: enc_output,
            decoder_super_graph: decoder_sg_result.super_graph,
            decoder_token_link: decoder_sg_result.token_link,
            decoder_weights_link: decoder_sg_result.weights_link,
            decoder_encoder_hidden_link: decoder_sg_result.encoder_hidden_link,
            decoder_logit_link: decoder_sg_result.logit_link,
            decoder_cache_key_link: decoder_sg_result.cache_key_link,
            tokenizer,
            sample_rate: 16000,
            num_mel_bins: whisper_config.num_mel_bins as u32,
            decoder_start_token_id,
            eos_token_id,
        };

        let models = vec![
            LoadedModel {
                name: "whisper-encoder".to_string(),
                model: encoder_model,
            },
            LoadedModel {
                name: "whisper-decoder".to_string(),
                model: decoder_model,
            },
        ];

        Ok(LoaderOutput {
            models,
            interfaces: vec![LoadedInterface {
                name: "whisper-SpeechToText".to_string(),
                interface: interface.to_any(),
            }],
        })
    }
}

/// Build the encoder SuperGraph: mel → model execution → encoder_hidden_states.
fn build_encoder_supergraph(
    rng: &mut impl rand::Rng,
) -> (
    whisper_tensor::super_graph::SuperGraph,
    whisper_tensor::super_graph::links::SuperGraphLinkTensor,
    whisper_tensor::super_graph::links::SuperGraphLinkTensorMap,
    whisper_tensor::super_graph::links::SuperGraphLinkTensor,
) {
    let mut builder = SuperGraphBuilder::new();

    let mel_link = builder.new_tensor_link(rng);
    let enc_weights = builder.new_model_link(rng);
    let enc_output = builder.new_tensor_link(rng);

    builder.add_node(
        SuperGraphNodeModelExecution::new(
            rng,
            enc_weights,
            0, // model index 0 = encoder
            vec![(mel_link, "input_features".to_string())],
            vec![("last_hidden_state".to_string(), enc_output)],
        )
        .to_any(),
    );

    let sg_inputs = vec![mel_link.to_any(), enc_weights.to_any()];
    let sg_outputs = vec![enc_output.to_any()];
    let sg = builder.build(rng, &sg_inputs, &sg_outputs);

    (sg, mel_link, enc_weights, enc_output)
}

struct DecoderSuperGraphResult {
    super_graph: whisper_tensor::super_graph::SuperGraph,
    token_link: whisper_tensor::super_graph::links::SuperGraphLinkTensor,
    weights_link: whisper_tensor::super_graph::links::SuperGraphLinkTensorMap,
    encoder_hidden_link: whisper_tensor::super_graph::links::SuperGraphLinkTensor,
    logit_link: whisper_tensor::super_graph::links::SuperGraphLinkTensor,
    cache_key_link: whisper_tensor::super_graph::links::SuperGraphLinkHash,
}

/// Build the decoder SuperGraph with RNN-style cache for autoregressive decoding.
///
/// This follows the same pattern as `build_rnn_supergraph` but adds
/// encoder_hidden_states as an extra fixed input to each decoder step.
fn build_decoder_supergraph(
    config: &WhisperConfig,
    rng: &mut impl rand::Rng,
) -> DecoderSuperGraphResult {
    use std::collections::HashMap;
    use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
    use whisper_tensor::dtype::DType;
    use whisper_tensor::milli_graph::MilliOpGraph;
    use whisper_tensor::milli_graph::ops::{Cast, Constant, Shape, Squeeze, Unsqueeze};
    use whisper_tensor::super_graph::links::{SuperGraphLinkDouble, SuperGraphLinkTriple};
    use whisper_tensor::super_graph::nodes::{
        SuperGraphNodeMilliOpGraph, SuperGraphNodeRNNCacheRead, SuperGraphNodeRNNCacheWrite,
        SuperGraphNodeScan,
    };

    let mut builder = SuperGraphBuilder::new();
    let token_context_link = builder.new_tensor_link(rng);
    let model_weights_link = builder.new_model_link(rng);
    let encoder_hidden_link = builder.new_tensor_link(rng);
    let cache_key_link = builder.new_hash_link(rng);

    // State pairs: self-attn KV only (cross-attn K/V are recomputed each step)
    // 2 states per layer: self_k, self_v
    let num_states = config.decoder_layers * 2;
    let state_ids: Vec<usize> = (0..num_states).collect();

    let state_names: Vec<(String, String)> = {
        let mut pairs = Vec::new();
        for i in 0..config.decoder_layers {
            pairs.push((format!("self_k_cache_{i}"), format!("self_k_cache_out_{i}")));
            pairs.push((format!("self_v_cache_{i}"), format!("self_v_cache_out_{i}")));
        }
        pairs
    };

    // State init links (before cache read)
    let state_init_links: Vec<(
        usize,
        whisper_tensor::super_graph::links::SuperGraphLinkTensor,
    )> = state_ids
        .iter()
        .map(|&id| (id, builder.new_tensor_link(rng)))
        .collect();

    // Cache read node
    let post_cache_tokens = builder.new_tensor_link(rng);
    let post_cache_state_links: Vec<(
        usize,
        whisper_tensor::super_graph::links::SuperGraphLinkTensor,
    )> = state_ids
        .iter()
        .map(|&id| (id, builder.new_tensor_link(rng)))
        .collect();

    {
        let node = SuperGraphNodeRNNCacheRead::new(
            cache_key_link,
            token_context_link,
            post_cache_tokens,
            post_cache_state_links
                .iter()
                .map(|(id, link)| (id.to_string(), *link))
                .collect(),
            state_init_links
                .iter()
                .map(|(id, link)| (id.to_string(), *link))
                .collect(),
            rng,
        );
        builder.add_node(node.to_any());
    }

    // Loop count from token sequence length
    let loop_count_link = {
        let lc = builder.new_tensor_link(rng);
        let (mut mg, input_map) =
            MilliOpGraph::new(std::iter::once(post_cache_tokens.global_id()), rng);
        let input = *input_map.get(&post_cache_tokens.global_id()).unwrap();
        let shape_out = Shape::push_new(&mut mg, input, rng);
        mg.set_output_map(std::iter::once((shape_out, lc.global_id())));
        builder.add_node(SuperGraphNodeMilliOpGraph::new(mg, rng).to_any());
        lc
    };

    // State initialization (zeros)
    {
        let (mut mg, _) = MilliOpGraph::new(std::iter::empty(), rng);
        let mut output_map = HashMap::new();
        let mut output_order = vec![];
        let head_dim = config.d_model / config.decoder_attention_heads;
        for (_, link) in &state_init_links {
            // All KV cache states: [1, num_heads, 0, head_dim] (empty initial cache)
            let init = Constant::push_new(
                &mut mg,
                NDArrayNumericTensor::from_vec_shape(
                    Vec::<f32>::new(),
                    &vec![
                        1u64,
                        config.decoder_attention_heads as u64,
                        0,
                        head_dim as u64,
                    ],
                )
                .unwrap(),
                rng,
            );
            output_map.insert(init, link.global_id());
            output_order.push(link.global_id());
        }
        mg.set_output_map_ordered(output_map, output_order);
        builder.add_node(SuperGraphNodeMilliOpGraph::new(mg, rng).to_any());
    }

    // Build scan sub-graph
    let mut sub_builder = SuperGraphBuilder::new();
    let sub_model_link = sub_builder.new_model_link(rng);
    let sub_token_input = sub_builder.new_tensor_link(rng);
    let sub_encoder_hidden = sub_builder.new_tensor_link(rng);

    let sub_state_in: HashMap<usize, whisper_tensor::super_graph::links::SuperGraphLinkTensor> =
        state_ids
            .iter()
            .map(|&id| (id, sub_builder.new_tensor_link(rng)))
            .collect();
    let sub_state_out: HashMap<usize, whisper_tensor::super_graph::links::SuperGraphLinkTensor> =
        state_ids
            .iter()
            .map(|&id| (id, sub_builder.new_tensor_link(rng)))
            .collect();
    let final_state_links: Vec<(
        usize,
        whisper_tensor::super_graph::links::SuperGraphLinkTensor,
    )> = state_ids
        .iter()
        .map(|&id| (id, builder.new_tensor_link(rng)))
        .collect();

    // Input processing: cast + unsqueeze token to [1, 1]
    let processed_token = {
        let (mut mg, input_map) =
            MilliOpGraph::new(std::iter::once(sub_token_input.global_id()), rng);
        let input = *input_map.get(&sub_token_input.global_id()).unwrap();
        let x = Cast::push_new(&mut mg, input, DType::I64, rng);
        let zero = Constant::push_new(
            &mut mg,
            NDArrayNumericTensor::from_vec_shape(vec![0i64], &vec![1]).unwrap(),
            rng,
        );
        let x = Unsqueeze::push_new(&mut mg, x, zero, rng);
        let x = Unsqueeze::push_new(&mut mg, x, zero, rng);

        let out = sub_builder.new_tensor_link(rng);
        mg.set_output_map(std::iter::once((x, out.global_id())));
        sub_builder.add_node(SuperGraphNodeMilliOpGraph::new(mg, rng).to_any());
        out
    };

    // Model execution in sub-graph
    let sub_logit_output = sub_builder.new_tensor_link(rng);
    {
        let mut tensor_inputs = vec![
            (processed_token, "decoder_input_ids".to_string()),
            (sub_encoder_hidden, "encoder_hidden_states".to_string()),
        ];
        let mut tensor_outputs = vec![("logits".to_string(), sub_logit_output)];

        for (id, pair) in state_names.iter().enumerate() {
            tensor_inputs.push((*sub_state_in.get(&id).unwrap(), pair.0.clone()));
            tensor_outputs.push((pair.1.clone(), *sub_state_out.get(&id).unwrap()));
        }

        sub_builder.add_node(
            SuperGraphNodeModelExecution::new(
                rng,
                sub_model_link,
                1,
                tensor_inputs,
                tensor_outputs,
            )
            .to_any(),
        );
    }

    // Output processing: squeeze + cast to F32
    let processed_logit = {
        let (mut mg, input_map) =
            MilliOpGraph::new(std::iter::once(sub_logit_output.global_id()), rng);
        let input = *input_map.get(&sub_logit_output.global_id()).unwrap();
        // logits: [1, 1, vocab] → squeeze batch + seq → [vocab]
        let zero = Constant::push_new(
            &mut mg,
            NDArrayNumericTensor::from_vec_shape(vec![0i64], &vec![1]).unwrap(),
            rng,
        );
        let x = Squeeze::push_new(&mut mg, input, zero, rng);
        let x = Squeeze::push_new(&mut mg, x, zero, rng);
        let x = Cast::push_new(&mut mg, x, DType::F32, rng);

        let out = sub_builder.new_tensor_link(rng);
        mg.set_output_map(std::iter::once((x, out.global_id())));
        sub_builder.add_node(SuperGraphNodeMilliOpGraph::new(mg, rng).to_any());
        out
    };

    // Build sub-graph
    let mut sub_inputs = vec![
        sub_model_link.to_any(),
        sub_token_input.to_any(),
        sub_encoder_hidden.to_any(),
    ];
    for &id in &state_ids {
        sub_inputs.push(sub_state_in.get(&id).unwrap().to_any());
    }
    let mut sub_outputs = vec![processed_logit.to_any()];
    for &id in &state_ids {
        sub_outputs.push(sub_state_out.get(&id).unwrap().to_any());
    }
    let sub_graph = sub_builder.build(rng, &sub_inputs, &sub_outputs);

    // Create scan node
    let outer_logit_link = builder.new_tensor_link(rng);

    let state_links: Vec<SuperGraphLinkTriple> = post_cache_state_links
        .iter()
        .map(|(id, init_link)| {
            SuperGraphLinkTriple::Tensor(
                *init_link,
                *sub_state_in.get(id).unwrap(),
                *sub_state_out.get(id).unwrap(),
            )
        })
        .collect();

    let final_state_outputs: Vec<SuperGraphLinkDouble> = final_state_links
        .iter()
        .map(|(id, link)| SuperGraphLinkDouble::Tensor(*sub_state_out.get(id).unwrap(), *link))
        .collect();

    let scan_node = SuperGraphNodeScan::new(
        sub_graph,
        loop_count_link,
        // simple_inputs: model weights + encoder hidden states
        vec![
            SuperGraphLinkDouble::TensorMap(model_weights_link, sub_model_link),
            SuperGraphLinkDouble::Tensor(encoder_hidden_link, sub_encoder_hidden),
        ],
        state_links,
        // scan_inputs: token sequence
        vec![(post_cache_tokens, sub_token_input, 0)],
        // scan_outputs: logits accumulated
        vec![(processed_logit, outer_logit_link, 0)],
        final_state_outputs,
        rng,
    );
    builder.add_node(scan_node.to_any());

    // Cache write
    {
        let node = SuperGraphNodeRNNCacheWrite::new(
            cache_key_link,
            token_context_link,
            final_state_links
                .iter()
                .map(|(id, link)| (id.to_string(), *link))
                .collect(),
            rng,
        );
        builder.add_node(node.to_any());
    }

    // Build outer graph
    let sg_inputs = vec![
        cache_key_link.to_any(),
        model_weights_link.to_any(),
        token_context_link.to_any(),
        encoder_hidden_link.to_any(),
    ];
    let sg_outputs = vec![outer_logit_link.to_any()];
    let super_graph = builder.build(rng, &sg_inputs, &sg_outputs);

    DecoderSuperGraphResult {
        super_graph,
        token_link: token_context_link,
        weights_link: model_weights_link,
        encoder_hidden_link,
        logit_link: outer_logit_link,
        cache_key_link,
    }
}
