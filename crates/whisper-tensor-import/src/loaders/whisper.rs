use std::collections::HashMap;
use std::sync::Arc;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::dtype::DType;
use whisper_tensor::interfaces::SpeechToTextInterface;
use whisper_tensor::loader::*;
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::milli_graph::MilliOpGraph;
use whisper_tensor::milli_graph::ops::{
    ArgMax, Cast, Concat as MilliConcat, Constant, SimpleBinary, Squeeze, Unsqueeze, Where,
};
use whisper_tensor::model::Model;
use whisper_tensor::super_graph::SuperGraphBuilder;
use whisper_tensor::super_graph::links::{
    SuperGraphLink, SuperGraphLinkDouble, SuperGraphLinkTriple,
};
use whisper_tensor::super_graph::nodes::{
    SuperGraphNode, SuperGraphNodeAudioClipToTensor, SuperGraphNodeMilliOpGraph,
    SuperGraphNodeModelExecution, SuperGraphNodeScan,
};

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
        let decoder_prefix_token_ids = load_forced_decoder_ids(&dir, decoder_start_token_id);
        let max_decode_steps = whisper_config.max_target_positions.max(1) as u32;

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

        // Build unified STT supergraph.
        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer = if tokenizer_path.exists() {
            TokenizerInfo::HFTokenizerLocal(tokenizer_path.to_string_lossy().to_string())
        } else {
            return Err(LoaderError::LoadFailed(anyhow::anyhow!(
                "tokenizer.json not found in {}",
                dir.display()
            )));
        };

        let whisper_sg = build_whisper_supergraph(
            &whisper_config,
            decoder_prefix_token_ids.as_slice(),
            eos_token_id,
            max_decode_steps,
            &mut rng,
        );

        let interface = SpeechToTextInterface {
            super_graph: whisper_sg.super_graph,
            audio_input_link: whisper_sg.audio_link,
            encoder_weights_link: whisper_sg.encoder_weights_link,
            decoder_weights_link: whisper_sg.decoder_weights_link,
            output_token_link: whisper_sg.output_token_link,
            tokenizer,
            sample_rate: 16000,
            num_mel_bins: whisper_config.num_mel_bins as u32,
            decoder_start_token_id,
            eos_token_id,
            decoder_prefix_token_ids,
            max_decode_steps,
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

struct WhisperSuperGraphResult {
    super_graph: whisper_tensor::super_graph::SuperGraph,
    audio_link: SuperGraphLink,
    encoder_weights_link: SuperGraphLink,
    decoder_weights_link: SuperGraphLink,
    output_token_link: SuperGraphLink,
}

fn build_whisper_supergraph(
    config: &WhisperConfig,
    decoder_prefix_token_ids: &[u32],
    eos_token_id: u32,
    max_decode_steps: u32,
    rng: &mut impl rand::Rng,
) -> WhisperSuperGraphResult {
    let mut builder = SuperGraphBuilder::new();

    // Inputs and encoder pass.
    let audio_link = builder.new_audio_clip_link(rng);
    let mel_link =
        SuperGraphNodeAudioClipToTensor::new_and_add(&mut builder, audio_link, Some(16000), rng);
    let encoder_weights_link = builder.new_model_link(rng);
    let decoder_weights_link = builder.new_model_link(rng);
    let encoder_hidden_link = builder.new_tensor_link(rng);

    builder.add_node(
        SuperGraphNodeModelExecution::new(
            rng,
            encoder_weights_link,
            0, // model index 0 = encoder
            vec![(mel_link, "input_features".to_string())],
            vec![("last_hidden_state".to_string(), encoder_hidden_link)],
        )
        .to_any(),
    );

    // Decoder state layout:
    // - token state: [1] (current token id)
    // - finished state: [1] bool
    // - KV cache states: [1, heads, seq, head_dim] for each self-attn K/V tensor
    let num_states = config.decoder_layers * 2;
    let state_ids: Vec<usize> = (0..num_states).collect();
    let state_names: Vec<(String, String)> = {
        let mut pairs = Vec::with_capacity(num_states);
        for i in 0..config.decoder_layers {
            pairs.push((format!("self_k_cache_{i}"), format!("self_k_cache_out_{i}")));
            pairs.push((format!("self_v_cache_{i}"), format!("self_v_cache_out_{i}")));
        }
        pairs
    };

    let kv_state_zero_links: Vec<(usize, SuperGraphLink)> = state_ids
        .iter()
        .map(|&id| (id, builder.new_tensor_link(rng)))
        .collect();
    let prefix_tokens_link = builder.new_tensor_link(rng);
    let prefix_iteration_count_link = builder.new_tensor_link(rng);

    {
        let (mut mg, _) = MilliOpGraph::new(std::iter::empty(), rng);
        let mut output_map = HashMap::new();
        let mut output_order = Vec::new();

        let prefix_values: Vec<i64> = if decoder_prefix_token_ids.is_empty() {
            vec![eos_token_id as i64]
        } else {
            decoder_prefix_token_ids
                .iter()
                .copied()
                .map(|x| x as i64)
                .collect()
        };
        let prefix_token_tensor = Constant::push_new(
            &mut mg,
            NDArrayNumericTensor::from_vec_shape(
                prefix_values.clone(),
                &vec![prefix_values.len() as u64],
            )
            .unwrap(),
            rng,
        );
        output_map.insert(prefix_token_tensor, prefix_tokens_link.global_id());
        output_order.push(prefix_tokens_link.global_id());

        let prefix_iterations = Constant::push_new(
            &mut mg,
            NDArrayNumericTensor::from_vec_shape(vec![prefix_values.len() as i64], &vec![1])
                .unwrap(),
            rng,
        );
        output_map.insert(prefix_iterations, prefix_iteration_count_link.global_id());
        output_order.push(prefix_iteration_count_link.global_id());

        let head_dim = config.d_model / config.decoder_attention_heads;
        for (_, link) in &kv_state_zero_links {
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

    // Prefill scan over decoder prefix tokens to warm KV state and compute first generated token.
    let mut prefill_builder = SuperGraphBuilder::new();
    let prefill_model_link = prefill_builder.new_model_link(rng);
    let prefill_encoder_hidden = prefill_builder.new_tensor_link(rng);
    let prefill_token_in = prefill_builder.new_tensor_link(rng);

    let prefill_state_in: HashMap<usize, SuperGraphLink> = state_ids
        .iter()
        .map(|&id| (id, prefill_builder.new_tensor_link(rng)))
        .collect();
    let prefill_state_out: HashMap<usize, SuperGraphLink> = state_ids
        .iter()
        .map(|&id| (id, prefill_builder.new_tensor_link(rng)))
        .collect();

    let prefill_decoder_input_ids = {
        let (mut mg, input_map) =
            MilliOpGraph::new(std::iter::once(prefill_token_in.global_id()), rng);
        let token_in = *input_map.get(&prefill_token_in.global_id()).unwrap();
        let token_i64 = Cast::push_new(&mut mg, token_in, DType::I64, rng);
        let axis0 = Constant::push_new(
            &mut mg,
            NDArrayNumericTensor::from_vec_shape(vec![0i64], &vec![1]).unwrap(),
            rng,
        );
        let token_2d = Unsqueeze::push_new(&mut mg, token_i64, axis0, rng);
        let out = prefill_builder.new_tensor_link(rng);
        mg.set_output_map(std::iter::once((token_2d, out.global_id())));
        prefill_builder.add_node(SuperGraphNodeMilliOpGraph::new(mg, rng).to_any());
        out
    };

    let prefill_raw_logits = prefill_builder.new_tensor_link(rng);
    {
        let mut tensor_inputs = vec![
            (prefill_decoder_input_ids, "decoder_input_ids".to_string()),
            (prefill_encoder_hidden, "encoder_hidden_states".to_string()),
        ];
        let mut tensor_outputs = vec![("logits".to_string(), prefill_raw_logits)];
        for (id, pair) in state_names.iter().enumerate() {
            tensor_inputs.push((*prefill_state_in.get(&id).unwrap(), pair.0.clone()));
            tensor_outputs.push((pair.1.clone(), *prefill_state_out.get(&id).unwrap()));
        }

        prefill_builder.add_node(
            SuperGraphNodeModelExecution::new(
                rng,
                prefill_model_link,
                1, // model index 1 = decoder
                tensor_inputs,
                tensor_outputs,
            )
            .to_any(),
        );
    }

    let prefill_logits = {
        let out = prefill_builder.new_tensor_link(rng);
        let (mut mg, input_map) =
            MilliOpGraph::new(std::iter::once(prefill_raw_logits.global_id()), rng);
        let logits = *input_map.get(&prefill_raw_logits.global_id()).unwrap();
        let axis0 = Constant::push_new(
            &mut mg,
            NDArrayNumericTensor::from_vec_shape(vec![0i64], &vec![1]).unwrap(),
            rng,
        );
        let squeezed = Squeeze::push_new(&mut mg, logits, axis0, rng);
        let squeezed = Squeeze::push_new(&mut mg, squeezed, axis0, rng);
        let logits_f32 = Cast::push_new(&mut mg, squeezed, DType::F32, rng);
        mg.set_output_map(std::iter::once((logits_f32, out.global_id())));
        prefill_builder.add_node(SuperGraphNodeMilliOpGraph::new(mg, rng).to_any());
        out
    };

    let mut prefill_inputs = vec![
        prefill_model_link.to_any(),
        prefill_encoder_hidden.to_any(),
        prefill_token_in.to_any(),
    ];
    for id in &state_ids {
        prefill_inputs.push(prefill_state_in.get(id).unwrap().to_any());
    }

    let mut prefill_outputs = vec![prefill_logits.to_any()];
    for id in &state_ids {
        prefill_outputs.push(prefill_state_out.get(id).unwrap().to_any());
    }
    let prefill_graph = prefill_builder.build(rng, &prefill_inputs, &prefill_outputs);

    let prefill_last_logits_link = builder.new_tensor_link(rng);
    let prefill_final_state_links: Vec<(usize, SuperGraphLink)> = state_ids
        .iter()
        .map(|&id| (id, builder.new_tensor_link(rng)))
        .collect();

    let prefill_state_links: Vec<SuperGraphLinkTriple> = kv_state_zero_links
        .iter()
        .map(|(id, init_link)| {
            SuperGraphLinkTriple::new(
                *init_link,
                *prefill_state_in.get(id).unwrap(),
                *prefill_state_out.get(id).unwrap(),
            )
        })
        .collect();

    let mut prefill_simple_outputs = vec![SuperGraphLinkDouble::new(
        prefill_logits,
        prefill_last_logits_link,
    )];
    for (id, out_link) in &prefill_final_state_links {
        prefill_simple_outputs.push(SuperGraphLinkDouble::new(
            *prefill_state_out.get(id).unwrap(),
            *out_link,
        ));
    }

    builder.add_node(
        SuperGraphNodeScan::new(
            prefill_graph,
            prefix_iteration_count_link,
            vec![
                SuperGraphLinkDouble::new(decoder_weights_link, prefill_model_link),
                SuperGraphLinkDouble::new(encoder_hidden_link, prefill_encoder_hidden),
            ],
            prefill_state_links,
            vec![(prefix_tokens_link, prefill_token_in, 0)],
            vec![],
            prefill_simple_outputs,
            rng,
        )
        .to_any(),
    );

    let token_state_init_link = builder.new_tensor_link(rng);
    let finished_state_init_link = builder.new_tensor_link(rng);
    {
        let (mut mg, input_map) =
            MilliOpGraph::new(std::iter::once(prefill_last_logits_link.global_id()), rng);
        let last_logits = *input_map
            .get(&prefill_last_logits_link.global_id())
            .unwrap();
        let next_token_scalar = ArgMax::push_new(&mut mg, last_logits, 0, false, false, rng);
        let next_token_scalar = Cast::push_new(&mut mg, next_token_scalar, DType::I64, rng);
        let axis0 = Constant::push_new(
            &mut mg,
            NDArrayNumericTensor::from_vec_shape(vec![0i64], &vec![1]).unwrap(),
            rng,
        );
        let next_token = Unsqueeze::push_new(&mut mg, next_token_scalar, axis0, rng);
        let eos_token = Constant::push_new(
            &mut mg,
            NDArrayNumericTensor::from_vec_shape(vec![eos_token_id as i64], &vec![1]).unwrap(),
            rng,
        );
        let initial_finished = SimpleBinary::equal(&mut mg, next_token, eos_token, rng);
        mg.set_output_map([
            (next_token, token_state_init_link.global_id()),
            (initial_finished, finished_state_init_link.global_id()),
        ]);
        builder.add_node(SuperGraphNodeMilliOpGraph::new(mg, rng).to_any());
    }

    let generation_iteration_count_link = builder.new_tensor_link(rng);
    {
        let (mut mg, _) = MilliOpGraph::new(std::iter::empty(), rng);
        let loop_count = Constant::push_new(
            &mut mg,
            NDArrayNumericTensor::from_vec_shape(vec![max_decode_steps as i64], &vec![1]).unwrap(),
            rng,
        );
        mg.set_output_map(std::iter::once((
            loop_count,
            generation_iteration_count_link.global_id(),
        )));
        builder.add_node(SuperGraphNodeMilliOpGraph::new(mg, rng).to_any());
    }

    // Generation scan sub-graph.
    let mut sub_builder = SuperGraphBuilder::new();
    let sub_model_link = sub_builder.new_model_link(rng);
    let sub_encoder_hidden = sub_builder.new_tensor_link(rng);
    let sub_token_in = sub_builder.new_tensor_link(rng);
    let sub_finished_in = sub_builder.new_tensor_link(rng);
    let sub_token_out = sub_builder.new_tensor_link(rng);
    let sub_finished_out = sub_builder.new_tensor_link(rng);

    let sub_state_in: HashMap<usize, SuperGraphLink> = state_ids
        .iter()
        .map(|&id| (id, sub_builder.new_tensor_link(rng)))
        .collect();
    let sub_state_out: HashMap<usize, SuperGraphLink> = state_ids
        .iter()
        .map(|&id| (id, sub_builder.new_tensor_link(rng)))
        .collect();

    let decoder_input_ids = {
        let (mut mg, input_map) = MilliOpGraph::new(std::iter::once(sub_token_in.global_id()), rng);
        let token_in = *input_map.get(&sub_token_in.global_id()).unwrap();
        let token_i64 = Cast::push_new(&mut mg, token_in, DType::I64, rng);
        let axis0 = Constant::push_new(
            &mut mg,
            NDArrayNumericTensor::from_vec_shape(vec![0i64], &vec![1]).unwrap(),
            rng,
        );
        let token_2d = Unsqueeze::push_new(&mut mg, token_i64, axis0, rng);
        let out = sub_builder.new_tensor_link(rng);
        mg.set_output_map(std::iter::once((token_2d, out.global_id())));
        sub_builder.add_node(SuperGraphNodeMilliOpGraph::new(mg, rng).to_any());
        out
    };

    let sub_logits = sub_builder.new_tensor_link(rng);
    {
        let mut tensor_inputs = vec![
            (decoder_input_ids, "decoder_input_ids".to_string()),
            (sub_encoder_hidden, "encoder_hidden_states".to_string()),
        ];
        let mut tensor_outputs = vec![("logits".to_string(), sub_logits)];
        for (id, pair) in state_names.iter().enumerate() {
            tensor_inputs.push((*sub_state_in.get(&id).unwrap(), pair.0.clone()));
            tensor_outputs.push((pair.1.clone(), *sub_state_out.get(&id).unwrap()));
        }

        sub_builder.add_node(
            SuperGraphNodeModelExecution::new(
                rng,
                sub_model_link,
                1, // model index 1 = decoder
                tensor_inputs,
                tensor_outputs,
            )
            .to_any(),
        );
    }

    {
        let (mut mg, input_map) =
            MilliOpGraph::new([sub_logits.global_id(), sub_finished_in.global_id()], rng);
        let logits = *input_map.get(&sub_logits.global_id()).unwrap();
        let finished_in = *input_map.get(&sub_finished_in.global_id()).unwrap();

        let axis0 = Constant::push_new(
            &mut mg,
            NDArrayNumericTensor::from_vec_shape(vec![0i64], &vec![1]).unwrap(),
            rng,
        );
        let squeezed = Squeeze::push_new(&mut mg, logits, axis0, rng);
        let squeezed = Squeeze::push_new(&mut mg, squeezed, axis0, rng);
        let logits_f32 = Cast::push_new(&mut mg, squeezed, DType::F32, rng);
        let next_token_scalar = ArgMax::push_new(&mut mg, logits_f32, 0, false, false, rng);
        let next_token_scalar = Cast::push_new(&mut mg, next_token_scalar, DType::I64, rng);
        let next_token = Unsqueeze::push_new(&mut mg, next_token_scalar, axis0, rng);

        let eos_token = Constant::push_new(
            &mut mg,
            NDArrayNumericTensor::from_vec_shape(vec![eos_token_id as i64], &vec![1]).unwrap(),
            rng,
        );
        let eos_hit = SimpleBinary::equal(&mut mg, next_token, eos_token, rng);
        let next_finished = SimpleBinary::or(&mut mg, finished_in, eos_hit, rng);
        let emitted_token = Where::push_new(&mut mg, finished_in, eos_token, next_token, rng);

        mg.set_output_map([
            (emitted_token, sub_token_out.global_id()),
            (next_finished, sub_finished_out.global_id()),
        ]);
        sub_builder.add_node(SuperGraphNodeMilliOpGraph::new(mg, rng).to_any());
    }

    let mut sub_inputs = vec![
        sub_model_link.to_any(),
        sub_encoder_hidden.to_any(),
        sub_token_in.to_any(),
        sub_finished_in.to_any(),
    ];
    for id in &state_ids {
        sub_inputs.push(sub_state_in.get(id).unwrap().to_any());
    }

    let mut sub_outputs = vec![sub_token_out.to_any(), sub_finished_out.to_any()];
    for id in &state_ids {
        sub_outputs.push(sub_state_out.get(id).unwrap().to_any());
    }
    let sub_graph = sub_builder.build(rng, &sub_inputs, &sub_outputs);

    // Outer scan wiring.
    let generated_tokens_raw = builder.new_tensor_link(rng);
    let mut state_links = vec![
        SuperGraphLinkTriple::new(token_state_init_link, sub_token_in, sub_token_out),
        SuperGraphLinkTriple::new(finished_state_init_link, sub_finished_in, sub_finished_out),
    ];
    for (id, init_link) in &prefill_final_state_links {
        state_links.push(SuperGraphLinkTriple::new(
            *init_link,
            *sub_state_in.get(id).unwrap(),
            *sub_state_out.get(id).unwrap(),
        ));
    }

    builder.add_node(
        SuperGraphNodeScan::new(
            sub_graph,
            generation_iteration_count_link,
            vec![
                SuperGraphLinkDouble::new(decoder_weights_link, sub_model_link),
                SuperGraphLinkDouble::new(encoder_hidden_link, sub_encoder_hidden),
            ],
            state_links,
            vec![],
            vec![(sub_token_out, generated_tokens_raw, 0)],
            vec![],
            rng,
        )
        .to_any(),
    );

    let generated_tokens = {
        let out = builder.new_tensor_link(rng);
        let (mut mg, input_map) =
            MilliOpGraph::new(std::iter::once(generated_tokens_raw.global_id()), rng);
        let tokens = *input_map.get(&generated_tokens_raw.global_id()).unwrap();
        let axis1 = Constant::push_new(
            &mut mg,
            NDArrayNumericTensor::from_vec_shape(vec![1i64], &vec![1]).unwrap(),
            rng,
        );
        let squeezed = Squeeze::push_new(&mut mg, tokens, axis1, rng);
        mg.set_output_map(std::iter::once((squeezed, out.global_id())));
        builder.add_node(SuperGraphNodeMilliOpGraph::new(mg, rng).to_any());
        out
    };

    let output_token_link = {
        let out = builder.new_tensor_link(rng);
        let (mut mg, input_map) = MilliOpGraph::new(
            [generated_tokens.global_id(), prefix_tokens_link.global_id()],
            rng,
        );
        let generated = *input_map.get(&generated_tokens.global_id()).unwrap();
        let prefix = *input_map.get(&prefix_tokens_link.global_id()).unwrap();
        let full_tokens = MilliConcat::push_new(&mut mg, vec![prefix, generated], 0, rng);
        mg.set_output_map(std::iter::once((full_tokens, out.global_id())));
        builder.add_node(SuperGraphNodeMilliOpGraph::new(mg, rng).to_any());
        out
    };

    let super_graph = builder.build(
        rng,
        &[
            audio_link.to_any(),
            encoder_weights_link.to_any(),
            decoder_weights_link.to_any(),
        ],
        &[output_token_link.to_any()],
    );

    WhisperSuperGraphResult {
        super_graph,
        audio_link,
        encoder_weights_link,
        decoder_weights_link,
        output_token_link,
    }
}

/// Load forced decoder IDs from generation_config.json.
/// Returns `[start_token, forced_token_1, forced_token_2, ...]`.
fn load_forced_decoder_ids(model_dir: &std::path::Path, start_token: u32) -> Vec<u32> {
    let mut ids = vec![start_token];
    let config_path = model_dir.join("generation_config.json");
    if let Ok(data) = std::fs::read_to_string(&config_path)
        && let Ok(json) = serde_json::from_str::<serde_json::Value>(&data)
        && let Some(forced) = json["forced_decoder_ids"].as_array()
    {
        let mut pairs: Vec<(u64, u32)> = forced
            .iter()
            .filter_map(|pair| {
                let arr = pair.as_array()?;
                Some((arr[0].as_u64()?, arr[1].as_u64()? as u32))
            })
            .collect();
        pairs.sort_by_key(|p| p.0);
        for (_, tok) in pairs {
            ids.push(tok);
        }
        eprintln!("Forced decoder IDs: {:?}", ids);
    }
    ids
}
