use crate::gguf::parser::GgufFile;
use std::path::Path;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::dtype::DType;
use whisper_tensor::graph::GlobalId;
use whisper_tensor::model::Model;
use whisper_tensor::scalar_info::ScalarInfoTyped;
use whisper_tensor::symbolic_graph::tensor_store::StoredTensor;
use whisper_tensor::symbolic_graph::{SymbolicGraphMutator, TensorType};

use super::llama3::GgufModelInfo;

/// Configuration extracted from GGUF metadata for Phi-3 models.
///
/// Key differences from Llama:
/// - Fused QKV projection (`attn_qkv.weight` of shape [hidden, q_dim + 2*kv_dim])
/// - Fused gate_up projection (`ffn_up.weight` of shape [hidden, 2*intermediate])
/// - SU RoPE scaling with long/short frequency factors
/// - SentencePiece tokenizer (not BPE) — tokenizer synthesis not yet supported
pub struct GgufPhi3Config {
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub embedding_length: usize,
    pub feed_forward_length: usize,
    pub rope_theta: f32,
    pub rope_dimension_count: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub original_context_length: usize,
}

impl GgufPhi3Config {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, anyhow::Error> {
        let prefix = "phi3";

        let get_u64 = |key: &str| -> Result<u64, anyhow::Error> {
            gguf.get_metadata(key)
                .and_then(|v| v.as_u64_coerce())
                .ok_or_else(|| anyhow::anyhow!("missing metadata key: {key}"))
        };

        let num_hidden_layers = get_u64(&format!("{prefix}.block_count"))? as usize;
        let num_attention_heads = get_u64(&format!("{prefix}.attention.head_count"))? as usize;
        let num_key_value_heads = get_u64(&format!("{prefix}.attention.head_count_kv"))? as usize;
        let embedding_length = get_u64(&format!("{prefix}.embedding_length"))? as usize;
        let feed_forward_length = get_u64(&format!("{prefix}.feed_forward_length"))? as usize;
        let rope_theta = gguf
            .get_metadata(&format!("{prefix}.rope.freq_base"))
            .and_then(|v| v.as_f32_coerce())
            .unwrap_or(10000.0);
        let rope_dimension_count =
            gguf.get_metadata(&format!("{prefix}.rope.dimension_count"))
                .and_then(|v| v.as_u64_coerce())
                .unwrap_or((embedding_length / num_attention_heads) as u64) as usize;
        let max_position_embeddings = gguf
            .get_metadata(&format!("{prefix}.context_length"))
            .and_then(|v| v.as_u64_coerce())
            .unwrap_or(4096) as usize;
        let rms_norm_eps = gguf
            .get_metadata(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.as_f32_coerce())
            .unwrap_or(1e-5);
        let original_context_length = gguf
            .get_metadata(&format!("{prefix}.rope.scaling.original_context_length"))
            .and_then(|v| v.as_u64_coerce())
            .unwrap_or(4096) as usize;

        Ok(Self {
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            embedding_length,
            feed_forward_length,
            rope_theta,
            rope_dimension_count,
            max_position_embeddings,
            rms_norm_eps,
            original_context_length,
        })
    }
}

/// Build a Phi-3 Model from a pre-parsed GGUF file.
pub fn load_phi3_gguf(gguf: &GgufFile, gguf_path: &Path) -> Result<GgufModelInfo, anyhow::Error> {
    let config = GgufPhi3Config::from_gguf(gguf)?;
    let gguf_path_str = gguf_path
        .canonicalize()
        .unwrap_or_else(|_| gguf_path.to_path_buf())
        .to_string_lossy()
        .into_owned();

    let mut rng = rand::rng();
    let mut mutator = SymbolicGraphMutator::new(&mut rng);
    let build_info;
    {
        let mut b = GraphBuilder::new(&mut mutator, gguf, &gguf_path_str, &config);
        build_info = b.build(&mut rng)?;
    }
    let (graph, tensor_store) = mutator.get_inner();

    let model_name = gguf_path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    Ok(GgufModelInfo {
        model: Model::new_from_graph(model_name, graph, tensor_store),
        state_pairs: build_info.state_pairs,
        token_input_name: build_info.token_input_name,
        logit_output_name: build_info.logit_output_name,
    })
}

struct BuildInfo {
    state_pairs: Vec<(String, String)>,
    token_input_name: String,
    logit_output_name: String,
}

struct GraphBuilder<'a> {
    m: &'a mut SymbolicGraphMutator,
    gguf: &'a GgufFile,
    gguf_path: &'a str,
    config: &'a GgufPhi3Config,
}

impl<'a> GraphBuilder<'a> {
    fn new(
        m: &'a mut SymbolicGraphMutator,
        gguf: &'a GgufFile,
        gguf_path: &'a str,
        config: &'a GgufPhi3Config,
    ) -> Self {
        Self {
            m,
            gguf,
            gguf_path,
            config,
        }
    }

    fn load_weight(
        &mut self,
        name: &str,
        rng: &mut impl rand::Rng,
    ) -> Result<GlobalId, anyhow::Error> {
        let info = self
            .gguf
            .get_tensor(name)
            .ok_or_else(|| anyhow::anyhow!("missing GGUF tensor: {name}"))?;
        let offset = self.gguf.tensor_file_offset(info) as usize;
        let stored = StoredTensor::ExternalGGUF {
            path: self.gguf_path.to_string(),
            tensor_name: name.to_string(),
            offset,
            length: info.byte_length,
            dtype: info.dtype,
            shape: info.dimensions.clone(),
        };
        let store_id = self.m.tensor_store_mut().add_tensor(stored);
        Ok(self
            .m
            .push_stored_tensor(store_id, Some(name.to_string()), rng))
    }

    /// Read raw F32 values from a GGUF tensor (for RoPE scaling factors etc.)
    fn load_f32_weight(
        &mut self,
        name: &str,
        _rng: &mut impl rand::Rng,
    ) -> Result<Option<Vec<f32>>, anyhow::Error> {
        let info = match self.gguf.get_tensor(name) {
            Some(info) => info,
            None => return Ok(None),
        };
        // Read the raw f32 data from the file
        let offset = self.gguf.tensor_file_offset(info);
        let num_elements: u64 = info.dimensions.iter().product();
        let byte_len = num_elements * 4; // F32 = 4 bytes
        use std::io::{Read, Seek, SeekFrom};
        let mut file = std::fs::File::open(self.gguf_path)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut buf = vec![0u8; byte_len as usize];
        file.read_exact(&mut buf)?;
        let floats: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(Some(floats))
    }

    fn input_tensor(
        &mut self,
        name: &str,
        dtype: DType,
        shape: Vec<Option<u64>>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let shape_info: Vec<ScalarInfoTyped<u64>> = shape
            .iter()
            .map(|x| ScalarInfoTyped::Numeric(x.unwrap_or(0)))
            .collect();
        let id = self.m.push_typed_tensor(
            name,
            TensorType::Input(None),
            Some(dtype),
            Some(shape_info),
            rng,
        );
        self.m.push_input(id);
        id
    }

    fn output_tensor(&mut self, tensor_id: GlobalId) {
        self.m.push_output(tensor_id);
    }

    fn build(&mut self, rng: &mut impl rand::Rng) -> Result<BuildInfo, anyhow::Error> {
        let config = self.config;
        let head_dim = config.embedding_length / config.num_attention_heads;
        let half_head_dim = head_dim / 2;
        let q_dim = config.num_attention_heads * head_dim;
        let kv_dim = config.num_key_value_heads * head_dim;
        let eps = config.rms_norm_eps;
        let mut state_pairs = Vec::new();

        // -- Input --
        let input_ids = self.input_tensor("input_ids", DType::I64, vec![Some(1), None], rng);

        // -- Embedding --
        let embed_weight = self.load_weight("token_embd.weight", rng)?;
        let embed_weight = self
            .m
            .push_cast("embed_tokens/cast", embed_weight, DType::F32, rng);
        let x = self
            .m
            .push_gather("embed_tokens", embed_weight, input_ids, 0, rng);

        // -- Precompute RoPE cos/sin caches --
        // Phi-3 128K uses SU RoPE scaling with long/short frequency factors.
        // For inference we use the long factors (suitable for any sequence length).
        let max_len = config.max_position_embeddings;
        let rope_factors = self.load_f32_weight("rope_factors_long.weight", rng)?;

        let (cos_cache_id, sin_cache_id) = {
            let base_inv_freq: Vec<f64> = (0..half_head_dim)
                .map(|i| 1.0 / (config.rope_theta as f64).powf(i as f64 * 2.0 / head_dim as f64))
                .collect();

            // Apply SU scaling factors if present
            let inv_freq: Vec<f64> = if let Some(ref factors) = rope_factors {
                base_inv_freq
                    .iter()
                    .zip(factors.iter())
                    .map(|(&f, &scale)| f / scale as f64)
                    .collect()
            } else {
                base_inv_freq
            };

            let mut cos_vals = Vec::with_capacity(max_len * half_head_dim);
            let mut sin_vals = Vec::with_capacity(max_len * half_head_dim);
            for pos in 0..max_len {
                for &freq in &inv_freq {
                    let angle = pos as f64 * freq;
                    cos_vals.push(angle.cos() as f32);
                    sin_vals.push(angle.sin() as f32);
                }
            }
            let shape = vec![max_len as u64, half_head_dim as u64];
            let cos_nd = NDArrayNumericTensor::from_vec_shape(cos_vals, &shape).unwrap();
            let sin_nd = NDArrayNumericTensor::from_vec_shape(sin_vals, &shape).unwrap();
            let cos_id = self
                .m
                .push_constant_tensor(cos_nd, Some("cos_cache".to_string()), rng);
            let sin_id = self
                .m
                .push_constant_tensor(sin_nd, Some("sin_cache".to_string()), rng);
            (cos_id, sin_id)
        };

        // -- Scale constant for attention --
        let scale_val = (head_dim as f32).sqrt();
        let scale_const = self.m.push_constant_tensor(
            NDArrayNumericTensor::from_vec(vec![scale_val]).to_dyn(),
            None,
            rng,
        );

        // -- Transformer layers --
        let mut layer_output = x;
        for i in 0..config.num_hidden_layers {
            let layer_input = layer_output;

            // Attention norm
            let attn_norm_w = self.load_weight(&format!("blk.{i}.attn_norm.weight"), rng)?;
            let att_normed = self.m.push_rms_norm(
                &format!("blk.{i}.attn_norm"),
                layer_input,
                attn_norm_w,
                eps,
                rng,
            );

            // Fused QKV projection: attn_qkv.weight [hidden, q_dim + 2*kv_dim]
            let qkv_w = self.load_weight(&format!("blk.{i}.attn_qkv.weight"), rng)?;
            let qkv =
                self.m
                    .push_quant_linear(&format!("blk.{i}.attn_qkv"), att_normed, qkv_w, rng);

            // Split Q, K, V from fused output along last axis
            let q = self
                .m
                .push_slice(&format!("blk.{i}.q_slice"), qkv, -1, 0, q_dim as i64, rng);
            let k = self.m.push_slice(
                &format!("blk.{i}.k_slice"),
                qkv,
                -1,
                q_dim as i64,
                (q_dim + kv_dim) as i64,
                rng,
            );
            let v = self.m.push_slice(
                &format!("blk.{i}.v_slice"),
                qkv,
                -1,
                (q_dim + kv_dim) as i64,
                (q_dim + 2 * kv_dim) as i64,
                rng,
            );

            // Reshape to [B, S, H, D] then transpose to [B, H, S, D]
            let q = self.m.push_reshape(
                &format!("blk.{i}.q_reshape"),
                q,
                &[0, 0, config.num_attention_heads as i64, head_dim as i64],
                rng,
            );
            let q = self
                .m
                .push_transpose(&format!("blk.{i}.q_transpose"), q, &[0, 2, 1, 3], rng);
            let k = self.m.push_reshape(
                &format!("blk.{i}.k_reshape"),
                k,
                &[0, 0, config.num_key_value_heads as i64, head_dim as i64],
                rng,
            );
            let k = self
                .m
                .push_transpose(&format!("blk.{i}.k_transpose"), k, &[0, 2, 1, 3], rng);
            let v = self.m.push_reshape(
                &format!("blk.{i}.v_reshape"),
                v,
                &[0, 0, config.num_key_value_heads as i64, head_dim as i64],
                rng,
            );
            let v = self
                .m
                .push_transpose(&format!("blk.{i}.v_transpose"), v, &[0, 2, 1, 3], rng);

            // KV cache inputs
            let kv_cache_k = self.input_tensor(
                &format!("kv_cache_input_k_{i}"),
                DType::F32,
                vec![
                    Some(1),
                    Some(config.num_key_value_heads as u64),
                    None,
                    Some(head_dim as u64),
                ],
                rng,
            );
            let kv_cache_v = self.input_tensor(
                &format!("kv_cache_input_v_{i}"),
                DType::F32,
                vec![
                    Some(1),
                    Some(config.num_key_value_heads as u64),
                    None,
                    Some(head_dim as u64),
                ],
                rng,
            );

            // RoPE position from KV cache length
            let rope_pos = self.m.push_shape(
                &format!("blk.{i}.rope_pos"),
                kv_cache_k,
                Some(2),
                Some(3),
                rng,
            );
            let cos_at_pos = self.m.push_gather(
                &format!("blk.{i}.cos_gather"),
                cos_cache_id,
                rope_pos,
                0,
                rng,
            );
            let sin_at_pos = self.m.push_gather(
                &format!("blk.{i}.sin_gather"),
                sin_cache_id,
                rope_pos,
                0,
                rng,
            );
            let cos_at_pos = self.m.push_reshape(
                &format!("blk.{i}.cos_reshape"),
                cos_at_pos,
                &[1, 1, half_head_dim as i64],
                rng,
            );
            let sin_at_pos = self.m.push_reshape(
                &format!("blk.{i}.sin_reshape"),
                sin_at_pos,
                &[1, 1, half_head_dim as i64],
                rng,
            );

            // Apply RoPE
            let q = self.m.push_rotary_embedding(
                &format!("blk.{i}.rope_q"),
                q,
                cos_at_pos,
                sin_at_pos,
                rng,
            );
            let k = self.m.push_rotary_embedding(
                &format!("blk.{i}.rope_k"),
                k,
                cos_at_pos,
                sin_at_pos,
                rng,
            );

            // Concat with KV cache
            let k_out_name = format!("kv_cache_output_k_{i}");
            let v_out_name = format!("kv_cache_output_v_{i}");
            let k_out = self
                .m
                .push_unknown_tensor(&k_out_name, TensorType::Intermediate, rng);
            let v_out = self
                .m
                .push_unknown_tensor(&v_out_name, TensorType::Intermediate, rng);
            self.m.push_concat_into(
                &format!("blk.{i}.k_concat"),
                vec![kv_cache_k, k],
                2,
                k_out,
                rng,
            );
            self.m.push_concat_into(
                &format!("blk.{i}.v_concat"),
                vec![kv_cache_v, v],
                2,
                v_out,
                rng,
            );

            self.output_tensor(k_out);
            self.output_tensor(v_out);

            state_pairs.push((format!("kv_cache_input_k_{i}"), k_out_name));
            state_pairs.push((format!("kv_cache_input_v_{i}"), v_out_name));

            // GQA: repeat K/V if num_kv_heads != num_attention_heads
            let (k, v) = (k_out, v_out);
            let (k, v) = if config.num_key_value_heads != config.num_attention_heads {
                let n_rep = config.num_attention_heads / config.num_key_value_heads;

                let k_unsq = self.m.push_reshape(
                    &format!("blk.{i}.k_gqa_unsqueeze"),
                    k,
                    &[0, config.num_key_value_heads as i64, 1, -1, head_dim as i64],
                    rng,
                );
                let k_copies: Vec<GlobalId> = vec![k_unsq; n_rep];
                let k_repeated =
                    self.m
                        .push_concat(&format!("blk.{i}.k_gqa_repeat"), k_copies, 2, rng);
                let k = self.m.push_reshape(
                    &format!("blk.{i}.k_gqa_merge"),
                    k_repeated,
                    &[0, config.num_attention_heads as i64, -1, head_dim as i64],
                    rng,
                );

                let v_unsq = self.m.push_reshape(
                    &format!("blk.{i}.v_gqa_unsqueeze"),
                    v,
                    &[0, config.num_key_value_heads as i64, 1, -1, head_dim as i64],
                    rng,
                );
                let v_copies: Vec<GlobalId> = vec![v_unsq; n_rep];
                let v_repeated =
                    self.m
                        .push_concat(&format!("blk.{i}.v_gqa_repeat"), v_copies, 2, rng);
                let v = self.m.push_reshape(
                    &format!("blk.{i}.v_gqa_merge"),
                    v_repeated,
                    &[0, config.num_attention_heads as i64, -1, head_dim as i64],
                    rng,
                );

                (k, v)
            } else {
                (k, v)
            };

            // Attention: scores = Q @ K^T / sqrt(d)
            let kt = self
                .m
                .push_transpose(&format!("blk.{i}.kt"), k, &[0, 1, 3, 2], rng);
            let scores = self
                .m
                .push_matmul(&format!("blk.{i}.attn_scores"), q, kt, rng);
            let scores = self
                .m
                .push_div(&format!("blk.{i}.attn_scale"), scores, scale_const, rng);
            let scores = self
                .m
                .push_softmax(&format!("blk.{i}.attn_softmax"), scores, 3, rng);

            // Weighted sum: output = scores @ V
            let attn_out = self
                .m
                .push_matmul(&format!("blk.{i}.attn_weighted"), scores, v, rng);

            // Transpose back [B, H, S, D] -> [B, S, H, D] and reshape to [B, S, H*D]
            let attn_out = self.m.push_transpose(
                &format!("blk.{i}.attn_out_transpose"),
                attn_out,
                &[0, 2, 1, 3],
                rng,
            );
            let attn_out = self.m.push_reshape(
                &format!("blk.{i}.attn_out_reshape"),
                attn_out,
                &[0, 0, -1],
                rng,
            );

            // Output projection
            let o_w = self.load_weight(&format!("blk.{i}.attn_output.weight"), rng)?;
            let attn_proj =
                self.m
                    .push_quant_linear(&format!("blk.{i}.attn_output"), attn_out, o_w, rng);

            // Residual
            let h = self.m.push_add(
                &format!("blk.{i}.attn_residual"),
                layer_input,
                attn_proj,
                rng,
            );

            // FFN norm
            let ffn_norm_w = self.load_weight(&format!("blk.{i}.ffn_norm.weight"), rng)?;
            let ffn_normed =
                self.m
                    .push_rms_norm(&format!("blk.{i}.ffn_norm"), h, ffn_norm_w, eps, rng);

            // Fused gate_up projection: ffn_up.weight [hidden, 2*intermediate]
            let gate_up_w = self.load_weight(&format!("blk.{i}.ffn_up.weight"), rng)?;
            let gate_up = self.m.push_quant_linear(
                &format!("blk.{i}.ffn_gate_up"),
                ffn_normed,
                gate_up_w,
                rng,
            );

            // Split gate and up
            let gate = self.m.push_slice(
                &format!("blk.{i}.ffn_gate_slice"),
                gate_up,
                -1,
                0,
                config.feed_forward_length as i64,
                rng,
            );
            let up = self.m.push_slice(
                &format!("blk.{i}.ffn_up_slice"),
                gate_up,
                -1,
                config.feed_forward_length as i64,
                (2 * config.feed_forward_length) as i64,
                rng,
            );

            let gate = self.m.push_silu(&format!("blk.{i}.ffn_silu"), gate, rng);
            let hidden = self.m.push_mul(&format!("blk.{i}.ffn_mul"), gate, up, rng);

            let down_w = self.load_weight(&format!("blk.{i}.ffn_down.weight"), rng)?;
            let down = self
                .m
                .push_quant_linear(&format!("blk.{i}.ffn_down"), hidden, down_w, rng);

            // Residual
            layer_output = self
                .m
                .push_add(&format!("blk.{i}.ffn_residual"), h, down, rng);
        }

        // Final norm
        let output_norm_w = self.load_weight("output_norm.weight", rng)?;
        let h = self
            .m
            .push_rms_norm("output_norm", layer_output, output_norm_w, eps, rng);

        // LM head — use output.weight if present, otherwise tie with token_embd.weight
        let (output_w, vocab_size) = if let Some(t) = self.gguf.get_tensor("output.weight") {
            let vocab_size = t.dimensions[0];
            let w = self.load_weight("output.weight", rng)?;
            (w, vocab_size)
        } else {
            let t = self
                .gguf
                .get_tensor("token_embd.weight")
                .ok_or_else(|| anyhow::anyhow!("missing token_embd.weight for tied embeddings"))?;
            let vocab_size = t.dimensions[0];
            let w = self.load_weight("token_embd.weight", rng)?;
            (w, vocab_size)
        };

        let logits = self.m.push_typed_tensor(
            "logits",
            TensorType::Intermediate,
            Some(DType::F32),
            Some(vec![
                ScalarInfoTyped::Numeric(1),
                ScalarInfoTyped::Numeric(0),
                ScalarInfoTyped::Numeric(vocab_size),
            ]),
            rng,
        );
        self.m
            .push_quant_linear_into("output", h, output_w, logits, rng);
        self.output_tensor(logits);

        Ok(BuildInfo {
            state_pairs,
            token_input_name: "input_ids".to_string(),
            logit_output_name: "logits".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_phi3_medium_graph() {
        let path =
            Path::new("/ceph/public/neural_models/llms/Phi-3-medium-128k-instruct-Q5_K_S.gguf");
        if !path.exists() {
            eprintln!("Skipping test: GGUF file not found");
            return;
        }

        let gguf = GgufFile::open(path).unwrap();
        let info = load_phi3_gguf(&gguf, path).expect("Failed to build Phi-3 from GGUF");
        let graph = info.model.get_symbolic_graph();

        // 40 layers: 1 input_ids + 40*2 KV cache = 81 inputs
        let inputs = graph.get_inputs();
        assert_eq!(
            inputs.len(),
            81,
            "Expected 81 inputs (1 token + 40*2 KV cache)"
        );

        // 40*2 KV outputs + 1 logits = 81 outputs
        let outputs = graph.get_outputs();
        assert_eq!(
            outputs.len(),
            81,
            "Expected 81 outputs (40*2 KV cache + logits)"
        );

        let ops = graph.get_operations();
        assert!(
            ops.len() > 100,
            "Expected many operations, got {}",
            ops.len()
        );

        assert_eq!(info.state_pairs.len(), 80, "Expected 40*2 state pairs");
        assert_eq!(info.token_input_name, "input_ids");
        assert_eq!(info.logit_output_name, "logits");

        println!(
            "Phi-3 Medium Q5_K_S GGUF graph: {} inputs, {} outputs, {} operations, {} state pairs",
            inputs.len(),
            outputs.len(),
            ops.len(),
            info.state_pairs.len(),
        );
    }
}
