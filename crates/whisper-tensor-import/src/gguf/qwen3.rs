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

/// Configuration extracted from GGUF metadata for Qwen3 dense models.
///
/// Key differences from Qwen2:
/// - Explicit head_dim via `attention.key_length` (can differ from embedding_length / num_heads)
/// - QK-norm (per-head RMS norm on Q and K after projection)
/// - No QKV bias
/// - Tied embeddings (no output.weight)
pub struct GgufQwen3Config {
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub embedding_length: usize,
    pub head_dim: usize,
    pub feed_forward_length: usize,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
}

impl GgufQwen3Config {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, anyhow::Error> {
        let prefix = "qwen3";

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

        // Qwen3 provides explicit head_dim; fall back to embedding_length / num_heads
        let head_dim = gguf
            .get_metadata(&format!("{prefix}.attention.key_length"))
            .and_then(|v| v.as_u64_coerce())
            .unwrap_or((embedding_length / num_attention_heads) as u64)
            as usize;

        let rope_theta = gguf
            .get_metadata(&format!("{prefix}.rope.freq_base"))
            .and_then(|v| v.as_f32_coerce())
            .unwrap_or(1_000_000.0);
        let max_position_embeddings = gguf
            .get_metadata(&format!("{prefix}.context_length"))
            .and_then(|v| v.as_u64_coerce())
            .unwrap_or(40960) as usize;
        let rms_norm_eps = gguf
            .get_metadata(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.as_f32_coerce())
            .unwrap_or(1e-6);

        Ok(Self {
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            embedding_length,
            head_dim,
            feed_forward_length,
            rope_theta,
            max_position_embeddings,
            rms_norm_eps,
        })
    }
}

/// Build a Qwen3 (dense) Model from a pre-parsed GGUF file.
pub fn load_qwen3_gguf(gguf: &GgufFile, gguf_path: &Path) -> Result<GgufModelInfo, anyhow::Error> {
    let config = GgufQwen3Config::from_gguf(gguf)?;
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
    config: &'a GgufQwen3Config,
}

impl<'a> GraphBuilder<'a> {
    fn new(
        m: &'a mut SymbolicGraphMutator,
        gguf: &'a GgufFile,
        gguf_path: &'a str,
        config: &'a GgufQwen3Config,
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
        let head_dim = config.head_dim;
        let half_head_dim = head_dim / 2;
        let eps = config.rms_norm_eps;
        let mut state_pairs = Vec::new();

        // -- Input --
        let input_ids = self.input_tensor("input_ids", DType::I64, vec![Some(1), None], rng);

        // -- Embedding (dequantize packed weight, then gather) --
        let embed_weight = self.load_weight("token_embd.weight", rng)?;
        let embed_weight = self
            .m
            .push_cast("embed_tokens/cast", embed_weight, DType::F32, rng);
        let x = self
            .m
            .push_gather("embed_tokens", embed_weight, input_ids, 0, rng);

        // -- Precompute RoPE cos/sin caches --
        let max_len = config.max_position_embeddings;
        let (cos_cache_id, sin_cache_id) = {
            let inv_freq: Vec<f64> = (0..half_head_dim)
                .map(|i| 1.0 / (config.rope_theta as f64).powf(i as f64 * 2.0 / head_dim as f64))
                .collect();
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

            // QKV projections (no bias in Qwen3)
            let q_w = self.load_weight(&format!("blk.{i}.attn_q.weight"), rng)?;
            let k_w = self.load_weight(&format!("blk.{i}.attn_k.weight"), rng)?;
            let v_w = self.load_weight(&format!("blk.{i}.attn_v.weight"), rng)?;

            let q = self
                .m
                .push_quant_linear(&format!("blk.{i}.attn_q"), att_normed, q_w, rng);
            let k = self
                .m
                .push_quant_linear(&format!("blk.{i}.attn_k"), att_normed, k_w, rng);
            let v = self
                .m
                .push_quant_linear(&format!("blk.{i}.attn_v"), att_normed, v_w, rng);

            // Reshape Q/K to multi-head: [B, S, H, D]
            let q = self.m.push_reshape(
                &format!("blk.{i}.q_reshape"),
                q,
                &[0, 0, config.num_attention_heads as i64, head_dim as i64],
                rng,
            );
            let k = self.m.push_reshape(
                &format!("blk.{i}.k_reshape"),
                k,
                &[0, 0, config.num_key_value_heads as i64, head_dim as i64],
                rng,
            );

            // QK-norm: per-head RMS norm on Q and K (applied on last dim = head_dim)
            let q_norm_w = self.load_weight(&format!("blk.{i}.attn_q_norm.weight"), rng)?;
            let k_norm_w = self.load_weight(&format!("blk.{i}.attn_k_norm.weight"), rng)?;
            let q = self.m.push_rms_norm(
                &format!("blk.{i}.attn_q_norm"),
                q,
                q_norm_w,
                eps,
                rng,
            );
            let k = self.m.push_rms_norm(
                &format!("blk.{i}.attn_k_norm"),
                k,
                k_norm_w,
                eps,
                rng,
            );

            // Transpose to [B, H, S, D]
            let q = self
                .m
                .push_transpose(&format!("blk.{i}.q_transpose"), q, &[0, 2, 1, 3], rng);
            let k = self
                .m
                .push_transpose(&format!("blk.{i}.k_transpose"), k, &[0, 2, 1, 3], rng);

            // V: reshape + transpose (no QK-norm on V)
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

            // FFN: SwiGLU
            let gate_w = self.load_weight(&format!("blk.{i}.ffn_gate.weight"), rng)?;
            let up_w = self.load_weight(&format!("blk.{i}.ffn_up.weight"), rng)?;
            let down_w = self.load_weight(&format!("blk.{i}.ffn_down.weight"), rng)?;

            let gate =
                self.m
                    .push_quant_linear(&format!("blk.{i}.ffn_gate"), ffn_normed, gate_w, rng);
            let gate = self.m.push_silu(&format!("blk.{i}.ffn_silu"), gate, rng);
            let up = self
                .m
                .push_quant_linear(&format!("blk.{i}.ffn_up"), ffn_normed, up_w, rng);
            let hidden = self.m.push_mul(&format!("blk.{i}.ffn_mul"), gate, up, rng);
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
            // Tied embeddings: reuse token_embd.weight
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
    fn build_qwen3_0_6b_graph() {
        let path = Path::new(
            "/home/christian/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-GGUF/snapshots/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf",
        );
        if !path.exists() {
            eprintln!("Skipping test: GGUF file not found");
            return;
        }

        let gguf = GgufFile::open(path).unwrap();
        let info = load_qwen3_gguf(&gguf, path).expect("Failed to build Qwen3 from GGUF");
        let graph = info.model.get_symbolic_graph();

        // 28 layers: 1 input_ids + 28*2 KV cache = 57 inputs
        let inputs = graph.get_inputs();
        assert_eq!(
            inputs.len(),
            57,
            "Expected 57 inputs (1 token + 28*2 KV cache)"
        );

        // 28*2 KV outputs + 1 logits = 57 outputs
        let outputs = graph.get_outputs();
        assert_eq!(
            outputs.len(),
            57,
            "Expected 57 outputs (28*2 KV cache + logits)"
        );

        let ops = graph.get_operations();
        assert!(
            ops.len() > 100,
            "Expected many operations, got {}",
            ops.len()
        );

        assert_eq!(info.state_pairs.len(), 56, "Expected 28*2 state pairs");
        assert_eq!(info.token_input_name, "input_ids");
        assert_eq!(info.logit_output_name, "logits");

        println!(
            "Qwen3 0.6B GGUF graph: {} inputs, {} outputs, {} operations, {} state pairs",
            inputs.len(),
            outputs.len(),
            ops.len(),
            info.state_pairs.len(),
        );
    }
}
