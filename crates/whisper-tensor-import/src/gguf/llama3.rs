use crate::gguf::parser::GgufFile;
use std::path::Path;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::dtype::DType;
use whisper_tensor::graph::GlobalId;
use whisper_tensor::model::Model;
use whisper_tensor::scalar_info::ScalarInfoTyped;
use whisper_tensor::symbolic_graph::ops::*;
use whisper_tensor::symbolic_graph::tensor_store::StoredTensor;
use whisper_tensor::symbolic_graph::{SymbolicGraphMutator, TensorType};
use whisper_tensor::tensor_rank::DynRank;

/// Configuration extracted from GGUF metadata.
pub struct GgufLlama3Config {
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub embedding_length: usize,
    pub feed_forward_length: usize,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
}

impl GgufLlama3Config {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, anyhow::Error> {
        let arch = gguf
            .architecture()
            .ok_or_else(|| anyhow::anyhow!("missing general.architecture"))?;

        let prefix = arch; // e.g. "llama"

        let get_u64 = |key: &str| -> Result<u64, anyhow::Error> {
            gguf.get_metadata(key)
                .and_then(|v| v.as_u64_coerce())
                .ok_or_else(|| anyhow::anyhow!("missing metadata key: {key}"))
        };

        let num_hidden_layers = get_u64(&format!("{prefix}.block_count"))? as usize;
        let num_attention_heads =
            get_u64(&format!("{prefix}.attention.head_count"))? as usize;
        let num_key_value_heads =
            get_u64(&format!("{prefix}.attention.head_count_kv"))? as usize;
        let embedding_length = get_u64(&format!("{prefix}.embedding_length"))? as usize;
        let feed_forward_length =
            get_u64(&format!("{prefix}.feed_forward_length"))? as usize;
        let rope_theta = gguf
            .get_metadata(&format!("{prefix}.rope.freq_base"))
            .and_then(|v| v.as_f32_coerce())
            .unwrap_or(10000.0);
        let max_position_embeddings = gguf
            .get_metadata(&format!("{prefix}.context_length"))
            .and_then(|v| v.as_u64_coerce())
            .unwrap_or(8192) as usize;
        let rms_norm_eps = gguf
            .get_metadata(&format!("{prefix}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.as_f32_coerce())
            .unwrap_or(1e-5);

        Ok(Self {
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            embedding_length,
            feed_forward_length,
            rope_theta,
            max_position_embeddings,
            rms_norm_eps,
        })
    }
}

/// Information returned by the GGUF loader alongside the Model,
/// so the caller can build the SuperGraph / interface.
pub struct GgufModelInfo {
    pub model: Model,
    /// (input_name, output_name) pairs for KV cache state tensors.
    pub state_pairs: Vec<(String, String)>,
    /// Name of the token input tensor in the graph.
    pub token_input_name: String,
    /// Name of the logit output tensor in the graph.
    pub logit_output_name: String,
}

/// Build a Llama3 Model from a pre-parsed GGUF file.
pub fn load_llama3_gguf(gguf: &GgufFile, gguf_path: &Path) -> Result<GgufModelInfo, anyhow::Error> {
    let config = GgufLlama3Config::from_gguf(gguf)?;
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

/// Internal build result returned by GraphBuilder::build().
struct BuildInfo {
    state_pairs: Vec<(String, String)>,
    token_input_name: String,
    logit_output_name: String,
}

/// Helper that holds a mutable reference to the mutator during construction.
struct GraphBuilder<'a> {
    m: &'a mut SymbolicGraphMutator,
    gguf: &'a GgufFile,
    gguf_path: &'a str,
    config: &'a GgufLlama3Config,
}

impl<'a> GraphBuilder<'a> {
    fn new(
        m: &'a mut SymbolicGraphMutator,
        gguf: &'a GgufFile,
        gguf_path: &'a str,
        config: &'a GgufLlama3Config,
    ) -> Self {
        Self {
            m,
            gguf,
            gguf_path,
            config,
        }
    }

    /// Load a GGUF tensor as a stored constant in the graph.
    fn load_weight(&mut self, name: &str, rng: &mut impl rand::Rng) -> Result<GlobalId, anyhow::Error> {
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
        Ok(self.m.push_stored_tensor(store_id, Some(name.to_string()), rng))
    }

    /// Create an intermediate tensor (no shape/dtype info, used for operation outputs).
    fn intermediate(&mut self, name: &str, rng: &mut impl rand::Rng) -> GlobalId {
        self.m.push_unknown_tensor(name, TensorType::Intermediate, rng)
    }

    /// Create a constant tensor from an NDArray value.
    fn constant(
        &mut self,
        name: Option<String>,
        value: NDArrayNumericTensor<DynRank>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        self.m.push_constant_tensor(value, name, rng)
    }

    /// Add an operation and return its output tensor id.
    fn add_op(
        &mut self,
        name: Option<String>,
        op: AnyOperation,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        self.m.push_operation(name, op, rng)
    }

    /// MatMul: output = a @ b
    fn matmul(
        &mut self,
        name: &str,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let out = self.intermediate(&format!("{name}_out"), rng);
        self.matmul_into(name, a, b, out, rng);
        out
    }

    /// MatMul into a pre-existing output tensor.
    fn matmul_into(
        &mut self,
        name: &str,
        a: GlobalId,
        b: GlobalId,
        out: GlobalId,
        rng: &mut impl rand::Rng,
    ) {
        let op = BinaryOperation::new(a, b, out, WhichBinaryOperation::MatMul, rng);
        self.add_op(Some(name.to_string()), AnyOperation::Binary(op), rng);
    }

    /// Add: output = a + b
    fn add(
        &mut self,
        name: &str,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let out = self.intermediate(&format!("{name}_out"), rng);
        let op = BinaryOperation::new(a, b, out, WhichBinaryOperation::Add, rng);
        self.add_op(Some(name.to_string()), AnyOperation::Binary(op), rng);
        out
    }

    /// Mul: output = a * b
    fn mul(
        &mut self,
        name: &str,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let out = self.intermediate(&format!("{name}_out"), rng);
        let op = BinaryOperation::new(a, b, out, WhichBinaryOperation::Mul, rng);
        self.add_op(Some(name.to_string()), AnyOperation::Binary(op), rng);
        out
    }

    /// Div: output = a / b
    fn div(
        &mut self,
        name: &str,
        a: GlobalId,
        b: GlobalId,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let out = self.intermediate(&format!("{name}_out"), rng);
        let op = BinaryOperation::new(a, b, out, WhichBinaryOperation::Div, rng);
        self.add_op(Some(name.to_string()), AnyOperation::Binary(op), rng);
        out
    }

    /// Sigmoid
    fn sigmoid(
        &mut self,
        name: &str,
        input: GlobalId,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let out = self.intermediate(&format!("{name}_out"), rng);
        let op = UnaryOperation::new(input, out, WhichUnaryOperation::Sigmoid, rng);
        self.add_op(Some(name.to_string()), AnyOperation::Unary(op), rng);
        out
    }

    /// SiLU: output = x * sigmoid(x)
    fn silu(
        &mut self,
        name: &str,
        input: GlobalId,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let sig = self.sigmoid(&format!("{name}_sigmoid"), input, rng);
        self.mul(name, input, sig, rng)
    }

    /// RMS Normalization
    fn rms_norm(
        &mut self,
        name: &str,
        input: GlobalId,
        scale: GlobalId,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let out = self.intermediate(&format!("{name}_out"), rng);
        let op = RMSNormalizationOperation::new(
            input,
            scale,
            None,
            out,
            self.config.rms_norm_eps,
            rng,
        );
        self.add_op(
            Some(name.to_string()),
            AnyOperation::RMSNormalization(op),
            rng,
        );
        out
    }

    /// Reshape
    fn reshape(
        &mut self,
        name: &str,
        input: GlobalId,
        shape_values: Vec<i64>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let shape_tensor = self.constant(
            None,
            NDArrayNumericTensor::from_vec_shape(shape_values.clone(), &vec![shape_values.len() as u64])
                .unwrap(),
            rng,
        );
        let out = self.intermediate(&format!("{name}_out"), rng);
        let op = ReshapeOperation::new(input, shape_tensor, out, rng);
        self.add_op(Some(name.to_string()), AnyOperation::Reshape(op), rng);
        out
    }

    /// Transpose with explicit permutation
    fn transpose(
        &mut self,
        name: &str,
        input: GlobalId,
        perm: Vec<i64>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let out = self.intermediate(&format!("{name}_out"), rng);
        let op = TransposeOperation::new(input, out, Some(perm), rng);
        self.add_op(Some(name.to_string()), AnyOperation::Transpose(op), rng);
        out
    }

    /// Transpose (reverse all dims — for the last-two-dims transpose of K)
    fn transpose_last2(
        &mut self,
        name: &str,
        input: GlobalId,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        // [B, H, S, D] -> [B, H, D, S]
        self.transpose(name, input, vec![0, 1, 3, 2], rng)
    }

    /// Softmax on a given axis
    fn softmax(
        &mut self,
        name: &str,
        input: GlobalId,
        axis: i64,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let out = self.intermediate(&format!("{name}_out"), rng);
        let op = SoftmaxOperation::new(input, out, Some(axis), rng);
        self.add_op(Some(name.to_string()), AnyOperation::Softmax(op), rng);
        out
    }

    /// Gather (embedding lookup)
    fn gather(
        &mut self,
        name: &str,
        data: GlobalId,
        indices: GlobalId,
        axis: i64,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let out = self.intermediate(&format!("{name}_out"), rng);
        let op = GatherOperation::new(data, indices, out, axis, rng);
        self.add_op(Some(name.to_string()), AnyOperation::Gather(op), rng);
        out
    }

    /// Concat along an axis
    fn concat(
        &mut self,
        name: &str,
        inputs: Vec<GlobalId>,
        axis: i64,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let out = self.intermediate(&format!("{name}_out"), rng);
        self.concat_into(name, inputs, axis, out, rng);
        out
    }

    /// Concat into a pre-existing output tensor.
    fn concat_into(
        &mut self,
        name: &str,
        inputs: Vec<GlobalId>,
        axis: i64,
        out: GlobalId,
        rng: &mut impl rand::Rng,
    ) {
        let op = ConcatOperation::new(inputs, out, axis, rng);
        self.add_op(Some(name.to_string()), AnyOperation::Concat(op), rng);
    }

    /// RotaryEmbedding
    fn rotary_embedding(
        &mut self,
        name: &str,
        data: GlobalId,
        cos_cache: GlobalId,
        sin_cache: GlobalId,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let out = self.intermediate(&format!("{name}_out"), rng);
        let op = RotaryEmbeddingOperation::new(
            data,
            cos_cache,
            sin_cache,
            None, // no position_ids — we handle indexing externally
            out,
            false, // not interleaved
            None,
            0, // full head rotation
            rng,
        );
        self.add_op(
            Some(name.to_string()),
            AnyOperation::RotaryEmbedding(op),
            rng,
        );
        out
    }

    /// Shape operation: extract shape[start..end] as a 1D i64 tensor
    fn shape_op(
        &mut self,
        name: &str,
        input: GlobalId,
        start: Option<i64>,
        end: Option<i64>,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let out = self.intermediate(&format!("{name}_out"), rng);
        let op = ShapeOperation::new(input, out, start, end, rng);
        self.add_op(Some(name.to_string()), AnyOperation::Shape(op), rng);
        out
    }

    /// Linear layer: output = input @ weight^T (GGUF weights are [out_features, in_features])
    fn linear(
        &mut self,
        name: &str,
        input: GlobalId,
        weight: GlobalId,
        rng: &mut impl rand::Rng,
    ) -> GlobalId {
        let out = self.intermediate(&format!("{name}_out"), rng);
        self.linear_into(name, input, weight, out, rng);
        out
    }

    /// Linear into a pre-existing output tensor.
    fn linear_into(
        &mut self,
        name: &str,
        input: GlobalId,
        weight: GlobalId,
        out: GlobalId,
        rng: &mut impl rand::Rng,
    ) {
        // GGUF stores weights as [out_features, in_features], same as PyTorch.
        // We need input @ weight^T.
        let wt = self.transpose(
            &format!("{name}_wt"),
            weight,
            vec![1, 0],
            rng,
        );
        self.matmul_into(name, input, wt, out, rng);
    }

    /// Create an input tensor node with dtype and shape info.
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

    /// Register a tensor as a graph output.
    fn output_tensor(&mut self, tensor_id: GlobalId) {
        self.m.push_output(tensor_id);
    }

    /// Build the full Llama3 graph.
    fn build(&mut self, rng: &mut impl rand::Rng) -> Result<BuildInfo, anyhow::Error> {
        let config = self.config;
        let head_dim = config.embedding_length / config.num_attention_heads;
        let half_head_dim = head_dim / 2;
        let mut state_pairs = Vec::new();

        // -- Input --
        let input_ids = self.input_tensor("input_ids", DType::I64, vec![Some(1), None], rng);

        // -- Embedding --
        let embed_weight = self.load_weight("token_embd.weight", rng)?;
        let x = self.gather("embed_tokens", embed_weight, input_ids, 0, rng);

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
                    cos_vals.push(half::bf16::from_f64(angle.cos()));
                    sin_vals.push(half::bf16::from_f64(angle.sin()));
                }
            }
            let shape = vec![max_len as u64, half_head_dim as u64];
            let cos_nd = NDArrayNumericTensor::from_vec_shape(cos_vals, &shape).unwrap();
            let sin_nd = NDArrayNumericTensor::from_vec_shape(sin_vals, &shape).unwrap();
            let cos_id = self.constant(Some("cos_cache".to_string()), cos_nd, rng);
            let sin_id = self.constant(Some("sin_cache".to_string()), sin_nd, rng);
            (cos_id, sin_id)
        };

        // -- KV cache dtype: use the embedding output dtype (BF16 after dequant, or F32) --
        // For now we don't set explicit dtype/shape on intermediate tensors.

        // -- Scale constant for attention --
        let scale_val = (head_dim as f32).sqrt();
        let scale_const = self.constant(
            None,
            NDArrayNumericTensor::from_vec(vec![scale_val]).to_dyn(),
            rng,
        );

        // -- Transformer layers --
        let mut layer_output = x;
        for i in 0..config.num_hidden_layers {
            let layer_input = layer_output;

            // Attention norm
            let attn_norm_w = self.load_weight(&format!("blk.{i}.attn_norm.weight"), rng)?;
            let att_normed = self.rms_norm(
                &format!("blk.{i}.attn_norm"),
                layer_input,
                attn_norm_w,
                rng,
            );

            // QKV projections
            let q_w = self.load_weight(&format!("blk.{i}.attn_q.weight"), rng)?;
            let k_w = self.load_weight(&format!("blk.{i}.attn_k.weight"), rng)?;
            let v_w = self.load_weight(&format!("blk.{i}.attn_v.weight"), rng)?;

            let q = self.linear(&format!("blk.{i}.attn_q"), att_normed, q_w, rng);
            let k = self.linear(&format!("blk.{i}.attn_k"), att_normed, k_w, rng);
            let v = self.linear(&format!("blk.{i}.attn_v"), att_normed, v_w, rng);

            // Reshape to [B, S, H, D] then transpose to [B, H, S, D]
            let q = self.reshape(
                &format!("blk.{i}.q_reshape"),
                q,
                vec![0, 0, config.num_attention_heads as i64, head_dim as i64],
                rng,
            );
            let q = self.transpose(
                &format!("blk.{i}.q_transpose"),
                q,
                vec![0, 2, 1, 3],
                rng,
            );
            let k = self.reshape(
                &format!("blk.{i}.k_reshape"),
                k,
                vec![0, 0, config.num_key_value_heads as i64, head_dim as i64],
                rng,
            );
            let k = self.transpose(
                &format!("blk.{i}.k_transpose"),
                k,
                vec![0, 2, 1, 3],
                rng,
            );
            let v = self.reshape(
                &format!("blk.{i}.v_reshape"),
                v,
                vec![0, 0, config.num_key_value_heads as i64, head_dim as i64],
                rng,
            );
            let v = self.transpose(
                &format!("blk.{i}.v_transpose"),
                v,
                vec![0, 2, 1, 3],
                rng,
            );

            // KV cache inputs
            let kv_cache_k = self.input_tensor(
                &format!("kv_cache_input_k_{i}"),
                DType::BF16,
                vec![Some(1), Some(config.num_key_value_heads as u64), None, Some(head_dim as u64)],
                rng,
            );
            let kv_cache_v = self.input_tensor(
                &format!("kv_cache_input_v_{i}"),
                DType::BF16,
                vec![Some(1), Some(config.num_key_value_heads as u64), None, Some(head_dim as u64)],
                rng,
            );

            // RoPE position from KV cache length
            let rope_pos = self.shape_op(
                &format!("blk.{i}.rope_pos"),
                kv_cache_k,
                Some(2),
                Some(3),
                rng,
            );
            let cos_at_pos = self.gather(
                &format!("blk.{i}.cos_gather"),
                cos_cache_id,
                rope_pos,
                0,
                rng,
            );
            let sin_at_pos = self.gather(
                &format!("blk.{i}.sin_gather"),
                sin_cache_id,
                rope_pos,
                0,
                rng,
            );
            let cos_at_pos = self.reshape(
                &format!("blk.{i}.cos_reshape"),
                cos_at_pos,
                vec![1, 1, half_head_dim as i64],
                rng,
            );
            let sin_at_pos = self.reshape(
                &format!("blk.{i}.sin_reshape"),
                sin_at_pos,
                vec![1, 1, half_head_dim as i64],
                rng,
            );

            // Apply RoPE
            let q = self.rotary_embedding(
                &format!("blk.{i}.rope_q"),
                q,
                cos_at_pos,
                sin_at_pos,
                rng,
            );
            let k = self.rotary_embedding(
                &format!("blk.{i}.rope_k"),
                k,
                cos_at_pos,
                sin_at_pos,
                rng,
            );

            // Concat with KV cache — output tensors have explicit names
            // matching the convention used by the Transformers loader.
            let k_out_name = format!("kv_cache_output_k_{i}");
            let v_out_name = format!("kv_cache_output_v_{i}");
            let k_out = self.m.push_unknown_tensor(&k_out_name, TensorType::Intermediate, rng);
            let v_out = self.m.push_unknown_tensor(&v_out_name, TensorType::Intermediate, rng);
            self.concat_into(&format!("blk.{i}.k_concat"), vec![kv_cache_k, k], 2, k_out, rng);
            self.concat_into(&format!("blk.{i}.v_concat"), vec![kv_cache_v, v], 2, v_out, rng);

            self.output_tensor(k_out);
            self.output_tensor(v_out);

            state_pairs.push((format!("kv_cache_input_k_{i}"), k_out_name));
            state_pairs.push((format!("kv_cache_input_v_{i}"), v_out_name));

            // GQA: repeat K/V if num_kv_heads != num_attention_heads
            let (k, v) = (k_out, v_out);
            let (k, v) = if config.num_key_value_heads != config.num_attention_heads {
                let n_rep = config.num_attention_heads / config.num_key_value_heads;

                // Repeat K: [B, kv_heads, seq, D] -> [B, num_heads, seq, D]
                let k_unsq = self.reshape(
                    &format!("blk.{i}.k_gqa_unsqueeze"),
                    k,
                    vec![0, config.num_key_value_heads as i64, 1, -1, head_dim as i64],
                    rng,
                );
                let k_copies: Vec<GlobalId> = vec![k_unsq; n_rep];
                let k_repeated = self.concat(&format!("blk.{i}.k_gqa_repeat"), k_copies, 2, rng);
                let k = self.reshape(
                    &format!("blk.{i}.k_gqa_merge"),
                    k_repeated,
                    vec![0, config.num_attention_heads as i64, -1, head_dim as i64],
                    rng,
                );

                // Repeat V: same pattern
                let v_unsq = self.reshape(
                    &format!("blk.{i}.v_gqa_unsqueeze"),
                    v,
                    vec![0, config.num_key_value_heads as i64, 1, -1, head_dim as i64],
                    rng,
                );
                let v_copies: Vec<GlobalId> = vec![v_unsq; n_rep];
                let v_repeated = self.concat(&format!("blk.{i}.v_gqa_repeat"), v_copies, 2, rng);
                let v = self.reshape(
                    &format!("blk.{i}.v_gqa_merge"),
                    v_repeated,
                    vec![0, config.num_attention_heads as i64, -1, head_dim as i64],
                    rng,
                );

                (k, v)
            } else {
                (k, v)
            };

            // Attention: scores = Q @ K^T / sqrt(d)
            let kt = self.transpose_last2(&format!("blk.{i}.kt"), k, rng);
            let scores = self.matmul(&format!("blk.{i}.attn_scores"), q, kt, rng);
            let scores = self.div(&format!("blk.{i}.attn_scale"), scores, scale_const, rng);
            let scores = self.softmax(&format!("blk.{i}.attn_softmax"), scores, 3, rng);

            // Weighted sum: output = scores @ V
            let attn_out = self.matmul(&format!("blk.{i}.attn_weighted"), scores, v, rng);

            // Transpose back [B, H, S, D] -> [B, S, H, D] and reshape to [B, S, H*D]
            let attn_out = self.transpose(
                &format!("blk.{i}.attn_out_transpose"),
                attn_out,
                vec![0, 2, 1, 3],
                rng,
            );
            let attn_out = self.reshape(
                &format!("blk.{i}.attn_out_reshape"),
                attn_out,
                vec![0, 0, -1],
                rng,
            );

            // Output projection
            let o_w = self.load_weight(&format!("blk.{i}.attn_output.weight"), rng)?;
            let attn_proj = self.linear(&format!("blk.{i}.attn_output"), attn_out, o_w, rng);

            // Residual
            let h = self.add(&format!("blk.{i}.attn_residual"), layer_input, attn_proj, rng);

            // FFN norm
            let ffn_norm_w = self.load_weight(&format!("blk.{i}.ffn_norm.weight"), rng)?;
            let ffn_normed = self.rms_norm(&format!("blk.{i}.ffn_norm"), h, ffn_norm_w, rng);

            // FFN: SwiGLU
            let gate_w = self.load_weight(&format!("blk.{i}.ffn_gate.weight"), rng)?;
            let up_w = self.load_weight(&format!("blk.{i}.ffn_up.weight"), rng)?;
            let down_w = self.load_weight(&format!("blk.{i}.ffn_down.weight"), rng)?;

            let gate = self.linear(&format!("blk.{i}.ffn_gate"), ffn_normed, gate_w, rng);
            let gate = self.silu(&format!("blk.{i}.ffn_silu"), gate, rng);
            let up = self.linear(&format!("blk.{i}.ffn_up"), ffn_normed, up_w, rng);
            let hidden = self.mul(&format!("blk.{i}.ffn_mul"), gate, up, rng);
            let down = self.linear(&format!("blk.{i}.ffn_down"), hidden, down_w, rng);

            // Residual
            layer_output = self.add(&format!("blk.{i}.ffn_residual"), h, down, rng);
        }

        // Final norm
        let output_norm_w = self.load_weight("output_norm.weight", rng)?;
        let h = self.rms_norm("output_norm", layer_output, output_norm_w, rng);

        // LM head — create output tensor with explicit name and shape info
        let output_w = self.load_weight("output.weight", rng)?;
        let vocab_size = self
            .gguf
            .get_tensor("output.weight")
            .map(|t| t.dimensions[0])
            .unwrap_or(0);
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
        self.linear_into("output", h, output_w, logits, rng);
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
    fn build_llama3_q4_0_graph() {
        let path = Path::new("/ceph/public/neural_models/llms/Llama-3-8B.Q4_0.gguf");
        if !path.exists() {
            eprintln!("Skipping test: GGUF file not found");
            return;
        }

        let gguf = GgufFile::open(path).unwrap();
        let info = load_llama3_gguf(&gguf, path).expect("Failed to build Llama3 from GGUF");
        let graph = info.model.get_symbolic_graph();

        // Should have input_ids + 32 layers * 2 KV cache inputs = 65 inputs
        let inputs = graph.get_inputs();
        assert_eq!(inputs.len(), 65, "Expected 65 inputs (1 token + 32*2 KV cache)");

        // Should have 32 layers * 2 KV outputs + 1 logits = 65 outputs
        let outputs = graph.get_outputs();
        assert_eq!(outputs.len(), 65, "Expected 65 outputs (32*2 KV cache + logits)");

        // Should have operations
        let ops = graph.get_operations();
        assert!(ops.len() > 100, "Expected many operations, got {}", ops.len());

        // State pairs should match layer count
        assert_eq!(info.state_pairs.len(), 64, "Expected 32*2 state pairs");
        assert_eq!(info.token_input_name, "input_ids");
        assert_eq!(info.logit_output_name, "logits");

        println!(
            "Llama3 Q4_0 GGUF graph: {} inputs, {} outputs, {} operations, {} state pairs",
            inputs.len(),
            outputs.len(),
            ops.len(),
            info.state_pairs.len(),
        );
    }

    #[test]
    fn build_llama3_q4_k_m_graph() {
        let path = Path::new("/ceph/public/neural_models/llms/Llama-3-8B.Q4_K_M.gguf");
        if !path.exists() {
            eprintln!("Skipping test: GGUF file not found");
            return;
        }

        let gguf = GgufFile::open(path).unwrap();
        let info = load_llama3_gguf(&gguf, path).expect("Failed to build Llama3 from GGUF");
        let graph = info.model.get_symbolic_graph();
        let inputs = graph.get_inputs();
        let outputs = graph.get_outputs();
        let ops = graph.get_operations();

        println!(
            "Llama3 Q4_K_M GGUF graph: {} inputs, {} outputs, {} operations",
            inputs.len(),
            outputs.len(),
            ops.len()
        );
        assert_eq!(inputs.len(), 65);
        assert_eq!(outputs.len(), 65);
    }
}
