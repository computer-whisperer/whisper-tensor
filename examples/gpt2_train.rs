/// GPT-2 training example using whisper-tensor's training infrastructure.
///
/// Architecture: GPT-2 (configurable depth/width).
/// Loss: Cross-entropy on next-token prediction.
/// Optimizer: SGD (to start; can swap for Adam later).
///
/// This builds the GPT-2 graph explicitly using SymbolicGraphMutator helpers,
/// rather than importing from ONNX.
use std::collections::HashMap;

use rand::RngExt;
use whisper_tensor::graph::GlobalId;
use whisper_tensor::milli_graph::{
    BackwardGenOptions, LossInputSource, LossWiring, MilliGraphGenOptions, MilliOpGraph,
    OptimizerGenOptions, OptimizerKind,
};
use whisper_tensor::numeric_dtype::{NumericDType, ONNXDType};
use whisper_tensor::numeric_scalar::NumericScalar;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::pool::SystemPool;
use whisper_tensor::scalar_info::ScalarInfoTyped;
use whisper_tensor::symbolic_graph::{SymbolicGraphMutator, TensorType};
use whisper_tensor::tensor_rank::DynRank;

type PoolTensor = NumericTensor<'static, DynRank, SystemPool>;

static POOL: SystemPool = SystemPool;

// --- Config ---

struct Gpt2Config {
    vocab_size: usize,
    n_positions: usize,
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
    learning_rate: f32,
    batch_size: usize,
    seq_len: usize,
}

impl Gpt2Config {
    #[allow(dead_code)]
    fn tiny() -> Self {
        Self {
            vocab_size: 50257,
            n_positions: 1024,
            n_embd: 768,
            n_head: 12,
            n_layer: 12,
            learning_rate: 3e-4,
            batch_size: 4,
            seq_len: 128,
        }
    }

    fn debug() -> Self {
        Self {
            vocab_size: 256,
            n_positions: 64,
            n_embd: 64,
            n_head: 4,
            n_layer: 2,
            learning_rate: 1e-3,
            batch_size: 2,
            seq_len: 16,
        }
    }
}

// --- Graph builder ---

struct Gpt2Ids {
    input_ids: GlobalId,
    trainable: Vec<GlobalId>,
    logits: GlobalId,
    /// IDs + data for constants created via push_constant_tensor, needed as
    /// milli_graph external inputs.
    constants: HashMap<GlobalId, PoolTensor>,
}

fn make_f32_tensor(data: Vec<f32>, shape: Vec<usize>) -> PoolTensor {
    let shape_u64: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
    NumericTensor::from_fn(shape_u64, NumericDType::F32, &POOL, |i| {
        NumericScalar::from_f32(data[i])
    })
    .unwrap()
}

fn make_i64_tensor(data: Vec<i64>, shape: Vec<usize>) -> PoolTensor {
    let shape_u64: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
    NumericTensor::from_fn(shape_u64, NumericDType::I64, &POOL, |i| {
        NumericScalar::from_i64(data[i])
    })
    .unwrap()
}

fn rand_normal(rng: &mut impl rand::Rng) -> f32 {
    let u1: f32 = rng.random::<f32>().max(1e-7);
    let u2: f32 = rng.random::<f32>();
    (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * std::f32::consts::PI * u2).cos()
}

fn make_param(
    m: &mut SymbolicGraphMutator,
    trainable: &mut Vec<GlobalId>,
    name: &str,
    shape: Vec<ScalarInfoTyped<u64>>,
    rng: &mut impl rand::Rng,
) -> GlobalId {
    let id = m.push_typed_tensor(
        name,
        TensorType::Input(None),
        Some(ONNXDType::Numeric(NumericDType::F32)),
        Some(shape),
        rng,
    );
    m.push_input(id);
    trainable.push(id);
    id
}

fn build_gpt2_graph(
    config: &Gpt2Config,
    rng: &mut impl rand::Rng,
) -> (SymbolicGraphMutator, Gpt2Ids) {
    let mut m = SymbolicGraphMutator::new(rng);
    let s = |v: u64| ScalarInfoTyped::Numeric(v);
    let batch = config.batch_size as u64;
    let seq = config.seq_len as u64;
    let embd = config.n_embd as u64;
    let vocab = config.vocab_size as u64;
    let n_head = config.n_head as u64;
    let head_dim = embd / n_head;
    let ff_dim = 4 * embd;

    let mut trainable = Vec::new();
    let mut constants: HashMap<GlobalId, PoolTensor> = HashMap::new();

    // Input: token IDs [batch, seq_len] as I64
    let input_ids = m.push_typed_tensor(
        "input_ids",
        TensorType::Input(None),
        Some(ONNXDType::Numeric(NumericDType::I64)),
        Some(vec![s(batch), s(seq)]),
        rng,
    );
    m.push_input(input_ids);

    // Position indices [seq_len] -- constant
    let pos_data: Vec<i64> = (0..config.seq_len as i64).collect();
    let pos_pool = make_i64_tensor(pos_data.clone(), vec![config.seq_len]);
    let position_ids = m.push_constant_tensor(pos_data, vec![seq], None, rng);
    constants.insert(position_ids, pos_pool);

    // --- Embeddings ---
    let wte = make_param(&mut m, &mut trainable, "wte", vec![s(vocab), s(embd)], rng);
    let wpe = make_param(
        &mut m,
        &mut trainable,
        "wpe",
        vec![s(config.n_positions as u64), s(embd)],
        rng,
    );

    let tok_emb = m.push_gather("tok_emb", wte, input_ids, 0, rng);
    let pos_emb = m.push_gather("pos_emb", wpe, position_ids, 0, rng);
    let mut hidden = m.push_add("emb", tok_emb, pos_emb, rng);

    // --- Causal mask ---
    let mut mask_data = vec![0.0f32; (seq * seq) as usize];
    for i in 0..seq as usize {
        for j in (i + 1)..seq as usize {
            mask_data[i * seq as usize + j] = -1e9;
        }
    }
    let mask_pool = make_f32_tensor(mask_data.clone(), vec![1, 1, seq as usize, seq as usize]);
    let causal_mask = m.push_constant_tensor(mask_data, vec![1, 1, seq, seq], None, rng);
    constants.insert(causal_mask, mask_pool);

    // Attention scale constant
    let scale_val = 1.0 / (head_dim as f32).sqrt();
    let scale_pool = make_f32_tensor(vec![scale_val], vec![1]);
    let scale_tensor = m.push_constant_tensor(vec![scale_val], vec![1u64], None, rng);
    constants.insert(scale_tensor, scale_pool);

    // --- Transformer blocks ---
    for layer in 0..config.n_layer {
        let p = format!("h{layer}");

        // Layer norm 1
        let ln1_w = make_param(
            &mut m,
            &mut trainable,
            &format!("{p}.ln1.weight"),
            vec![s(embd)],
            rng,
        );
        let ln1_b = make_param(
            &mut m,
            &mut trainable,
            &format!("{p}.ln1.bias"),
            vec![s(embd)],
            rng,
        );
        let ln1 = m.push_layer_norm(&format!("{p}.ln1"), hidden, ln1_w, Some(ln1_b), 1e-5, rng);

        // Separate Q, K, V projections (avoids Slice which lacks backward)
        let q_w = make_param(
            &mut m,
            &mut trainable,
            &format!("{p}.attn.q.weight"),
            vec![s(embd), s(embd)],
            rng,
        );
        let q_b = make_param(
            &mut m,
            &mut trainable,
            &format!("{p}.attn.q.bias"),
            vec![s(embd)],
            rng,
        );
        let k_w = make_param(
            &mut m,
            &mut trainable,
            &format!("{p}.attn.k.weight"),
            vec![s(embd), s(embd)],
            rng,
        );
        let k_b = make_param(
            &mut m,
            &mut trainable,
            &format!("{p}.attn.k.bias"),
            vec![s(embd)],
            rng,
        );
        let v_w = make_param(
            &mut m,
            &mut trainable,
            &format!("{p}.attn.v.weight"),
            vec![s(embd), s(embd)],
            rng,
        );
        let v_b = make_param(
            &mut m,
            &mut trainable,
            &format!("{p}.attn.v.bias"),
            vec![s(embd)],
            rng,
        );

        let q = m.push_linear(&format!("{p}.attn.q"), ln1, q_w, rng);
        let q = m.push_add(&format!("{p}.attn.q_b"), q, q_b, rng);
        let k = m.push_linear(&format!("{p}.attn.k"), ln1, k_w, rng);
        let k = m.push_add(&format!("{p}.attn.k_b"), k, k_b, rng);
        let v = m.push_linear(&format!("{p}.attn.v"), ln1, v_w, rng);
        let v = m.push_add(&format!("{p}.attn.v_b"), v, v_b, rng);

        // Reshape to multi-head and transpose: [B,S,E] -> [B,S,H,D] -> [B,H,S,D]
        let mut reshape = |m: &mut SymbolicGraphMutator, name: &str, x: GlobalId| -> GlobalId {
            let r = m.push_reshape(
                &format!("{name}_mh"),
                x,
                &[batch as i64, seq as i64, n_head as i64, head_dim as i64],
                rng,
            );
            m.push_transpose(&format!("{name}_t"), r, &[0, 2, 1, 3], rng)
        };
        let q = reshape(&mut m, &format!("{p}.q"), q);
        let k = reshape(&mut m, &format!("{p}.k"), k);
        let v = reshape(&mut m, &format!("{p}.v"), v);

        // Attention: softmax((Q @ K^T) / sqrt(d) + mask) @ V
        let kt = m.push_transpose(&format!("{p}.kt"), k, &[0, 1, 3, 2], rng);
        let scores = m.push_matmul(&format!("{p}.scores"), q, kt, rng);
        let scores = m.push_mul(&format!("{p}.scale"), scores, scale_tensor, rng);
        let scores = m.push_add(&format!("{p}.mask"), scores, causal_mask, rng);
        let weights = m.push_softmax(&format!("{p}.attn_w"), scores, -1, rng);
        let ctx = m.push_matmul(&format!("{p}.ctx"), weights, v, rng);

        // Reshape back: [B,H,S,D] -> [B,S,H,D] -> [B,S,E]
        let ctx = m.push_transpose(&format!("{p}.ctx_t"), ctx, &[0, 2, 1, 3], rng);
        let ctx = m.push_reshape(
            &format!("{p}.ctx_r"),
            ctx,
            &[batch as i64, seq as i64, embd as i64],
            rng,
        );

        // Output projection + residual
        let proj_w = make_param(
            &mut m,
            &mut trainable,
            &format!("{p}.attn.proj.weight"),
            vec![s(embd), s(embd)],
            rng,
        );
        let proj_b = make_param(
            &mut m,
            &mut trainable,
            &format!("{p}.attn.proj.bias"),
            vec![s(embd)],
            rng,
        );
        let proj = m.push_linear(&format!("{p}.attn.proj"), ctx, proj_w, rng);
        let proj = m.push_add(&format!("{p}.attn.proj_b"), proj, proj_b, rng);
        hidden = m.push_add(&format!("{p}.res1"), hidden, proj, rng);

        // Layer norm 2
        let ln2_w = make_param(
            &mut m,
            &mut trainable,
            &format!("{p}.ln2.weight"),
            vec![s(embd)],
            rng,
        );
        let ln2_b = make_param(
            &mut m,
            &mut trainable,
            &format!("{p}.ln2.bias"),
            vec![s(embd)],
            rng,
        );
        let ln2 = m.push_layer_norm(&format!("{p}.ln2"), hidden, ln2_w, Some(ln2_b), 1e-5, rng);

        // MLP: fc -> gelu -> proj
        let fc_w = make_param(
            &mut m,
            &mut trainable,
            &format!("{p}.mlp.fc.weight"),
            vec![s(ff_dim), s(embd)],
            rng,
        );
        let fc_b = make_param(
            &mut m,
            &mut trainable,
            &format!("{p}.mlp.fc.bias"),
            vec![s(ff_dim)],
            rng,
        );
        let h = m.push_linear(&format!("{p}.mlp.fc"), ln2, fc_w, rng);
        let h = m.push_add(&format!("{p}.mlp.fc_b"), h, fc_b, rng);
        let h = m.push_gelu(&format!("{p}.mlp.gelu"), h, rng);

        let mp_w = make_param(
            &mut m,
            &mut trainable,
            &format!("{p}.mlp.proj.weight"),
            vec![s(embd), s(ff_dim)],
            rng,
        );
        let mp_b = make_param(
            &mut m,
            &mut trainable,
            &format!("{p}.mlp.proj.bias"),
            vec![s(embd)],
            rng,
        );
        let mlp = m.push_linear(&format!("{p}.mlp.proj"), h, mp_w, rng);
        let mlp = m.push_add(&format!("{p}.mlp.proj_b"), mlp, mp_b, rng);
        hidden = m.push_add(&format!("{p}.res2"), hidden, mlp, rng);
    }

    // Final layer norm
    let ln_f_w = make_param(&mut m, &mut trainable, "ln_f.weight", vec![s(embd)], rng);
    let ln_f_b = make_param(&mut m, &mut trainable, "ln_f.bias", vec![s(embd)], rng);
    let hidden = m.push_layer_norm("ln_f", hidden, ln_f_w, Some(ln_f_b), 1e-5, rng);

    // LM head (weight-tied with wte): logits = hidden @ wte^T
    // Reshape to [batch*seq, embd] for the matmul, then loss expects [N, C]
    let hidden_2d = m.push_reshape("lm_head_reshape", hidden, &[-1, embd as i64], rng);
    let logits = m.push_linear("lm_head", hidden_2d, wte, rng);
    m.push_output(logits);

    (
        m,
        Gpt2Ids {
            input_ids,
            trainable,
            logits,
            constants,
        },
    )
}

// --- Weight initialization ---

fn init_param(name: &str, shape: &[usize], rng: &mut impl rand::Rng) -> PoolTensor {
    let n: usize = shape.iter().product();
    let std_dev = 0.02f32;
    let data: Vec<f32> = if name.ends_with(".bias") {
        vec![0.0; n]
    } else if name.contains("ln1.weight") || name.contains("ln2.weight") || name == "ln_f.weight" {
        vec![1.0; n]
    } else {
        (0..n).map(|_| rand_normal(rng) * std_dev).collect()
    };
    make_f32_tensor(data, shape.to_vec())
}

fn pool_eval_milli<'p>(
    graph: &MilliOpGraph,
    inputs: &HashMap<GlobalId, PoolTensor>,
    pool: &'p SystemPool,
) -> HashMap<GlobalId, NumericTensor<'p, DynRank, SystemPool>> {
    let views: HashMap<GlobalId, _> = inputs.iter().map(|(&id, t)| (id, t.view())).collect();
    let view_refs: HashMap<GlobalId, _> = views.iter().map(|(&id, v)| (id, v)).collect();
    graph.pool_eval(&view_refs, pool).unwrap()
}

fn read_f32_vec(t: &NumericTensor<'_, DynRank, SystemPool>) -> Vec<f32> {
    let view = t.view();
    (0..view.numel())
        .map(|i| view.read_element(i).to_f64() as f32)
        .collect()
}

// --- Main ---

fn main() {
    let rng = &mut rand::rng();
    let config = Gpt2Config::debug();

    println!(
        "Building GPT-2: {} layers, d_model={}, {} heads, vocab={}, seq={}",
        config.n_layer, config.n_embd, config.n_head, config.vocab_size, config.seq_len
    );

    let (mutator, ids) = build_gpt2_graph(&config, rng);
    let (graph, _store) = mutator.get_inner();
    println!("SymbolicGraph: {} ops", graph.get_operations().len());

    // Build training graph
    let (loss_graph, loss_info) = MilliOpGraph::cross_entropy_loss(rng);

    let options = MilliGraphGenOptions {
        backward: Some(BackwardGenOptions {
            loss_graph,
            loss_wiring: vec![
                LossWiring {
                    loss_input: loss_info.predictions_input,
                    source: LossInputSource::ForwardOutput(ids.logits),
                },
                LossWiring {
                    loss_input: loss_info.targets_input,
                    source: LossInputSource::ExternalInput {
                        name: "targets".into(),
                    },
                },
            ],
            loss_output: loss_info.loss_output,
            trainable_params: ids.trainable.clone(),
            stop_gradients: std::collections::HashSet::new(),
            external_gradients: vec![],
        }),
        optimizer: Some(OptimizerGenOptions {
            kind: OptimizerKind::SGD {
                lr: config.learning_rate,
            },
        }),
    };

    println!("Generating training graph...");
    let training_graph = match graph.generate_milli_graph_with_options(&options, rng) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed: {e}");
            return;
        }
    };
    let meta = training_graph.training_metadata.as_ref().unwrap();
    println!(
        "Training graph: {} trainable params",
        meta.param_updates.len(),
    );

    // Initialize parameters
    let mut param_values: HashMap<GlobalId, PoolTensor> = HashMap::new();
    for &pid in &ids.trainable {
        let info = graph.get_tensor_info(pid).unwrap();
        let name = info.onnx_name.as_deref().unwrap_or("?");
        let shape: Vec<usize> = info
            .shape
            .as_ref()
            .unwrap()
            .iter()
            .map(|s| match s {
                ScalarInfoTyped::Numeric(v) => *v as usize,
                _ => panic!("dynamic dim"),
            })
            .collect();
        param_values.insert(pid, init_param(name, &shape, rng));
    }

    // Initialize optimizer state
    for (_key, state) in &meta.state_updates {
        let src = &param_values[&_key.0];
        let n = src.numel();
        param_values.insert(
            state.input,
            make_f32_tensor(
                vec![0.0f32; n],
                src.shape().iter().map(|&x| x as usize).collect(),
            ),
        );
    }

    // Synthetic training data
    let num_batches = 10;
    let mut all_tokens: Vec<Vec<i64>> = Vec::new();
    for _ in 0..num_batches * config.batch_size {
        let tokens: Vec<i64> = (0..config.seq_len + 1)
            .map(|_| (rng.random::<u32>() % config.vocab_size as u32) as i64)
            .collect();
        all_tokens.push(tokens);
    }

    println!("\nTraining...");
    for epoch in 0..3 {
        let mut epoch_loss = 0.0f32;
        for batch_idx in 0..num_batches {
            let batch_start = batch_idx * config.batch_size;
            let mut input_data = Vec::new();
            let mut target_data = Vec::new();

            for i in 0..config.batch_size {
                let tokens = &all_tokens[batch_start + i];
                input_data.extend_from_slice(&tokens[..config.seq_len]);
                for t in 0..config.seq_len {
                    let target = tokens[t + 1] as usize;
                    let mut one_hot = vec![0.0f32; config.vocab_size];
                    one_hot[target] = 1.0;
                    target_data.extend_from_slice(&one_hot);
                }
            }

            let input_tensor = make_i64_tensor(input_data, vec![config.batch_size, config.seq_len]);
            let target_tensor = make_f32_tensor(
                target_data,
                vec![config.batch_size * config.seq_len, config.vocab_size],
            );

            let mut inputs: HashMap<GlobalId, PoolTensor> = HashMap::new();
            inputs.insert(ids.input_ids, input_tensor);
            inputs.insert(meta.external_inputs[0], target_tensor);
            // Supply graph constants
            for (k, v) in &ids.constants {
                inputs.insert(*k, v.clone());
            }
            for (&pid, val) in &param_values {
                inputs.insert(pid, val.clone());
            }

            let results = pool_eval_milli(&training_graph, &inputs, &POOL);

            let loss_val = read_f32_vec(&results[&meta.loss.unwrap()])[0];
            epoch_loss += loss_val;

            for (&ext, &new_out) in &meta.param_updates {
                let result_view = results[&new_out].view();
                param_values.insert(ext, result_view.to_tensor(&POOL).unwrap());
            }
            for state in meta.state_updates.values() {
                let result_view = results[&state.output].view();
                param_values.insert(state.input, result_view.to_tensor(&POOL).unwrap());
            }

            println!(
                "  epoch {} batch {}/{}: loss = {:.4}",
                epoch, batch_idx, num_batches, loss_val,
            );
        }
        println!(
            "Epoch {} avg loss: {:.4}\n",
            epoch,
            epoch_loss / num_batches as f32,
        );
    }
}
