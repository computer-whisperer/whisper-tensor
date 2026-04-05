/// RWKV7 LoRA fine-tuning example.
///
/// Loads a real RWKV7 .pth checkpoint, injects low-rank adapters on the main
/// linear layers, then trains them with teacher forcing on a short text.
///
/// The base model weights stay frozen (constants in the graph). Only the LoRA
/// A/B matrices are trainable. RNN state is threaded forward between tokens
/// but gradients do not flow through state (no BPTT -- that's a future extension
/// using `external_gradients`).
///
/// STATUS: Currently broken at runtime due to BF16/F32 dtype mismatches in the
/// backward pass and optimizer. The RWKV model operates in BF16, but the backward
/// generation and optimizer create F32 constants (learning rate, relu threshold,
/// sqrt/tanh constants) that interact with BF16 forward values. The NDArray backend
/// does not support mixed-dtype binary ops. This will be resolved by upcoming
/// changes to how milli-ops handle dtypes.
///
/// Usage:
///   cargo run --example rwkv_lora_train --release -- <path-to-rwkv7.pth>
use std::collections::{HashMap, HashSet};
use std::path::Path;

use half::bf16;
use rand::RngExt;
use whisper_tensor::graph::GlobalId;
use whisper_tensor::metadata::TokenizerInfo;
use whisper_tensor::milli_graph::{
    BackwardGenOptions, LossInputSource, LossWiring, MilliGraphGenOptions, MilliOpGraph,
    OptimizerGenOptions, OptimizerKind,
};
use whisper_tensor::numeric_dtype::{NumericDType, ONNXDType};
use whisper_tensor::numeric_scalar::NumericScalar;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::pool::{Pool, SystemPool};
use whisper_tensor::scalar_info::ScalarInfoTyped;
use whisper_tensor::super_graph::cache::SuperGraphCache;
use whisper_tensor::super_graph::data::SuperGraphData;
use whisper_tensor::super_graph::links::{SuperGraphLinkDouble, SuperGraphLinkTriple};
use whisper_tensor::super_graph::nodes::{
    SuperGraphNode, SuperGraphNodeMilliOpGraph, SuperGraphNodeScan,
};
use whisper_tensor::super_graph::{
    SuperGraphAnyLink, SuperGraphBuilder, SuperGraphContext, SuperGraphLink, SuperGraphLinkKind,
};
use whisper_tensor::symbolic_graph::{SymbolicGraphMutator, TensorType};
use whisper_tensor::tensor_rank::DynRank;
use whisper_tensor::tokenizer::{AnyTokenizer, Tokenizer};

type PoolTensor = NumericTensor<'static, DynRank, SystemPool>;

static POOL: SystemPool = SystemPool;

// ---------------------------------------------------------------------------
// Tensor resolution helper (replaces deleted get_initialized_tensors)
// ---------------------------------------------------------------------------

/// Walk all tensors in the SymbolicGraph and resolve Constants and initialized
/// Inputs to pool-based tensors, returning a map from GlobalId to PoolTensor.
/// This replicates the old `get_initialized_tensors(&store)` behavior.
fn resolve_initialized_tensors(
    graph: &whisper_tensor::symbolic_graph::SymbolicGraph,
    store: &whisper_tensor::symbolic_graph::tensor_store::TensorStore,
) -> HashMap<GlobalId, PoolTensor> {
    use whisper_tensor::numeric_tensor::TensorLayout;
    use whisper_tensor::symbolic_graph::{StoredOrNotTensor, TensorType};

    let mut result = HashMap::new();
    let tensors_by_name = graph.get_tensors_by_name();
    // Get all tensor IDs from the name map (covers all named tensors)
    let tensor_ids: Vec<GlobalId> = tensors_by_name.values().copied().collect();

    for &tensor_id in &tensor_ids {
        let Some(info) = graph.get_tensor_info(tensor_id) else {
            continue;
        };
        let stored_ref = match &info.tensor_type {
            TensorType::Constant(s) | TensorType::Input(Some(s)) => Some(s),
            _ => None,
        };
        let Some(stored_ref) = stored_ref else {
            continue;
        };
        match stored_ref {
            StoredOrNotTensor::Stored(store_id) => {
                if let Some(stored) = store.get_tensor(*store_id)
                    && let Some(t) = stored.to_pool_tensor(&POOL)
                {
                    result.insert(tensor_id, t);
                }
            }
            StoredOrNotTensor::Inline(shared) => {
                let src = &*shared.0;
                let layout = TensorLayout::<DynRank>::row_major(src.shape().clone(), src.dtype());
                if let Ok(buf) = POOL.allocate(layout.buffer_size_bytes()) {
                    let mut tensor = NumericTensor::from_parts(buf, layout);
                    for i in 0..src.numel() {
                        tensor.write_element(i, src.read_element(i));
                    }
                    result.insert(tensor_id, tensor);
                }
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// LoRA injection
// ---------------------------------------------------------------------------

/// Target linear layers for LoRA injection (per block).
const LORA_TARGETS: &[&str] = &[
    "att.receptance",
    "att.key",
    "att.value",
    "att.output",
    "ffn.key",
    "ffn.value",
];

const LORA_RANK: usize = 16;

/// Inject LoRA adapters on all target linear layers in the graph.
fn inject_lora(m: &mut SymbolicGraphMutator, rng: &mut impl rand::Rng) -> Vec<GlobalId> {
    let mut trainable = Vec::new();

    let mut targets: Vec<(GlobalId, String)> = Vec::new();
    for pattern in LORA_TARGETS {
        for (op_id, name) in m.find_ops_by_name(pattern) {
            if name.ends_with(".bias") {
                continue;
            }
            targets.push((op_id, name));
        }
    }

    eprintln!("LoRA injection targets: {}", targets.len());

    for (op_id, name) in &targets {
        let inputs = m.op_inputs(*op_id);
        let outputs = m.op_outputs(*op_id);

        if inputs.len() != 2 || outputs.len() != 1 {
            eprintln!(
                "  Skip {name}: {}/{} inputs/outputs",
                inputs.len(),
                outputs.len()
            );
            continue;
        }

        let activation = inputs[0];
        let y = outputs[0];

        let lora_a = m.push_typed_tensor(
            &format!("{name}.lora_a"),
            TensorType::Input(None),
            Some(ONNXDType::Numeric(NumericDType::BF16)),
            None,
            rng,
        );
        m.push_input(lora_a);
        trainable.push(lora_a);

        let lora_b = m.push_typed_tensor(
            &format!("{name}.lora_b"),
            TensorType::Input(None),
            Some(ONNXDType::Numeric(NumericDType::BF16)),
            None,
            rng,
        );
        m.push_input(lora_b);
        trainable.push(lora_b);

        let h = m.push_matmul(&format!("{name}.lora_down"), activation, lora_a, rng);
        let lora_out = m.push_matmul(&format!("{name}.lora_up"), h, lora_b, rng);
        let combined = m.push_add(&format!("{name}.lora_residual"), y, lora_out, rng);

        m.replace_tensor(y, combined);
        eprintln!("  {name}");
    }

    trainable
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn rand_normal(rng: &mut impl rand::Rng) -> f32 {
    let u1: f32 = rng.random::<f32>().max(1e-7);
    let u2: f32 = rng.random::<f32>();
    (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * std::f32::consts::PI * u2).cos()
}

fn make_i32_tensor(data: Vec<i32>, shape: Vec<usize>) -> PoolTensor {
    let shape_u64: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
    NumericTensor::from_fn(shape_u64, NumericDType::I32, &POOL, |i| {
        NumericScalar::from_i32(data[i])
    })
    .unwrap()
}

fn make_f32_tensor(data: Vec<f32>, shape: Vec<usize>) -> PoolTensor {
    let shape_u64: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
    NumericTensor::from_fn(shape_u64, NumericDType::F32, &POOL, |i| {
        NumericScalar::from_f32(data[i])
    })
    .unwrap()
}

fn read_f32_vec(t: &PoolTensor) -> Vec<f32> {
    let view = t.view();
    (0..view.numel())
        .map(|i| view.read_element(i).to_f64() as f32)
        .collect()
}

/// Determine LoRA weight shapes from the base model's weight tensors.
fn compute_lora_shapes(
    graph: &whisper_tensor::symbolic_graph::SymbolicGraph,
    store: &whisper_tensor::symbolic_graph::tensor_store::TensorStore,
    trainable: &[GlobalId],
) -> HashMap<GlobalId, (usize, usize)> {
    let initialized = resolve_initialized_tensors(graph, store);
    let tensors_by_name = graph.get_tensors_by_name();
    let mut shapes = HashMap::new();

    for chunk in trainable.chunks(2) {
        let (lora_a_id, lora_b_id) = (chunk[0], chunk[1]);
        let a_name = graph.get_tensor_name(lora_a_id).unwrap().to_string();
        let base_name = a_name.strip_suffix(".lora_a").unwrap();
        let weight_name = format!("{base_name}.weight");

        if let Some(&weight_id) = tensors_by_name.get(&weight_name)
            && let Some(weight_tensor) = initialized.get(&weight_id)
        {
            let s = weight_tensor.shape();
            // PyTorch weight shape [out, in]; linear() transposes before matmul
            // LoRA: activation @ A @ B, so A=[in, rank], B=[rank, out]
            shapes.insert(lora_a_id, (s[1] as usize, LORA_RANK));
            shapes.insert(lora_b_id, (LORA_RANK, s[0] as usize));
            continue;
        }
        eprintln!("  Warning: no shape info for {base_name}");
    }
    shapes
}

/// Initialize LoRA weights: A gets Kaiming init, B gets zeros.
fn init_lora_weights(
    graph: &whisper_tensor::symbolic_graph::SymbolicGraph,
    trainable: &[GlobalId],
    shapes: &HashMap<GlobalId, (usize, usize)>,
    rng: &mut impl rand::Rng,
) -> HashMap<GlobalId, PoolTensor> {
    let mut params = HashMap::new();
    for (i, &id) in trainable.iter().enumerate() {
        let &(rows, cols) = &shapes[&id];
        let is_a = i % 2 == 0;
        let values: Vec<bf16> = if is_a {
            let std = (1.0 / rows as f32).sqrt();
            (0..rows * cols)
                .map(|_| bf16::from_f32(rand_normal(rng) * std))
                .collect()
        } else {
            vec![bf16::ZERO; rows * cols]
        };
        let name = graph.get_tensor_name(id).unwrap_or("?");
        eprintln!("  {name}: [{rows}, {cols}]");
        let shape_u64 = vec![rows as u64, cols as u64];
        let tensor = NumericTensor::from_fn(shape_u64, NumericDType::BF16, &POOL, |i| {
            NumericScalar::from_bf16(values[i])
        })
        .unwrap();
        params.insert(id, tensor);
    }
    params
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path-to-rwkv7.pth>", args[0]);
        std::process::exit(1);
    }
    let pth_path = Path::new(&args[1]);
    let mut rng = rand::rng();

    // -- 1. Load model --
    eprintln!("Loading RWKV7 from {}...", pth_path.display());
    let onnx_data = whisper_tensor_import::models::llm::rwkv7::load_rwkv7_pth(
        pth_path,
        whisper_tensor_import::onnx_graph::WeightStorageStrategy::OriginReference,
    )?;
    let mut mutator = SymbolicGraphMutator::from_onnx_bytes(
        &onnx_data,
        &mut rng,
        Some(pth_path.parent().unwrap()),
    )?;
    eprintln!("Graph: {} ops", mutator.graph_ref().get_operations().len());

    // -- 2. Inject LoRA --
    eprintln!("\nInjecting LoRA (rank={LORA_RANK})...");
    let trainable = inject_lora(&mut mutator, &mut rng);
    eprintln!("{} trainable LoRA params", trainable.len());

    // -- 3. Extract graph, compute shapes, init LoRA weights --
    let (graph, store) = mutator.get_inner();
    let lora_shapes = compute_lora_shapes(&graph, &store, &trainable);
    eprintln!("\nInitializing LoRA weights...");
    let lora_params = init_lora_weights(&graph, &trainable, &lora_shapes, &mut rng);

    // -- 4. Discover graph structure --
    let names_by_id = graph.get_tensors_by_name();
    let id_by_name: HashMap<&str, GlobalId> =
        names_by_id.iter().map(|(k, v)| (k.as_str(), *v)).collect();

    let token_input_id = id_by_name["token_input"];
    let output_id = id_by_name["output"];

    // State pairs: (in_id, out_id)
    let mut state_pairs: Vec<(GlobalId, GlobalId)> = Vec::new();
    let mut layer = 0;
    loop {
        let mut found = false;
        for prefix in ["time_mixer_x", "channel_mixer_x", "vk_state"] {
            let in_name = format!("{prefix}_in_{layer}");
            let out_name = format!("{prefix}_out_{layer}");
            if let (Some(&i), Some(&o)) = (names_by_id.get(&in_name), names_by_id.get(&out_name)) {
                state_pairs.push((i, o));
                found = true;
            }
        }
        if !found {
            break;
        }
        layer += 1;
    }
    eprintln!("{} state pairs ({layer} layers)", state_pairs.len());

    // -- 5. Build training milli graph --
    eprintln!("\nBuilding training graph...");
    let (loss_graph, loss_info) = MilliOpGraph::cross_entropy_loss(&mut rng);
    let options = MilliGraphGenOptions {
        backward: Some(BackwardGenOptions {
            loss_graph,
            loss_wiring: vec![
                LossWiring {
                    loss_input: loss_info.predictions_input,
                    source: LossInputSource::ForwardOutput(output_id),
                },
                LossWiring {
                    loss_input: loss_info.targets_input,
                    source: LossInputSource::ExternalInput {
                        name: "targets".into(),
                    },
                },
            ],
            loss_output: loss_info.loss_output,
            trainable_params: trainable.clone(),
            stop_gradients: HashSet::new(),
            external_gradients: vec![],
        }),
        optimizer: Some(OptimizerGenOptions {
            kind: OptimizerKind::SGD { lr: 0.01 },
        }),
    };

    let training_graph = graph.generate_milli_graph_with_options(&options, &mut rng)?;
    let meta = training_graph.training_metadata.clone().unwrap();
    let loss_id = meta.loss.unwrap();
    let targets_ext_id = meta.external_inputs[0];

    // -- 6. Categorize all training graph inputs --
    let all_milli_inputs: Vec<GlobalId> = training_graph.get_inputs();
    let base_constants = resolve_initialized_tensors(&graph, &store);

    let trainable_set: HashSet<GlobalId> = trainable.iter().copied().collect();
    let state_in_set: HashSet<GlobalId> = state_pairs.iter().map(|&(i, _)| i).collect();

    let special_ids: HashSet<GlobalId> = {
        let mut s = trainable_set.clone();
        s.extend(&state_in_set);
        s.insert(token_input_id);
        s.insert(targets_ext_id);
        s
    };

    let constant_input_ids: Vec<GlobalId> = all_milli_inputs
        .iter()
        .filter(|id| !special_ids.contains(id))
        .copied()
        .collect();

    eprintln!(
        "Milli graph inputs: {} total ({} constants, {} params, {} state, 2 data)",
        all_milli_inputs.len(),
        constant_input_ids.len(),
        trainable.len(),
        state_pairs.len()
    );

    // -- 7. Tokenize training text --
    let training_text = "The quick brown fox jumps over the lazy dog. \
        A journey of a thousand miles begins with a single step. \
        To be or not to be, that is the question.";

    let tokenizer = AnyTokenizer::from_tokenizer_info(&TokenizerInfo::RWKVWorld);
    let tokens: Vec<u32> = tokenizer.encode(training_text);
    let seq_len = tokens.len() - 1;
    eprintln!("{} tokens -> {seq_len} training steps", tokens.len());

    let vocab_size = {
        let info = graph.get_tensor_info(output_id);
        info.and_then(|i| i.shape())
            .and_then(|s| s.last().cloned())
            .and_then(|d| match d {
                ScalarInfoTyped::Numeric(v) => Some(v as usize),
                _ => None,
            })
            .unwrap_or(65536)
    };

    // Batched for Scan: [seq_len, 1, 1] tokens, [seq_len, 1, 1, vocab_size] targets
    let batched_tokens = make_i32_tensor(
        tokens[..seq_len].iter().map(|&t| t as i32).collect(),
        vec![seq_len, 1, 1],
    );

    let mut one_hot = vec![0.0f32; seq_len * vocab_size];
    for (i, &t) in tokens[1..].iter().enumerate() {
        one_hot[i * vocab_size + t as usize] = 1.0;
    }
    let batched_targets = make_f32_tensor(one_hot, vec![seq_len, 1, 1, vocab_size]);

    // -- 8. Build SuperGraph with Scan --
    eprintln!("\nBuilding SuperGraph...");

    let inner_input_links: Vec<_> = all_milli_inputs
        .iter()
        .map(|&id| SuperGraphAnyLink::tensor(id))
        .collect();

    let mut inner_output_links: Vec<_> = meta
        .param_updates
        .values()
        .map(|&v| SuperGraphAnyLink::tensor(v))
        .collect();
    inner_output_links.push(SuperGraphAnyLink::tensor(loss_id));
    for &(_, state_out) in &state_pairs {
        inner_output_links.push(SuperGraphAnyLink::tensor(state_out));
    }

    let mut inner_builder = SuperGraphBuilder::new();
    inner_builder.add_node(SuperGraphNodeMilliOpGraph::new(training_graph, &mut rng).to_any());
    let inner_graph = inner_builder.build(&mut rng, &inner_input_links, &inner_output_links);

    // Outer links
    let iter_count_link = SuperGraphLink::new(SuperGraphLinkKind::Tensor, &mut rng);
    let outer_tokens_link = SuperGraphLink::new(SuperGraphLinkKind::Tensor, &mut rng);
    let outer_targets_link = SuperGraphLink::new(SuperGraphLinkKind::Tensor, &mut rng);
    let collected_losses_link = SuperGraphLink::new(SuperGraphLinkKind::Tensor, &mut rng);

    // Simple inputs: base constants
    let constant_simple_inputs: Vec<(SuperGraphLink, GlobalId)> = constant_input_ids
        .iter()
        .map(|&id| {
            (
                SuperGraphLink::new(SuperGraphLinkKind::Tensor, &mut rng),
                id,
            )
        })
        .collect();

    let simple_inputs: Vec<SuperGraphLinkDouble> = constant_simple_inputs
        .iter()
        .map(|&(outer, inner_id)| {
            SuperGraphLinkDouble::new(outer, SuperGraphLink::tensor(inner_id))
        })
        .collect();

    // State: LoRA params
    let outer_param_links: Vec<(GlobalId, SuperGraphLink, SuperGraphLink)> = meta
        .param_updates
        .keys()
        .map(|&ext| {
            (
                ext,
                SuperGraphLink::new(SuperGraphLinkKind::Tensor, &mut rng),
                SuperGraphLink::new(SuperGraphLinkKind::Tensor, &mut rng),
            )
        })
        .collect();

    // State: RNN state
    let outer_rnn_links: Vec<(GlobalId, GlobalId, SuperGraphLink, SuperGraphLink)> = state_pairs
        .iter()
        .map(|&(si, so)| {
            (
                si,
                so,
                SuperGraphLink::new(SuperGraphLinkKind::Tensor, &mut rng),
                SuperGraphLink::new(SuperGraphLinkKind::Tensor, &mut rng),
            )
        })
        .collect();

    // Build state triples
    let mut state_links: Vec<SuperGraphLinkTriple> = Vec::new();
    for &(ext_param, outer_init, _) in &outer_param_links {
        let new_param = meta.param_updates[&ext_param];
        state_links.push(SuperGraphLinkTriple::new(
            outer_init,
            SuperGraphLink::tensor(ext_param),
            SuperGraphLink::tensor(new_param),
        ));
    }
    for &(si, so, outer_init, _) in &outer_rnn_links {
        state_links.push(SuperGraphLinkTriple::new(
            outer_init,
            SuperGraphLink::tensor(si),
            SuperGraphLink::tensor(so),
        ));
    }

    // Scan inputs
    let scan_inputs = vec![
        (
            outer_tokens_link,
            SuperGraphLink::tensor(token_input_id),
            0u32,
        ),
        (
            outer_targets_link,
            SuperGraphLink::tensor(targets_ext_id),
            0,
        ),
    ];

    // Scan outputs: per-token loss
    let scan_outputs = vec![(SuperGraphLink::tensor(loss_id), collected_losses_link, 0u32)];

    // Simple outputs: final param + state values
    let mut simple_outputs: Vec<SuperGraphLinkDouble> = outer_param_links
        .iter()
        .map(|&(ext, _, outer_final)| {
            SuperGraphLinkDouble::new(
                SuperGraphLink::tensor(meta.param_updates[&ext]),
                outer_final,
            )
        })
        .collect();
    for &(_, so, _, outer_final) in &outer_rnn_links {
        simple_outputs.push(SuperGraphLinkDouble::new(
            SuperGraphLink::tensor(so),
            outer_final,
        ));
    }

    let scan_node = SuperGraphNodeScan::new(
        inner_graph,
        iter_count_link,
        simple_inputs,
        state_links,
        scan_inputs,
        scan_outputs,
        simple_outputs,
        &mut rng,
    );

    let mut outer_builder = SuperGraphBuilder::new();
    outer_builder.add_node(scan_node.to_any());

    // Outer inputs
    let mut outer_input_links = vec![iter_count_link, outer_tokens_link, outer_targets_link];
    for &(outer, _) in &constant_simple_inputs {
        outer_input_links.push(outer);
    }
    for &(_, outer_init, _) in &outer_param_links {
        outer_input_links.push(outer_init);
    }
    for &(_, _, outer_init, _) in &outer_rnn_links {
        outer_input_links.push(outer_init);
    }

    // Outer outputs
    let mut outer_output_links = vec![collected_losses_link];
    for &(_, _, outer_final) in &outer_param_links {
        outer_output_links.push(outer_final);
    }
    for &(_, _, _, outer_final) in &outer_rnn_links {
        outer_output_links.push(outer_final);
    }

    let epoch_graph = outer_builder.build(&mut rng, &outer_input_links, &outer_output_links);
    eprintln!("SuperGraph built.");

    // -- 9. Load base constants --
    eprintln!("\nLoading base model weights...");
    eprintln!("  {} constant tensors", base_constants.len());

    // Initialize RNN state (zeros from the model's initialized tensors)
    let initial_state: HashMap<GlobalId, PoolTensor> = state_pairs
        .iter()
        .map(|&(si, _)| {
            let t = base_constants
                .get(&si)
                .expect("state tensor should be in initialized tensors")
                .clone();
            (si, t)
        })
        .collect();

    // -- 10. Training loop --
    let num_epochs = 3;
    let mut current_params = lora_params;
    let mut current_state = initial_state;

    let iter_count_tensor =
        NumericTensor::<DynRank, _>::from_fn(vec![1u64], NumericDType::I64, &POOL, |_| {
            NumericScalar::from_i64(seq_len as i64)
        })
        .unwrap();

    eprintln!("\nTraining ({num_epochs} epochs, {seq_len} steps each)...\n");

    for epoch in 0..num_epochs {
        let mut sg_data: SuperGraphData<'_, '_, SystemPool> = SuperGraphData::new();

        sg_data
            .tensors
            .insert(iter_count_link, iter_count_tensor.clone());
        sg_data
            .tensors
            .insert(outer_tokens_link, batched_tokens.clone());
        sg_data
            .tensors
            .insert(outer_targets_link, batched_targets.clone());

        // Constants (simple inputs)
        for &(outer_link, inner_id) in &constant_simple_inputs {
            if let Some(t) = base_constants.get(&inner_id) {
                sg_data.tensors.insert(outer_link, t.clone());
            }
        }

        // LoRA params (state init)
        for &(ext_param, outer_init, _) in &outer_param_links {
            sg_data
                .tensors
                .insert(outer_init, current_params[&ext_param].clone());
        }

        // RNN state (state init)
        for &(si, _, outer_init, _) in &outer_rnn_links {
            sg_data
                .tensors
                .insert(outer_init, current_state[&si].clone());
        }

        let mut observer = ();
        let mut caches = SuperGraphCache::new();
        let mut context = SuperGraphContext::new(&POOL, &mut observer);
        context.caches = Some(&mut caches);

        let results = epoch_graph.run(sg_data, &mut context)?;

        // Extract results
        let losses = read_f32_vec(&results.tensors[&collected_losses_link]);
        let avg_loss = losses.iter().map(|&v| v as f64).sum::<f64>() / losses.len() as f64;
        let first_loss = losses.first().copied().unwrap_or(0.0);
        let last_loss = losses.last().copied().unwrap_or(0.0);

        eprintln!(
            "Epoch {}/{}: loss {first_loss:.4} -> {last_loss:.4} (avg {avg_loss:.4})",
            epoch + 1,
            num_epochs
        );

        // Update params
        for &(ext_param, _, outer_final) in &outer_param_links {
            let view = results.tensors[&outer_final].view();
            current_params.insert(ext_param, view.to_tensor(&POOL).unwrap());
        }

        // Update state for next epoch
        for &(si, _, _, outer_final) in &outer_rnn_links {
            let view = results.tensors[&outer_final].view();
            current_state.insert(si, view.to_tensor(&POOL).unwrap());
        }
    }

    eprintln!("\nDone!");
    Ok(())
}
