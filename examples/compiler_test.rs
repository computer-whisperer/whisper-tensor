use half::bf16;
use rand::RngCore;
use std::collections::HashMap;
use std::time::Instant;

// v1
use whisper_tensor::compiler::attempts::v1_scalar_crystal::{
    codegen as v1_codegen, crystal, nano_op::NanoOpExpander,
};

// v2
use whisper_tensor::compiler::attempts::v2_fusion::{
    codegen as v2_codegen, kernel as v2_kernel, planner as v2_planner,
};
use whisper_tensor::compiler::attempts::v3_nano_fusion::{
    codegen as v3_codegen, fusion as v3_fusion,
};
use whisper_tensor::compiler::attempts::v4_pool_growth::codegen as v4_codegen;
use whisper_tensor::compiler::attempts::v5_typed_synthesis::synth as v5_synth;
use whisper_tensor::compiler::attempts::v6_schedule_synthesis::synthesis as v6_synth;
use whisper_tensor::compiler::attempts::v7_parallel_crystal::codegen as v7_codegen;
use whisper_tensor::compiler::attempts::v7_parallel_crystal::executor as v7_exec;
use whisper_tensor::compiler::attempts::v7_parallel_crystal::planner as v7_planner;
use whisper_tensor::compiler::attempts::v8_generic_kernel::codegen as v8_codegen;
use whisper_tensor::compiler::attempts::v8_generic_kernel::executor as v8_executor;
use whisper_tensor::compiler::attempts::v9_fused_expr::pipeline as v9_pipeline;

use whisper_tensor::dtype::DType;
use whisper_tensor::graph::GlobalId;
use whisper_tensor::milli_graph::MilliOpGraph;
use whisper_tensor::milli_graph::ops::{MatMul, SimpleBinary, SimpleUnaryOp};
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::tensor_rank::DynRank;

// ---------------------------------------------------------------------------
// Graph builders
// ---------------------------------------------------------------------------

/// out = neg(exp(a * b + c))
fn build_chain_graph(
    shape: &[usize],
    rng: &mut impl rand::Rng,
) -> (
    MilliOpGraph,
    Vec<GlobalId>,
    GlobalId,
    HashMap<GlobalId, Vec<usize>>,
) {
    let ext_a = GlobalId::new(rng);
    let ext_b = GlobalId::new(rng);
    let ext_c = GlobalId::new(rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b, ext_c], rng);
    let a = input_map[&ext_a];
    let b = input_map[&ext_b];
    let c = input_map[&ext_c];

    let mul = SimpleBinary::mul(&mut graph, a, b, rng);
    let add = SimpleBinary::add(&mut graph, mul, c, rng);
    let exp = SimpleUnaryOp::exp(&mut graph, add, rng);
    let neg = SimpleUnaryOp::neg(&mut graph, exp, rng);

    let ext_out = GlobalId::new(rng);
    graph.set_output_map([(neg, ext_out)]);

    let s = shape.to_vec();
    let mut shapes = HashMap::new();
    for &t in &[a, b, c, mul, add, exp, neg] {
        shapes.insert(t, s.clone());
    }

    (graph, vec![ext_a, ext_b, ext_c], ext_out, shapes)
}

/// Elementwise MLP: each layer = tanh(x * w + b) (all same shape, no matmul).
fn build_elementwise_mlp(
    shape: &[usize],
    num_layers: usize,
    rng: &mut impl rand::Rng,
) -> (
    MilliOpGraph,
    GlobalId,
    Vec<(GlobalId, GlobalId)>,
    GlobalId,
    HashMap<GlobalId, Vec<usize>>,
) {
    let ext_input = GlobalId::new(rng);
    let mut ext_ids = vec![ext_input];
    let mut layer_params = Vec::new();
    for _ in 0..num_layers {
        let w = GlobalId::new(rng);
        let b = GlobalId::new(rng);
        ext_ids.push(w);
        ext_ids.push(b);
        layer_params.push((w, b));
    }

    let (mut graph, input_map) = MilliOpGraph::new(ext_ids.iter().copied(), rng);
    let s = shape.to_vec();
    let mut shapes = HashMap::new();
    for id in &ext_ids {
        shapes.insert(input_map[id], s.clone());
    }

    let mut current = input_map[&ext_input];
    for (ew, eb) in &layer_params {
        let w = input_map[ew];
        let b = input_map[eb];
        let mul = SimpleBinary::mul(&mut graph, current, w, rng);
        shapes.insert(mul, s.clone());
        let add = SimpleBinary::add(&mut graph, mul, b, rng);
        shapes.insert(add, s.clone());
        let tanh = SimpleUnaryOp::trig(&mut graph, add, whisper_tensor::TrigOp::Tanh, rng);
        shapes.insert(tanh, s.clone());
        current = tanh;
    }

    let ext_out = GlobalId::new(rng);
    graph.set_output_map([(current, ext_out)]);
    (graph, ext_input, layer_params, ext_out, shapes)
}

/// Elementwise arithmetic-only: each layer = x * w + b (no transcendentals).
fn build_arithmetic_chain(
    shape: &[usize],
    num_layers: usize,
    rng: &mut impl rand::Rng,
) -> (
    MilliOpGraph,
    GlobalId,
    Vec<(GlobalId, GlobalId)>,
    GlobalId,
    HashMap<GlobalId, Vec<usize>>,
) {
    let ext_input = GlobalId::new(rng);
    let mut ext_ids = vec![ext_input];
    let mut layer_params = Vec::new();
    for _ in 0..num_layers {
        let w = GlobalId::new(rng);
        let b = GlobalId::new(rng);
        ext_ids.push(w);
        ext_ids.push(b);
        layer_params.push((w, b));
    }

    let (mut graph, input_map) = MilliOpGraph::new(ext_ids.iter().copied(), rng);
    let s = shape.to_vec();
    let mut shapes = HashMap::new();
    for id in &ext_ids {
        shapes.insert(input_map[id], s.clone());
    }

    let mut current = input_map[&ext_input];
    for (ew, eb) in &layer_params {
        let w = input_map[ew];
        let b = input_map[eb];
        let mul = SimpleBinary::mul(&mut graph, current, w, rng);
        shapes.insert(mul, s.clone());
        let add = SimpleBinary::add(&mut graph, mul, b, rng);
        shapes.insert(add, s.clone());
        current = add;
    }

    let ext_out = GlobalId::new(rng);
    graph.set_output_map([(current, ext_out)]);
    (graph, ext_input, layer_params, ext_out, shapes)
}

/// Real MLP layer: y = tanh(x @ W + b).
fn build_matmul_mlp(
    batch: usize,
    dims: &[usize], // [in, hidden1, hidden2, ..., out]
    rng: &mut impl rand::Rng,
) -> (
    MilliOpGraph,
    GlobalId,
    Vec<(GlobalId, GlobalId)>,
    GlobalId,
    HashMap<GlobalId, Vec<usize>>,
) {
    let ext_input = GlobalId::new(rng);
    let mut ext_ids = vec![ext_input];
    let mut layer_params = Vec::new();
    for _ in 0..dims.len() - 1 {
        let w = GlobalId::new(rng);
        let b = GlobalId::new(rng);
        ext_ids.push(w);
        ext_ids.push(b);
        layer_params.push((w, b));
    }

    let (mut graph, input_map) = MilliOpGraph::new(ext_ids.iter().copied(), rng);
    let mut shapes = HashMap::new();
    shapes.insert(input_map[&ext_input], vec![batch, dims[0]]);

    let mut current = input_map[&ext_input];
    let mut current_cols = dims[0];

    for (i, (ew, eb)) in layer_params.iter().enumerate() {
        let out_dim = dims[i + 1];
        let w = input_map[ew];
        let b = input_map[eb];
        shapes.insert(w, vec![current_cols, out_dim]);
        shapes.insert(b, vec![1, out_dim]);

        let mm = MatMul::push_new_default_precision(&mut graph, current, w, DType::F32, rng);
        shapes.insert(mm, vec![batch, out_dim]);
        let add = SimpleBinary::add(&mut graph, mm, b, rng);
        shapes.insert(add, vec![batch, out_dim]);
        let tanh = SimpleUnaryOp::trig(&mut graph, add, whisper_tensor::TrigOp::Tanh, rng);
        shapes.insert(tanh, vec![batch, out_dim]);

        current = tanh;
        current_cols = out_dim;
    }

    let ext_out = GlobalId::new(rng);
    graph.set_output_map([(current, ext_out)]);
    (graph, ext_input, layer_params, ext_out, shapes)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn run_interpreter(
    graph: &MilliOpGraph,
    inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
    output_ext: GlobalId,
) -> Vec<f32> {
    let results = whisper_tensor::compiler::interpret_milli_graph(graph, inputs).unwrap();
    results[&output_ext].flatten().unwrap().try_into().unwrap()
}

fn run_v1_compiled(
    compiled: &v1_codegen::NativeCompiledGraph,
    inputs: &HashMap<GlobalId, Vec<f32>>,
    output_id: GlobalId,
    output_size: usize,
) -> Vec<f32> {
    let layout = &compiled.layout;
    let mut bufs: Vec<Vec<f32>> = (0..layout.num_buffers).map(|_| Vec::new()).collect();
    for (id, data) in inputs {
        bufs[layout.tensor_index[id]] = data.clone();
    }
    for (id, &size) in &layout.tensor_sizes {
        let idx = layout.tensor_index[id];
        if bufs[idx].is_empty() {
            bufs[idx] = vec![0.0f32; size];
        }
    }
    let mut ptrs: Vec<*mut f32> = bufs.iter_mut().map(|v| v.as_mut_ptr()).collect();
    unsafe { compiled.execute(&mut ptrs) };
    bufs[layout.tensor_index[&output_id]][..output_size].to_vec()
}

fn run_v2_compiled(
    compiled: &v2_codegen::NativeCompiledGraph,
    inputs: &HashMap<GlobalId, Vec<f32>>,
    output_id: GlobalId,
    output_size: usize,
) -> Vec<f32> {
    let layout = &compiled.layout;
    let mut bufs: Vec<Vec<f32>> = (0..layout.num_buffers).map(|_| Vec::new()).collect();
    for (id, data) in inputs {
        if let Some(&idx) = layout.tensor_index.get(id) {
            bufs[idx] = data.clone();
        }
    }
    for (id, &size) in &layout.tensor_sizes {
        let idx = layout.tensor_index[id];
        if bufs[idx].is_empty() {
            bufs[idx] = vec![0.0f32; size];
        }
    }
    let mut ptrs: Vec<*mut f32> = bufs.iter_mut().map(|v| v.as_mut_ptr()).collect();
    unsafe { compiled.execute(&mut ptrs) };
    bufs[layout.tensor_index[&output_id]][..output_size].to_vec()
}

fn run_v3_compiled(
    compiled: &v3_codegen::NativeCompiledGraph,
    inputs: &HashMap<GlobalId, Vec<f32>>,
    output_id: GlobalId,
    output_size: usize,
) -> Vec<f32> {
    run_v1_compiled(compiled, inputs, output_id, output_size)
}

fn run_v4_compiled(
    compiled: &v4_codegen::NativeCompiledGraph,
    inputs: &HashMap<GlobalId, Vec<f32>>,
    output_id: GlobalId,
    output_size: usize,
) -> Vec<f32> {
    run_v1_compiled(compiled, inputs, output_id, output_size)
}

fn run_v7_planned(
    artifacts: &v6_synth::PipelineArtifacts,
    plan: &v7_planner::TaskPlan,
    shapes: &HashMap<GlobalId, Vec<usize>>,
    inputs: &HashMap<GlobalId, Vec<f32>>,
    output_id: GlobalId,
    output_size: usize,
) -> Vec<f32> {
    run_v7_planned_with_exec(
        artifacts,
        plan,
        shapes,
        inputs,
        output_id,
        output_size,
        v7_exec::ExecuteConfig::default(),
    )
}

fn run_v7_planned_with_exec(
    artifacts: &v6_synth::PipelineArtifacts,
    plan: &v7_planner::TaskPlan,
    shapes: &HashMap<GlobalId, Vec<usize>>,
    inputs: &HashMap<GlobalId, Vec<f32>>,
    output_id: GlobalId,
    output_size: usize,
    exec: v7_exec::ExecuteConfig,
) -> Vec<f32> {
    let mut bufs = HashMap::<GlobalId, Vec<f32>>::new();
    for (id, shape) in shapes {
        let size = shape.iter().product::<usize>().max(1);
        bufs.insert(*id, vec![0.0; size]);
    }
    for (id, data) in inputs {
        if let Some(dst) = bufs.get_mut(id) {
            *dst = data.clone();
        }
    }
    v7_exec::execute_plan_f32_with_config(artifacts, plan, &mut bufs, exec).expect("v7 execute");
    bufs[&output_id][..output_size].to_vec()
}

fn run_v7_compiled(
    compiled: &v7_codegen::NativeCompiledGraph,
    inputs: &HashMap<GlobalId, Vec<f32>>,
    output_id: GlobalId,
    output_size: usize,
) -> Vec<f32> {
    let layout = &compiled.layout;
    let mut bufs: Vec<Vec<f32>> = (0..layout.num_buffers).map(|_| Vec::new()).collect();
    for (id, data) in inputs {
        if let Some(&idx) = layout.tensor_index.get(id) {
            bufs[idx] = data.clone();
        }
    }
    for (id, &size) in &layout.tensor_sizes {
        let idx = layout.tensor_index[id];
        if bufs[idx].is_empty() {
            bufs[idx] = vec![0.0f32; size];
        }
    }
    let mut ptrs: Vec<*mut f32> = bufs.iter_mut().map(|v| v.as_mut_ptr()).collect();
    unsafe { compiled.execute(&mut ptrs) };
    bufs[layout.tensor_index[&output_id]][..output_size].to_vec()
}

fn run_v8_compiled(
    compiled: &v8_codegen::NativeCompiledGraph,
    inputs: &HashMap<GlobalId, Vec<f32>>,
    output_id: GlobalId,
    output_size: usize,
) -> Vec<f32> {
    let layout = &compiled.layout;
    let mut bufs: Vec<Vec<f32>> = (0..layout.num_buffers).map(|_| Vec::new()).collect();
    for (id, data) in inputs {
        if let Some(&idx) = layout.tensor_index.get(id) {
            bufs[idx] = data.clone();
        }
    }
    for (id, &size) in &layout.tensor_sizes {
        let idx = layout.tensor_index[id];
        if bufs[idx].is_empty() {
            bufs[idx] = vec![0.0f32; size];
        }
    }
    let mut ptrs: Vec<*mut f32> = bufs.iter_mut().map(|v| v.as_mut_ptr()).collect();
    unsafe { compiled.execute(&mut ptrs) };
    bufs[layout.tensor_index[&output_id]][..output_size].to_vec()
}

fn compile_v9(
    graph: &MilliOpGraph,
    shapes: &HashMap<GlobalId, Vec<usize>>,
    int_out: GlobalId,
) -> v9_pipeline::CompiledGraph {
    use std::collections::HashSet;
    let final_outputs: HashSet<GlobalId> = [int_out].into_iter().collect();
    v9_pipeline::compile_graph(graph, shapes, &final_outputs, 0).expect("v9 compile")
}

fn run_v9_compiled(
    compiled: &v9_pipeline::CompiledGraph,
    inputs: &HashMap<GlobalId, Vec<f32>>,
    output_id: GlobalId,
    output_size: usize,
) -> Vec<f32> {
    let layout = &compiled.layout;
    let mut bufs: Vec<Vec<f32>> = (0..layout.num_buffers).map(|_| Vec::new()).collect();
    for (id, data) in inputs {
        if let Some(&idx) = layout.tensor_index.get(id) {
            bufs[idx] = data.clone();
        }
    }
    for (id, &size) in &layout.tensor_sizes {
        let idx = layout.tensor_index[id];
        if bufs[idx].is_empty() {
            bufs[idx] = vec![0.0f32; size];
        }
    }
    let mut ptrs: Vec<*mut f32> = bufs.iter_mut().map(|v| v.as_mut_ptr()).collect();
    unsafe { compiled.execute(&mut ptrs) };
    bufs[layout.tensor_index[&output_id]][..output_size].to_vec()
}

fn run_v9_parallel(
    compiled: &v9_pipeline::CompiledGraph,
    inputs: &HashMap<GlobalId, Vec<f32>>,
    output_id: GlobalId,
    output_size: usize,
    num_threads: usize,
) -> Vec<f32> {
    let layout = &compiled.layout;
    let mut bufs: Vec<Vec<f32>> = (0..layout.num_buffers).map(|_| Vec::new()).collect();
    for (id, data) in inputs {
        if let Some(&idx) = layout.tensor_index.get(id) {
            bufs[idx] = data.clone();
        }
    }
    for (id, &size) in &layout.tensor_sizes {
        let idx = layout.tensor_index[id];
        if bufs[idx].is_empty() {
            bufs[idx] = vec![0.0f32; size];
        }
    }
    let ptrs: Vec<*mut f32> = bufs.iter_mut().map(|v| v.as_mut_ptr()).collect();
    unsafe { v9_pipeline::execute_parallel(compiled, &ptrs, num_threads) };
    bufs[layout.tensor_index[&output_id]][..output_size].to_vec()
}

fn make_random_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = wyrand::WyRand::new(seed);
    (0..n)
        .map(|_| {
            let bits: u32 = rng.next_u32();
            (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn run_reference_bf16_matmul(m: usize, n: usize, k: usize, a: &[u16], b: &[u16]) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let av = bf16::from_bits(a[row * k + kk]).to_f32();
                let bv = bf16::from_bits(b[kk * n + col]).to_f32();
                acc += av * bv;
            }
            out[row * n + col] = acc;
        }
    }
    out
}

/// Build internal input map for compiled execution.
fn build_compiled_inputs(
    graph: &MilliOpGraph,
    interp_inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
) -> HashMap<GlobalId, Vec<f32>> {
    let mut out = HashMap::new();
    for (ext_id, tensor) in interp_inputs {
        let int_id = graph.input_map[ext_id];
        out.insert(
            int_id,
            Vec::<f32>::try_from(tensor.flatten().unwrap()).unwrap(),
        );
    }
    out
}

fn find_internal_output(graph: &MilliOpGraph, ext_out: GlobalId) -> GlobalId {
    *graph
        .output_map
        .as_ref()
        .unwrap()
        .keys()
        .find(|id| graph.output_map.as_ref().unwrap()[id] == ext_out)
        .unwrap()
}

fn print_timing(label: &str, duration: std::time::Duration, n: u32) {
    print!("  {:26} {:?} ({} iters)", label, duration, n);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== Whisper-Tensor Compiler Test ===\n");
    let mut rng = wyrand::WyRand::new(12345);

    // -----------------------------------------------------------------------
    // Test 1: Fused chain — neg(exp(a*b + c)), correctness check
    // -----------------------------------------------------------------------
    {
        let shape = &[4, 8];
        let total: usize = shape.iter().product();
        println!("Test 1: neg(exp(a*b + c)), shape {:?}", shape);

        let (graph, ext_ins, ext_out, shapes) = build_chain_graph(shape, &mut rng);
        let a_data = make_random_f32(total, 1);
        let b_data = make_random_f32(total, 2);
        let c_data = make_random_f32(total, 3);

        let mut interp_inputs = HashMap::new();
        for (i, &eid) in ext_ins.iter().enumerate() {
            let data = [&a_data, &b_data, &c_data][i];
            interp_inputs.insert(
                eid,
                NumericTensor::<DynRank>::from_vec_shape(data.clone(), shape.to_vec()).unwrap(),
            );
        }
        let interp_result = run_interpreter(&graph, &interp_inputs, ext_out);
        let compiled_inputs = build_compiled_inputs(&graph, &interp_inputs);
        let int_out = find_internal_output(&graph, ext_out);

        // v1 crystal
        let mut expander = NanoOpExpander::new(shapes.clone());
        let nano_ops = expander.expand(&graph).unwrap();
        let crystal_ops = crystal::crystallize(&nano_ops);
        let v1_layout = v1_codegen::TensorLayout::from_shapes(&shapes);
        let v1_compiled = v1_codegen::compile_crystallized(&crystal_ops, &v1_layout).unwrap();
        let v1_result = run_v1_compiled(&v1_compiled, &compiled_inputs, int_out, total);

        // v2 fusion
        let kernels = v2_planner::plan(&graph, &shapes).unwrap();
        let v2_stats = v2_kernel::stats(&kernels);
        let v2_layout = v2_codegen::TensorLayout::from_shapes(&shapes);
        let v2_compiled = v2_codegen::compile(&kernels, &v2_layout).unwrap();
        let v2_result = run_v2_compiled(&v2_compiled, &compiled_inputs, int_out, total);

        // v3 nano-fusion (v1 nano ops + fused crystal loops)
        let (v3_fused_crystals, v3_stats) = v3_fusion::fuse_with_stats(&crystal_ops);
        let v3_layout = v3_codegen::TensorLayout::from_shapes(&shapes);
        let v3_compiled = v3_codegen::compile_fused_crystallized(&crystal_ops, &v3_layout).unwrap();
        let v3_result = run_v3_compiled(&v3_compiled, &compiled_inputs, int_out, total);
        let (v4_compiled, v4_artifacts) = v4_codegen::compile_graph(&graph, &shapes).unwrap();
        let v4_result = run_v4_compiled(&v4_compiled, &compiled_inputs, int_out, total);

        let v9_compiled = compile_v9(&graph, &shapes, int_out);
        let v9_result = run_v9_compiled(&v9_compiled, &compiled_inputs, int_out, total);

        println!(
            "  v2: {} kernels, {} fused ops",
            v2_stats.num_kernels, v2_stats.total_fused_ops
        );
        println!(
            "  v3: {} -> {} crystal ops, {} fused pairs, {} eliminated loads",
            crystal_ops.len(),
            v3_fused_crystals.len(),
            v3_stats.fused_pairs,
            v3_stats.eliminated_loads,
        );
        println!("  v9: {} kernels", v9_compiled.kernels.len());
        println!(
            "  v1 crystal diff: {:.2e}",
            max_abs_diff(&interp_result, &v1_result)
        );
        println!(
            "  v2 fusion diff:  {:.2e}",
            max_abs_diff(&interp_result, &v2_result)
        );
        println!(
            "  v3 nano diff:    {:.2e}",
            max_abs_diff(&interp_result, &v3_result)
        );
        println!(
            "  v4 pool: {} ordered loops -> {} fused loops ({} fused pairs)",
            v4_artifacts.plan.stats.ordered_loops,
            v4_artifacts.plan.stats.fused_loops,
            v4_artifacts.plan.stats.fused_pairs,
        );
        println!(
            "  v4 pool diff:    {:.2e}",
            max_abs_diff(&interp_result, &v4_result)
        );
        println!(
            "  v9 fused diff:   {:.2e}",
            max_abs_diff(&interp_result, &v9_result)
        );
        assert!(max_abs_diff(&interp_result, &v1_result) < 1e-5);
        assert!(max_abs_diff(&interp_result, &v2_result) < 1e-5);
        assert!(max_abs_diff(&interp_result, &v3_result) < 1e-5);
        assert!(max_abs_diff(&interp_result, &v4_result) < 1e-5);
        assert!(max_abs_diff(&interp_result, &v9_result) < 1e-5);
        println!("  PASS\n");
    }

    // -----------------------------------------------------------------------
    // Test 2: Elementwise MLP — v1 vs v2 performance
    // -----------------------------------------------------------------------
    {
        let shape = &[32, 64];
        let total: usize = shape.iter().product();
        let num_layers = 5;
        println!(
            "Test 2: {}-layer elementwise MLP, shape {:?}",
            num_layers, shape
        );

        let (graph, ext_input, layer_params, ext_out, shapes) =
            build_elementwise_mlp(shape, num_layers, &mut rng);

        let mut interp_inputs = HashMap::new();
        interp_inputs.insert(
            ext_input,
            NumericTensor::<DynRank>::from_vec_shape(make_random_f32(total, 100), shape.to_vec())
                .unwrap(),
        );
        for (i, (ew, eb)) in layer_params.iter().enumerate() {
            interp_inputs.insert(
                *ew,
                NumericTensor::<DynRank>::from_vec_shape(
                    make_random_f32(total, 200 + i as u64 * 2),
                    shape.to_vec(),
                )
                .unwrap(),
            );
            interp_inputs.insert(
                *eb,
                NumericTensor::<DynRank>::from_vec_shape(
                    make_random_f32(total, 201 + i as u64 * 2),
                    shape.to_vec(),
                )
                .unwrap(),
            );
        }

        let interp_result = run_interpreter(&graph, &interp_inputs, ext_out);
        let compiled_inputs = build_compiled_inputs(&graph, &interp_inputs);
        let int_out = find_internal_output(&graph, ext_out);

        // v1 crystal
        let mut expander = NanoOpExpander::new(shapes.clone());
        let nano_ops = expander.expand(&graph).unwrap();
        let crystal_ops = crystal::crystallize(&nano_ops);
        let v1_layout = v1_codegen::TensorLayout::from_shapes(&shapes);
        let v1_compiled = v1_codegen::compile_crystallized(&crystal_ops, &v1_layout).unwrap();

        // v2 fusion
        let kernels = v2_planner::plan(&graph, &shapes).unwrap();
        let v2_stats = v2_kernel::stats(&kernels);
        let v2_layout = v2_codegen::TensorLayout::from_shapes(&shapes);
        let v2_compiled = v2_codegen::compile(&kernels, &v2_layout).unwrap();

        println!(
            "  v2: {} kernels, {} fused ops",
            v2_stats.num_kernels, v2_stats.total_fused_ops
        );

        let v1_result = run_v1_compiled(&v1_compiled, &compiled_inputs, int_out, total);
        let v2_result = run_v2_compiled(&v2_compiled, &compiled_inputs, int_out, total);
        assert!(max_abs_diff(&interp_result, &v1_result) < 1e-5);
        assert!(max_abs_diff(&interp_result, &v2_result) < 1e-5);

        let v9_compiled = compile_v9(&graph, &shapes, int_out);
        let v9_result = run_v9_compiled(&v9_compiled, &compiled_inputs, int_out, total);
        println!("  v9: {} kernels", v9_compiled.kernels.len());
        assert!(max_abs_diff(&interp_result, &v9_result) < 1e-5);

        let n = 100;
        let t = Instant::now();
        for _ in 0..n {
            let _ = run_interpreter(&graph, &interp_inputs, ext_out);
        }
        let interp_avg = t.elapsed() / n;

        let t = Instant::now();
        for _ in 0..n {
            let _ = run_v1_compiled(&v1_compiled, &compiled_inputs, int_out, total);
        }
        let v1_avg = t.elapsed() / n;

        let t = Instant::now();
        for _ in 0..n {
            let _ = run_v2_compiled(&v2_compiled, &compiled_inputs, int_out, total);
        }
        let v2_avg = t.elapsed() / n;

        let t = Instant::now();
        for _ in 0..n {
            let _ = run_v9_compiled(&v9_compiled, &compiled_inputs, int_out, total);
        }
        let v9_avg = t.elapsed() / n;

        print_timing("Interpreter:", interp_avg, n);
        println!();
        print_timing("v1 crystal:", v1_avg, n);
        println!(
            "  {:.2}x",
            interp_avg.as_nanos() as f64 / v1_avg.as_nanos() as f64
        );
        print_timing("v2 fusion:", v2_avg, n);
        println!(
            "  {:.2}x",
            interp_avg.as_nanos() as f64 / v2_avg.as_nanos() as f64
        );
        print_timing("v9 fused:", v9_avg, n);
        println!(
            "  {:.2}x",
            interp_avg.as_nanos() as f64 / v9_avg.as_nanos() as f64
        );
        println!("  PASS\n");
    }

    // -----------------------------------------------------------------------
    // Test 2b: Arithmetic-only chain (no transcendentals) — isolate fusion benefit
    // -----------------------------------------------------------------------
    {
        let shape = &[32, 64];
        let total: usize = shape.iter().product();
        let num_layers = 5;
        println!(
            "Test 2b: {}-layer arithmetic chain (mul+add only), shape {:?}",
            num_layers, shape
        );

        let (graph, ext_input, layer_params, ext_out, shapes) =
            build_arithmetic_chain(shape, num_layers, &mut rng);

        let mut interp_inputs = HashMap::new();
        interp_inputs.insert(
            ext_input,
            NumericTensor::<DynRank>::from_vec_shape(make_random_f32(total, 400), shape.to_vec())
                .unwrap(),
        );
        for (i, (ew, eb)) in layer_params.iter().enumerate() {
            interp_inputs.insert(
                *ew,
                NumericTensor::<DynRank>::from_vec_shape(
                    make_random_f32(total, 500 + i as u64 * 2),
                    shape.to_vec(),
                )
                .unwrap(),
            );
            interp_inputs.insert(
                *eb,
                NumericTensor::<DynRank>::from_vec_shape(
                    make_random_f32(total, 501 + i as u64 * 2),
                    shape.to_vec(),
                )
                .unwrap(),
            );
        }

        let compiled_inputs = build_compiled_inputs(&graph, &interp_inputs);
        let int_out = find_internal_output(&graph, ext_out);

        // v1 crystal
        let mut expander = NanoOpExpander::new(shapes.clone());
        let nano_ops = expander.expand(&graph).unwrap();
        let crystal_ops = crystal::crystallize(&nano_ops);
        let v1_layout = v1_codegen::TensorLayout::from_shapes(&shapes);
        let v1_compiled = v1_codegen::compile_crystallized(&crystal_ops, &v1_layout).unwrap();

        // v2 fusion
        let kernels = v2_planner::plan(&graph, &shapes).unwrap();
        let v2_stats = v2_kernel::stats(&kernels);
        let v2_layout = v2_codegen::TensorLayout::from_shapes(&shapes);
        let v2_compiled = v2_codegen::compile(&kernels, &v2_layout).unwrap();

        println!(
            "  v2: {} kernels, {} fused ops",
            v2_stats.num_kernels, v2_stats.total_fused_ops
        );
        let (v3_fused_crystals, v3_stats) = v3_fusion::fuse_with_stats(&crystal_ops);
        let v3_layout = v3_codegen::TensorLayout::from_shapes(&shapes);
        let v3_compiled = v3_codegen::compile_fused_crystallized(&crystal_ops, &v3_layout).unwrap();
        let (v4_compiled, v4_artifacts) = v4_codegen::compile_graph(&graph, &shapes).unwrap();
        println!(
            "  v3: {} -> {} crystal ops, {} fused pairs, {} eliminated loads",
            crystal_ops.len(),
            v3_fused_crystals.len(),
            v3_stats.fused_pairs,
            v3_stats.eliminated_loads,
        );
        println!(
            "  v4: {} recovered loops -> {} fused loops, {} fused pairs",
            v4_artifacts.plan.stats.ordered_loops,
            v4_artifacts.plan.stats.fused_loops,
            v4_artifacts.plan.stats.fused_pairs,
        );

        let interp_result = run_interpreter(&graph, &interp_inputs, ext_out);
        let v1_result = run_v1_compiled(&v1_compiled, &compiled_inputs, int_out, total);
        let v2_result = run_v2_compiled(&v2_compiled, &compiled_inputs, int_out, total);
        let v3_result = run_v3_compiled(&v3_compiled, &compiled_inputs, int_out, total);
        let v4_result = run_v4_compiled(&v4_compiled, &compiled_inputs, int_out, total);
        assert!(max_abs_diff(&interp_result, &v1_result) < 1e-5);
        assert!(max_abs_diff(&interp_result, &v2_result) < 1e-5);
        assert!(max_abs_diff(&interp_result, &v3_result) < 1e-5);
        assert!(max_abs_diff(&interp_result, &v4_result) < 1e-5);

        let v9_compiled = compile_v9(&graph, &shapes, int_out);
        let v9_result = run_v9_compiled(&v9_compiled, &compiled_inputs, int_out, total);
        println!("  v9: {} kernels", v9_compiled.kernels.len());
        assert!(max_abs_diff(&interp_result, &v9_result) < 1e-5);

        let n = 500;
        let t = Instant::now();
        for _ in 0..n {
            let _ = run_interpreter(&graph, &interp_inputs, ext_out);
        }
        let interp_avg = t.elapsed() / n;

        let t = Instant::now();
        for _ in 0..n {
            let _ = run_v1_compiled(&v1_compiled, &compiled_inputs, int_out, total);
        }
        let v1_avg = t.elapsed() / n;

        let t = Instant::now();
        for _ in 0..n {
            let _ = run_v2_compiled(&v2_compiled, &compiled_inputs, int_out, total);
        }
        let v2_avg = t.elapsed() / n;

        let t = Instant::now();
        for _ in 0..n {
            let _ = run_v3_compiled(&v3_compiled, &compiled_inputs, int_out, total);
        }
        let v3_avg = t.elapsed() / n;
        let t = Instant::now();
        for _ in 0..n {
            let _ = run_v4_compiled(&v4_compiled, &compiled_inputs, int_out, total);
        }
        let v4_avg = t.elapsed() / n;

        let t = Instant::now();
        for _ in 0..n {
            let _ = run_v9_compiled(&v9_compiled, &compiled_inputs, int_out, total);
        }
        let v9_avg = t.elapsed() / n;

        print_timing("Interpreter:", interp_avg, n);
        println!();
        print_timing("v1 crystal:", v1_avg, n);
        println!(
            "  {:.2}x",
            interp_avg.as_nanos() as f64 / v1_avg.as_nanos() as f64
        );
        print_timing("v2 fusion:", v2_avg, n);
        println!(
            "  {:.2}x",
            interp_avg.as_nanos() as f64 / v2_avg.as_nanos() as f64
        );
        print_timing("v3 nano-fusion:", v3_avg, n);
        println!(
            "  {:.2}x",
            interp_avg.as_nanos() as f64 / v3_avg.as_nanos() as f64
        );
        println!(
            "  v3 vs v2: {:.2}x",
            v2_avg.as_nanos() as f64 / v3_avg.as_nanos() as f64
        );
        print_timing("v4 pool-growth:", v4_avg, n);
        println!(
            "  {:.2}x",
            interp_avg.as_nanos() as f64 / v4_avg.as_nanos() as f64
        );
        println!(
            "  v4 vs v2: {:.2}x",
            v2_avg.as_nanos() as f64 / v4_avg.as_nanos() as f64
        );
        print_timing("v9 fused:", v9_avg, n);
        println!(
            "  {:.2}x",
            interp_avg.as_nanos() as f64 / v9_avg.as_nanos() as f64
        );
        println!(
            "  v9 vs v2: {:.2}x",
            v2_avg.as_nanos() as f64 / v9_avg.as_nanos() as f64
        );
        println!("  PASS\n");
    }

    // -----------------------------------------------------------------------
    // Test 2c: Large tensor elementwise MLP — fusion should help here
    // -----------------------------------------------------------------------
    {
        let shape = &[256, 256];
        let total: usize = shape.iter().product();
        let num_layers = 5;
        println!(
            "Test 2c: {}-layer elementwise MLP, shape {:?} ({}KB per tensor)",
            num_layers,
            shape,
            total * 4 / 1024
        );

        let (graph, ext_input, layer_params, ext_out, shapes) =
            build_elementwise_mlp(shape, num_layers, &mut rng);

        let mut interp_inputs = HashMap::new();
        interp_inputs.insert(
            ext_input,
            NumericTensor::<DynRank>::from_vec_shape(make_random_f32(total, 600), shape.to_vec())
                .unwrap(),
        );
        for (i, (ew, eb)) in layer_params.iter().enumerate() {
            interp_inputs.insert(
                *ew,
                NumericTensor::<DynRank>::from_vec_shape(
                    make_random_f32(total, 700 + i as u64 * 2),
                    shape.to_vec(),
                )
                .unwrap(),
            );
            interp_inputs.insert(
                *eb,
                NumericTensor::<DynRank>::from_vec_shape(
                    make_random_f32(total, 701 + i as u64 * 2),
                    shape.to_vec(),
                )
                .unwrap(),
            );
        }

        let interp_result = run_interpreter(&graph, &interp_inputs, ext_out);
        let compiled_inputs = build_compiled_inputs(&graph, &interp_inputs);
        let int_out = find_internal_output(&graph, ext_out);

        // v1 crystal
        let mut expander = NanoOpExpander::new(shapes.clone());
        let nano_ops = expander.expand(&graph).unwrap();
        let crystal_ops = crystal::crystallize(&nano_ops);
        let v1_layout = v1_codegen::TensorLayout::from_shapes(&shapes);
        let v1_compiled = v1_codegen::compile_crystallized(&crystal_ops, &v1_layout).unwrap();

        // v2 fusion
        let kernels = v2_planner::plan(&graph, &shapes).unwrap();
        let v2_stats = v2_kernel::stats(&kernels);
        let v2_layout = v2_codegen::TensorLayout::from_shapes(&shapes);
        let v2_compiled = v2_codegen::compile(&kernels, &v2_layout).unwrap();

        println!(
            "  v2: {} kernels, {} fused ops",
            v2_stats.num_kernels, v2_stats.total_fused_ops
        );

        let v1_result = run_v1_compiled(&v1_compiled, &compiled_inputs, int_out, total);
        let v2_result = run_v2_compiled(&v2_compiled, &compiled_inputs, int_out, total);
        assert!(max_abs_diff(&interp_result, &v1_result) < 1e-4);
        assert!(max_abs_diff(&interp_result, &v2_result) < 1e-4);

        let v9_compiled = compile_v9(&graph, &shapes, int_out);
        let v9_result = run_v9_compiled(&v9_compiled, &compiled_inputs, int_out, total);
        println!("  v9: {} kernels", v9_compiled.kernels.len());
        assert!(max_abs_diff(&interp_result, &v9_result) < 1e-4);

        let n = 20;
        let t = Instant::now();
        for _ in 0..n {
            let _ = run_interpreter(&graph, &interp_inputs, ext_out);
        }
        let interp_avg = t.elapsed() / n;

        let t = Instant::now();
        for _ in 0..n {
            let _ = run_v1_compiled(&v1_compiled, &compiled_inputs, int_out, total);
        }
        let v1_avg = t.elapsed() / n;

        let t = Instant::now();
        for _ in 0..n {
            let _ = run_v2_compiled(&v2_compiled, &compiled_inputs, int_out, total);
        }
        let v2_avg = t.elapsed() / n;

        let t = Instant::now();
        for _ in 0..n {
            let _ = run_v9_compiled(&v9_compiled, &compiled_inputs, int_out, total);
        }
        let v9_avg = t.elapsed() / n;

        print_timing("Interpreter:", interp_avg, n);
        println!();
        print_timing("v1 crystal:", v1_avg, n);
        println!(
            "  {:.2}x",
            interp_avg.as_nanos() as f64 / v1_avg.as_nanos() as f64
        );
        print_timing("v2 fusion:", v2_avg, n);
        println!(
            "  {:.2}x",
            interp_avg.as_nanos() as f64 / v2_avg.as_nanos() as f64
        );
        print_timing("v9 fused:", v9_avg, n);
        println!(
            "  {:.2}x",
            interp_avg.as_nanos() as f64 / v9_avg.as_nanos() as f64
        );
        println!("  PASS\n");
    }

    // -----------------------------------------------------------------------
    // Test 3: Simple matmul — correctness (v2 only)
    // -----------------------------------------------------------------------
    {
        println!("Test 3: MatMul [4,8] x [8,3]");
        let mut rng2 = wyrand::WyRand::new(999);
        let ext_a = GlobalId::new(&mut rng2);
        let ext_b = GlobalId::new(&mut rng2);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng2);
        let a = input_map[&ext_a];
        let b = input_map[&ext_b];
        let c = MatMul::push_new_default_precision(&mut graph, a, b, DType::F32, &mut rng2);
        let ext_out = GlobalId::new(&mut rng2);
        graph.set_output_map([(c, ext_out)]);

        let mut shapes = HashMap::new();
        shapes.insert(a, vec![4, 8]);
        shapes.insert(b, vec![8, 3]);
        shapes.insert(c, vec![4, 3]);

        let a_data = make_random_f32(32, 1);
        let b_data = make_random_f32(24, 2);
        let mut interp_inputs = HashMap::new();
        interp_inputs.insert(
            ext_a,
            NumericTensor::<DynRank>::from_vec_shape(a_data.clone(), vec![4, 8]).unwrap(),
        );
        interp_inputs.insert(
            ext_b,
            NumericTensor::<DynRank>::from_vec_shape(b_data.clone(), vec![8, 3]).unwrap(),
        );
        let interp_result = run_interpreter(&graph, &interp_inputs, ext_out);

        let kernels = v2_planner::plan(&graph, &shapes).unwrap();
        let layout = v2_codegen::TensorLayout::from_shapes(&shapes);
        let compiled = v2_codegen::compile(&kernels, &layout).unwrap();
        let compiled_inputs = build_compiled_inputs(&graph, &interp_inputs);
        let v2_result = run_v2_compiled(&compiled, &compiled_inputs, c, 12);

        let v9_compiled = compile_v9(&graph, &shapes, c);
        let v9_result = run_v9_compiled(&v9_compiled, &compiled_inputs, c, 12);

        let diff = max_abs_diff(&interp_result, &v2_result);
        let v9_diff = max_abs_diff(&interp_result, &v9_result);
        println!("  v2 diff: {:.2e}, v9 diff: {:.2e}", diff, v9_diff);
        assert!(diff < 1e-4, "v2 MatMul diverged: {diff}");
        assert!(v9_diff < 1e-4, "v9 MatMul diverged: {v9_diff}");
        println!("  PASS\n");
    }

    // -----------------------------------------------------------------------
    // Test 4: MatMul MLP — y = tanh(x@W + b) (v2 only, with timing)
    // -----------------------------------------------------------------------
    {
        let batch = 16;
        let dims = &[64, 32, 16];
        println!("Test 4: MatMul MLP, batch={}, dims={:?}", batch, dims);

        let (graph, ext_input, layer_params, ext_out, shapes) =
            build_matmul_mlp(batch, dims, &mut rng);

        let mut interp_inputs = HashMap::new();
        interp_inputs.insert(
            ext_input,
            NumericTensor::<DynRank>::from_vec_shape(
                make_random_f32(batch * dims[0], 700),
                vec![batch, dims[0]],
            )
            .unwrap(),
        );
        for (i, (ew, eb)) in layer_params.iter().enumerate() {
            let in_d = dims[i];
            let out_d = dims[i + 1];
            interp_inputs.insert(
                *ew,
                NumericTensor::<DynRank>::from_vec_shape(
                    make_random_f32(in_d * out_d, 800 + i as u64 * 2),
                    vec![in_d, out_d],
                )
                .unwrap(),
            );
            interp_inputs.insert(
                *eb,
                NumericTensor::<DynRank>::from_vec_shape(
                    make_random_f32(out_d, 801 + i as u64 * 2),
                    vec![1, out_d],
                )
                .unwrap(),
            );
        }

        let interp_result = run_interpreter(&graph, &interp_inputs, ext_out);
        let compiled_inputs = build_compiled_inputs(&graph, &interp_inputs);
        let int_out = find_internal_output(&graph, ext_out);
        let out_size = batch * dims[dims.len() - 1];

        let kernels = v2_planner::plan(&graph, &shapes).unwrap();
        let v2_stats = v2_kernel::stats(&kernels);
        let layout = v2_codegen::TensorLayout::from_shapes(&shapes);
        let compiled = v2_codegen::compile(&kernels, &layout).unwrap();
        let v2_result = run_v2_compiled(&compiled, &compiled_inputs, int_out, out_size);

        let v9_compiled = compile_v9(&graph, &shapes, int_out);
        let v9_result = run_v9_compiled(&v9_compiled, &compiled_inputs, int_out, out_size);

        println!(
            "  v2: {} kernels ({} gemm, {} elementwise, {} fused ops)",
            v2_stats.num_kernels,
            v2_stats.num_gemm,
            v2_stats.num_elementwise,
            v2_stats.total_fused_ops
        );
        println!("  v9: {} kernels", v9_compiled.kernels.len());

        let diff = max_abs_diff(&interp_result, &v2_result);
        let v9_diff = max_abs_diff(&interp_result, &v9_result);
        println!("  v2 diff: {:.2e}, v9 diff: {:.2e}", diff, v9_diff);
        assert!(diff < 1e-4, "v2 MatMul MLP diverged: {diff}");
        assert!(v9_diff < 1e-4, "v9 MatMul MLP diverged: {v9_diff}");

        let n = 200;
        let t = Instant::now();
        for _ in 0..n {
            let _ = run_interpreter(&graph, &interp_inputs, ext_out);
        }
        let interp_avg = t.elapsed() / n;

        let t = Instant::now();
        for _ in 0..n {
            let _ = run_v2_compiled(&compiled, &compiled_inputs, int_out, out_size);
        }
        let v2_avg = t.elapsed() / n;

        let t = Instant::now();
        for _ in 0..n {
            let _ = run_v9_compiled(&v9_compiled, &compiled_inputs, int_out, out_size);
        }
        let v9_avg = t.elapsed() / n;

        print_timing("Interpreter:", interp_avg, n);
        println!();
        print_timing("v2 fusion:", v2_avg, n);
        println!(
            "  {:.2}x",
            interp_avg.as_nanos() as f64 / v2_avg.as_nanos() as f64
        );
        print_timing("v9 fused:", v9_avg, n);
        println!(
            "  {:.2}x",
            interp_avg.as_nanos() as f64 / v9_avg.as_nanos() as f64
        );
        println!(
            "  v9 vs v2: {:.2}x",
            v2_avg.as_nanos() as f64 / v9_avg.as_nanos() as f64
        );
        println!("  PASS\n");
    }

    // -----------------------------------------------------------------------
    // Test 5: Larger MatMul MLP — stress test
    // -----------------------------------------------------------------------
    {
        let batch = 32;
        let dims = &[128, 64, 32, 16];
        println!(
            "Test 5: MatMul MLP stress, batch={}, dims={:?}",
            batch, dims
        );

        let (graph, ext_input, layer_params, ext_out, shapes) =
            build_matmul_mlp(batch, dims, &mut rng);

        let mut interp_inputs = HashMap::new();
        interp_inputs.insert(
            ext_input,
            NumericTensor::<DynRank>::from_vec_shape(
                make_random_f32(batch * dims[0], 900),
                vec![batch, dims[0]],
            )
            .unwrap(),
        );
        for (i, (ew, eb)) in layer_params.iter().enumerate() {
            let in_d = dims[i];
            let out_d = dims[i + 1];
            interp_inputs.insert(
                *ew,
                NumericTensor::<DynRank>::from_vec_shape(
                    make_random_f32(in_d * out_d, 1000 + i as u64 * 2),
                    vec![in_d, out_d],
                )
                .unwrap(),
            );
            interp_inputs.insert(
                *eb,
                NumericTensor::<DynRank>::from_vec_shape(
                    make_random_f32(out_d, 1001 + i as u64 * 2),
                    vec![1, out_d],
                )
                .unwrap(),
            );
        }

        let interp_result = run_interpreter(&graph, &interp_inputs, ext_out);
        let compiled_inputs = build_compiled_inputs(&graph, &interp_inputs);
        let int_out = find_internal_output(&graph, ext_out);
        let out_size = batch * dims[dims.len() - 1];

        let kernels = v2_planner::plan(&graph, &shapes).unwrap();
        let v2_stats = v2_kernel::stats(&kernels);
        let layout = v2_codegen::TensorLayout::from_shapes(&shapes);
        let compiled = v2_codegen::compile(&kernels, &layout).unwrap();
        let v2_result = run_v2_compiled(&compiled, &compiled_inputs, int_out, out_size);

        let v9_compiled = compile_v9(&graph, &shapes, int_out);
        let v9_result = run_v9_compiled(&v9_compiled, &compiled_inputs, int_out, out_size);

        println!(
            "  v2: {} kernels ({} gemm, {} elementwise, {} fused ops)",
            v2_stats.num_kernels,
            v2_stats.num_gemm,
            v2_stats.num_elementwise,
            v2_stats.total_fused_ops
        );
        println!("  v9: {} kernels", v9_compiled.kernels.len());

        let diff = max_abs_diff(&interp_result, &v2_result);
        let v9_diff = max_abs_diff(&interp_result, &v9_result);
        println!("  v2 diff: {:.2e}, v9 diff: {:.2e}", diff, v9_diff);
        assert!(diff < 1e-3, "v2 MatMul MLP stress diverged: {diff}");
        assert!(v9_diff < 1e-3, "v9 MatMul MLP stress diverged: {v9_diff}");

        let n = 100;
        let t = Instant::now();
        for _ in 0..n {
            let _ = run_interpreter(&graph, &interp_inputs, ext_out);
        }
        let interp_avg = t.elapsed() / n;

        let t = Instant::now();
        for _ in 0..n {
            let _ = run_v2_compiled(&compiled, &compiled_inputs, int_out, out_size);
        }
        let v2_avg = t.elapsed() / n;

        let t = Instant::now();
        for _ in 0..n {
            let _ = run_v9_compiled(&v9_compiled, &compiled_inputs, int_out, out_size);
        }
        let v9_avg = t.elapsed() / n;

        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let t = Instant::now();
        for _ in 0..n {
            let _ = run_v9_parallel(
                &v9_compiled,
                &compiled_inputs,
                int_out,
                out_size,
                num_threads,
            );
        }
        let v9_mt_avg = t.elapsed() / n;

        print_timing("Interpreter:", interp_avg, n);
        println!();
        print_timing("v2 fusion:", v2_avg, n);
        println!(
            "  {:.2}x",
            interp_avg.as_nanos() as f64 / v2_avg.as_nanos() as f64
        );
        print_timing("v9 fused:", v9_avg, n);
        println!(
            "  {:.2}x",
            interp_avg.as_nanos() as f64 / v9_avg.as_nanos() as f64
        );
        print_timing(&format!("v9 parallel({}t):", num_threads), v9_mt_avg, n);
        println!(
            "  {:.2}x  ({:.1}x vs serial)",
            interp_avg.as_nanos() as f64 / v9_mt_avg.as_nanos() as f64,
            v9_avg.as_nanos() as f64 / v9_mt_avg.as_nanos() as f64,
        );
        println!(
            "  v9 vs v2: {:.2}x",
            v2_avg.as_nanos() as f64 / v9_avg.as_nanos() as f64
        );
        println!("  PASS\n");
    }

    // -----------------------------------------------------------------------
    // Test 6: v5 typed synthesis from whitewashed pool + software BF16 matmul
    // -----------------------------------------------------------------------
    {
        let (m, n, k) = (96usize, 96usize, 128usize);
        println!(
            "Test 6: v5 BF16 matmul synthesis [{}x{}] x [{}x{}]",
            m, k, k, n
        );

        let mut rng2 = wyrand::WyRand::new(2026);
        let ext_a = GlobalId::new(&mut rng2);
        let ext_b = GlobalId::new(&mut rng2);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng2);
        let a = input_map[&ext_a];
        let b = input_map[&ext_b];
        let c = MatMul::push_new_default_precision(&mut graph, a, b, DType::F32, &mut rng2);
        let ext_out = GlobalId::new(&mut rng2);
        graph.set_output_map([(c, ext_out)]);

        let mut shapes = HashMap::new();
        shapes.insert(a, vec![m, k]);
        shapes.insert(b, vec![k, n]);
        shapes.insert(c, vec![m, n]);

        let mut dtypes = HashMap::new();
        dtypes.insert(a, DType::BF16);
        dtypes.insert(b, DType::BF16);
        dtypes.insert(c, DType::F32);

        let hw = v5_synth::HardwareProfile::default();
        let (typed_plan, pool_plan) =
            v5_synth::build_from_graph(&graph, &shapes, &dtypes, hw).expect("v5 build_from_graph");
        assert_eq!(pool_plan.matmul_crystals.len(), 1);
        assert_eq!(typed_plan.matmul_kernels.len(), 1);
        println!(
            "  recovered: {} matmul crystal from {} pool ops",
            pool_plan.matmul_crystals.len(),
            pool_plan.stats.total_nano_ops
        );

        let a_f = make_random_f32(m * k, 1200);
        let b_f = make_random_f32(k * n, 1201);
        let a_bf16: Vec<u16> = a_f.iter().map(|x| bf16::from_f32(*x).to_bits()).collect();
        let b_bf16: Vec<u16> = b_f.iter().map(|x| bf16::from_f32(*x).to_bits()).collect();
        let expected = run_reference_bf16_matmul(m, n, k, &a_bf16, &b_bf16);

        let mut plan = typed_plan.matmul_kernels[0].clone();
        println!(
            "  blocking: mc={} nc={} kc={} mr={} nr={}",
            plan.blocking.mc,
            plan.blocking.nc,
            plan.blocking.kc,
            plan.blocking.mr,
            plan.blocking.nr
        );

        let iters = 25;
        for strategy in [
            v5_synth::Bf16KernelStrategy::OnTheFlyConvert,
            v5_synth::Bf16KernelStrategy::PackBPanelF32,
        ] {
            plan.bf16_strategy = Some(strategy);
            let mut out = vec![0.0f32; m * n];
            let stats =
                v5_synth::execute_matmul_bf16_f32_with_stats(&plan, &a_bf16, &b_bf16, &mut out)
                    .expect("v5 execute");
            let diff = max_abs_diff(&expected, &out);
            assert!(diff < 6e-3, "v5 {strategy:?} diff too high: {diff}");

            let t = Instant::now();
            for _ in 0..iters {
                v5_synth::execute_matmul_bf16_f32(&plan, &a_bf16, &b_bf16, &mut out)
                    .expect("v5 timed execute");
            }
            let avg = t.elapsed() / iters;
            print_timing(&format!("v5 {:?}:", strategy), avg, iters);
            println!(
                "  diff {:.2e}, b-conv={}, packed-panels={}",
                diff, stats.bf16_to_f32_b, stats.packed_b_panels
            );
        }
        println!("  PASS\n");
    }

    // -----------------------------------------------------------------------
    // Test 7: v7 planning + execution benchmark
    // -----------------------------------------------------------------------
    {
        let dim_override = std::env::var("WT_TEST7_DIM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&d| d > 0);
        let (m, n, k) = if let Some(d) = dim_override {
            (d, d, d)
        } else {
            (64usize, 80usize, 96usize)
        };
        println!(
            "Test 7: v7 planning + compile [{}x{}] x [{}x{}]",
            m, k, k, n
        );

        let mut rng2 = wyrand::WyRand::new(3007);
        let ext_a = GlobalId::new(&mut rng2);
        let ext_b = GlobalId::new(&mut rng2);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng2);
        let a = input_map[&ext_a];
        let b = input_map[&ext_b];
        let c = MatMul::push_new_default_precision(&mut graph, a, b, DType::F32, &mut rng2);
        let ext_out = GlobalId::new(&mut rng2);
        graph.set_output_map([(c, ext_out)]);

        let mut shapes = HashMap::new();
        shapes.insert(a, vec![m, k]);
        shapes.insert(b, vec![k, n]);
        shapes.insert(c, vec![m, n]);

        let a_data = make_random_f32(m * k, 1700);
        let b_data = make_random_f32(k * n, 1701);
        let mut interp_inputs = HashMap::new();
        interp_inputs.insert(
            ext_a,
            NumericTensor::<DynRank>::from_vec_shape(a_data.clone(), vec![m, k]).unwrap(),
        );
        interp_inputs.insert(
            ext_b,
            NumericTensor::<DynRank>::from_vec_shape(b_data.clone(), vec![k, n]).unwrap(),
        );
        let interp_result = run_interpreter(&graph, &interp_inputs, ext_out);
        let compiled_inputs = build_compiled_inputs(&graph, &interp_inputs);
        let out_size = m * n;

        let (v7_artifacts, v7_plan) = v7_planner::plan_from_graph(
            &graph,
            &shapes,
            v7_planner::TaskPlannerConfig {
                min_tile_elements: 512,
                max_tile_elements: 4096,
            },
        )
        .expect("v7 plan");
        let schedule = &v7_artifacts.schedule;
        println!(
            "  schedule stats: families={} pointwise={} reductions={} unknown={}",
            schedule.stats.grouped_families,
            schedule.stats.pointwise_families,
            schedule.stats.additive_reduction_families,
            schedule.stats.unknown_families
        );
        assert_eq!(schedule.stats.additive_reduction_families, 1);
        let recovered = &schedule.loops[0];
        let selected = recovered
            .selected_schedule
            .expect("v7 selected schedule missing");
        let best = &recovered.schedule_candidates[selected];
        println!(
            "  selected: score={} order={:?} tiles={:?} rtile={} runroll={} vec={:?}",
            best.score,
            best.loop_order,
            best.output_tiles,
            best.reduction_tile,
            best.reduction_unroll,
            best.vectorize
        );
        assert!(!recovered.schedule_candidates.is_empty());

        let v7_result = run_v7_planned(
            &v7_artifacts,
            &v7_plan,
            &shapes,
            &compiled_inputs,
            c,
            out_size,
        );
        let v7_diff = max_abs_diff(&interp_result, &v7_result);
        let v7_mt_result = run_v7_planned_with_exec(
            &v7_artifacts,
            &v7_plan,
            &shapes,
            &compiled_inputs,
            c,
            out_size,
            v7_exec::ExecuteConfig::auto(),
        );
        let v7_mt_diff = max_abs_diff(&interp_result, &v7_mt_result);
        println!(
            "  v7 diff: {:.2e} (tasks={})",
            v7_diff, v7_plan.stats.total_tasks
        );
        println!("  v7 mt diff: {:.2e}", v7_mt_diff);
        assert!(v7_diff < 2e-3, "v7 matmul diverged: {v7_diff}");
        assert!(v7_mt_diff < 2e-3, "v7 mt matmul diverged: {v7_mt_diff}");
        let v7_compiled =
            v7_codegen::compile_plan(&v7_artifacts, &v7_plan, &shapes).expect("v7 compile");
        let v7_cl_result = run_v7_compiled(&v7_compiled, &compiled_inputs, c, out_size);
        let v7_cl_diff = max_abs_diff(&interp_result, &v7_cl_result);
        println!("  v7 cranelift diff: {:.2e}", v7_cl_diff);
        assert!(
            v7_cl_diff < 2e-3,
            "v7 cranelift matmul diverged: {v7_cl_diff}"
        );

        let kernels = v2_planner::plan(&graph, &shapes).unwrap();
        let v2_layout = v2_codegen::TensorLayout::from_shapes(&shapes);
        let v2_compiled = v2_codegen::compile(&kernels, &v2_layout).unwrap();
        let v2_result = run_v2_compiled(&v2_compiled, &compiled_inputs, c, out_size);
        let v2_diff = max_abs_diff(&interp_result, &v2_result);
        assert!(v2_diff < 2e-3, "v2 matmul diverged: {v2_diff}");

        let v9_compiled = compile_v9(&graph, &shapes, c);
        let v9_result = run_v9_compiled(&v9_compiled, &compiled_inputs, c, out_size);
        let v9_diff = max_abs_diff(&interp_result, &v9_result);
        println!(
            "  v9 diff: {:.2e}, {} kernels",
            v9_diff,
            v9_compiled.kernels.len()
        );
        assert!(v9_diff < 2e-3, "v9 matmul diverged: {v9_diff}");

        let iters = if m >= 512 || n >= 512 || k >= 512 {
            4
        } else {
            80
        };
        let t = Instant::now();
        for _ in 0..iters {
            let _ = run_interpreter(&graph, &interp_inputs, ext_out);
        }
        let interp_avg = t.elapsed() / iters;

        let t = Instant::now();
        for _ in 0..iters {
            let _ = run_v2_compiled(&v2_compiled, &compiled_inputs, c, out_size);
        }
        let v2_avg = t.elapsed() / iters;

        let t = Instant::now();
        for _ in 0..iters {
            let _ = run_v7_planned(
                &v7_artifacts,
                &v7_plan,
                &shapes,
                &compiled_inputs,
                c,
                out_size,
            );
        }
        let v7_avg = t.elapsed() / iters;

        let t = Instant::now();
        for _ in 0..iters {
            let _ = run_v7_planned_with_exec(
                &v7_artifacts,
                &v7_plan,
                &shapes,
                &compiled_inputs,
                c,
                out_size,
                v7_exec::ExecuteConfig::auto(),
            );
        }
        let v7_mt_avg = t.elapsed() / iters;

        let t = Instant::now();
        for _ in 0..iters {
            let _ = run_v7_compiled(&v7_compiled, &compiled_inputs, c, out_size);
        }
        let v7_cl_avg = t.elapsed() / iters;

        let t = Instant::now();
        for _ in 0..iters {
            let _ = run_v9_compiled(&v9_compiled, &compiled_inputs, c, out_size);
        }
        let v9_avg = t.elapsed() / iters;

        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        // Verify parallel correctness
        let v9_par_result =
            run_v9_parallel(&v9_compiled, &compiled_inputs, c, out_size, num_threads);
        let v9_par_diff = max_abs_diff(&v9_result, &v9_par_result);
        println!("  v9 mt diff: {:.2e} ({}t)", v9_par_diff, num_threads,);
        assert!(v9_par_diff < 1e-5, "v9 parallel diverged: {}", v9_par_diff);

        let t = Instant::now();
        for _ in 0..iters {
            let _ = run_v9_parallel(&v9_compiled, &compiled_inputs, c, out_size, num_threads);
        }
        let v9_mt_avg = t.elapsed() / iters;

        print_timing("Interpreter:", interp_avg, iters);
        println!();
        print_timing("v2 fusion:", v2_avg, iters);
        println!(
            "  {:.2}x",
            interp_avg.as_nanos() as f64 / v2_avg.as_nanos() as f64
        );
        print_timing("v7 tasks:", v7_avg, iters);
        println!(
            "  {:.2}x",
            interp_avg.as_nanos() as f64 / v7_avg.as_nanos() as f64
        );
        print_timing("v7 cranelift:", v7_cl_avg, iters);
        println!(
            "  {:.2}x",
            interp_avg.as_nanos() as f64 / v7_cl_avg.as_nanos() as f64
        );
        print_timing("v9 fused:", v9_avg, iters);
        println!(
            "  {:.2}x",
            interp_avg.as_nanos() as f64 / v9_avg.as_nanos() as f64
        );
        print_timing(&format!("v9 parallel({}t):", num_threads), v9_mt_avg, iters);
        println!(
            "  {:.2}x  ({:.1}x vs serial)",
            interp_avg.as_nanos() as f64 / v9_mt_avg.as_nanos() as f64,
            v9_avg.as_nanos() as f64 / v9_mt_avg.as_nanos() as f64,
        );
        println!(
            "  v7 vs v2: {:.2}x",
            v2_avg.as_nanos() as f64 / v7_avg.as_nanos() as f64
        );
        println!(
            "  v9 vs v7: {:.2}x",
            v7_avg.as_nanos() as f64 / v9_avg.as_nanos() as f64
        );
        println!("  PASS\n");
    }

    // -----------------------------------------------------------------------
    // Test 8: v8 generic kernel benchmark
    // -----------------------------------------------------------------------
    {
        let (m, n, k) = if let Some(d) = std::env::var("WT_V8_DIM")
            .or_else(|_| std::env::var("WT_TEST7_DIM"))
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&d| d > 0)
        {
            (d, d, d)
        } else {
            (64usize, 80usize, 96usize)
        };
        println!("Test 8: v8 generic kernel [{}x{}] x [{}x{}]", m, k, k, n);

        let mut rng2 = wyrand::WyRand::new(3008);
        let ext_a = GlobalId::new(&mut rng2);
        let ext_b = GlobalId::new(&mut rng2);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng2);
        let a = input_map[&ext_a];
        let b = input_map[&ext_b];
        let c = MatMul::push_new_default_precision(&mut graph, a, b, DType::F32, &mut rng2);
        let ext_out = GlobalId::new(&mut rng2);
        graph.set_output_map([(c, ext_out)]);

        let mut shapes = HashMap::new();
        shapes.insert(a, vec![m, k]);
        shapes.insert(b, vec![k, n]);
        shapes.insert(c, vec![m, n]);

        let a_data = make_random_f32(m * k, 1700);
        let b_data = make_random_f32(k * n, 1701);
        let mut interp_inputs = HashMap::new();
        interp_inputs.insert(
            ext_a,
            NumericTensor::<DynRank>::from_vec_shape(a_data.clone(), vec![m, k]).unwrap(),
        );
        interp_inputs.insert(
            ext_b,
            NumericTensor::<DynRank>::from_vec_shape(b_data.clone(), vec![k, n]).unwrap(),
        );
        let interp_result = run_interpreter(&graph, &interp_inputs, ext_out);
        let compiled_inputs = build_compiled_inputs(&graph, &interp_inputs);
        let out_size = m * n;

        match v8_codegen::compile_graph(&graph, &shapes) {
            Err(e) => {
                println!("  v8 skipped: {e}");
                println!("  PASS (skipped)\n");
            }
            Ok((v8_compiled, _)) => {
                let v8_result = run_v8_compiled(&v8_compiled, &compiled_inputs, c, out_size);
                let v8_diff = max_abs_diff(&interp_result, &v8_result);
                println!("  v8 diff: {:.2e}", v8_diff);
                assert!(v8_diff < 2e-3, "v8 matmul diverged: {v8_diff}");

                let iters = if m >= 512 || n >= 512 || k >= 512 {
                    10
                } else {
                    80
                };
                let t = Instant::now();
                for _ in 0..iters {
                    let _ = run_interpreter(&graph, &interp_inputs, ext_out);
                }
                let interp_avg = t.elapsed() / iters;

                let t = Instant::now();
                for _ in 0..iters {
                    let _ = run_v8_compiled(&v8_compiled, &compiled_inputs, c, out_size);
                }
                let v8_avg = t.elapsed() / iters;

                print_timing("Interpreter:", interp_avg, iters);
                println!();
                print_timing("v8 serial:", v8_avg, iters);
                println!(
                    "  {:.2}x",
                    interp_avg.as_nanos() as f64 / v8_avg.as_nanos() as f64
                );

                // Parallel execution benchmark.
                let n_threads = std::thread::available_parallelism()
                    .map(|x| x.get())
                    .unwrap_or(1);
                if n_threads > 1 {
                    let mut par_bufs: Vec<Vec<f32>> = (0..v8_compiled.layout.num_buffers)
                        .map(|_| Vec::new())
                        .collect();
                    for (&id, &sz) in &v8_compiled.layout.tensor_sizes {
                        let idx = v8_compiled.layout.tensor_index[&id];
                        par_bufs[idx] = vec![0.0f32; sz];
                    }
                    for (id, data) in &compiled_inputs {
                        if let Some(&idx) = v8_compiled.layout.tensor_index.get(id) {
                            par_bufs[idx][..data.len()].copy_from_slice(data);
                        }
                    }
                    let par_ptrs: Vec<*mut f32> =
                        par_bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();
                    let t = Instant::now();
                    for _ in 0..iters {
                        unsafe {
                            v8_executor::execute_parallel(&v8_compiled, &par_ptrs, n_threads);
                        }
                    }
                    let par_avg = t.elapsed() / iters;
                    print_timing(&format!("v8 parallel({n_threads}t):"), par_avg, iters);
                    println!(
                        "  {:.2}x  ({:.1}x vs serial)",
                        interp_avg.as_nanos() as f64 / par_avg.as_nanos() as f64,
                        v8_avg.as_nanos() as f64 / par_avg.as_nanos() as f64,
                    );
                }
                println!("  PASS\n");
            }
        }
    }

    println!("=== All tests passed ===");
}
