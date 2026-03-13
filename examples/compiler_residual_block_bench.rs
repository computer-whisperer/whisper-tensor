use rand::RngCore;
use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

use whisper_tensor::compiler::attempts::v1_scalar_crystal::{
    codegen as v1_codegen, crystal, nano_op::NanoOpExpander,
};
use whisper_tensor::compiler::attempts::v2_fusion::{
    codegen as v2_codegen, kernel as v2_kernel, planner as v2_planner,
};
use whisper_tensor::compiler::attempts::v3_nano_fusion::{
    codegen as v3_codegen, fusion as v3_fusion,
};
use whisper_tensor::compiler::attempts::v4_pool_growth::codegen as v4_codegen;
use whisper_tensor::compiler::attempts::v6_schedule_synthesis::codegen as v6_codegen;
use whisper_tensor::compiler::attempts::v6_schedule_synthesis::synthesis as v6_synth;
use whisper_tensor::compiler::attempts::v7_parallel_crystal::codegen as v7_codegen;
use whisper_tensor::compiler::attempts::v7_parallel_crystal::planner as v7_planner;

use whisper_tensor::dtype::DType;
use whisper_tensor::graph::GlobalId;
use whisper_tensor::milli_graph::MilliOpGraph;
use whisper_tensor::milli_graph::ops::{MatMul, SimpleBinary, SimpleUnaryOp};
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::tensor_rank::DynRank;

#[derive(Clone, Copy)]
struct ResidualGatedBlockExternals {
    input: GlobalId,
    w_up: GlobalId,
    w_gate: GlobalId,
    w_down: GlobalId,
    b_up: GlobalId,
    b_gate: GlobalId,
    b_down: GlobalId,
    ones_ff: GlobalId,
    out_scale: GlobalId,
    out_bias: GlobalId,
    output: GlobalId,
}

fn build_residual_gated_block(
    batch: usize,
    d_model: usize,
    d_ff: usize,
    rng: &mut impl rand::Rng,
) -> (
    MilliOpGraph,
    ResidualGatedBlockExternals,
    HashMap<GlobalId, Vec<usize>>,
) {
    let ext = ResidualGatedBlockExternals {
        input: GlobalId::new(rng),
        w_up: GlobalId::new(rng),
        w_gate: GlobalId::new(rng),
        w_down: GlobalId::new(rng),
        b_up: GlobalId::new(rng),
        b_gate: GlobalId::new(rng),
        b_down: GlobalId::new(rng),
        ones_ff: GlobalId::new(rng),
        out_scale: GlobalId::new(rng),
        out_bias: GlobalId::new(rng),
        output: GlobalId::new(rng),
    };

    let (mut graph, input_map) = MilliOpGraph::new(
        [
            ext.input,
            ext.w_up,
            ext.w_gate,
            ext.w_down,
            ext.b_up,
            ext.b_gate,
            ext.b_down,
            ext.ones_ff,
            ext.out_scale,
            ext.out_bias,
        ],
        rng,
    );

    let x = input_map[&ext.input];
    let w_up = input_map[&ext.w_up];
    let w_gate = input_map[&ext.w_gate];
    let w_down = input_map[&ext.w_down];
    let b_up = input_map[&ext.b_up];
    let b_gate = input_map[&ext.b_gate];
    let b_down = input_map[&ext.b_down];
    let ones_ff = input_map[&ext.ones_ff];
    let out_scale = input_map[&ext.out_scale];
    let out_bias = input_map[&ext.out_bias];

    let mut shapes = HashMap::new();
    shapes.insert(x, vec![batch, d_model]);
    shapes.insert(w_up, vec![d_model, d_ff]);
    shapes.insert(w_gate, vec![d_model, d_ff]);
    shapes.insert(w_down, vec![d_ff, d_model]);
    shapes.insert(b_up, vec![1, d_ff]);
    shapes.insert(b_gate, vec![1, d_ff]);
    shapes.insert(b_down, vec![1, d_model]);
    shapes.insert(ones_ff, vec![1, d_ff]);
    shapes.insert(out_scale, vec![1, d_model]);
    shapes.insert(out_bias, vec![1, d_model]);

    let up_mm = MatMul::push_new_default_precision(&mut graph, x, w_up, DType::F32, rng);
    shapes.insert(up_mm, vec![batch, d_ff]);
    let up = SimpleBinary::add(&mut graph, up_mm, b_up, rng);
    shapes.insert(up, vec![batch, d_ff]);

    let gate_mm = MatMul::push_new_default_precision(&mut graph, x, w_gate, DType::F32, rng);
    shapes.insert(gate_mm, vec![batch, d_ff]);
    let gate_bias = SimpleBinary::add(&mut graph, gate_mm, b_gate, rng);
    shapes.insert(gate_bias, vec![batch, d_ff]);
    let gate_neg = SimpleUnaryOp::neg(&mut graph, gate_bias, rng);
    shapes.insert(gate_neg, vec![batch, d_ff]);
    let gate_exp = SimpleUnaryOp::exp(&mut graph, gate_neg, rng);
    shapes.insert(gate_exp, vec![batch, d_ff]);
    let gate_denom = SimpleBinary::add(&mut graph, gate_exp, ones_ff, rng);
    shapes.insert(gate_denom, vec![batch, d_ff]);
    let gate = SimpleUnaryOp::reciprocal(&mut graph, gate_denom, rng);
    shapes.insert(gate, vec![batch, d_ff]);

    let ff = SimpleBinary::mul(&mut graph, up, gate, rng);
    shapes.insert(ff, vec![batch, d_ff]);
    let down_mm = MatMul::push_new_default_precision(&mut graph, ff, w_down, DType::F32, rng);
    shapes.insert(down_mm, vec![batch, d_model]);
    let down = SimpleBinary::add(&mut graph, down_mm, b_down, rng);
    shapes.insert(down, vec![batch, d_model]);
    let residual = SimpleBinary::add(&mut graph, down, x, rng);
    shapes.insert(residual, vec![batch, d_model]);
    let scaled = SimpleBinary::mul(&mut graph, residual, out_scale, rng);
    shapes.insert(scaled, vec![batch, d_model]);
    let shifted = SimpleBinary::add(&mut graph, scaled, out_bias, rng);
    shapes.insert(shifted, vec![batch, d_model]);
    let out = SimpleUnaryOp::trig(&mut graph, shifted, whisper_tensor::TrigOp::Tanh, rng);
    shapes.insert(out, vec![batch, d_model]);

    graph.set_output_map([(out, ext.output)]);
    (graph, ext, shapes)
}

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

fn run_v4_compiled(
    compiled: &v4_codegen::NativeCompiledGraph,
    inputs: &HashMap<GlobalId, Vec<f32>>,
    output_id: GlobalId,
    output_size: usize,
) -> Vec<f32> {
    run_v1_compiled(compiled, inputs, output_id, output_size)
}

fn run_v6_compiled(
    compiled: &v6_codegen::NativeCompiledGraph,
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

fn run_catch_silent<F, R>(f: F) -> std::thread::Result<R>
where
    F: FnOnce() -> R,
{
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let out = catch_unwind(AssertUnwindSafe(f));
    std::panic::set_hook(hook);
    out
}

fn main() {
    println!("=== Residual Block Standalone Benchmark ===\n");
    let (batch, d_model, d_ff) = (24usize, 96usize, 192usize);
    println!(
        "Graph: residual gated block, batch={}, d_model={}, d_ff={}",
        batch, d_model, d_ff
    );

    let mut rng = wyrand::WyRand::new(6001);
    let (graph, ext, shapes) = build_residual_gated_block(batch, d_model, d_ff, &mut rng);
    let int_out = find_internal_output(&graph, ext.output);
    let out_size = batch * d_model;

    let mut interp_inputs = HashMap::new();
    interp_inputs.insert(
        ext.input,
        NumericTensor::<DynRank>::from_vec_shape(
            make_random_f32(batch * d_model, 7001),
            vec![batch, d_model],
        )
        .unwrap(),
    );
    interp_inputs.insert(
        ext.w_up,
        NumericTensor::<DynRank>::from_vec_shape(
            make_random_f32(d_model * d_ff, 7002),
            vec![d_model, d_ff],
        )
        .unwrap(),
    );
    interp_inputs.insert(
        ext.w_gate,
        NumericTensor::<DynRank>::from_vec_shape(
            make_random_f32(d_model * d_ff, 7003),
            vec![d_model, d_ff],
        )
        .unwrap(),
    );
    interp_inputs.insert(
        ext.w_down,
        NumericTensor::<DynRank>::from_vec_shape(
            make_random_f32(d_ff * d_model, 7004),
            vec![d_ff, d_model],
        )
        .unwrap(),
    );
    interp_inputs.insert(
        ext.b_up,
        NumericTensor::<DynRank>::from_vec_shape(make_random_f32(d_ff, 7005), vec![1, d_ff])
            .unwrap(),
    );
    interp_inputs.insert(
        ext.b_gate,
        NumericTensor::<DynRank>::from_vec_shape(make_random_f32(d_ff, 7006), vec![1, d_ff])
            .unwrap(),
    );
    interp_inputs.insert(
        ext.b_down,
        NumericTensor::<DynRank>::from_vec_shape(make_random_f32(d_model, 7007), vec![1, d_model])
            .unwrap(),
    );
    interp_inputs.insert(
        ext.ones_ff,
        NumericTensor::<DynRank>::from_vec_shape(vec![1.0f32; d_ff], vec![1, d_ff]).unwrap(),
    );
    interp_inputs.insert(
        ext.out_scale,
        NumericTensor::<DynRank>::from_vec_shape(make_random_f32(d_model, 7008), vec![1, d_model])
            .unwrap(),
    );
    interp_inputs.insert(
        ext.out_bias,
        NumericTensor::<DynRank>::from_vec_shape(make_random_f32(d_model, 7009), vec![1, d_model])
            .unwrap(),
    );

    let interp_result = run_interpreter(&graph, &interp_inputs, ext.output);
    let compiled_inputs = build_compiled_inputs(&graph, &interp_inputs);
    let iters = 40u32;

    let t = Instant::now();
    for _ in 0..iters {
        let _ = run_interpreter(&graph, &interp_inputs, ext.output);
    }
    let interp_avg = t.elapsed() / iters;

    println!("\nBaseline:");
    print_timing("Interpreter:", interp_avg, iters);
    println!();

    let mut expander = NanoOpExpander::new(shapes.clone());
    let nano_ops = expander.expand(&graph).expect("v1 expand");
    let crystal_ops = crystal::crystallize(&nano_ops);

    println!("\nAttempt 1 (v1 scalar crystal):");
    match run_catch_silent(|| {
        let t = Instant::now();
        let v1_layout = v1_codegen::TensorLayout::from_shapes(&shapes);
        let v1_compiled =
            v1_codegen::compile_crystallized(&crystal_ops, &v1_layout).expect("v1 compile");
        let v1_compile = t.elapsed();
        let v1_result = run_v1_compiled(&v1_compiled, &compiled_inputs, int_out, out_size);
        let v1_diff = max_abs_diff(&interp_result, &v1_result);
        let t = Instant::now();
        for _ in 0..iters {
            let _ = run_v1_compiled(&v1_compiled, &compiled_inputs, int_out, out_size);
        }
        let v1_avg = t.elapsed() / iters;
        (v1_compile, v1_diff, v1_avg)
    }) {
        Ok((v1_compile, v1_diff, v1_avg)) => {
            println!("  diff: {:.2e}", v1_diff);
            print_timing("compile:", v1_compile, 1);
            println!();
            print_timing("execute:", v1_avg, iters);
            println!(
                "  {:.2}x vs interp",
                interp_avg.as_nanos() as f64 / v1_avg.as_nanos() as f64
            );
        }
        Err(_) => {
            println!("  unsupported: panicked during compile/execute");
        }
    }

    println!("\nAttempt 2 (v2 fusion):");
    let t = Instant::now();
    let v2_kernels = v2_planner::plan(&graph, &shapes).expect("v2 plan");
    let v2_stats = v2_kernel::stats(&v2_kernels);
    let v2_layout = v2_codegen::TensorLayout::from_shapes(&shapes);
    let v2_compiled = v2_codegen::compile(&v2_kernels, &v2_layout).expect("v2 compile");
    let v2_compile = t.elapsed();
    let v2_result = run_v2_compiled(&v2_compiled, &compiled_inputs, int_out, out_size);
    let v2_diff = max_abs_diff(&interp_result, &v2_result);
    let t = Instant::now();
    for _ in 0..iters {
        let _ = run_v2_compiled(&v2_compiled, &compiled_inputs, int_out, out_size);
    }
    let v2_avg = t.elapsed() / iters;
    println!(
        "  kernels: {} (gemm={}, elementwise={}, fused_ops={})",
        v2_stats.num_kernels, v2_stats.num_gemm, v2_stats.num_elementwise, v2_stats.total_fused_ops
    );
    println!("  diff: {:.2e}", v2_diff);
    print_timing("compile:", v2_compile, 1);
    println!();
    print_timing("execute:", v2_avg, iters);
    println!(
        "  {:.2}x vs interp",
        interp_avg.as_nanos() as f64 / v2_avg.as_nanos() as f64
    );

    println!("\nAttempt 3 (v3 nano fusion):");
    let (_v3_fused, v3_stats) = v3_fusion::fuse_with_stats(&crystal_ops);
    match run_catch_silent(|| {
        let t = Instant::now();
        let v3_layout = v3_codegen::TensorLayout::from_shapes(&shapes);
        let v3_compiled =
            v3_codegen::compile_fused_crystallized(&crystal_ops, &v3_layout).expect("v3 compile");
        let v3_compile = t.elapsed();
        let v3_result = run_v1_compiled(&v3_compiled, &compiled_inputs, int_out, out_size);
        let v3_diff = max_abs_diff(&interp_result, &v3_result);
        let t = Instant::now();
        for _ in 0..iters {
            let _ = run_v1_compiled(&v3_compiled, &compiled_inputs, int_out, out_size);
        }
        let v3_avg = t.elapsed() / iters;
        (v3_compile, v3_diff, v3_avg)
    }) {
        Ok((v3_compile, v3_diff, v3_avg)) => {
            println!(
                "  fused_pairs={}, eliminated_loads={}",
                v3_stats.fused_pairs, v3_stats.eliminated_loads
            );
            println!("  diff: {:.2e}", v3_diff);
            print_timing("compile:", v3_compile, 1);
            println!();
            print_timing("execute:", v3_avg, iters);
            println!(
                "  {:.2}x vs interp",
                interp_avg.as_nanos() as f64 / v3_avg.as_nanos() as f64
            );
        }
        Err(_) => {
            println!("  unsupported: panicked during compile/execute");
        }
    }

    println!("\nAttempt 4 (v4 pool growth):");
    let t = Instant::now();
    match v4_codegen::compile_graph(&graph, &shapes) {
        Ok((v4_compiled, v4_artifacts)) => {
            let v4_compile = t.elapsed();
            let v4_result = run_v4_compiled(&v4_compiled, &compiled_inputs, int_out, out_size);
            let v4_diff = max_abs_diff(&interp_result, &v4_result);
            let t = Instant::now();
            for _ in 0..iters {
                let _ = run_v4_compiled(&v4_compiled, &compiled_inputs, int_out, out_size);
            }
            let v4_avg = t.elapsed() / iters;
            println!(
                "  loops: ordered={}, fused={}, fused_pairs={}",
                v4_artifacts.plan.stats.ordered_loops,
                v4_artifacts.plan.stats.fused_loops,
                v4_artifacts.plan.stats.fused_pairs
            );
            println!("  diff: {:.2e}", v4_diff);
            print_timing("compile:", v4_compile, 1);
            println!();
            print_timing("execute:", v4_avg, iters);
            println!(
                "  {:.2}x vs interp",
                interp_avg.as_nanos() as f64 / v4_avg.as_nanos() as f64
            );
        }
        Err(err) => {
            println!("  unsupported: {err}");
        }
    }

    println!("\nAttempt 6 (v6 schedule synthesis):");
    let t = Instant::now();
    let v6_artifacts = v6_synth::build_from_graph(&graph, &shapes).expect("v6 build");
    let v6_build = t.elapsed();
    println!(
        "  schedule: families={} pointwise={} reductions={} unknown={}",
        v6_artifacts.schedule.stats.grouped_families,
        v6_artifacts.schedule.stats.pointwise_families,
        v6_artifacts.schedule.stats.additive_reduction_families,
        v6_artifacts.schedule.stats.unknown_families
    );
    for (i, recovered) in v6_artifacts.schedule.loops.iter().enumerate() {
        let kind = match &recovered.intent {
            v6_synth::LoopIntent::Pointwise(pointwise) => {
                if pointwise.contiguous_output {
                    "pointwise(contig)"
                } else {
                    "pointwise(strided)"
                }
            }
            v6_synth::LoopIntent::AdditiveReduction(_) => "reduction",
            v6_synth::LoopIntent::Unknown => "unknown",
        };
        println!(
            "    loop[{i:02}] kind={kind:17} shape={:?} covered={} loads={}",
            recovered.output_shape,
            recovered.covered_outputs,
            recovered.load_tensors.len()
        );
    }
    print_timing("build:", v6_build, 1);
    println!();

    let t = Instant::now();
    match v6_codegen::compile_graph(&graph, &shapes) {
        Ok((v6_compiled, _)) => {
            let v6_compile = t.elapsed();
            let v6_result = run_v6_compiled(&v6_compiled, &compiled_inputs, int_out, out_size);
            let v6_diff = max_abs_diff(&interp_result, &v6_result);
            let t = Instant::now();
            for _ in 0..iters {
                let _ = run_v6_compiled(&v6_compiled, &compiled_inputs, int_out, out_size);
            }
            let v6_avg = t.elapsed() / iters;
            println!("  diff: {:.2e}", v6_diff);
            print_timing("compile:", v6_compile, 1);
            println!();
            print_timing("execute:", v6_avg, iters);
            println!(
                "  {:.2}x vs interp",
                interp_avg.as_nanos() as f64 / v6_avg.as_nanos() as f64
            );
        }
        Err(err) => {
            println!("  compile unsupported: {err}");
        }
    }

    println!("\nAttempt 7 (v7 planner/codegen):");
    let t = Instant::now();
    let (v7_artifacts, v7_plan) = v7_planner::plan_from_graph(
        &graph,
        &shapes,
        v7_planner::TaskPlannerConfig {
            min_tile_elements: 512,
            max_tile_elements: 4096,
        },
    )
    .expect("v7 plan");
    let v7_plan_t = t.elapsed();
    println!(
        "  plan: loops={} red_loops={} planned_loops={} tasks={} est_fmas={}",
        v7_plan.stats.loop_count,
        v7_plan.stats.additive_reduction_loops,
        v7_plan.stats.planned_loops,
        v7_plan.stats.total_tasks,
        v7_plan.stats.total_estimated_fmas
    );
    print_timing("plan:", v7_plan_t, 1);
    println!();
    let t = Instant::now();
    match v7_codegen::compile_plan(&v7_artifacts, &v7_plan, &shapes) {
        Ok(v7_compiled) => {
            let v7_compile = t.elapsed();
            let v7_result = run_v7_compiled(&v7_compiled, &compiled_inputs, int_out, out_size);
            let v7_diff = max_abs_diff(&interp_result, &v7_result);
            let t = Instant::now();
            for _ in 0..iters {
                let _ = run_v7_compiled(&v7_compiled, &compiled_inputs, int_out, out_size);
            }
            let v7_avg = t.elapsed() / iters;
            print_timing("compile (strict):", v7_compile, 1);
            println!();
            println!("  diff: {:.2e}", v7_diff);
            print_timing("execute (strict):", v7_avg, iters);
            println!(
                "  {:.2}x vs interp",
                interp_avg.as_nanos() as f64 / v7_avg.as_nanos() as f64
            );
        }
        Err(err) => {
            println!("  compile (strict) unsupported: {err}");
        }
    }
    let t = Instant::now();
    match v7_codegen::compile_plan_reduction_only(&v7_artifacts, &v7_plan, &shapes) {
        Ok((_compiled, coverage)) => {
            let v7_compile = t.elapsed();
            print_timing("compile (reduction-only):", v7_compile, 1);
            println!();
            println!(
                "  reduction coverage: compiled={} planned={} schedule_loops={} pointwise={} unknown={}",
                coverage.compiled_reduction_loops,
                coverage.planned_reduction_loops,
                coverage.schedule_loops,
                coverage.pointwise_loops,
                coverage.unknown_loops
            );
        }
        Err(err) => {
            println!("  compile (reduction-only) unsupported: {err}");
        }
    }
}
