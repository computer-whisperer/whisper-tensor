use std::collections::HashMap;
use std::time::Instant;
use rand::RngCore;
use whisper_tensor::compiler::attempts::v1_scalar_crystal::{codegen, crystal};
use whisper_tensor::compiler::attempts::v1_scalar_crystal::codegen::TensorLayout;
use whisper_tensor::compiler::attempts::v1_scalar_crystal::nano_op::NanoOpExpander;
use whisper_tensor::graph::GlobalId;
use whisper_tensor::milli_graph::MilliOpGraph;
use whisper_tensor::milli_graph::ops::{SimpleBinary, SimpleUnaryOp};
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::tensor_rank::DynRank;

/// Build a graph that chains: out = neg(exp(a * b + c))
/// where a, b, c are all the same shape.
fn build_chain_graph(
    shape: &[usize],
    rng: &mut impl rand::Rng,
) -> (
    MilliOpGraph,
    GlobalId,  // ext_a
    GlobalId,  // ext_b
    GlobalId,  // ext_c
    GlobalId,  // ext_out
    HashMap<GlobalId, Vec<usize>>,
) {
    let ext_a = GlobalId::new(rng);
    let ext_b = GlobalId::new(rng);
    let ext_c = GlobalId::new(rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b, ext_c], rng);
    let int_a = input_map[&ext_a];
    let int_b = input_map[&ext_b];
    let int_c = input_map[&ext_c];

    // a * b
    let mul_out = SimpleBinary::mul(&mut graph, int_a, int_b, rng);
    // (a * b) + c
    let add_out = SimpleBinary::add(&mut graph, mul_out, int_c, rng);
    // exp((a * b) + c)
    let exp_out = SimpleUnaryOp::exp(&mut graph, add_out, rng);
    // neg(exp(...))
    let neg_out = SimpleUnaryOp::neg(&mut graph, exp_out, rng);

    let ext_out = GlobalId::new(rng);
    graph.set_output_map([(neg_out, ext_out)]);

    let shape_vec = shape.to_vec();
    let mut shapes = HashMap::new();
    shapes.insert(int_a, shape_vec.clone());
    shapes.insert(int_b, shape_vec.clone());
    shapes.insert(int_c, shape_vec.clone());
    shapes.insert(mul_out, shape_vec.clone());
    shapes.insert(add_out, shape_vec.clone());
    shapes.insert(exp_out, shape_vec.clone());
    shapes.insert(neg_out, shape_vec);

    (graph, ext_a, ext_b, ext_c, ext_out, shapes)
}

/// Build a deeper graph: a multi-layer "MLP-like" computation.
/// Each layer: out = tanh(x * weights + bias)
/// All elementwise (no matmul), so weights/bias are same shape as x.
fn build_deep_graph(
    shape: &[usize],
    num_layers: usize,
    rng: &mut impl rand::Rng,
) -> (
    MilliOpGraph,
    GlobalId,                       // ext_input
    Vec<(GlobalId, GlobalId)>,      // ext_(weight, bias) per layer
    GlobalId,                       // ext_out
    HashMap<GlobalId, Vec<usize>>,
) {
    let ext_input = GlobalId::new(rng);
    let mut ext_ids = vec![ext_input];
    let mut layer_params = Vec::new();

    for _ in 0..num_layers {
        let ext_w = GlobalId::new(rng);
        let ext_b = GlobalId::new(rng);
        ext_ids.push(ext_w);
        ext_ids.push(ext_b);
        layer_params.push((ext_w, ext_b));
    }

    let (mut graph, input_map) = MilliOpGraph::new(ext_ids.iter().copied(), rng);
    let shape_vec = shape.to_vec();
    let mut shapes = HashMap::new();

    // Register input shapes
    for ext_id in &ext_ids {
        let int_id = input_map[ext_id];
        shapes.insert(int_id, shape_vec.clone());
    }

    let mut current = input_map[&ext_input];

    for (ext_w, ext_b) in &layer_params {
        let int_w = input_map[ext_w];
        let int_b = input_map[ext_b];

        // x * w
        let mul_out = SimpleBinary::mul(&mut graph, current, int_w, rng);
        shapes.insert(mul_out, shape_vec.clone());

        // (x * w) + b
        let add_out = SimpleBinary::add(&mut graph, mul_out, int_b, rng);
        shapes.insert(add_out, shape_vec.clone());

        // tanh(...)
        let tanh_out = SimpleUnaryOp::trig(
            &mut graph,
            add_out,
            whisper_tensor::TrigOp::Tanh,
            rng,
        );
        shapes.insert(tanh_out, shape_vec.clone());

        current = tanh_out;
    }

    let ext_out = GlobalId::new(rng);
    graph.set_output_map([(current, ext_out)]);

    (graph, ext_input, layer_params, ext_out, shapes)
}

/// Run interpreter on a graph and return output as Vec<f32>.
fn run_interpreter(
    graph: &MilliOpGraph,
    inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
    output_ext: GlobalId,
) -> Vec<f32> {
    let results = whisper_tensor::compiler::interpret_milli_graph(graph, inputs).unwrap();
    let output = &results[&output_ext];
    let flat: Vec<f32> = output.flatten().unwrap().try_into().unwrap();
    flat
}

/// Run compiled graph and return output as Vec<f32>.
fn run_compiled(
    compiled: &codegen::NativeCompiledGraph,
    inputs: &HashMap<GlobalId, Vec<f32>>,
    output_id: GlobalId,
    output_size: usize,
) -> Vec<f32> {
    let layout = &compiled.layout;
    let mut buffers_storage: Vec<Vec<f32>> = (0..layout.num_buffers)
        .map(|_| Vec::new())
        .collect();

    // Fill input/constant buffers
    for (id, data) in inputs {
        let idx = layout.tensor_index[id];
        buffers_storage[idx] = data.clone();
    }

    // Allocate intermediate and output buffers
    for (id, &size) in &layout.tensor_sizes {
        let idx = layout.tensor_index[id];
        if buffers_storage[idx].is_empty() {
            buffers_storage[idx] = vec![0.0f32; size];
        }
    }

    let mut buffer_ptrs: Vec<*mut f32> = buffers_storage
        .iter_mut()
        .map(|v| v.as_mut_ptr())
        .collect();

    unsafe {
        compiled.execute(&mut buffer_ptrs);
    }

    let out_idx = layout.tensor_index[&output_id];
    buffers_storage[out_idx][..output_size].to_vec()
}

fn make_random_f32(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = wyrand::WyRand::new(seed);
    (0..n)
        .map(|_| {
            // Small values to keep exp() from blowing up
            let bits: u32 = rng.next_u32();
            (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn main() {
    println!("=== Whisper-Tensor Compiler Test ===\n");

    let mut rng = wyrand::WyRand::new(12345);

    // -----------------------------------------------------------------------
    // Test 1: Small chain graph (neg(exp(a*b + c)))
    // -----------------------------------------------------------------------
    {
        let shape = &[4, 8];
        let total: usize = shape.iter().product();
        println!("Test 1: neg(exp(a*b + c)), shape {:?} ({} elements)", shape, total);

        let (graph, ext_a, ext_b, ext_c, ext_out, shapes) =
            build_chain_graph(shape, &mut rng);

        let a_data = make_random_f32(total, 1);
        let b_data = make_random_f32(total, 2);
        let c_data = make_random_f32(total, 3);

        // Interpreter
        let int_a = graph.input_map[&ext_a];
        let int_b = graph.input_map[&ext_b];
        let int_c = graph.input_map[&ext_c];

        let mut interp_inputs = HashMap::new();
        interp_inputs.insert(ext_a, NumericTensor::<DynRank>::from_vec_shape(a_data.clone(), shape.to_vec()).unwrap());
        interp_inputs.insert(ext_b, NumericTensor::<DynRank>::from_vec_shape(b_data.clone(), shape.to_vec()).unwrap());
        interp_inputs.insert(ext_c, NumericTensor::<DynRank>::from_vec_shape(c_data.clone(), shape.to_vec()).unwrap());

        let interp_result = run_interpreter(&graph, &interp_inputs, ext_out);

        // Compile (scalar)
        let mut expander = NanoOpExpander::new(shapes.clone());
        let nano_ops = expander.expand(&graph).unwrap();
        let layout = TensorLayout::from_shapes(&shapes);
        let compiled = codegen::compile(&nano_ops, &layout).unwrap();

        // Compile (crystal)
        let crystal_ops = crystal::crystallize(&nano_ops);
        let crystal_stats = crystal::stats(&crystal_ops);
        let crystal_compiled = codegen::compile_crystallized(&crystal_ops, &layout).unwrap();

        println!("  Nano ops: {}", nano_ops.len());
        println!("  Crystal: {} loops, {} scalars, {}/{} nano ops in loops",
            crystal_stats.num_loops, crystal_stats.num_scalars,
            crystal_stats.nano_ops_in_loops, nano_ops.len());

        // Run both compiled versions
        let mut compiled_inputs = HashMap::new();
        compiled_inputs.insert(int_a, a_data);
        compiled_inputs.insert(int_b, b_data);
        compiled_inputs.insert(int_c, c_data);

        let int_out = *graph.output_map.as_ref().unwrap().keys()
            .find(|id| graph.output_map.as_ref().unwrap()[id] == ext_out)
            .unwrap();

        let scalar_result = run_compiled(&compiled, &compiled_inputs, int_out, total);
        let crystal_result = run_compiled(&crystal_compiled, &compiled_inputs, int_out, total);

        let diff_scalar = max_abs_diff(&interp_result, &scalar_result);
        let diff_crystal = max_abs_diff(&interp_result, &crystal_result);
        println!("  Max abs diff (scalar):  {:.2e}", diff_scalar);
        println!("  Max abs diff (crystal): {:.2e}", diff_crystal);
        assert!(diff_scalar < 1e-5, "Scalar results diverged! diff={}", diff_scalar);
        assert!(diff_crystal < 1e-5, "Crystal results diverged! diff={}", diff_crystal);
        println!("  PASS\n");
    }

    // -----------------------------------------------------------------------
    // Test 2: Deep elementwise MLP, small shape
    // -----------------------------------------------------------------------
    {
        let shape = &[8, 16];
        let total: usize = shape.iter().product();
        let num_layers = 10;
        println!(
            "Test 2: {}-layer elementwise MLP, shape {:?} ({} elements)",
            num_layers, shape, total
        );

        let (graph, ext_input, layer_params, ext_out, shapes) =
            build_deep_graph(shape, num_layers, &mut rng);

        // Prepare interpreter inputs
        let mut interp_inputs = HashMap::new();
        let input_data = make_random_f32(total, 100);
        interp_inputs.insert(
            ext_input,
            NumericTensor::<DynRank>::from_vec_shape(input_data.clone(), shape.to_vec()).unwrap(),
        );
        for (i, (ext_w, ext_b)) in layer_params.iter().enumerate() {
            let w_data = make_random_f32(total, 200 + i as u64 * 2);
            let b_data = make_random_f32(total, 201 + i as u64 * 2);
            interp_inputs.insert(
                *ext_w,
                NumericTensor::<DynRank>::from_vec_shape(w_data, shape.to_vec()).unwrap(),
            );
            interp_inputs.insert(
                *ext_b,
                NumericTensor::<DynRank>::from_vec_shape(b_data, shape.to_vec()).unwrap(),
            );
        }

        // Run interpreter with timing
        let t0 = Instant::now();
        let interp_result = run_interpreter(&graph, &interp_inputs, ext_out);
        let _interp_time = t0.elapsed();

        // Compile (scalar + crystal)
        let mut expander = NanoOpExpander::new(shapes.clone());
        let nano_ops = expander.expand(&graph).unwrap();
        let layout = TensorLayout::from_shapes(&shapes);
        let compiled = codegen::compile(&nano_ops, &layout).unwrap();

        let crystal_ops = crystal::crystallize(&nano_ops);
        let crystal_stats = crystal::stats(&crystal_ops);
        let crystal_compiled = codegen::compile_crystallized(&crystal_ops, &layout).unwrap();

        println!("  Nano ops: {}", nano_ops.len());
        println!("  Crystal: {} loops, {} scalars, {}/{} nano ops in loops",
            crystal_stats.num_loops, crystal_stats.num_scalars,
            crystal_stats.nano_ops_in_loops, nano_ops.len());

        // Build compiled inputs using internal IDs
        let mut compiled_inputs = HashMap::new();
        let int_input = graph.input_map[&ext_input];
        compiled_inputs.insert(
            int_input,
            interp_inputs[&ext_input].flatten().unwrap().try_into().unwrap(),
        );
        for (ext_w, ext_b) in &layer_params {
            let int_w = graph.input_map[ext_w];
            let int_b = graph.input_map[ext_b];
            compiled_inputs.insert(
                int_w,
                Vec::<f32>::try_from(interp_inputs[ext_w].flatten().unwrap()).unwrap(),
            );
            compiled_inputs.insert(
                int_b,
                Vec::<f32>::try_from(interp_inputs[ext_b].flatten().unwrap()).unwrap(),
            );
        }

        // Find internal output tensor ID
        let int_out = *graph.output_map.as_ref().unwrap().keys()
            .find(|id| graph.output_map.as_ref().unwrap()[id] == ext_out)
            .unwrap();

        let compiled_result = run_compiled(&compiled, &compiled_inputs, int_out, total);
        let crystal_result = run_compiled(&crystal_compiled, &compiled_inputs, int_out, total);

        // Run all three multiple times for timing
        let n_iters = 100;
        let t0 = Instant::now();
        for _ in 0..n_iters {
            let _ = run_interpreter(&graph, &interp_inputs, ext_out);
        }
        let interp_avg = t0.elapsed() / n_iters;

        let t0 = Instant::now();
        for _ in 0..n_iters {
            let _ = run_compiled(&compiled, &compiled_inputs, int_out, total);
        }
        let scalar_avg = t0.elapsed() / n_iters;

        let t0 = Instant::now();
        for _ in 0..n_iters {
            let _ = run_compiled(&crystal_compiled, &compiled_inputs, int_out, total);
        }
        let crystal_avg = t0.elapsed() / n_iters;

        let diff_scalar = max_abs_diff(&interp_result, &compiled_result);
        let diff_crystal = max_abs_diff(&interp_result, &crystal_result);
        println!("  Max abs diff (scalar):  {:.2e}", diff_scalar);
        println!("  Max abs diff (crystal): {:.2e}", diff_crystal);
        println!("  Interpreter avg:     {:?} ({} iters)", interp_avg, n_iters);
        println!("  Scalar compiled avg: {:?} ({} iters)", scalar_avg, n_iters);
        println!("  Crystal compiled avg:{:?} ({} iters)", crystal_avg, n_iters);
        if scalar_avg.as_nanos() > 0 {
            println!("  Scalar speedup:  {:.2}x vs interp", interp_avg.as_nanos() as f64 / scalar_avg.as_nanos() as f64);
        }
        if crystal_avg.as_nanos() > 0 {
            println!("  Crystal speedup: {:.2}x vs interp, {:.2}x vs scalar",
                interp_avg.as_nanos() as f64 / crystal_avg.as_nanos() as f64,
                scalar_avg.as_nanos() as f64 / crystal_avg.as_nanos() as f64);
        }
        assert!(diff_scalar < 1e-5, "Scalar results diverged! diff={}", diff_scalar);
        assert!(diff_crystal < 1e-5, "Crystal results diverged! diff={}", diff_crystal);
        println!("  PASS\n");
    }

    // -----------------------------------------------------------------------
    // Test 3: Larger shape, deep graph — stress test
    // -----------------------------------------------------------------------
    {
        let shape = &[32, 64];
        let total: usize = shape.iter().product();
        let num_layers = 5;
        println!(
            "Test 3: {}-layer elementwise MLP, shape {:?} ({} elements) — stress test",
            num_layers, shape, total
        );

        let (graph, ext_input, layer_params, ext_out, shapes) =
            build_deep_graph(shape, num_layers, &mut rng);

        let mut interp_inputs = HashMap::new();
        let input_data = make_random_f32(total, 500);
        interp_inputs.insert(
            ext_input,
            NumericTensor::<DynRank>::from_vec_shape(input_data.clone(), shape.to_vec()).unwrap(),
        );
        for (i, (ext_w, ext_b)) in layer_params.iter().enumerate() {
            interp_inputs.insert(
                *ext_w,
                NumericTensor::<DynRank>::from_vec_shape(
                    make_random_f32(total, 600 + i as u64 * 2),
                    shape.to_vec(),
                )
                .unwrap(),
            );
            interp_inputs.insert(
                *ext_b,
                NumericTensor::<DynRank>::from_vec_shape(
                    make_random_f32(total, 601 + i as u64 * 2),
                    shape.to_vec(),
                )
                .unwrap(),
            );
        }

        let interp_result = run_interpreter(&graph, &interp_inputs, ext_out);

        let mut expander = NanoOpExpander::new(shapes.clone());
        let nano_ops = expander.expand(&graph).unwrap();
        let layout = TensorLayout::from_shapes(&shapes);
        let compiled = codegen::compile(&nano_ops, &layout).unwrap();

        let crystal_ops = crystal::crystallize(&nano_ops);
        let crystal_stats = crystal::stats(&crystal_ops);
        let crystal_compiled = codegen::compile_crystallized(&crystal_ops, &layout).unwrap();

        println!("  Nano ops: {}", nano_ops.len());
        println!("  Crystal: {} loops, {} scalars, {}/{} nano ops in loops",
            crystal_stats.num_loops, crystal_stats.num_scalars,
            crystal_stats.nano_ops_in_loops, nano_ops.len());

        let mut compiled_inputs = HashMap::new();
        compiled_inputs.insert(
            graph.input_map[&ext_input],
            Vec::<f32>::try_from(interp_inputs[&ext_input].flatten().unwrap()).unwrap(),
        );
        for (ext_w, ext_b) in &layer_params {
            compiled_inputs.insert(
                graph.input_map[ext_w],
                Vec::<f32>::try_from(interp_inputs[ext_w].flatten().unwrap()).unwrap(),
            );
            compiled_inputs.insert(
                graph.input_map[ext_b],
                Vec::<f32>::try_from(interp_inputs[ext_b].flatten().unwrap()).unwrap(),
            );
        }

        let int_out = *graph.output_map.as_ref().unwrap().keys()
            .find(|id| graph.output_map.as_ref().unwrap()[id] == ext_out)
            .unwrap();

        let compiled_result = run_compiled(&compiled, &compiled_inputs, int_out, total);
        let crystal_result = run_compiled(&crystal_compiled, &compiled_inputs, int_out, total);

        let n_iters = 50;
        let t0 = Instant::now();
        for _ in 0..n_iters {
            let _ = run_interpreter(&graph, &interp_inputs, ext_out);
        }
        let interp_avg = t0.elapsed() / n_iters;

        let t0 = Instant::now();
        for _ in 0..n_iters {
            let _ = run_compiled(&compiled, &compiled_inputs, int_out, total);
        }
        let scalar_avg = t0.elapsed() / n_iters;

        let t0 = Instant::now();
        for _ in 0..n_iters {
            let _ = run_compiled(&crystal_compiled, &compiled_inputs, int_out, total);
        }
        let crystal_avg = t0.elapsed() / n_iters;

        let diff_scalar = max_abs_diff(&interp_result, &compiled_result);
        let diff_crystal = max_abs_diff(&interp_result, &crystal_result);
        println!("  Max abs diff (scalar):  {:.2e}", diff_scalar);
        println!("  Max abs diff (crystal): {:.2e}", diff_crystal);
        println!("  Interpreter avg:     {:?} ({} iters)", interp_avg, n_iters);
        println!("  Scalar compiled avg: {:?} ({} iters)", scalar_avg, n_iters);
        println!("  Crystal compiled avg:{:?} ({} iters)", crystal_avg, n_iters);
        if scalar_avg.as_nanos() > 0 {
            println!("  Scalar speedup:  {:.2}x vs interp", interp_avg.as_nanos() as f64 / scalar_avg.as_nanos() as f64);
        }
        if crystal_avg.as_nanos() > 0 {
            println!("  Crystal speedup: {:.2}x vs interp, {:.2}x vs scalar",
                interp_avg.as_nanos() as f64 / crystal_avg.as_nanos() as f64,
                scalar_avg.as_nanos() as f64 / crystal_avg.as_nanos() as f64);
        }
        assert!(diff_scalar < 1e-5, "Scalar results diverged! diff={}", diff_scalar);
        assert!(diff_crystal < 1e-5, "Crystal results diverged! diff={}", diff_crystal);
        println!("  PASS\n");
    }

    println!("=== All tests passed ===");
}
