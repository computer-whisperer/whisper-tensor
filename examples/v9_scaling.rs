//! Focused v9 scaling benchmark: v9 vs interpreter (BLAS) at various sizes.
//! Only tests v9 compilation + execution — no v2/v7/v8 overhead.
//!
//! Usage: cargo run --release --features cranelift --example v9_scaling
//!        WT_SIZES="256,512,1024,2048" cargo run --release ...

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use whisper_tensor::compiler::attempts::v9_fused_expr::pipeline as v9_pipeline;
use whisper_tensor::graph::GlobalId;
use whisper_tensor::milli_graph::ops::MatMul;
use whisper_tensor::milli_graph::MilliOpGraph;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::tensor_rank::DynRank;

fn make_random_f32(n: usize, seed: u64) -> Vec<f32> {
    use rand::RngCore;
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
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn run_interpreter(
    graph: &MilliOpGraph,
    inputs: &HashMap<GlobalId, NumericTensor<DynRank>>,
    output_ext: GlobalId,
) -> Vec<f32> {
    let results = whisper_tensor::compiler::interpret_milli_graph(graph, inputs).unwrap();
    results[&output_ext].flatten().unwrap().try_into().unwrap()
}

fn run_v9(
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

fn bench_matmul(m: usize, k: usize, n: usize) {
    println!("\n--- Matmul [{m}x{k}] x [{k}x{n}] = [{m}x{n}] ---");
    println!(
        "    data: A={:.1}MB B={:.1}MB C={:.1}MB",
        (m * k * 4) as f64 / 1e6,
        (k * n * 4) as f64 / 1e6,
        (m * n * 4) as f64 / 1e6,
    );

    let mut rng = wyrand::WyRand::new(42);
    let ext_a = GlobalId::new(&mut rng);
    let ext_b = GlobalId::new(&mut rng);
    let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
    let a = input_map[&ext_a];
    let b = input_map[&ext_b];
    let c = MatMul::push_new(&mut graph, a, b, &mut rng);
    let ext_out = GlobalId::new(&mut rng);
    graph.set_output_map([(c, ext_out)]);

    let mut shapes = HashMap::new();
    shapes.insert(a, vec![m, k]);
    shapes.insert(b, vec![k, n]);
    shapes.insert(c, vec![m, n]);

    let a_data = make_random_f32(m * k, 1700);
    let b_data = make_random_f32(k * n, 1701);

    // Interpreter (BLAS) reference
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

    let mut compiled_inputs: HashMap<GlobalId, Vec<f32>> = HashMap::new();
    compiled_inputs.insert(a, a_data);
    compiled_inputs.insert(b, b_data);
    let out_size = m * n;

    // V9 compile
    let t = Instant::now();
    let final_outputs: HashSet<GlobalId> = [c].into_iter().collect();
    let v9_compiled =
        v9_pipeline::compile_graph(&graph, &shapes, &final_outputs, 0).expect("v9 compile");
    let compile_time = t.elapsed();
    let kernel = &v9_compiled.kernels[0];
    println!(
        "    compile: {:?}, {} kernels, parallel_extent={} step={}",
        compile_time,
        v9_compiled.kernels.len(),
        kernel.parallel_extent,
        kernel.parallel_step,
    );

    // V9 serial correctness
    let v9_result = run_v9(&v9_compiled, &compiled_inputs, c, out_size);
    let v9_diff = max_abs_diff(&interp_result, &v9_result);
    println!("    v9 diff vs BLAS: {:.2e}", v9_diff);
    assert!(v9_diff < 0.01, "v9 diverged: {v9_diff}");

    // V9 parallel correctness
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let v9_par_result =
        run_v9_parallel(&v9_compiled, &compiled_inputs, c, out_size, num_threads);
    let v9_par_diff = max_abs_diff(&v9_result, &v9_par_result);
    println!("    v9 parallel diff: {:.2e} ({num_threads}t)", v9_par_diff);
    assert!(
        v9_par_diff < 1e-5,
        "v9 parallel diverged: {}",
        v9_par_diff
    );

    // Benchmark iterations — WT_ITERS overrides default
    let total_flops = 2.0 * m as f64 * n as f64 * k as f64;
    let iters: u32 = std::env::var("WT_ITERS")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or_else(|| {
            if m * n * k > 512 * 512 * 512 { 3 } else if m * n * k > 128 * 128 * 128 { 10 } else { 50 }
        });

    // V9 parallel only — warmup + measure
    // Warmup run
    let _ = run_v9_parallel(&v9_compiled, &compiled_inputs, c, out_size, num_threads);

    println!("    running {iters} iterations, parallel only ({num_threads}t)...");
    let t = Instant::now();
    for i in 0..iters {
        let _ = run_v9_parallel(&v9_compiled, &compiled_inputs, c, out_size, num_threads);
        if i == 0 {
            println!("    first iter: {:?}", t.elapsed());
        }
    }
    let total_time = t.elapsed();
    let v9_mt_avg = total_time / iters;
    let v9_mt_gflops = total_flops / v9_mt_avg.as_secs_f64() / 1e9;

    println!(
        "    v9 parallel({}t): {:>10?} avg  {:.1} GFLOPS  ({:?} total, {} iters)",
        num_threads, v9_mt_avg, v9_mt_gflops, total_time, iters,
    );
}

fn cpu_warmup() {
    use std::hint::black_box;
    let t = Instant::now();
    let mut x = 1.0f64;
    while t.elapsed().as_millis() < 500 {
        for _ in 0..10000 {
            x = black_box(x * 1.0000001 + 0.0000001);
        }
    }
    let _ = black_box(x);
    println!("CPU warmup done ({:?})", t.elapsed());
}

fn main() {
    println!("=== V9 Scaling Benchmark ===");
    cpu_warmup();

    let sizes: Vec<usize> = std::env::var("WT_SIZES")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|v| v.trim().parse().ok())
                .collect()
        })
        .unwrap_or_else(|| vec![64, 128, 256, 512, 1024]);

    for &d in &sizes {
        bench_matmul(d, d, d);
    }

    println!("\n=== Done ===");
}
