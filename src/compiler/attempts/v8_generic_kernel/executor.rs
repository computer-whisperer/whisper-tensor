#![allow(clippy::all, dead_code, unreachable_patterns)]
//! Tile-task DAG executor for v8 generic kernel.
//!
//! Decomposes compiled kernels into row-tiles and wires up dependencies
//! at tile granularity, enabling:
//! - **Intra-op parallelism**: one large kernel's rows run across many threads
//! - **Inter-op parallelism**: independent kernels' tiles run concurrently
//! - **Pipelining**: as producer kernel tile i completes, consumer kernel
//!   tile i becomes ready immediately (no barrier between kernels)
//!
//! The executor is a simple work-stealing loop: threads dequeue ready tiles,
//! execute them, and decrement dependents' counters.  When a counter hits
//! zero, that tile is enqueued.

use super::codegen::NativeCompiledGraph;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};

/// Minimum number of output rows per tile.  Tiles smaller than this create
/// more scheduling overhead than they save from parallelism.
const MIN_TILE_ROWS: usize = 16;

// ---- Task graph types ----

struct TileTask {
    kernel_idx: usize,
    i_start: usize,
    i_end: usize,
}

struct TaskNode {
    task: TileTask,
    /// Number of predecessor tiles that must complete before this tile is ready.
    remaining_deps: AtomicUsize,
    /// Indices of task nodes that depend on this one completing.
    dependents: Vec<usize>,
}

struct TaskGraph {
    nodes: Vec<TaskNode>,
    /// Task indices that are initially ready (no dependencies).
    initial_ready: Vec<usize>,
}

// ---- Task graph construction ----

/// Build a tile-task DAG from the compiled graph.
///
/// Each kernel is decomposed into tiles of roughly `target_tile_rows` rows
/// on the parallel axis.  Dependencies are wired at tile granularity:
///
/// - If a consumer kernel reads a producer kernel's output and both have
///   the same parallel_extent, tiles are matched 1:1 (consumer tile [s,e)
///   depends only on producer tile [s,e)).
/// - Otherwise, every consumer tile depends on all producer tiles
///   (conservative but correct).
fn build_task_graph(graph: &NativeCompiledGraph, num_threads: usize) -> TaskGraph {
    let n_kernels = graph.kernels.len();

    // Map output tensor → kernel index.
    let mut output_to_kernel = std::collections::HashMap::new();
    for (ki, kernel) in graph.kernels.iter().enumerate() {
        output_to_kernel.insert(kernel.output_tensor, ki);
    }

    // Determine tiles per kernel.
    struct KernelTiles {
        tile_ranges: Vec<(usize, usize)>, // (i_start, i_end) per tile
        first_node: usize,                // index of first TaskNode for this kernel
    }
    let mut kernel_tiles = Vec::with_capacity(n_kernels);
    let mut total_nodes = 0usize;
    for kernel in &graph.kernels {
        let extent = kernel.parallel_extent;
        let n_tiles = compute_tile_count(extent, num_threads);
        let tile_size = extent.div_ceil(n_tiles);
        let mut ranges = Vec::with_capacity(n_tiles);
        let mut start = 0;
        while start < extent {
            let end = (start + tile_size).min(extent);
            ranges.push((start, end));
            start = end;
        }
        kernel_tiles.push(KernelTiles {
            tile_ranges: ranges,
            first_node: total_nodes,
        });
        total_nodes += kernel_tiles.last().unwrap().tile_ranges.len();
    }

    // Build task nodes.
    let mut nodes: Vec<TaskNode> = Vec::with_capacity(total_nodes);
    for (ki, kt) in kernel_tiles.iter().enumerate() {
        for &(i_start, i_end) in &kt.tile_ranges {
            nodes.push(TaskNode {
                task: TileTask {
                    kernel_idx: ki,
                    i_start,
                    i_end,
                },
                remaining_deps: AtomicUsize::new(0),
                dependents: Vec::new(),
            });
        }
    }

    // Wire dependencies.
    for (consumer_ki, consumer_kernel) in graph.kernels.iter().enumerate() {
        let consumer_kt = &kernel_tiles[consumer_ki];
        for input_tensor in &consumer_kernel.input_tensors {
            let producer_ki = match output_to_kernel.get(input_tensor) {
                Some(&ki) => ki,
                None => continue, // external input, no dependency
            };
            let producer_kernel = &graph.kernels[producer_ki];
            let producer_kt = &kernel_tiles[producer_ki];

            let one_to_one = producer_kernel.parallel_extent == consumer_kernel.parallel_extent
                && producer_kt.tile_ranges.len() == consumer_kt.tile_ranges.len()
                && producer_kt
                    .tile_ranges
                    .iter()
                    .zip(consumer_kt.tile_ranges.iter())
                    .all(|(p, c)| p.0 == c.0 && p.1 == c.1);

            if one_to_one {
                // 1:1 tile mapping — consumer tile i depends on producer tile i.
                for ti in 0..consumer_kt.tile_ranges.len() {
                    let producer_node = producer_kt.first_node + ti;
                    let consumer_node = consumer_kt.first_node + ti;
                    nodes[consumer_node]
                        .remaining_deps
                        .fetch_add(1, Ordering::Relaxed);
                    nodes[producer_node].dependents.push(consumer_node);
                }
            } else {
                // All-to-all: every consumer tile depends on every producer tile.
                for cti in 0..consumer_kt.tile_ranges.len() {
                    let consumer_node = consumer_kt.first_node + cti;
                    nodes[consumer_node]
                        .remaining_deps
                        .fetch_add(producer_kt.tile_ranges.len(), Ordering::Relaxed);
                }
                for pti in 0..producer_kt.tile_ranges.len() {
                    let producer_node = producer_kt.first_node + pti;
                    for cti in 0..consumer_kt.tile_ranges.len() {
                        let consumer_node = consumer_kt.first_node + cti;
                        nodes[producer_node].dependents.push(consumer_node);
                    }
                }
            }
        }
    }

    // Find initially ready tasks.
    let initial_ready: Vec<usize> = nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| n.remaining_deps.load(Ordering::Relaxed) == 0)
        .map(|(i, _)| i)
        .collect();

    TaskGraph {
        nodes,
        initial_ready,
    }
}

fn compute_tile_count(extent: usize, num_threads: usize) -> usize {
    if extent == 0 {
        return 1;
    }
    // Aim for ~2× oversubscription so that load balancing can smooth out
    // uneven tile durations. But never make tiles smaller than MIN_TILE_ROWS.
    let max_tiles = extent.div_ceil(MIN_TILE_ROWS);
    let target = num_threads * 2;
    target.min(max_tiles).max(1)
}

// ---- Executor ----

struct WorkQueue {
    queue: Mutex<VecDeque<usize>>,
    condvar: Condvar,
    tasks_remaining: AtomicUsize,
}

/// Execute the compiled graph using `num_threads` worker threads.
///
/// Tiles are dispatched through a shared work queue.  When a tile completes,
/// its dependents' counters are decremented; newly-ready tiles are enqueued.
///
/// # Safety
/// `buffers` must contain valid pointers for all tensors in the layout.
pub unsafe fn execute_parallel(
    graph: &NativeCompiledGraph,
    buffers: &[*mut f32],
    num_threads: usize,
) {
    assert!(
        buffers.len() >= graph.layout.num_buffers,
        "Expected at least {} buffers, got {}",
        graph.layout.num_buffers,
        buffers.len()
    );

    if num_threads <= 1 || graph.kernels.len() == 0 {
        // Fall back to serial execution.
        let ptr = buffers.as_ptr();
        for kernel in &graph.kernels {
            unsafe { kernel.execute_range(ptr, 0, kernel.parallel_extent) };
        }
        return;
    }

    let task_graph = build_task_graph(graph, num_threads);
    let total_tasks = task_graph.nodes.len();

    if total_tasks == 0 {
        return;
    }

    let wq = WorkQueue {
        queue: Mutex::new(VecDeque::from(task_graph.initial_ready.clone())),
        condvar: Condvar::new(),
        tasks_remaining: AtomicUsize::new(total_tasks),
    };

    let buf_addr = buffers.as_ptr() as usize;

    std::thread::scope(|scope| {
        for _ in 0..num_threads {
            let wq_ref = &wq;
            let nodes = &task_graph.nodes;
            let graph_ref = graph;
            let ba = buf_addr;
            scope.spawn(move || {
                let kernels = &graph_ref.kernels;
                let buf_ptr = ba as *const *mut f32;
                loop {
                    // Try to dequeue a ready task.
                    let task_idx = {
                        let mut q = wq_ref.queue.lock().unwrap();
                        loop {
                            if let Some(idx) = q.pop_front() {
                                break Some(idx);
                            }
                            if wq_ref.tasks_remaining.load(Ordering::Acquire) == 0 {
                                break None;
                            }
                            q = wq_ref.condvar.wait(q).unwrap();
                        }
                    };

                    let task_idx = match task_idx {
                        Some(idx) => idx,
                        None => return, // all done
                    };

                    // Execute the tile.
                    let node = &nodes[task_idx];
                    let kernel = &kernels[node.task.kernel_idx];
                    unsafe {
                        kernel.execute_range(buf_ptr, node.task.i_start, node.task.i_end);
                    }

                    // Decrement dependents and enqueue newly ready.
                    let mut newly_ready = Vec::new();
                    for &dep_idx in &node.dependents {
                        let prev = nodes[dep_idx].remaining_deps.fetch_sub(1, Ordering::AcqRel);
                        if prev == 1 {
                            newly_ready.push(dep_idx);
                        }
                    }

                    // Mark this task as done.
                    let prev_remaining = wq_ref.tasks_remaining.fetch_sub(1, Ordering::AcqRel);

                    if !newly_ready.is_empty() {
                        let mut q = wq_ref.queue.lock().unwrap();
                        for idx in newly_ready {
                            q.push_back(idx);
                        }
                        wq_ref.condvar.notify_all();
                    } else if prev_remaining == 1 {
                        // Last task — wake everyone to exit.
                        wq_ref.condvar.notify_all();
                    }
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::attempts::v8_generic_kernel::codegen;
    use crate::graph::GlobalId;
    use crate::milli_graph::MilliOpGraph;
    use crate::milli_graph::ops::{MatMul, SimpleBinary, SimpleUnaryOp};
    use crate::dtype::DType;
    use std::collections::HashMap;

    fn make_random_f32(n: usize, seed: u64) -> Vec<f32> {
        // Simple xorshift for test determinism — no external trait needed.
        let mut state = seed | 1;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn alloc_buffers(
        compiled: &NativeCompiledGraph,
        inputs: &HashMap<GlobalId, Vec<f32>>,
    ) -> Vec<Vec<f32>> {
        let layout = &compiled.layout;
        let mut bufs: Vec<Vec<f32>> = (0..layout.num_buffers).map(|_| Vec::new()).collect();
        // Allocate all tensors to their required sizes.
        for (id, &size) in &layout.tensor_sizes {
            let idx = layout.tensor_index[id];
            bufs[idx] = vec![0.0f32; size];
        }
        // Copy input data.
        for (id, data) in inputs {
            if let Some(&idx) = layout.tensor_index.get(id) {
                bufs[idx][..data.len()].copy_from_slice(data);
            }
        }
        bufs
    }

    fn run_compiled(
        compiled: &NativeCompiledGraph,
        inputs: &HashMap<GlobalId, Vec<f32>>,
        output_id: GlobalId,
        _output_size: usize,
    ) -> Vec<f32> {
        let mut bufs = alloc_buffers(compiled, inputs);
        let mut ptrs: Vec<*mut f32> = bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();
        unsafe { compiled.execute(&mut ptrs) };
        let out_idx = compiled.layout.tensor_index[&output_id];
        bufs[out_idx].clone()
    }

    fn run_compiled_parallel(
        compiled: &NativeCompiledGraph,
        inputs: &HashMap<GlobalId, Vec<f32>>,
        output_id: GlobalId,
        _output_size: usize,
        num_threads: usize,
    ) -> Vec<f32> {
        let mut bufs = alloc_buffers(compiled, inputs);
        let ptrs: Vec<*mut f32> = bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();
        unsafe { execute_parallel(compiled, &ptrs, num_threads) };
        let out_idx = compiled.layout.tensor_index[&output_id];
        bufs[out_idx].clone()
    }

    #[test]
    fn test_parallel_matmul_matches_serial() {
        let mut rng = wyrand::WyRand::new(5001);
        let (m, k, n) = (128, 96, 64);
        let ext_a = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let a = input_map[&ext_a];
        let b = input_map[&ext_b];
        let c = MatMul::push_new_default_precision(&mut graph, a, b, DType::F32, &mut rng);

        let mut shapes = HashMap::new();
        shapes.insert(a, vec![m, k]);
        shapes.insert(b, vec![k, n]);
        shapes.insert(c, vec![m, n]);

        let a_data = make_random_f32(m * k, 5002);
        let b_data = make_random_f32(k * n, 5003);
        let mut inputs = HashMap::new();
        inputs.insert(a, a_data);
        inputs.insert(b, b_data);

        let (compiled, _) = codegen::compile_graph(&graph, &shapes).expect("compile");

        let serial = run_compiled(&compiled, &inputs, c, m * n);

        for threads in [2, 4, 8] {
            let parallel = run_compiled_parallel(&compiled, &inputs, c, m * n, threads);
            let diff = max_diff(&serial, &parallel);
            assert!(
                diff < 1e-5,
                "parallel ({threads} threads) diverged from serial: max diff {diff}"
            );
        }
    }

    #[test]
    fn test_parallel_fused_pipeline() {
        // matmul → neg → exp → add(ones) → reciprocal: multi-kernel pipeline.
        let mut rng = wyrand::WyRand::new(6001);
        let (m, k, n) = (64, 48, 32);
        let ext_a = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let ext_bias = GlobalId::new(&mut rng);
        let ext_ones = GlobalId::new(&mut rng);
        let (mut graph, input_map) =
            MilliOpGraph::new([ext_a, ext_b, ext_bias, ext_ones], &mut rng);
        let a = input_map[&ext_a];
        let b = input_map[&ext_b];
        let bias = input_map[&ext_bias];
        let ones = input_map[&ext_ones];
        let mm = MatMul::push_new_default_precision(&mut graph, a, b, DType::F32, &mut rng);
        let bias_add = SimpleBinary::add(&mut graph, mm, bias, &mut rng);
        let neg = SimpleUnaryOp::neg(&mut graph, bias_add, &mut rng);
        let exp = SimpleUnaryOp::exp(&mut graph, neg, &mut rng);
        let denom = SimpleBinary::add(&mut graph, exp, ones, &mut rng);
        let out = SimpleUnaryOp::reciprocal(&mut graph, denom, &mut rng);

        let mut shapes = HashMap::new();
        shapes.insert(a, vec![m, k]);
        shapes.insert(b, vec![k, n]);
        shapes.insert(bias, vec![1, n]);
        shapes.insert(ones, vec![1, n]);
        shapes.insert(mm, vec![m, n]);
        shapes.insert(bias_add, vec![m, n]);
        shapes.insert(neg, vec![m, n]);
        shapes.insert(exp, vec![m, n]);
        shapes.insert(denom, vec![m, n]);
        shapes.insert(out, vec![m, n]);

        let a_data = make_random_f32(m * k, 6002);
        let b_data = make_random_f32(k * n, 6003);
        let bias_data = make_random_f32(n, 6004);
        let ones_data = vec![1.0f32; n];
        let mut inputs = HashMap::new();
        inputs.insert(a, a_data);
        inputs.insert(b, b_data);
        inputs.insert(bias, bias_data);
        inputs.insert(ones, ones_data);

        let (compiled, _) = codegen::compile_graph(&graph, &shapes).expect("compile");

        let serial = run_compiled(&compiled, &inputs, out, m * n);

        for threads in [2, 4, 8] {
            let parallel = run_compiled_parallel(&compiled, &inputs, out, m * n, threads);
            let diff = max_diff(&serial, &parallel);
            assert!(
                diff < 1e-4,
                "fused pipeline ({threads} threads) diverged from serial: max diff {diff}"
            );
        }
    }
}
