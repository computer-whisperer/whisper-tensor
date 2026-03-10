//! Tile-task DAG executor for v9 fused expression compiler.
//!
//! Decomposes compiled kernels into tiles and wires up dependencies
//! at tile granularity, enabling:
//! - **Intra-op parallelism**: one kernel's output range split across threads
//! - **Inter-op parallelism**: independent kernels' tiles run concurrently
//! - **Pipelining**: as producer tile i completes, consumer tile i becomes
//!   ready immediately (no barrier between kernels)

#[cfg(feature = "cranelift")]
pub mod dag {
    use crate::compiler::attempts::v9_fused_expr::codegen::jit::CompiledGraph;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Condvar, Mutex};

    /// Minimum parallel axis units per tile. For blocked kernels (step=NR),
    /// this is measured in NR-blocks; for unblocked, in elements.
    const MIN_TILE_STEPS: usize = 2;

    // ---- Task graph types ----

    struct TileTask {
        kernel_idx: usize,
        start: usize,
        end: usize,
    }

    struct TaskNode {
        task: TileTask,
        remaining_deps: AtomicUsize,
        dependents: Vec<usize>,
    }

    struct TaskGraph {
        nodes: Vec<TaskNode>,
        initial_ready: Vec<usize>,
    }

    // ---- Task graph construction ----

    fn build_task_graph(graph: &CompiledGraph, num_threads: usize) -> TaskGraph {
        let n_kernels = graph.kernels.len();

        // Map output tensor → kernel index.
        let mut output_to_kernel = std::collections::HashMap::new();
        for (ki, kernel) in graph.kernels.iter().enumerate() {
            output_to_kernel.insert(kernel.output_tensor, ki);
        }

        // Determine tiles per kernel.
        struct KernelTiles {
            tile_ranges: Vec<(usize, usize)>,
            first_node: usize,
        }
        let mut kernel_tiles = Vec::with_capacity(n_kernels);
        let mut total_nodes = 0usize;
        for kernel in &graph.kernels {
            let extent = kernel.parallel_extent;
            let step = kernel.parallel_step;
            let n_tiles = compute_tile_count(extent, step, num_threads);
            let tile_size = align_up(extent.div_ceil(n_tiles), step);
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
        for kt in &kernel_tiles {
            for &(start, end) in &kt.tile_ranges {
                nodes.push(TaskNode {
                    task: TileTask {
                        kernel_idx: 0, // filled below
                        start,
                        end,
                    },
                    remaining_deps: AtomicUsize::new(0),
                    dependents: Vec::new(),
                });
            }
        }
        // Fix kernel_idx
        for (ki, kt) in kernel_tiles.iter().enumerate() {
            for ti in 0..kt.tile_ranges.len() {
                nodes[kt.first_node + ti].task.kernel_idx = ki;
            }
        }

        // Wire dependencies.
        for (consumer_ki, consumer_kernel) in graph.kernels.iter().enumerate() {
            let consumer_kt = &kernel_tiles[consumer_ki];
            for input_tensor in &consumer_kernel.input_tensors {
                let producer_ki = match output_to_kernel.get(input_tensor) {
                    Some(&ki) => ki,
                    None => continue,
                };
                let producer_kernel = &graph.kernels[producer_ki];
                let producer_kt = &kernel_tiles[producer_ki];

                // 1:1 tile mapping when extents and tile counts match
                let one_to_one = producer_kernel.parallel_extent
                    == consumer_kernel.parallel_extent
                    && producer_kt.tile_ranges.len() == consumer_kt.tile_ranges.len()
                    && producer_kt
                        .tile_ranges
                        .iter()
                        .zip(consumer_kt.tile_ranges.iter())
                        .all(|(p, c)| p.0 == c.0 && p.1 == c.1);

                if one_to_one {
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

    fn align_up(val: usize, step: usize) -> usize {
        if step <= 1 {
            val
        } else {
            ((val + step - 1) / step) * step
        }
    }

    fn compute_tile_count(extent: usize, step: usize, num_threads: usize) -> usize {
        if extent == 0 {
            return 1;
        }
        let s = step.max(1);
        let n_steps = extent / s;
        // Each tile needs at least MIN_TILE_STEPS steps.
        let max_tiles = n_steps / MIN_TILE_STEPS;
        // Target 2× oversubscription for load balancing.
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
    /// # Safety
    /// `buffers` must contain valid pointers for all tensors in the layout.
    pub unsafe fn execute_parallel(
        graph: &CompiledGraph,
        buffers: &[*mut f32],
        num_threads: usize,
    ) {
        if num_threads <= 1 || graph.kernels.is_empty() {
            let ptr = buffers.as_ptr();
            for kernel in &graph.kernels {
                unsafe { kernel.execute(ptr) };
            }
            return;
        }

        let task_graph = build_task_graph(graph, num_threads);
        let total_tasks = task_graph.nodes.len();

        if total_tasks == 0 {
            return;
        }

        // Cap thread count: never spawn more threads than tasks, and fall
        // back to serial if there aren't enough tasks to justify overhead.
        let effective_threads = total_tasks.min(num_threads);
        if effective_threads <= 1 {
            let ptr = buffers.as_ptr();
            for kernel in &graph.kernels {
                unsafe { kernel.execute(ptr) };
            }
            return;
        }

        let wq = WorkQueue {
            queue: Mutex::new(VecDeque::from(task_graph.initial_ready.clone())),
            condvar: Condvar::new(),
            tasks_remaining: AtomicUsize::new(total_tasks),
        };

        let buf_addr = buffers.as_ptr() as usize;

        std::thread::scope(|scope| {
            for _ in 0..effective_threads {
                let wq_ref = &wq;
                let nodes = &task_graph.nodes;
                let graph_ref = graph;
                let ba = buf_addr;
                scope.spawn(move || {
                    let kernels = &graph_ref.kernels;
                    let buf_ptr = ba as *const *mut f32;
                    loop {
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
                            None => return,
                        };

                        let node = &nodes[task_idx];
                        let kernel = &kernels[node.task.kernel_idx];
                        unsafe {
                            kernel.execute_range(buf_ptr, node.task.start, node.task.end);
                        }

                        let mut newly_ready = Vec::new();
                        for &dep_idx in &node.dependents {
                            let prev = nodes[dep_idx]
                                .remaining_deps
                                .fetch_sub(1, Ordering::AcqRel);
                            if prev == 1 {
                                newly_ready.push(dep_idx);
                            }
                        }

                        let prev_remaining =
                            wq_ref.tasks_remaining.fetch_sub(1, Ordering::AcqRel);

                        if !newly_ready.is_empty() {
                            let mut q = wq_ref.queue.lock().unwrap();
                            for idx in newly_ready {
                                q.push_back(idx);
                            }
                            wq_ref.condvar.notify_all();
                        } else if prev_remaining == 1 {
                            wq_ref.condvar.notify_all();
                        }
                    }
                });
            }
        });
    }
}
