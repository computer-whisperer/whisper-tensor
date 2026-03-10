//! v9 fused expression compiler.
//!
//! First attempt consuming the v2 frontend's boundaryless expression trees.
//! Key differences from v8:
//! - No per-milli-op kernel boundaries — intermediate tensors can be fused away
//! - Pattern recovery operates on fully-inlined expression trees
//! - Single-threaded, no tiling — correctness first

pub mod codegen;
pub mod executor;
pub mod inline;
pub mod pattern;

#[cfg(feature = "cranelift")]
pub mod pipeline {
    use crate::backends::eval_backend::EvalBackend;
    use crate::backends::ndarray_backend::NDArrayNumericTensor;
    use crate::compiler::attempts::v1_scalar_crystal::codegen::TensorLayout;
    pub use crate::compiler::attempts::v9_fused_expr::codegen::jit::{
        CompiledGraph, V9Error,
    };
    pub use crate::compiler::attempts::v9_fused_expr::executor::dag::execute_parallel;
    use crate::compiler::attempts::v9_fused_expr::executor::dag::execute_kernel_batch;
    use crate::compiler::attempts::v9_fused_expr::codegen::jit::compile_patterns;
    use crate::compiler::attempts::v9_fused_expr::inline::inline_intermediates;
    use crate::compiler::attempts::v9_fused_expr::pattern::recover_patterns;
    use crate::compiler::common::v2_frontend::ExprExpander;
    use crate::graph::{GlobalId, Graph, Node};
    use crate::milli_graph::MilliOpGraph;
    use crate::milli_graph::ops::{AnyMilliOp, MilliOp};
    use crate::numeric_tensor::NumericTensor;
    use crate::tensor_rank::DynRank;
    use std::collections::{HashMap, HashSet};

    // ---- Interpreted op support ----

    /// An op that v9 couldn't compile, to be run via the interpreter at execution time.
    pub struct InterpretedOp {
        pub op: AnyMilliOp,
        pub input_ids: Vec<GlobalId>,
        pub output_ids: Vec<GlobalId>,
    }

    /// A step in the hybrid execution plan.
    pub enum ExecStep {
        /// Run a compiled kernel.
        Kernel(usize),
        /// Run an interpreted op.
        Eval(usize),
    }

    /// Hybrid execution plan: compiled kernels interleaved with interpreted ops.
    pub struct ExecutionPlan {
        pub compiled: CompiledGraph,
        pub interpreted_ops: Vec<InterpretedOp>,
        pub steps: Vec<ExecStep>,
    }

    impl ExecutionPlan {
        /// Read a tensor from a raw buffer slot.
        fn read_tensor(
            layout: &TensorLayout,
            buffers: &[*mut f32],
            tensor_id: &GlobalId,
        ) -> NumericTensor<DynRank> {
            let idx = layout.tensor_index[tensor_id];
            let size = layout.tensor_sizes[tensor_id];
            let dtype = layout.dtype_of(tensor_id);
            let byte_size = dtype.size().unwrap_or(4);
            let buf_ptr = buffers[idx] as *const u8;
            let raw = unsafe { std::slice::from_raw_parts(buf_ptr, size * byte_size) };
            let shape: Vec<u64> = layout.tensor_shapes.as_ref()
                .and_then(|s| s.get(tensor_id))
                .expect("interpreted op tensor missing shape")
                .iter()
                .map(|&d| d as u64)
                .collect();
            NDArrayNumericTensor::<DynRank>::from_raw_data(raw, dtype, shape)
                .expect("failed to deserialize tensor from buffer")
                .into()
        }

        /// Write a tensor into a raw buffer slot.
        fn write_tensor(
            layout: &TensorLayout,
            buffers: &[*mut f32],
            tensor_id: &GlobalId,
            tensor: &NumericTensor<DynRank>,
        ) {
            let idx = layout.tensor_index[tensor_id];
            let size = layout.tensor_sizes[tensor_id];
            let layout_dtype = layout.dtype_of(tensor_id);
            let byte_size = layout_dtype.size().unwrap_or(4);
            let buf_ptr = buffers[idx] as *mut u8;
            let buf = unsafe { std::slice::from_raw_parts_mut(buf_ptr, size * byte_size) };

            // Cast to the layout's dtype if the eval produced a different type
            let nd = tensor.to_ndarray().expect("can convert to ndarray");
            let actual_dtype = nd.dtype();
            let nd = if actual_dtype != layout_dtype {
                nd.cast(layout_dtype).unwrap_or(nd)
            } else {
                nd
            };

            let bytes = nd.to_contiguous_bytes();
            let copy_len = bytes.len().min(buf.len());
            if bytes.len() != buf.len() {
                eprintln!(
                    "WARN: write_tensor size mismatch for {:?}: eval produced {} bytes ({:?}), buffer is {} bytes ({:?})",
                    tensor_id, bytes.len(), actual_dtype, buf.len(), layout_dtype
                );
            }
            buf[..copy_len].copy_from_slice(&bytes[..copy_len]);
        }

        /// Run an interpreted op: read inputs from buffers, eval, write outputs back.
        pub fn run_interpreted_op(
            &self,
            op_idx: usize,
            buffers: &[*mut f32],
        ) {
            let iop = &self.interpreted_ops[op_idx];
            let layout = &self.compiled.layout;

            // Build input map from buffers
            let mut inputs: HashMap<GlobalId, NumericTensor<DynRank>> = HashMap::new();
            for &id in &iop.input_ids {
                if layout.tensor_index.contains_key(&id) {
                    inputs.insert(id, Self::read_tensor(layout, buffers, &id));
                }
            }

            // Eval
            let mut backend = EvalBackend::NDArray;
            let results: Vec<_> = iop.op.eval(&inputs, &mut backend)
                .expect("interpreted op eval failed")
                .collect();

            // Write outputs back
            for (tensor_id, value) in results {
                if layout.tensor_index.contains_key(&tensor_id) {
                    Self::write_tensor(layout, buffers, &tensor_id, &value);
                }
            }
        }

        /// Execute the plan serially.
        ///
        /// # Safety
        /// `buffers` must contain valid pointers for all tensors in the layout.
        pub unsafe fn execute(&self, buffers: &mut [*mut f32]) {
            let buf_ptr = buffers.as_ptr();
            for step in &self.steps {
                match step {
                    ExecStep::Kernel(ki) => {
                        unsafe { self.compiled.kernels[*ki].execute(buf_ptr) };
                    }
                    ExecStep::Eval(ei) => {
                        self.run_interpreted_op(*ei, buffers);
                    }
                }
            }
        }

        /// Execute the plan with parallel kernel execution.
        /// Interpreted ops run serially at their topological position;
        /// compiled kernels between interpreted ops run in parallel.
        ///
        /// # Safety
        /// `buffers` must contain valid pointers for all tensors in the layout.
        pub unsafe fn execute_parallel(&self, buffers: &[*mut f32], num_threads: usize) {
            // Group steps into runs of consecutive kernels separated by eval steps.
            // Each kernel run can be parallelized; eval steps are barriers.
            let mut i = 0;
            while i < self.steps.len() {
                match &self.steps[i] {
                    ExecStep::Eval(ei) => {
                        self.run_interpreted_op(*ei, buffers);
                        i += 1;
                    }
                    ExecStep::Kernel(_) => {
                        // Collect consecutive kernel steps
                        let start = i;
                        while i < self.steps.len() && matches!(&self.steps[i], ExecStep::Kernel(_)) {
                            i += 1;
                        }
                        // Execute this batch of kernels in parallel
                        let kernel_indices: Vec<usize> = self.steps[start..i]
                            .iter()
                            .map(|s| match s { ExecStep::Kernel(ki) => *ki, _ => unreachable!() })
                            .collect();
                        unsafe {
                            execute_kernel_batch(
                                &self.compiled,
                                &kernel_indices,
                                buffers,
                                num_threads,
                            );
                        }
                    }
                }
            }
        }
    }

    // ---- Plan construction ----

    /// Build a hybrid execution plan from a compiled graph and the original milli graph.
    ///
    /// Identifies ops that v9 couldn't compile (their output tensors have no producing
    /// kernel) and inserts them as interpreted steps at the correct topological position.
    pub fn build_execution_plan(
        graph: &MilliOpGraph,
        compiled: CompiledGraph,
    ) -> ExecutionPlan {
        // Which tensors are produced by compiled kernels?
        let compiled_outputs: HashSet<GlobalId> = compiled.kernels
            .iter()
            .map(|k| k.output_tensor)
            .collect();

        // Which tensors are external inputs? (values in input_map)
        let external_inputs: HashSet<GlobalId> = graph.input_map.values().cloned().collect();

        // Walk ops in topological order. For each op:
        // - If its output is in compiled_outputs → find and emit the kernel
        // - If its output is NOT compiled and NOT an external input → interpreted step
        // - Constants are pre-filled, skip them
        let mut interpreted_ops = Vec::new();
        let mut steps = Vec::new();
        let mut emitted_kernels: HashSet<usize> = HashSet::new();

        for op_id in graph.op_ordering() {
            let op = graph.get_node_by_id(op_id).unwrap();
            let output_ids: Vec<GlobalId> = op.outputs().collect();

            // Check if any output is produced by a compiled kernel
            let mut has_compiled = false;
            for &out_id in &output_ids {
                if compiled_outputs.contains(&out_id) {
                    // Find which kernel produces this output
                    for (ki, kernel) in compiled.kernels.iter().enumerate() {
                        if kernel.output_tensor == out_id && !emitted_kernels.contains(&ki) {
                            steps.push(ExecStep::Kernel(ki));
                            emitted_kernels.insert(ki);
                            has_compiled = true;
                        }
                    }
                }
            }

            if !has_compiled {
                // Check if this op produces any tensor in the layout that isn't an input
                let produces_needed_tensor = output_ids.iter().any(|id| {
                    compiled.layout.tensor_index.contains_key(id)
                        && !external_inputs.contains(id)
                });

                if produces_needed_tensor {
                    let input_ids: Vec<GlobalId> = op.inputs().collect();
                    let eval_idx = interpreted_ops.len();
                    interpreted_ops.push(InterpretedOp {
                        op: op.clone(),
                        input_ids,
                        output_ids,
                    });
                    steps.push(ExecStep::Eval(eval_idx));
                }
            }
        }

        // Emit any remaining compiled kernels that weren't matched to an op
        // (e.g., kernels from fused ops where the "original" op was consumed)
        for ki in 0..compiled.kernels.len() {
            if !emitted_kernels.contains(&ki) {
                steps.push(ExecStep::Kernel(ki));
            }
        }

        let n_interp = interpreted_ops.len();
        let n_kern = compiled.kernels.len();
        if n_interp > 0 {
            eprintln!(
                "  ExecutionPlan: {} compiled kernels, {} interpreted ops",
                n_kern, n_interp
            );
        }

        ExecutionPlan {
            compiled,
            interpreted_ops,
            steps,
        }
    }

    // ---- Structural verification ----

    /// Verify execution plan integrity: op coverage and input provenance.
    ///
    /// Checks:
    /// 1. Every op whose output is in the layout is either compiled, interpreted,
    ///    or an external input.
    /// 2. Every tensor consumed by a step is produced by a prior step or is an
    ///    external input (no dangling reads from uninitialized buffers).
    pub fn verify_plan_integrity(
        plan: &ExecutionPlan,
        graph: &MilliOpGraph,
    ) -> Vec<String> {
        let mut issues = Vec::new();
        let layout = &plan.compiled.layout;

        let compiled_outputs: HashSet<GlobalId> = plan.compiled.kernels
            .iter()
            .map(|k| k.output_tensor)
            .collect();
        let interpreted_outputs: HashSet<GlobalId> = plan.interpreted_ops
            .iter()
            .flat_map(|iop| iop.output_ids.iter().cloned())
            .collect();
        let external_inputs: HashSet<GlobalId> = graph.input_map.values().cloned().collect();

        // Check 1: Op coverage — every tensor in the layout must be produced by something.
        for op_id in graph.op_ordering() {
            let op = graph.get_node_by_id(op_id).unwrap();
            for out_id in op.outputs() {
                if external_inputs.contains(&out_id) { continue; }
                if !layout.tensor_index.contains_key(&out_id) { continue; }
                if compiled_outputs.contains(&out_id) { continue; }
                if interpreted_outputs.contains(&out_id) { continue; }
                issues.push(format!(
                    "COVERAGE GAP: tensor {:?} is in the layout but not produced by any kernel or interpreted op",
                    out_id
                ));
            }
        }

        // Check 2: Input provenance — walk steps in order, verify all inputs are produced.
        let mut produced: HashSet<GlobalId> = external_inputs;
        for (step_idx, step) in plan.steps.iter().enumerate() {
            match step {
                ExecStep::Kernel(ki) => {
                    let kernel = &plan.compiled.kernels[*ki];
                    for input_id in &kernel.input_tensors {
                        if !produced.contains(input_id) {
                            issues.push(format!(
                                "DANGLING INPUT: step {} (kernel {}) reads {:?} not yet produced",
                                step_idx, ki, input_id
                            ));
                        }
                    }
                    produced.insert(kernel.output_tensor);
                }
                ExecStep::Eval(ei) => {
                    let iop = &plan.interpreted_ops[*ei];
                    for input_id in &iop.input_ids {
                        if !produced.contains(input_id)
                            && layout.tensor_index.contains_key(input_id)
                        {
                            issues.push(format!(
                                "DANGLING INPUT: step {} (eval {}) reads {:?} not yet produced",
                                step_idx, ei, input_id
                            ));
                        }
                    }
                    for out_id in &iop.output_ids {
                        produced.insert(*out_id);
                    }
                }
            }
        }

        issues
    }

    /// Streaming op-by-op expansion check. Verifies every op in the graph
    /// is either expandable by the compiler or covered by an interpreted op.
    ///
    /// Uses max_samples=1 and processes one op at a time to avoid realizing
    /// the full binding list (which can be millions of entries for large models).
    ///
    /// Checks:
    /// 1. Every op whose output is in the layout either produces bindings
    ///    (is expandable) or is in the interpreted ops list.
    /// 2. Constants are either expanded or pre-filled (external inputs).
    pub fn verify_expansion_coverage(
        plan: &ExecutionPlan,
        graph: &MilliOpGraph,
        shapes: &HashMap<GlobalId, Vec<usize>>,
        dtypes: &HashMap<GlobalId, crate::dtype::DType>,
    ) -> Vec<String> {
        let mut issues = Vec::new();
        let layout = &plan.compiled.layout;

        // Build a single-sample expander for cheap per-op probing.
        // skip_unsupported=true mirrors what the compilation pass does when
        // dtypes are provided (i.e., real model with BF16/etc).
        let probe = ExprExpander::new_sampled(shapes.clone(), 1)
            .with_skip_unsupported(!dtypes.is_empty());

        let interpreted_outputs: HashSet<GlobalId> = plan.interpreted_ops
            .iter()
            .flat_map(|iop| iop.output_ids.iter().cloned())
            .collect();
        let external_inputs: HashSet<GlobalId> = graph.input_map.values().cloned().collect();

        let mut scratch = Vec::new();
        let mut n_expanded = 0usize;
        let mut n_interpreted = 0usize;
        let mut n_skipped = 0usize;

        for op_id in graph.op_ordering() {
            let op = graph.get_node_by_id(op_id).unwrap();
            let output_ids: Vec<GlobalId> = op.outputs().collect();

            // Try expanding this single op.
            scratch.clear();
            let expanded = probe.expand_op(op, &mut scratch).is_ok() && !scratch.is_empty();

            for &out_id in &output_ids {
                if external_inputs.contains(&out_id) { continue; }
                if !layout.tensor_index.contains_key(&out_id) {
                    // Not in layout — either a shape-only tensor or fully inlined.
                    n_skipped += 1;
                    continue;
                }

                let is_interpreted = interpreted_outputs.contains(&out_id);

                if expanded {
                    n_expanded += 1;
                } else if is_interpreted {
                    n_interpreted += 1;
                } else {
                    issues.push(format!(
                        "EXPANSION GAP: {} op output {:?} is in the layout, \
                         not expandable, and not interpreted",
                        op.op_kind(), out_id
                    ));
                }
            }
        }

        eprintln!(
            "  expansion coverage: {} expanded, {} interpreted, {} not-in-layout",
            n_expanded, n_interpreted, n_skipped
        );

        issues
    }

    // ---- Compilation entry points ----

    /// Compile a MilliOpGraph into a v9 CompiledGraph.
    ///
    /// `shapes` must contain concrete shapes for every tensor in the graph.
    /// `final_output_ids` are the internal tensor IDs that must be materialized
    /// (everything else is eligible for fusion/inlining).
    pub fn compile_graph(
        graph: &MilliOpGraph,
        shapes: &HashMap<GlobalId, Vec<usize>>,
        final_output_ids: &HashSet<GlobalId>,
        max_samples: usize,
    ) -> Result<CompiledGraph, V9Error> {
        compile_graph_typed(graph, shapes, final_output_ids, max_samples, &HashMap::new())
    }

    /// Like `compile_graph` but with explicit dtype info for BF16/F16 support.
    pub fn compile_graph_typed(
        graph: &MilliOpGraph,
        shapes: &HashMap<GlobalId, Vec<usize>>,
        final_output_ids: &HashSet<GlobalId>,
        max_samples: usize,
        dtypes: &HashMap<GlobalId, crate::dtype::DType>,
    ) -> Result<CompiledGraph, V9Error> {
        // Phase 1: v2 frontend — produce expression trees.
        let max_output_elements: usize = shapes
            .values()
            .map(|s| s.iter().product::<usize>())
            .max()
            .unwrap_or(0);
        const SAMPLE_THRESHOLD: usize = 100_000;
        let expander = if max_samples > 0 {
            ExprExpander::new_sampled(shapes.clone(), max_samples)
        } else if max_output_elements > SAMPLE_THRESHOLD {
            ExprExpander::new_sampled(shapes.clone(), 64)
        } else {
            ExprExpander::new(shapes.clone())
        };
        let expander = if !dtypes.is_empty() {
            expander.with_skip_unsupported(true)
        } else {
            expander
        };
        let bindings = expander.expand(graph)?;

        // Phase 2: inline intermediates
        let inlined = inline_intermediates(&bindings, final_output_ids, dtypes);

        // Phase 3: recover patterns from inlined expressions
        let patterns = recover_patterns(&inlined.bindings, shapes);

        // Helper to format expression tree
        fn format_pattern_expr(expr: &crate::compiler::attempts::v9_fused_expr::pattern::PatternExpr) -> String {
            use crate::compiler::attempts::v9_fused_expr::pattern::PatternExpr;
            match expr {
                PatternExpr::Load { load_ordinal, .. } => format!("L{}", load_ordinal),
                PatternExpr::Literal { value_bits } => {
                    let v = f64::from_bits(*value_bits);
                    format!("{:.6}", v)
                }
                PatternExpr::Binary { op, a, b } => {
                    format!("({:?} {} {})", op, format_pattern_expr(a), format_pattern_expr(b))
                }
                PatternExpr::Unary { op, input } => {
                    format!("({:?} {})", op, format_pattern_expr(input))
                }
                PatternExpr::Reduce { accum_op, reduce_idx, body, .. } => {
                    format!("(Reduce[{}]{:?} {})", reduce_idx, accum_op, format_pattern_expr(body))
                }
            }
        }

        // Log pattern details if V9_LOG_PATTERNS is set
        if std::env::var("V9_LOG_PATTERNS").is_ok() {
            for (ki, p) in patterns.iter().enumerate() {
                let input_tensors: Vec<GlobalId> = {
                    let mut ids: Vec<GlobalId> = p.accesses.iter().map(|a| a.tensor).collect();
                    for red in &p.reductions {
                        for a in &red.accesses {
                            ids.push(a.tensor);
                        }
                    }
                    ids.sort();
                    ids.dedup();
                    ids
                };
                eprintln!("  K{:>3}: out={:?} shape={:?} inputs={} reds={} accesses={}",
                    ki, p.output_tensor, p.output_shape,
                    input_tensors.len(), p.reductions.len(), p.accesses.len());
                for (ai, a) in p.accesses.iter().enumerate() {
                    eprintln!("    access[{}]: tensor={:?} base={} coeffs={:?}",
                        ai, a.tensor, a.base, a.axis_coeffs);
                }
                for (ri, r) in p.reductions.iter().enumerate() {
                    eprintln!("    reduce[{}]: depth={} loads={}", ri, r.depth, r.accesses.len());
                    for (ai, a) in r.accesses.iter().enumerate() {
                        eprintln!("      red_access[{}]: tensor={:?} base={} coeffs={:?} red_coeff={}",
                            ai, a.tensor, a.base, a.axis_coeffs, r.reduction_coeffs[ai]);
                    }
                }
                eprintln!("    expr: {}", format_pattern_expr(&p.pattern));
            }
        }

        // Phase 4: build tensor layout and compile
        let layout = TensorLayout::from_shapes_and_dtypes(shapes, dtypes);
        compile_patterns(&patterns, layout)
    }

    /// Compile and build a hybrid execution plan in one step.
    ///
    /// Runs structural verification after building the plan:
    /// - Plan integrity (op coverage + input provenance)
    /// - Full expansion cross-check (verifies sparse sampling correctness)
    pub fn compile_hybrid(
        graph: &MilliOpGraph,
        shapes: &HashMap<GlobalId, Vec<usize>>,
        final_output_ids: &HashSet<GlobalId>,
        max_samples: usize,
        dtypes: &HashMap<GlobalId, crate::dtype::DType>,
    ) -> Result<ExecutionPlan, V9Error> {
        let compiled = compile_graph_typed(graph, shapes, final_output_ids, max_samples, dtypes)?;
        let plan = build_execution_plan(graph, compiled);

        // Structural verification pass 1: plan integrity
        let integrity_issues = verify_plan_integrity(&plan, graph);
        if !integrity_issues.is_empty() {
            eprintln!("  verify_plan_integrity: {} issues", integrity_issues.len());
            for issue in &integrity_issues {
                eprintln!("    {}", issue);
            }
        }

        // Structural verification pass 2: streaming expansion coverage check
        let t0 = std::time::Instant::now();
        let expansion_issues = verify_expansion_coverage(
            &plan, graph, shapes, dtypes,
        );
        let verify_ms = t0.elapsed().as_secs_f64() * 1e3;
        if !expansion_issues.is_empty() {
            eprintln!("  verify_expansion_coverage: {} issues ({:.1}ms)", expansion_issues.len(), verify_ms);
            for issue in &expansion_issues {
                eprintln!("    {}", issue);
            }
        } else {
            eprintln!("  verify_expansion_coverage: OK ({:.1}ms)", verify_ms);
        }

        let total_issues = integrity_issues.len() + expansion_issues.len();
        if total_issues > 0 {
            eprintln!("  WARNING: {} structural verification issues found", total_issues);
        }

        Ok(plan)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::graph::GlobalId;
        use crate::milli_graph::MilliOpGraph;
        use crate::milli_graph::ops::{SimpleBinary, SimpleUnaryOp};

        #[test]
        fn test_fused_mul_neg() {
            // Graph: out = neg(a * b), where a*b is an intermediate (fused away)
            // Expected: -(a*b) = [-5, -12, -21, -32]
            let mut rng = wyrand::WyRand::new(42);
            let ext_a = GlobalId::new(&mut rng);
            let ext_b = GlobalId::new(&mut rng);
            let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
            let int_a = input_map[&ext_a];
            let int_b = input_map[&ext_b];
            let mul_out = SimpleBinary::mul(&mut graph, int_a, int_b, &mut rng);
            let neg_out = SimpleUnaryOp::neg(&mut graph, mul_out, &mut rng);

            let mut shapes = HashMap::new();
            shapes.insert(int_a, vec![4]);
            shapes.insert(int_b, vec![4]);
            shapes.insert(mul_out, vec![4]);
            shapes.insert(neg_out, vec![4]);

            let final_outputs: HashSet<GlobalId> = [neg_out].into_iter().collect();
            let compiled = compile_graph(&graph, &shapes, &final_outputs, 0).unwrap();

            // Should fuse into 1 kernel
            assert_eq!(compiled.kernels.len(), 1, "Expected 1 fused kernel");

            let mut buffers: Vec<Vec<f32>> = (0..compiled.layout.num_buffers)
                .map(|_| vec![0.0f32; 4])
                .collect();
            buffers[compiled.layout.tensor_index[&int_a]].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
            buffers[compiled.layout.tensor_index[&int_b]].copy_from_slice(&[5.0, 6.0, 7.0, 8.0]);

            let mut ptrs: Vec<*mut f32> = buffers.iter_mut().map(|b| b.as_mut_ptr()).collect();
            unsafe { compiled.execute(&mut ptrs) };

            let out_slot = compiled.layout.tensor_index[&neg_out];
            let expected = [-5.0, -12.0, -21.0, -32.0f32];
            assert_eq!(&buffers[out_slot], &expected);
        }

        #[test]
        fn test_pointwise_add() {
            // Simple: out = a + b (no fusion, just verifying basic codegen)
            let mut rng = wyrand::WyRand::new(42);
            let ext_a = GlobalId::new(&mut rng);
            let ext_b = GlobalId::new(&mut rng);
            let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
            let int_a = input_map[&ext_a];
            let int_b = input_map[&ext_b];
            let add_out = SimpleBinary::add(&mut graph, int_a, int_b, &mut rng);

            let mut shapes = HashMap::new();
            shapes.insert(int_a, vec![8]);
            shapes.insert(int_b, vec![8]);
            shapes.insert(add_out, vec![8]);

            let final_outputs: HashSet<GlobalId> = [add_out].into_iter().collect();
            let compiled = compile_graph(&graph, &shapes, &final_outputs, 0).unwrap();

            let a_data: Vec<f32> = (0..8).map(|i| i as f32).collect();
            let b_data: Vec<f32> = (0..8).map(|i| (i * 10) as f32).collect();

            let mut buffers: Vec<Vec<f32>> = (0..compiled.layout.num_buffers)
                .map(|_| vec![0.0f32; 8])
                .collect();
            let a_slot = compiled.layout.tensor_index[&int_a];
            let b_slot = compiled.layout.tensor_index[&int_b];
            buffers[a_slot].copy_from_slice(&a_data);
            buffers[b_slot].copy_from_slice(&b_data);

            let mut ptrs: Vec<*mut f32> = buffers.iter_mut().map(|b| b.as_mut_ptr()).collect();
            unsafe { compiled.execute(&mut ptrs) };

            let out_slot = compiled.layout.tensor_index[&add_out];
            let expected: Vec<f32> = (0..8).map(|i| i as f32 + (i * 10) as f32).collect();
            assert_eq!(&buffers[out_slot], &expected);
        }

        #[test]
        fn test_fused_add_mul_chain() {
            // out = (a + b) * c, where (a+b) is intermediate
            let mut rng = wyrand::WyRand::new(42);
            let ext_a = GlobalId::new(&mut rng);
            let ext_b = GlobalId::new(&mut rng);
            let ext_c = GlobalId::new(&mut rng);
            let (mut graph, input_map) =
                MilliOpGraph::new([ext_a, ext_b, ext_c], &mut rng);
            let int_a = input_map[&ext_a];
            let int_b = input_map[&ext_b];
            let int_c = input_map[&ext_c];
            let add_out = SimpleBinary::add(&mut graph, int_a, int_b, &mut rng);
            let mul_out = SimpleBinary::mul(&mut graph, add_out, int_c, &mut rng);

            let n = 6;
            let mut shapes = HashMap::new();
            shapes.insert(int_a, vec![n]);
            shapes.insert(int_b, vec![n]);
            shapes.insert(int_c, vec![n]);
            shapes.insert(add_out, vec![n]);
            shapes.insert(mul_out, vec![n]);

            let final_outputs: HashSet<GlobalId> = [mul_out].into_iter().collect();
            let compiled = compile_graph(&graph, &shapes, &final_outputs, 0).unwrap();

            // Verify: only 1 kernel (fusion happened)
            assert_eq!(
                compiled.kernels.len(),
                1,
                "Expected 1 fused kernel, got {}",
                compiled.kernels.len()
            );

            let a_data: Vec<f32> = (0..n).map(|i| (i + 1) as f32).collect();
            let b_data: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
            let c_data: Vec<f32> = (0..n).map(|i| (i + 3) as f32).collect();

            let mut buffers: Vec<Vec<f32>> = (0..compiled.layout.num_buffers)
                .map(|_| vec![0.0f32; n])
                .collect();
            buffers[compiled.layout.tensor_index[&int_a]].copy_from_slice(&a_data);
            buffers[compiled.layout.tensor_index[&int_b]].copy_from_slice(&b_data);
            buffers[compiled.layout.tensor_index[&int_c]].copy_from_slice(&c_data);

            let mut ptrs: Vec<*mut f32> = buffers.iter_mut().map(|b| b.as_mut_ptr()).collect();
            unsafe { compiled.execute(&mut ptrs) };

            let expected: Vec<f32> = (0..n)
                .map(|i| ((i + 1) as f32 + (i * 2) as f32) * (i + 3) as f32)
                .collect();
            let out_slot = compiled.layout.tensor_index[&mul_out];
            assert_eq!(&buffers[out_slot], &expected);
        }

        #[test]
        fn test_matmul_simple() {
            // out = A @ B, 2x3 * 3x2 = 2x2
            let mut rng = wyrand::WyRand::new(42);
            let ext_a = GlobalId::new(&mut rng);
            let ext_b = GlobalId::new(&mut rng);
            let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
            let int_a = input_map[&ext_a];
            let int_b = input_map[&ext_b];
            let mm_out =
                crate::milli_graph::ops::MatMul::push_new(&mut graph, int_a, int_b, &mut rng);

            let mut shapes = HashMap::new();
            shapes.insert(int_a, vec![2, 3]);
            shapes.insert(int_b, vec![3, 2]);
            shapes.insert(mm_out, vec![2, 2]);

            let final_outputs: HashSet<GlobalId> = [mm_out].into_iter().collect();
            let compiled = compile_graph(&graph, &shapes, &final_outputs, 0).unwrap();

            // A = [[1,2,3],[4,5,6]], B = [[1,0],[0,1],[1,1]]
            // A@B = [[1+0+3, 0+2+3],[4+0+6, 0+5+6]] = [[4,5],[10,11]]
            let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
            let b_data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0f32];
            let expected = vec![4.0, 5.0, 10.0, 11.0f32];

            let mut buffers: Vec<Vec<f32>> = (0..compiled.layout.num_buffers)
                .map(|_| vec![0.0f32; 8])
                .collect();
            buffers[compiled.layout.tensor_index[&int_a]][..6].copy_from_slice(&a_data);
            buffers[compiled.layout.tensor_index[&int_b]][..6].copy_from_slice(&b_data);

            let mut ptrs: Vec<*mut f32> = buffers.iter_mut().map(|b| b.as_mut_ptr()).collect();
            unsafe { compiled.execute(&mut ptrs) };

            let out_slot = compiled.layout.tensor_index[&mm_out];
            assert_eq!(&buffers[out_slot][..4], &expected);
        }

        #[test]
        fn test_fused_matmul_bias_tanh() {
            // out = tanh(A @ W + bias), 2x3 * 3x2 + [1,2] broadcast = 2x2
            // Tests the embedded-reduction architecture: reduction (matmul)
            // fused inside pointwise ops (add + tanh).
            let mut rng = wyrand::WyRand::new(42);
            let ext_a = GlobalId::new(&mut rng);
            let ext_w = GlobalId::new(&mut rng);
            let ext_bias = GlobalId::new(&mut rng);
            let (mut graph, input_map) =
                MilliOpGraph::new([ext_a, ext_w, ext_bias], &mut rng);
            let int_a = input_map[&ext_a];
            let int_w = input_map[&ext_w];
            let int_bias = input_map[&ext_bias];

            let mm_out =
                crate::milli_graph::ops::MatMul::push_new(&mut graph, int_a, int_w, &mut rng);
            let add_out = SimpleBinary::add(&mut graph, mm_out, int_bias, &mut rng);
            let tanh_out = SimpleUnaryOp::trig(
                &mut graph,
                add_out,
                crate::TrigOp::Tanh,
                &mut rng,
            );

            let mut shapes = HashMap::new();
            shapes.insert(int_a, vec![2, 3]);
            shapes.insert(int_w, vec![3, 2]);
            shapes.insert(int_bias, vec![1, 2]); // broadcast bias
            shapes.insert(mm_out, vec![2, 2]);
            shapes.insert(add_out, vec![2, 2]);
            shapes.insert(tanh_out, vec![2, 2]);

            // Only tanh_out is a final output — mm_out and add_out are intermediates
            let final_outputs: HashSet<GlobalId> = [tanh_out].into_iter().collect();
            let compiled = compile_graph(&graph, &shapes, &final_outputs, 0).unwrap();

            // Should fuse into 1 kernel (matmul + bias + tanh all fused)
            assert_eq!(compiled.kernels.len(), 1, "Expected 1 fused kernel");

            // A = [[1,2,3],[4,5,6]], W = [[1,0],[0,1],[1,1]], bias = [10, 20]
            // A@W = [[4,5],[10,11]]
            // A@W + bias = [[14,25],[20,31]]
            // tanh(A@W + bias) ≈ [[1.0, 1.0],[1.0, 1.0]] (saturated)
            let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32];
            let w_data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0f32];
            let bias_data = vec![10.0, 20.0f32];

            let mut buffers: Vec<Vec<f32>> = (0..compiled.layout.num_buffers)
                .map(|_| vec![0.0f32; 8])
                .collect();
            buffers[compiled.layout.tensor_index[&int_a]][..6].copy_from_slice(&a_data);
            buffers[compiled.layout.tensor_index[&int_w]][..6].copy_from_slice(&w_data);
            buffers[compiled.layout.tensor_index[&int_bias]][..2].copy_from_slice(&bias_data);

            let mut ptrs: Vec<*mut f32> = buffers.iter_mut().map(|b| b.as_mut_ptr()).collect();
            unsafe { compiled.execute(&mut ptrs) };

            let out_slot = compiled.layout.tensor_index[&tanh_out];
            let result = &buffers[out_slot][..4];

            // Compute expected: tanh(mm + bias) for each element
            let mm = [4.0f32, 5.0, 10.0, 11.0];
            let biases = [10.0f32, 20.0, 10.0, 20.0]; // broadcast [10,20] to 2x2
            let expected: Vec<f32> = mm
                .iter()
                .zip(biases.iter())
                .map(|(m, b)| (m + b).tanh())
                .collect();

            for i in 0..4 {
                assert!(
                    (result[i] - expected[i]).abs() < 1e-6,
                    "element {}: got {}, expected {}",
                    i,
                    result[i],
                    expected[i]
                );
            }
        }

        #[test]
        fn test_two_layer_matmul_mlp() {
            // Two-layer MLP: out = tanh(tanh(x @ W1 + b1) @ W2 + b2)
            // x: 2x3, W1: 3x4, b1: 1x4, W2: 4x2, b2: 1x2 → out: 2x2
            // Tests materialization boundaries: act1 must be materialized
            // between the two matmul reductions.
            let mut rng = wyrand::WyRand::new(99);
            let ext_x = GlobalId::new(&mut rng);
            let ext_w1 = GlobalId::new(&mut rng);
            let ext_b1 = GlobalId::new(&mut rng);
            let ext_w2 = GlobalId::new(&mut rng);
            let ext_b2 = GlobalId::new(&mut rng);
            let (mut graph, input_map) =
                MilliOpGraph::new([ext_x, ext_w1, ext_b1, ext_w2, ext_b2], &mut rng);
            let x = input_map[&ext_x];
            let w1 = input_map[&ext_w1];
            let b1 = input_map[&ext_b1];
            let w2 = input_map[&ext_w2];
            let b2 = input_map[&ext_b2];

            // Layer 1: act1 = tanh(x @ W1 + b1)
            let mm1 = crate::milli_graph::ops::MatMul::push_new(&mut graph, x, w1, &mut rng);
            let add1 = SimpleBinary::add(&mut graph, mm1, b1, &mut rng);
            let act1 = SimpleUnaryOp::trig(&mut graph, add1, crate::TrigOp::Tanh, &mut rng);

            // Layer 2: out = tanh(act1 @ W2 + b2)
            let mm2 = crate::milli_graph::ops::MatMul::push_new(&mut graph, act1, w2, &mut rng);
            let add2 = SimpleBinary::add(&mut graph, mm2, b2, &mut rng);
            let out = SimpleUnaryOp::trig(&mut graph, add2, crate::TrigOp::Tanh, &mut rng);

            let mut shapes = HashMap::new();
            shapes.insert(x, vec![2, 3]);
            shapes.insert(w1, vec![3, 4]);
            shapes.insert(b1, vec![1, 4]);
            shapes.insert(mm1, vec![2, 4]);
            shapes.insert(add1, vec![2, 4]);
            shapes.insert(act1, vec![2, 4]);
            shapes.insert(w2, vec![4, 2]);
            shapes.insert(b2, vec![1, 2]);
            shapes.insert(mm2, vec![2, 2]);
            shapes.insert(add2, vec![2, 2]);
            shapes.insert(out, vec![2, 2]);

            let final_outputs: HashSet<GlobalId> = [out].into_iter().collect();
            let compiled = compile_graph(&graph, &shapes, &final_outputs, 0).unwrap();

            // Should produce multiple kernels (materialization boundary at act1)
            assert!(
                compiled.kernels.len() >= 2,
                "Expected >= 2 kernels (materialization boundary), got {}",
                compiled.kernels.len()
            );

            // Fill with simple data and verify against reference computation
            let x_data: Vec<f32> = (0..6).map(|i| (i + 1) as f32 * 0.1).collect();
            let w1_data: Vec<f32> = (0..12).map(|i| (i as f32 - 6.0) * 0.1).collect();
            let b1_data: Vec<f32> = vec![0.1, -0.1, 0.2, -0.2];
            let w2_data: Vec<f32> = (0..8).map(|i| (i as f32 - 4.0) * 0.1).collect();
            let b2_data: Vec<f32> = vec![0.05, -0.05];

            // Reference computation
            // mm1 = x @ W1 (2x3 * 3x4 = 2x4)
            let mut mm1_ref = vec![0.0f32; 8];
            for i in 0..2 {
                for j in 0..4 {
                    let mut sum = 0.0f32;
                    for k in 0..3 {
                        sum += x_data[i * 3 + k] * w1_data[k * 4 + j];
                    }
                    mm1_ref[i * 4 + j] = sum;
                }
            }
            // act1 = tanh(mm1 + b1)
            let mut act1_ref = vec![0.0f32; 8];
            for i in 0..2 {
                for j in 0..4 {
                    act1_ref[i * 4 + j] = (mm1_ref[i * 4 + j] + b1_data[j]).tanh();
                }
            }
            // mm2 = act1 @ W2 (2x4 * 4x2 = 2x2)
            let mut mm2_ref = vec![0.0f32; 4];
            for i in 0..2 {
                for j in 0..2 {
                    let mut sum = 0.0f32;
                    for k in 0..4 {
                        sum += act1_ref[i * 4 + k] * w2_data[k * 2 + j];
                    }
                    mm2_ref[i * 2 + j] = sum;
                }
            }
            // out = tanh(mm2 + b2)
            let expected: Vec<f32> = (0..4)
                .map(|idx| {
                    let i = idx / 2;
                    let j = idx % 2;
                    (mm2_ref[i * 2 + j] + b2_data[j]).tanh()
                })
                .collect();

            // Allocate buffers
            let max_size = 12; // largest tensor is 3x4=12
            let mut buffers: Vec<Vec<f32>> = (0..compiled.layout.num_buffers)
                .map(|_| vec![0.0f32; max_size])
                .collect();
            buffers[compiled.layout.tensor_index[&x]][..6].copy_from_slice(&x_data);
            buffers[compiled.layout.tensor_index[&w1]][..12].copy_from_slice(&w1_data);
            buffers[compiled.layout.tensor_index[&b1]][..4].copy_from_slice(&b1_data);
            buffers[compiled.layout.tensor_index[&w2]][..8].copy_from_slice(&w2_data);
            buffers[compiled.layout.tensor_index[&b2]][..2].copy_from_slice(&b2_data);

            let mut ptrs: Vec<*mut f32> = buffers.iter_mut().map(|b| b.as_mut_ptr()).collect();
            unsafe { compiled.execute(&mut ptrs) };

            let out_slot = compiled.layout.tensor_index[&out];
            let result = &buffers[out_slot][..4];

            for i in 0..4 {
                assert!(
                    (result[i] - expected[i]).abs() < 1e-5,
                    "element {}: got {}, expected {}",
                    i,
                    result[i],
                    expected[i]
                );
            }
        }

        fn make_random_f32(n: usize, seed: u64) -> Vec<f32> {
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

        fn alloc_and_run(
            compiled: &CompiledGraph,
            inputs: &HashMap<GlobalId, Vec<f32>>,
            output_id: GlobalId,
        ) -> Vec<f32> {
            let layout = &compiled.layout;
            let mut bufs: Vec<Vec<f32>> = (0..layout.num_buffers)
                .map(|i| vec![0.0f32; layout.tensor_sizes.values()
                    .filter(|&&s| layout.tensor_index.values().any(|&idx| idx == i && s > 0))
                    .copied().next().unwrap_or(64)])
                .collect();
            // Properly size buffers
            for (id, &size) in &layout.tensor_sizes {
                let idx = layout.tensor_index[id];
                if bufs[idx].len() < size {
                    bufs[idx].resize(size, 0.0);
                }
            }
            for (id, data) in inputs {
                if let Some(&idx) = layout.tensor_index.get(id) {
                    bufs[idx][..data.len()].copy_from_slice(data);
                }
            }
            let mut ptrs: Vec<*mut f32> = bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();
            unsafe { compiled.execute(&mut ptrs) };
            let out_idx = layout.tensor_index[&output_id];
            bufs[out_idx].clone()
        }

        fn alloc_and_run_parallel(
            compiled: &CompiledGraph,
            inputs: &HashMap<GlobalId, Vec<f32>>,
            output_id: GlobalId,
            num_threads: usize,
        ) -> Vec<f32> {
            let layout = &compiled.layout;
            let mut bufs: Vec<Vec<f32>> = (0..layout.num_buffers)
                .map(|_| vec![0.0f32; 64])
                .collect();
            for (id, &size) in &layout.tensor_sizes {
                let idx = layout.tensor_index[id];
                if bufs[idx].len() < size {
                    bufs[idx].resize(size, 0.0);
                }
            }
            for (id, data) in inputs {
                if let Some(&idx) = layout.tensor_index.get(id) {
                    bufs[idx][..data.len()].copy_from_slice(data);
                }
            }
            let ptrs: Vec<*mut f32> = bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();
            unsafe { execute_parallel(compiled, &ptrs, num_threads) };
            let out_idx = layout.tensor_index[&output_id];
            bufs[out_idx].clone()
        }

        #[test]
        fn test_parallel_matmul() {
            let mut rng = wyrand::WyRand::new(7001);
            let (m, k, n) = (64, 96, 80);
            let ext_a = GlobalId::new(&mut rng);
            let ext_b = GlobalId::new(&mut rng);
            let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
            let a = input_map[&ext_a];
            let b = input_map[&ext_b];
            let c = crate::milli_graph::ops::MatMul::push_new(&mut graph, a, b, &mut rng);

            let mut shapes = HashMap::new();
            shapes.insert(a, vec![m, k]);
            shapes.insert(b, vec![k, n]);
            shapes.insert(c, vec![m, n]);

            let final_outputs: HashSet<GlobalId> = [c].into_iter().collect();
            let compiled = compile_graph(&graph, &shapes, &final_outputs, 0).unwrap();

            let a_data = make_random_f32(m * k, 7002);
            let b_data = make_random_f32(k * n, 7003);
            let mut inputs = HashMap::new();
            inputs.insert(a, a_data);
            inputs.insert(b, b_data);

            let serial = alloc_and_run(&compiled, &inputs, c);

            for threads in [2, 4, 8] {
                let parallel = alloc_and_run_parallel(&compiled, &inputs, c, threads);
                let diff = max_diff(&serial, &parallel);
                assert!(
                    diff < 1e-5,
                    "parallel ({threads}t) diverged from serial: max diff {diff}"
                );
            }
        }

        #[test]
        fn test_parallel_fused_mlp() {
            // Two-layer MLP: out = tanh(tanh(x @ W1 + b1) @ W2 + b2)
            // Tests inter-op parallelism (two kernels with dependency)
            let mut rng = wyrand::WyRand::new(8001);
            let (m, k1, k2, n) = (32, 128, 64, 32);
            let ext_x = GlobalId::new(&mut rng);
            let ext_w1 = GlobalId::new(&mut rng);
            let ext_b1 = GlobalId::new(&mut rng);
            let ext_w2 = GlobalId::new(&mut rng);
            let ext_b2 = GlobalId::new(&mut rng);
            let (mut graph, input_map) =
                MilliOpGraph::new([ext_x, ext_w1, ext_b1, ext_w2, ext_b2], &mut rng);
            let x = input_map[&ext_x];
            let w1 = input_map[&ext_w1];
            let b1 = input_map[&ext_b1];
            let w2 = input_map[&ext_w2];
            let b2 = input_map[&ext_b2];

            let mm1 = crate::milli_graph::ops::MatMul::push_new(&mut graph, x, w1, &mut rng);
            let add1 = SimpleBinary::add(&mut graph, mm1, b1, &mut rng);
            let act1 = SimpleUnaryOp::trig(&mut graph, add1, crate::TrigOp::Tanh, &mut rng);
            let mm2 = crate::milli_graph::ops::MatMul::push_new(&mut graph, act1, w2, &mut rng);
            let add2 = SimpleBinary::add(&mut graph, mm2, b2, &mut rng);
            let out = SimpleUnaryOp::trig(&mut graph, add2, crate::TrigOp::Tanh, &mut rng);

            let mut shapes = HashMap::new();
            shapes.insert(x, vec![m, k1]);
            shapes.insert(w1, vec![k1, k2]);
            shapes.insert(b1, vec![1, k2]);
            shapes.insert(mm1, vec![m, k2]);
            shapes.insert(add1, vec![m, k2]);
            shapes.insert(act1, vec![m, k2]);
            shapes.insert(w2, vec![k2, n]);
            shapes.insert(b2, vec![1, n]);
            shapes.insert(mm2, vec![m, n]);
            shapes.insert(add2, vec![m, n]);
            shapes.insert(out, vec![m, n]);

            let final_outputs: HashSet<GlobalId> = [out].into_iter().collect();
            let compiled = compile_graph(&graph, &shapes, &final_outputs, 0).unwrap();

            let mut inputs = HashMap::new();
            inputs.insert(x, make_random_f32(m * k1, 8002));
            inputs.insert(w1, make_random_f32(k1 * k2, 8003));
            inputs.insert(b1, make_random_f32(k2, 8004));
            inputs.insert(w2, make_random_f32(k2 * n, 8005));
            inputs.insert(b2, make_random_f32(n, 8006));

            let serial = alloc_and_run(&compiled, &inputs, out);

            for threads in [2, 4, 8] {
                let parallel = alloc_and_run_parallel(&compiled, &inputs, out, threads);
                let diff = max_diff(&serial, &parallel);
                assert!(
                    diff < 1e-4,
                    "fused MLP ({threads}t) diverged from serial: max diff {diff}"
                );
            }
        }
        #[test]
        fn test_sampled_large_add_correctness() {
            // Regression test: sampled [768, 768] pointwise add.
            // With flat-stride sampling (stride=9216=12*768), all samples had
            // column=0, so the recovered pattern loaded from column 0 for
            // every output element.
            let mut rng = wyrand::WyRand::new(9100);
            let (rows, cols) = (768, 768);
            let ext_a = GlobalId::new(&mut rng);
            let ext_b = GlobalId::new(&mut rng);
            let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
            let a = input_map[&ext_a];
            let b = input_map[&ext_b];
            let c = SimpleBinary::add(&mut graph, a, b, &mut rng);

            let mut shapes = HashMap::new();
            shapes.insert(a, vec![rows, cols]);
            shapes.insert(b, vec![rows, cols]);
            shapes.insert(c, vec![rows, cols]);

            let final_outputs: HashSet<GlobalId> = [c].into_iter().collect();
            let compiled = compile_graph(&graph, &shapes, &final_outputs, 64).unwrap();

            let a_data = make_random_f32(rows * cols, 9101);
            let b_data = make_random_f32(rows * cols, 9102);
            let mut inputs = HashMap::new();
            inputs.insert(a, a_data.clone());
            inputs.insert(b, b_data.clone());

            let result = alloc_and_run(&compiled, &inputs, c);

            let expected: Vec<f32> = a_data.iter().zip(&b_data)
                .map(|(x, y)| x + y).collect();

            let max_err = result.iter().zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err < 1e-6,
                "sampled [{rows}x{cols}] add max error {:.4e} (expected < 1e-6)",
                max_err
            );
        }

        #[test]
        fn test_sampled_large_matmul_correctness() {
            // Regression test: [256, 256] matmul with 64-sample expansion.
            // This would have failed with flat-stride sampling because stride=1024=4*256
            // means all samples have column 0, making the B-access coefficient wrong.
            let mut rng = wyrand::WyRand::new(9001);
            let (m, k, n) = (256, 128, 256);
            let ext_a = GlobalId::new(&mut rng);
            let ext_b = GlobalId::new(&mut rng);
            let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
            let a = input_map[&ext_a];
            let b = input_map[&ext_b];
            let c = crate::milli_graph::ops::MatMul::push_new(&mut graph, a, b, &mut rng);

            let mut shapes = HashMap::new();
            shapes.insert(a, vec![m, k]);
            shapes.insert(b, vec![k, n]);
            shapes.insert(c, vec![m, n]);

            let final_outputs: HashSet<GlobalId> = [c].into_iter().collect();
            // Use max_samples=64 to force sampling
            let compiled = compile_graph(&graph, &shapes, &final_outputs, 64).unwrap();

            let a_data = make_random_f32(m * k, 9002);
            let b_data = make_random_f32(k * n, 9003);
            let mut inputs = HashMap::new();
            inputs.insert(a, a_data.clone());
            inputs.insert(b, b_data.clone());

            let result = alloc_and_run(&compiled, &inputs, c);

            // Reference: naive matmul
            let mut expected = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for kk in 0..k {
                        sum += a_data[i * k + kk] * b_data[kk * n + j];
                    }
                    expected[i * n + j] = sum;
                }
            }

            let max_err = result.iter().zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err < 1e-2,
                "sampled [{}x{}] matmul max error {:.4} (expected < 0.01)",
                m, n, max_err
            );
        }
    }
}
