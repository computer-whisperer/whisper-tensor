//! v9 codegen: single-threaded Cranelift JIT from recovered patterns.
//!
//! Emits nested loops matching the output shape. Reduce nodes in the pattern
//! tree emit their own inner loops inline. Register-blocked reductions use
//! F32X4 SIMD when the block axis is innermost and all varying loads are
//! contiguous. No tiling yet.

#[cfg(feature = "cranelift")]
pub mod jit {
    use crate::compiler::attempts::v1_scalar_crystal::codegen::{CodegenError, TensorLayout};
    use crate::compiler::attempts::v9_fused_expr::pattern::*;
    use crate::compiler::common::v2_frontend::{ScalarBinOp, ScalarUnaryOp};
    use crate::graph::GlobalId;
    use std::collections::HashMap;
    use cranelift_codegen::ir::types;
    use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Value};
    use cranelift_codegen::settings::{self, Configurable};
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::{Linkage, Module};

    #[derive(Debug, thiserror::Error)]
    pub enum V9Error {
        #[error("v9 expand: {0}")]
        Expand(#[from] crate::compiler::common::v2_frontend::ExpandError),
        #[error("v9 codegen: {0}")]
        Codegen(#[from] CodegenError),
        #[error("v9: {0}")]
        Other(String),
    }

    pub struct CompiledKernel {
        func_ptr: *const u8,
        pub output_tensor: GlobalId,
        pub input_tensors: Vec<GlobalId>,
        pub count: usize,
    }

    impl CompiledKernel {
        /// # Safety
        pub unsafe fn execute(&self, buffers: *const *mut f32) {
            let func: unsafe extern "C" fn(*const *mut f32) =
                unsafe { std::mem::transmute(self.func_ptr) };
            unsafe { func(buffers) };
        }
    }

    pub struct CompiledGraph {
        _module: JITModule,
        pub layout: TensorLayout,
        pub kernels: Vec<CompiledKernel>,
    }

    unsafe impl Send for CompiledKernel {}
    unsafe impl Sync for CompiledKernel {}
    unsafe impl Send for CompiledGraph {}
    unsafe impl Sync for CompiledGraph {}

    impl CompiledGraph {
        /// # Safety
        pub unsafe fn execute(&self, buffers: &mut [*mut f32]) {
            let ptr = buffers.as_ptr();
            for kernel in &self.kernels {
                unsafe { kernel.execute(ptr) };
            }
        }
    }

    pub fn compile_patterns(
        patterns: &[RecoveredPattern],
        layout: TensorLayout,
    ) -> Result<CompiledGraph, V9Error> {
        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed").unwrap();
        let isa_builder = cranelift_native::builder()
            .map_err(|e| V9Error::Other(format!("cranelift native ISA: {}", e)))?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| V9Error::Other(format!("ISA finish: {}", e)))?;

        let mut jit_builder =
            JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        jit_builder.symbol("wt9_expf", wt9_expf as *const u8);
        jit_builder.symbol("wt9_sqrtf", wt9_sqrtf as *const u8);
        jit_builder.symbol("wt9_tanhf", wt9_tanhf as *const u8);
        jit_builder.symbol("wt9_logf", wt9_logf as *const u8);
        jit_builder.symbol("wt9_floorf", wt9_floorf as *const u8);
        jit_builder.symbol("wt9_ceilf", wt9_ceilf as *const u8);
        jit_builder.symbol("wt9_fabsf", wt9_fabsf as *const u8);
        jit_builder.symbol("wt9_erff", wt9_erff as *const u8);

        let mut module = JITModule::new(jit_builder);
        let mut ctx = module.make_context();
        let mut func_ctx = FunctionBuilderContext::new();

        let mut kernels = Vec::new();

        for (ki, pattern) in patterns.iter().enumerate() {
            let func_name = format!("v9_kernel_{}", ki);

            ctx.func.signature.params.push(AbiParam::new(types::I64));

            let func_id = module
                .declare_function(&func_name, Linkage::Local, &ctx.func.signature)
                .map_err(|e| V9Error::Other(format!("declare: {}", e)))?;

            let math_funcs = declare_math_funcs(&mut module)?;

            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let ptr_table = builder.block_params(entry_block)[0];

            emit_pattern(
                &mut builder,
                &mut module,
                pattern,
                &layout,
                ptr_table,
                &math_funcs,
            )?;

            builder.ins().return_(&[]);
            builder.finalize();

            module
                .define_function(func_id, &mut ctx)
                .map_err(|e| V9Error::Other(format!("define: {}", e)))?;
            ctx.clear();

            let mut input_tensors: Vec<GlobalId> =
                pattern.accesses.iter().map(|a| a.tensor).collect();
            for red in &pattern.reductions {
                for a in &red.accesses {
                    input_tensors.push(a.tensor);
                }
            }
            input_tensors.sort();
            input_tensors.dedup();

            let count: usize = pattern.output_shape.iter().product();
            kernels.push((func_id, pattern.output_tensor, input_tensors, count));
        }

        module
            .finalize_definitions()
            .map_err(|e| V9Error::Other(format!("finalize: {}", e)))?;

        let compiled_kernels: Vec<CompiledKernel> = kernels
            .into_iter()
            .map(|(func_id, output_tensor, input_tensors, count)| {
                let func_ptr = module.get_finalized_function(func_id);
                CompiledKernel {
                    func_ptr,
                    output_tensor,
                    input_tensors,
                    count,
                }
            })
            .collect();

        Ok(CompiledGraph {
            _module: module,
            layout,
            kernels: compiled_kernels,
        })
    }

    // ------------------------------------------------------------------
    // Nested loop emission
    // ------------------------------------------------------------------

    struct VarAlloc(u32);
    impl VarAlloc {
        fn new() -> Self {
            VarAlloc(0)
        }
        fn next(&mut self) -> Variable {
            let v = Variable::from_u32(self.0);
            self.0 += 1;
            v
        }
    }

    fn emit_pattern(
        builder: &mut FunctionBuilder,
        module: &mut JITModule,
        pattern: &RecoveredPattern,
        layout: &TensorLayout,
        ptr_table: Value,
        math_funcs: &MathFuncs,
    ) -> Result<(), V9Error> {
        // Try register blocking for patterns with reductions
        if let Some(blocking) = analyze_blocking(pattern) {
            return emit_pattern_blocked(
                builder, module, pattern, layout, ptr_table, math_funcs, &blocking,
            );
        }

        let ndim = pattern.output_shape.len();
        let mut vars = VarAlloc::new();

        // Collect all tensor IDs referenced by this pattern (for pointer preloading)
        let all_tensors = collect_pattern_tensors(pattern);

        // Pre-load all buffer pointers before any loops
        let buf_ptrs = preload_buffer_ptrs(builder, layout, ptr_table, &all_tensors);

        // Allocate loop variables for each output axis
        let mut axis_vars = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let v = vars.next();
            builder.declare_var(v, types::I64);
            axis_vars.push(v);
        }

        // Build nested loops
        let loop_blocks = build_nested_loops(builder, &axis_vars, &pattern.output_shape);

        // Compute flat output index
        let out_strides = row_major_strides(&pattern.output_shape);
        let flat_out = compute_flat_index(builder, &axis_vars, &out_strides);

        // Load outer (pointwise) values
        let outer_vals = load_outer_values(builder, pattern, layout, &axis_vars, &buf_ptrs);

        // Evaluate expression tree (Reduce nodes emit their own inner loops)
        let result = eval_pattern_expr(
            builder,
            module,
            &pattern.pattern,
            &outer_vals,
            &axis_vars,
            pattern,
            layout,
            ptr_table,
            math_funcs,
            &mut vars,
            &buf_ptrs,
        )?;

        // Store result
        store_result(builder, layout, pattern.output_tensor, flat_out, result, &buf_ptrs);

        // Close nested loops
        close_nested_loops(builder, &axis_vars, &loop_blocks);

        Ok(())
    }

    // ------------------------------------------------------------------
    // Register blocking analysis
    // ------------------------------------------------------------------

    struct BlockingInfo {
        block_axis: usize,
        nr: usize,
        reduce_idx: usize,
        /// Per reduction load: true if invariant (coefficient 0) on block axis.
        invariant: Vec<bool>,
    }

    /// Analyze a pattern for register-blocking opportunities.
    ///
    /// For a reduction `out[i,j] = sum_k(body(loads...))`, register blocking on
    /// axis `a` is beneficial when at least one load has `axis_coeffs[a] == 0`
    /// (invariant across that axis). That load gets loaded once per k and reused
    /// across NR accumulator lanes — exactly what a matmul micro-kernel does.
    fn analyze_blocking(pattern: &RecoveredPattern) -> Option<BlockingInfo> {
        // Only handle single-reduction patterns for now
        if pattern.reductions.len() != 1 {
            return None;
        }
        let red = &pattern.reductions[0];
        if red.accesses.is_empty() {
            return None;
        }
        let ndim = pattern.output_shape.len();

        // Prefer innermost axis (better cache behavior for stores)
        for axis in (0..ndim).rev() {
            let extent = pattern.output_shape[axis];
            if extent < 2 {
                continue;
            }
            let invariant: Vec<bool> = red
                .accesses
                .iter()
                .map(|a| a.axis_coeffs.get(axis).copied().unwrap_or(0) == 0)
                .collect();
            // Need at least one invariant AND at least one varying
            let has_inv = invariant.iter().any(|&v| v);
            let has_var = invariant.iter().any(|&v| !v);
            if has_inv && has_var {
                let nr = choose_nr(extent);
                if nr >= 2 {
                    return Some(BlockingInfo {
                        block_axis: axis,
                        nr,
                        reduce_idx: 0,
                        invariant,
                    });
                }
            }
        }
        None
    }

    fn choose_nr(extent: usize) -> usize {
        // Pick largest divisor of extent that's <= MAX_NR
        const MAX_NR: usize = 8;
        for nr in (2..=MAX_NR).rev() {
            if extent % nr == 0 {
                return nr;
            }
        }
        1
    }

    fn collect_pattern_tensors(pattern: &RecoveredPattern) -> Vec<GlobalId> {
        let mut all = vec![pattern.output_tensor];
        for a in &pattern.accesses {
            all.push(a.tensor);
        }
        for red in &pattern.reductions {
            for a in &red.accesses {
                all.push(a.tensor);
            }
        }
        all
    }

    /// Find the Reduce node with the given index in the pattern tree.
    /// Returns (accum_op, identity_bits, body).
    fn find_reduce_node(
        pattern: &PatternExpr,
        target_idx: usize,
    ) -> Option<(ScalarBinOp, u64, &PatternExpr)> {
        match pattern {
            PatternExpr::Reduce {
                accum_op,
                identity_bits,
                body,
                reduce_idx,
            } if *reduce_idx == target_idx => Some((*accum_op, *identity_bits, body.as_ref())),
            PatternExpr::Binary { a, b, .. } => {
                find_reduce_node(a, target_idx).or_else(|| find_reduce_node(b, target_idx))
            }
            PatternExpr::Unary { input, .. } => find_reduce_node(input, target_idx),
            _ => None,
        }
    }

    /// Evaluate a pattern expression, but substitute a pre-computed value for
    /// the Reduce node with the given index. Used in the post-reduction phase
    /// of register-blocked emission.
    fn eval_expr_with_reduce_override(
        builder: &mut FunctionBuilder,
        module: &mut JITModule,
        pattern: &PatternExpr,
        outer_load_vals: &[Value],
        math_funcs: &MathFuncs,
        target_reduce_idx: usize,
        reduce_value: Value,
    ) -> Result<Value, V9Error> {
        match pattern {
            PatternExpr::Load { load_ordinal, .. } => Ok(outer_load_vals[*load_ordinal]),
            PatternExpr::Literal { value_bits } => {
                let val = f64::from_bits(*value_bits) as f32;
                Ok(builder.ins().f32const(val))
            }
            PatternExpr::Binary { op, a, b } => {
                let va = eval_expr_with_reduce_override(
                    builder, module, a, outer_load_vals, math_funcs,
                    target_reduce_idx, reduce_value,
                )?;
                let vb = eval_expr_with_reduce_override(
                    builder, module, b, outer_load_vals, math_funcs,
                    target_reduce_idx, reduce_value,
                )?;
                Ok(emit_binop(builder, *op, va, vb))
            }
            PatternExpr::Unary { op, input } => {
                let vi = eval_expr_with_reduce_override(
                    builder, module, input, outer_load_vals, math_funcs,
                    target_reduce_idx, reduce_value,
                )?;
                emit_unaryop(builder, module, *op, vi, math_funcs)
            }
            PatternExpr::Reduce { reduce_idx, .. } => {
                if *reduce_idx == target_reduce_idx {
                    Ok(reduce_value)
                } else {
                    Err(V9Error::Other(
                        "unexpected non-target Reduce in blocked pattern".into(),
                    ))
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Register-blocked emission with panel packing
    // ------------------------------------------------------------------

    /// Check if a varying reduction load can be packed: it must only depend
    /// on the block axis and k, not on any non-block output axis.
    fn is_packable(access: &AffineAccess, block_axis: usize, is_invariant: bool, red_coeff: i64) -> bool {
        if is_invariant || red_coeff == 0 {
            return false;
        }
        // All non-block axis coefficients must be zero
        for (d, &coeff) in access.axis_coeffs.iter().enumerate() {
            if d != block_axis && coeff != 0 {
                return false;
            }
        }
        true
    }

    /// Check if a pattern expression body contains function calls (unary ops
    /// other than Neg, which is a native instruction). Used to gate SIMD:
    /// transcendentals like exp/tanh require scalar function calls.
    fn body_has_func_calls(pattern: &PatternExpr) -> bool {
        match pattern {
            PatternExpr::Unary { op, input } => match op {
                ScalarUnaryOp::Neg => body_has_func_calls(input),
                _ => true,
            },
            PatternExpr::Binary { a, b, .. } => body_has_func_calls(a) || body_has_func_calls(b),
            _ => false,
        }
    }

    fn emit_pattern_blocked(
        builder: &mut FunctionBuilder,
        module: &mut JITModule,
        pattern: &RecoveredPattern,
        layout: &TensorLayout,
        ptr_table: Value,
        math_funcs: &MathFuncs,
        blocking: &BlockingInfo,
    ) -> Result<(), V9Error> {
        use cranelift_codegen::ir::condcodes::IntCC;
        use cranelift_codegen::ir::{StackSlotData, StackSlotKind};

        let ndim = pattern.output_shape.len();
        let block_axis = blocking.block_axis;
        let nr = blocking.nr;
        let block_extent = pattern.output_shape[block_axis];
        let block_main_limit = block_extent - block_extent % nr;
        let mut vars = VarAlloc::new();

        // Pre-load buffer pointers
        let all_tensors = collect_pattern_tensors(pattern);
        let buf_ptrs = preload_buffer_ptrs(builder, layout, ptr_table, &all_tensors);

        // Allocate axis variables
        let mut axis_vars = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let v = vars.next();
            builder.declare_var(v, types::I64);
            axis_vars.push(v);
        }

        let red = &pattern.reductions[blocking.reduce_idx];
        let (accum_op, identity_bits, body_pattern) =
            find_reduce_node(&pattern.pattern, blocking.reduce_idx)
                .ok_or_else(|| V9Error::Other("Reduce node not found".into()))?;

        // Identify packable loads
        let packable: Vec<bool> = red
            .accesses
            .iter()
            .enumerate()
            .map(|(pos, access)| {
                is_packable(access, block_axis, blocking.invariant[pos], red.reduction_coeffs[pos])
            })
            .collect();
        let any_packable = packable.iter().any(|&p| p);

        // SIMD eligibility: NR must be multiple of 4, all varying loads
        // must be packable, body must not contain function calls, and
        // block_axis must be innermost (for contiguous stores).
        const SIMD_WIDTH: usize = 4;
        let use_simd = nr % SIMD_WIDTH == 0
            && block_axis + 1 == ndim
            && red.accesses.iter().enumerate().all(|(pos, _)| {
                blocking.invariant[pos] || packable[pos]
            })
            && !body_has_func_calls(body_pattern);

        // Allocate stack slots for pack panels (before any loops).
        // Use 16-byte alignment when SIMD is active for aligned vector loads.
        let panel_align = if use_simd { 16 } else { 0 };
        let panel_ptrs: Vec<Option<Value>> = if any_packable {
            red.accesses
                .iter()
                .enumerate()
                .map(|(pos, _)| {
                    if packable[pos] {
                        let panel_bytes = (red.depth * nr * 4) as u32;
                        let slot = builder.create_sized_stack_slot(StackSlotData::new(
                            StackSlotKind::ExplicitSlot,
                            panel_bytes,
                            panel_align,
                        ));
                        Some(builder.ins().stack_addr(types::I64, slot, 0))
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            vec![None; red.accesses.len()]
        };

        // ===== Loop order: block_axis (step NR) → pack → non-block axes → k =====

        // --- Block axis loop (outermost) ---
        let zero_ba = builder.ins().iconst(types::I64, 0);
        builder.def_var(axis_vars[block_axis], zero_ba);

        let ba_header = builder.create_block();
        let ba_body = builder.create_block();
        let ba_exit = builder.create_block();

        builder.ins().jump(ba_header, &[]);
        builder.switch_to_block(ba_header);

        let ba_iv = builder.use_var(axis_vars[block_axis]);
        let ba_limit = builder.ins().iconst(types::I64, block_main_limit as i64);
        let ba_cmp = builder.ins().icmp(IntCC::SignedLessThan, ba_iv, ba_limit);
        builder.ins().brif(ba_cmp, ba_body, &[], ba_exit, &[]);

        builder.switch_to_block(ba_body);
        builder.seal_block(ba_body);

        // --- Panel packing (inside block axis loop, before non-block axes) ---
        if any_packable {
            let j0_pack = builder.use_var(axis_vars[block_axis]);
            for (pos, access) in red.accesses.iter().enumerate() {
                if !packable[pos] {
                    continue;
                }
                let panel_ptr = panel_ptrs[pos].unwrap();
                let block_coeff = access.axis_coeffs.get(block_axis).copied().unwrap_or(0);
                let red_coeff = red.reduction_coeffs[pos];

                // Pack loop: for pk in 0..K
                let pk_var = vars.next();
                builder.declare_var(pk_var, types::I64);
                let zero_pk = builder.ins().iconst(types::I64, 0);
                builder.def_var(pk_var, zero_pk);

                let pk_header = builder.create_block();
                let pk_body = builder.create_block();
                let pk_exit = builder.create_block();

                builder.ins().jump(pk_header, &[]);
                builder.switch_to_block(pk_header);

                let pk = builder.use_var(pk_var);
                let pk_limit = builder.ins().iconst(types::I64, red.depth as i64);
                let pk_cmp = builder.ins().icmp(IntCC::SignedLessThan, pk, pk_limit);
                builder.ins().brif(pk_cmp, pk_body, &[], pk_exit, &[]);

                builder.switch_to_block(pk_body);
                builder.seal_block(pk_body);

                let slot = layout.tensor_index[&access.tensor];
                let src_buf = buf_ptrs[&slot];

                // Unroll NR lanes in packing
                for lane in 0..nr {
                    // src_flat = base + block_coeff * (j0 + lane) + red_coeff * pk
                    let j_lane = builder.ins().iadd_imm(j0_pack, lane as i64);
                    let mut src_flat = builder.ins().iconst(types::I64, access.base);
                    if block_coeff == 1 {
                        src_flat = builder.ins().iadd(src_flat, j_lane);
                    } else if block_coeff != 0 {
                        let c = builder.ins().iconst(types::I64, block_coeff);
                        let contrib = builder.ins().imul(c, j_lane);
                        src_flat = builder.ins().iadd(src_flat, contrib);
                    }
                    if red_coeff == 1 {
                        src_flat = builder.ins().iadd(src_flat, pk);
                    } else {
                        let c = builder.ins().iconst(types::I64, red_coeff);
                        let contrib = builder.ins().imul(c, pk);
                        src_flat = builder.ins().iadd(src_flat, contrib);
                    }

                    let src_byte = builder.ins().ishl_imm(src_flat, 2);
                    let src_ptr = builder.ins().iadd(src_buf, src_byte);
                    let val =
                        builder
                            .ins()
                            .load(types::F32, MemFlags::trusted(), src_ptr, 0);

                    // dst_idx = pk * NR + lane
                    let nr_val = builder.ins().iconst(types::I64, nr as i64);
                    let pk_times_nr = builder.ins().imul(pk, nr_val);
                    let dst_idx = builder.ins().iadd_imm(pk_times_nr, lane as i64);
                    let dst_byte = builder.ins().ishl_imm(dst_idx, 2);
                    let dst_ptr = builder.ins().iadd(panel_ptr, dst_byte);
                    builder.ins().store(MemFlags::trusted(), val, dst_ptr, 0);
                }

                // pk++
                let pk = builder.use_var(pk_var);
                let pk_next = builder.ins().iadd_imm(pk, 1);
                builder.def_var(pk_var, pk_next);
                builder.ins().jump(pk_header, &[]);

                builder.switch_to_block(pk_exit);
                builder.seal_block(pk_header);
                builder.seal_block(pk_exit);
            }
        }

        // --- Non-block axis loops ---
        let mut inner_loop_blocks = Vec::new();
        for (d, &extent) in pattern.output_shape.iter().enumerate() {
            if d == block_axis {
                continue;
            }
            let zero = builder.ins().iconst(types::I64, 0);
            builder.def_var(axis_vars[d], zero);

            let header = builder.create_block();
            let body = builder.create_block();
            let exit = builder.create_block();

            builder.ins().jump(header, &[]);
            builder.switch_to_block(header);

            let iv = builder.use_var(axis_vars[d]);
            let limit = builder.ins().iconst(types::I64, extent as i64);
            let cmp = builder.ins().icmp(IntCC::SignedLessThan, iv, limit);
            builder.ins().brif(cmp, body, &[], exit, &[]);

            builder.switch_to_block(body);
            builder.seal_block(body);

            inner_loop_blocks.push((header, body, exit, d));
        }

        let identity = f64::from_bits(identity_bits) as f32;

        if use_simd {
            // ===== SIMD path: F32X4 vector accumulators =====
            let vty = types::F32X4;
            let num_vec = nr / SIMD_WIDTH;

            let vec_acc_vars: Vec<Variable> = (0..num_vec)
                .map(|_| {
                    let v = vars.next();
                    builder.declare_var(v, vty);
                    v
                })
                .collect();

            let id_scalar = builder.ins().f32const(identity);
            let id_vec = builder.ins().splat(vty, id_scalar);
            for &acc in &vec_acc_vars {
                builder.def_var(acc, id_vec);
            }

            // K-loop
            let k_var = vars.next();
            builder.declare_var(k_var, types::I64);
            let zero_k = builder.ins().iconst(types::I64, 0);
            builder.def_var(k_var, zero_k);

            let k_header = builder.create_block();
            let k_body = builder.create_block();
            let k_exit = builder.create_block();

            builder.ins().jump(k_header, &[]);
            builder.switch_to_block(k_header);

            let k = builder.use_var(k_var);
            let k_limit = builder.ins().iconst(types::I64, red.depth as i64);
            let k_cmp = builder.ins().icmp(IntCC::SignedLessThan, k, k_limit);
            builder.ins().brif(k_cmp, k_body, &[], k_exit, &[]);

            builder.switch_to_block(k_body);
            builder.seal_block(k_body);

            // Load invariant values (scalar) and splat to F32X4
            let invariant_vecs: Vec<Option<Value>> = red
                .accesses
                .iter()
                .enumerate()
                .map(|(pos, access)| {
                    if blocking.invariant[pos] {
                        let scalar = load_affine_value(
                            builder, access, layout, &axis_vars, Some(k_var),
                            red.reduction_coeffs[pos], &buf_ptrs,
                        );
                        Some(builder.ins().splat(vty, scalar))
                    } else {
                        None
                    }
                })
                .collect();

            // Process num_vec groups of 4 lanes
            for vi in 0..num_vec {
                let lane_base = vi * SIMD_WIDTH;

                let body_vals: Vec<Value> = red
                    .accesses
                    .iter()
                    .enumerate()
                    .map(|(pos, _)| {
                        if let Some(inv_vec) = invariant_vecs[pos] {
                            inv_vec
                        } else {
                            // Packed panel: vector load 4 contiguous f32
                            let panel_ptr = panel_ptrs[pos].unwrap();
                            let k_val = builder.use_var(k_var);
                            let nr_val = builder.ins().iconst(types::I64, nr as i64);
                            let k_times_nr = builder.ins().imul(k_val, nr_val);
                            let idx = builder.ins().iadd_imm(k_times_nr, lane_base as i64);
                            let byte_off = builder.ins().ishl_imm(idx, 2);
                            let ptr = builder.ins().iadd(panel_ptr, byte_off);
                            builder.ins().load(vty, MemFlags::trusted(), ptr, 0)
                        }
                    })
                    .collect();

                let term = eval_body_expr_vec(
                    builder, module, body_pattern, &body_vals, math_funcs,
                )?;
                let acc = builder.use_var(vec_acc_vars[vi]);
                let new_acc = emit_binop(builder, accum_op, acc, term);
                builder.def_var(vec_acc_vars[vi], new_acc);
            }

            // k++
            let k = builder.use_var(k_var);
            let k_next = builder.ins().iadd_imm(k, 1);
            builder.def_var(k_var, k_next);
            builder.ins().jump(k_header, &[]);

            builder.switch_to_block(k_exit);
            builder.seal_block(k_header);
            builder.seal_block(k_exit);

            // Post-reduction: extract scalar lanes, evaluate outer expression
            let out_strides = row_major_strides(&pattern.output_shape);
            let j0_post = builder.use_var(axis_vars[block_axis]);
            for vi in 0..num_vec {
                let acc_vec = builder.use_var(vec_acc_vars[vi]);
                for si in 0..SIMD_WIDTH {
                    let lane = vi * SIMD_WIDTH + si;
                    let scalar = builder.ins().extractlane(acc_vec, si as u8);

                    let j_lane = builder.ins().iadd_imm(j0_post, lane as i64);
                    builder.def_var(axis_vars[block_axis], j_lane);

                    let outer_vals =
                        load_outer_values(builder, pattern, layout, &axis_vars, &buf_ptrs);

                    let result = eval_expr_with_reduce_override(
                        builder, module, &pattern.pattern, &outer_vals, math_funcs,
                        blocking.reduce_idx, scalar,
                    )?;

                    let flat_out = compute_flat_index(builder, &axis_vars, &out_strides);
                    store_result(
                        builder, layout, pattern.output_tensor, flat_out, result, &buf_ptrs,
                    );
                }
            }
            builder.def_var(axis_vars[block_axis], j0_post);
        } else {
            // ===== Scalar path: NR individual accumulators =====
            let acc_vars: Vec<Variable> = (0..nr)
                .map(|_| {
                    let v = vars.next();
                    builder.declare_var(v, types::F32);
                    v
                })
                .collect();

            let id_val = builder.ins().f32const(identity);
            for &acc in &acc_vars {
                builder.def_var(acc, id_val);
            }

            // K-loop
            let k_var = vars.next();
            builder.declare_var(k_var, types::I64);
            let zero_k = builder.ins().iconst(types::I64, 0);
            builder.def_var(k_var, zero_k);

            let k_header = builder.create_block();
            let k_body = builder.create_block();
            let k_exit = builder.create_block();

            builder.ins().jump(k_header, &[]);
            builder.switch_to_block(k_header);

            let k = builder.use_var(k_var);
            let k_limit = builder.ins().iconst(types::I64, red.depth as i64);
            let k_cmp = builder.ins().icmp(IntCC::SignedLessThan, k, k_limit);
            builder.ins().brif(k_cmp, k_body, &[], k_exit, &[]);

            builder.switch_to_block(k_body);
            builder.seal_block(k_body);

            // Load invariant values once per k
            let j0_k = builder.use_var(axis_vars[block_axis]);
            let invariant_vals: Vec<Option<Value>> = red
                .accesses
                .iter()
                .enumerate()
                .map(|(pos, access)| {
                    if blocking.invariant[pos] {
                        Some(load_affine_value(
                            builder, access, layout, &axis_vars, Some(k_var),
                            red.reduction_coeffs[pos], &buf_ptrs,
                        ))
                    } else {
                        None
                    }
                })
                .collect();

            // Unrolled lane loop
            for lane in 0..nr {
                let j_lane = builder.ins().iadd_imm(j0_k, lane as i64);
                builder.def_var(axis_vars[block_axis], j_lane);

                let body_vals: Vec<Value> = red
                    .accesses
                    .iter()
                    .enumerate()
                    .map(|(pos, access)| {
                        if let Some(inv_val) = invariant_vals[pos] {
                            inv_val
                        } else if packable[pos] {
                            // Load from panel[k * NR + lane]
                            let panel_ptr = panel_ptrs[pos].unwrap();
                            let k_val = builder.use_var(k_var);
                            let nr_val = builder.ins().iconst(types::I64, nr as i64);
                            let k_times_nr = builder.ins().imul(k_val, nr_val);
                            let idx = builder.ins().iadd_imm(k_times_nr, lane as i64);
                            let byte_off = builder.ins().ishl_imm(idx, 2);
                            let ptr = builder.ins().iadd(panel_ptr, byte_off);
                            builder
                                .ins()
                                .load(types::F32, MemFlags::trusted(), ptr, 0)
                        } else {
                            load_affine_value(
                                builder, access, layout, &axis_vars, Some(k_var),
                                red.reduction_coeffs[pos], &buf_ptrs,
                            )
                        }
                    })
                    .collect();

                let term =
                    eval_body_expr(builder, module, body_pattern, &body_vals, math_funcs)?;
                let acc = builder.use_var(acc_vars[lane]);
                let new_acc = emit_binop(builder, accum_op, acc, term);
                builder.def_var(acc_vars[lane], new_acc);
            }

            // Restore block axis var
            builder.def_var(axis_vars[block_axis], j0_k);

            // k++
            let k = builder.use_var(k_var);
            let k_next = builder.ins().iadd_imm(k, 1);
            builder.def_var(k_var, k_next);
            builder.ins().jump(k_header, &[]);

            builder.switch_to_block(k_exit);
            builder.seal_block(k_header);
            builder.seal_block(k_exit);

            // Post-reduction: evaluate outer expression for each lane
            let out_strides = row_major_strides(&pattern.output_shape);
            let j0_post = builder.use_var(axis_vars[block_axis]);
            for lane in 0..nr {
                let j_lane = builder.ins().iadd_imm(j0_post, lane as i64);
                builder.def_var(axis_vars[block_axis], j_lane);

                let outer_vals =
                    load_outer_values(builder, pattern, layout, &axis_vars, &buf_ptrs);

                let acc_val = builder.use_var(acc_vars[lane]);
                let result = eval_expr_with_reduce_override(
                    builder, module, &pattern.pattern, &outer_vals, math_funcs,
                    blocking.reduce_idx, acc_val,
                )?;

                let flat_out = compute_flat_index(builder, &axis_vars, &out_strides);
                store_result(
                    builder, layout, pattern.output_tensor, flat_out, result, &buf_ptrs,
                );
            }
            builder.def_var(axis_vars[block_axis], j0_post);
        }

        // --- Close non-block axis loops ---
        for &(header, _body, exit, d) in inner_loop_blocks.iter().rev() {
            let iv = builder.use_var(axis_vars[d]);
            let iv_next = builder.ins().iadd_imm(iv, 1);
            builder.def_var(axis_vars[d], iv_next);
            builder.ins().jump(header, &[]);

            builder.switch_to_block(exit);
            builder.seal_block(header);
            builder.seal_block(exit);
        }

        // --- Close block axis loop ---
        let ba_iv = builder.use_var(axis_vars[block_axis]);
        let ba_next = builder.ins().iadd_imm(ba_iv, nr as i64);
        builder.def_var(axis_vars[block_axis], ba_next);
        builder.ins().jump(ba_header, &[]);

        builder.switch_to_block(ba_exit);
        builder.seal_block(ba_header);
        builder.seal_block(ba_exit);

        // --- Scalar tail ---
        if block_extent % nr != 0 {
            emit_scalar_tail(
                builder, module, pattern, layout, ptr_table, math_funcs,
                &axis_vars, &buf_ptrs, &mut vars, block_axis,
            )?;
        }

        Ok(())
    }

    /// Emit a scalar tail loop for the blocked axis remainder.
    /// All non-blocked axis loops have already exited, so we re-emit
    /// the full nested loop structure with the blocked axis covering
    /// only the tail elements.
    fn emit_scalar_tail(
        builder: &mut FunctionBuilder,
        module: &mut JITModule,
        pattern: &RecoveredPattern,
        layout: &TensorLayout,
        _ptr_table: Value,
        math_funcs: &MathFuncs,
        axis_vars: &[Variable],
        buf_ptrs: &HashMap<usize, Value>,
        vars: &mut VarAlloc,
        block_axis: usize,
    ) -> Result<(), V9Error> {
        use cranelift_codegen::ir::condcodes::IntCC;

        let _ndim = pattern.output_shape.len();

        // Re-emit loops for all axes.
        // Non-blocked axes iterate fully; blocked axis covers only the tail.
        let mut loop_blocks = Vec::new();
        for (d, &extent) in pattern.output_shape.iter().enumerate() {
            let start = if d == block_axis {
                let nr = analyze_blocking(pattern).map(|b| b.nr).unwrap_or(1);
                builder
                    .ins()
                    .iconst(types::I64, (extent - extent % nr) as i64)
            } else {
                builder.ins().iconst(types::I64, 0)
            };
            builder.def_var(axis_vars[d], start);

            let header = builder.create_block();
            let body = builder.create_block();
            let exit = builder.create_block();

            builder.ins().jump(header, &[]);
            builder.switch_to_block(header);

            let iv = builder.use_var(axis_vars[d]);
            let limit = builder.ins().iconst(types::I64, extent as i64);
            let cmp = builder.ins().icmp(IntCC::SignedLessThan, iv, limit);
            builder.ins().brif(cmp, body, &[], exit, &[]);

            builder.switch_to_block(body);
            builder.seal_block(body);

            loop_blocks.push((header, body, exit));
        }

        // Scalar body: same as unblocked emit_pattern innards
        let out_strides = row_major_strides(&pattern.output_shape);
        let flat_out = compute_flat_index(builder, axis_vars, &out_strides);
        let outer_vals = load_outer_values(builder, pattern, layout, axis_vars, buf_ptrs);
        // eval_pattern_expr takes ptr_table but only uses it for the Reduce path
        // which calls load_affine_value. buf_ptrs already has all pointers, so
        // ptr_table won't actually be dereferenced. Pass a dummy value.
        let dummy_ptr_table = builder.ins().iconst(types::I64, 0);
        let result = eval_pattern_expr(
            builder, module, &pattern.pattern, &outer_vals, axis_vars, pattern, layout,
            dummy_ptr_table, math_funcs, vars, buf_ptrs,
        )?;
        store_result(builder, layout, pattern.output_tensor, flat_out, result, buf_ptrs);

        // Close tail loops (all step by 1)
        for d in (0..loop_blocks.len()).rev() {
            let (header, _body, exit) = loop_blocks[d];
            let iv = builder.use_var(axis_vars[d]);
            let iv_next = builder.ins().iadd_imm(iv, 1);
            builder.def_var(axis_vars[d], iv_next);
            builder.ins().jump(header, &[]);

            builder.switch_to_block(exit);
            builder.seal_block(header);
            builder.seal_block(exit);
        }

        Ok(())
    }

    fn build_nested_loops(
        builder: &mut FunctionBuilder,
        axis_vars: &[Variable],
        output_shape: &[usize],
    ) -> Vec<(
        cranelift_codegen::ir::Block,
        cranelift_codegen::ir::Block,
        cranelift_codegen::ir::Block,
    )> {
        let mut blocks = Vec::new();

        for (d, &extent) in output_shape.iter().enumerate() {
            let zero = builder.ins().iconst(types::I64, 0);
            builder.def_var(axis_vars[d], zero);

            let header = builder.create_block();
            let body = builder.create_block();
            let exit = builder.create_block();

            builder.ins().jump(header, &[]);
            builder.switch_to_block(header);

            let iv = builder.use_var(axis_vars[d]);
            let limit = builder.ins().iconst(types::I64, extent as i64);
            let cmp = builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::SignedLessThan,
                iv,
                limit,
            );
            builder.ins().brif(cmp, body, &[], exit, &[]);

            builder.switch_to_block(body);
            builder.seal_block(body);

            blocks.push((header, body, exit));
        }

        blocks
    }

    fn close_nested_loops(
        builder: &mut FunctionBuilder,
        axis_vars: &[Variable],
        blocks: &[(
            cranelift_codegen::ir::Block,
            cranelift_codegen::ir::Block,
            cranelift_codegen::ir::Block,
        )],
    ) {
        for d in (0..blocks.len()).rev() {
            let (header, _body, exit) = blocks[d];
            let iv = builder.use_var(axis_vars[d]);
            let iv_next = builder.ins().iadd_imm(iv, 1);
            builder.def_var(axis_vars[d], iv_next);
            builder.ins().jump(header, &[]);

            builder.switch_to_block(exit);
            builder.seal_block(header);
            builder.seal_block(exit);
        }
    }

    fn compute_flat_index(
        builder: &mut FunctionBuilder,
        axis_vars: &[Variable],
        strides: &[usize],
    ) -> Value {
        let mut flat = builder.ins().iconst(types::I64, 0);
        for (d, &stride) in strides.iter().enumerate() {
            let iv = builder.use_var(axis_vars[d]);
            let s = builder.ins().iconst(types::I64, stride as i64);
            let contrib = builder.ins().imul(iv, s);
            flat = builder.ins().iadd(flat, contrib);
        }
        flat
    }

    /// Pre-load buffer base pointers from the pointer table.
    /// Returns a map from tensor slot index to the IR Value of the buffer pointer.
    /// This should be called once before any loops so pointers are hoisted.
    fn preload_buffer_ptrs(
        builder: &mut FunctionBuilder,
        layout: &TensorLayout,
        ptr_table: Value,
        tensors: &[GlobalId],
    ) -> HashMap<usize, Value> {
        let mut ptrs = HashMap::new();
        for &t in tensors {
            let slot = layout.tensor_index[&t];
            if ptrs.contains_key(&slot) {
                continue;
            }
            let slot_offset = builder.ins().iconst(types::I64, (slot * 8) as i64);
            let buf_ptr_ptr = builder.ins().iadd(ptr_table, slot_offset);
            let buf_ptr =
                builder
                    .ins()
                    .load(types::I64, MemFlags::trusted(), buf_ptr_ptr, 0);
            ptrs.insert(slot, buf_ptr);
        }
        ptrs
    }

    fn load_outer_values(
        builder: &mut FunctionBuilder,
        pattern: &RecoveredPattern,
        layout: &TensorLayout,
        axis_vars: &[Variable],
        buf_ptrs: &HashMap<usize, Value>,
    ) -> Vec<Value> {
        pattern
            .accesses
            .iter()
            .map(|access| load_affine_value(builder, access, layout, axis_vars, None, 0, buf_ptrs))
            .collect()
    }

    fn load_reduction_values(
        builder: &mut FunctionBuilder,
        accesses: &[AffineAccess],
        reduction_coeffs: &[i64],
        layout: &TensorLayout,
        axis_vars: &[Variable],
        k_var: Variable,
        buf_ptrs: &HashMap<usize, Value>,
    ) -> Vec<Value> {
        accesses
            .iter()
            .enumerate()
            .map(|(pos, access)| {
                load_affine_value(
                    builder,
                    access,
                    layout,
                    axis_vars,
                    Some(k_var),
                    reduction_coeffs[pos],
                    buf_ptrs,
                )
            })
            .collect()
    }

    fn load_affine_value(
        builder: &mut FunctionBuilder,
        access: &AffineAccess,
        layout: &TensorLayout,
        axis_vars: &[Variable],
        k_var: Option<Variable>,
        red_coeff: i64,
        buf_ptrs: &HashMap<usize, Value>,
    ) -> Value {
        let slot = layout.tensor_index[&access.tensor];
        let buf_ptr = buf_ptrs[&slot];

        let mut flat_idx = builder.ins().iconst(types::I64, access.base);
        for (d, &coeff) in access.axis_coeffs.iter().enumerate() {
            if coeff != 0 {
                let iv = builder.use_var(axis_vars[d]);
                if coeff == 1 {
                    flat_idx = builder.ins().iadd(flat_idx, iv);
                } else {
                    let c = builder.ins().iconst(types::I64, coeff);
                    let contrib = builder.ins().imul(c, iv);
                    flat_idx = builder.ins().iadd(flat_idx, contrib);
                }
            }
        }
        if let Some(kv) = k_var {
            if red_coeff != 0 {
                let k = builder.use_var(kv);
                if red_coeff == 1 {
                    flat_idx = builder.ins().iadd(flat_idx, k);
                } else {
                    let rc = builder.ins().iconst(types::I64, red_coeff);
                    let contrib = builder.ins().imul(rc, k);
                    flat_idx = builder.ins().iadd(flat_idx, contrib);
                }
            }
        }

        let byte_offset = builder.ins().ishl_imm(flat_idx, 2);
        let elem_ptr = builder.ins().iadd(buf_ptr, byte_offset);
        builder
            .ins()
            .load(types::F32, MemFlags::trusted(), elem_ptr, 0)
    }

    fn store_result(
        builder: &mut FunctionBuilder,
        layout: &TensorLayout,
        tensor: GlobalId,
        flat_out: Value,
        value: Value,
        buf_ptrs: &HashMap<usize, Value>,
    ) {
        let slot = layout.tensor_index[&tensor];
        let ptr = buf_ptrs[&slot];
        let byte_offset = builder.ins().ishl_imm(flat_out, 2);
        let elem_ptr = builder.ins().iadd(ptr, byte_offset);
        builder
            .ins()
            .store(MemFlags::trusted(), value, elem_ptr, 0);
    }

    fn row_major_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    // ------------------------------------------------------------------
    // Expression evaluation
    // ------------------------------------------------------------------

    fn eval_pattern_expr(
        builder: &mut FunctionBuilder,
        module: &mut JITModule,
        pattern: &PatternExpr,
        outer_load_vals: &[Value],
        axis_vars: &[Variable],
        recovered: &RecoveredPattern,
        layout: &TensorLayout,
        ptr_table: Value,
        math_funcs: &MathFuncs,
        vars: &mut VarAlloc,
        buf_ptrs: &HashMap<usize, Value>,
    ) -> Result<Value, V9Error> {
        match pattern {
            PatternExpr::Load { load_ordinal, .. } => Ok(outer_load_vals[*load_ordinal]),
            PatternExpr::Literal { value_bits } => {
                let val = f64::from_bits(*value_bits) as f32;
                Ok(builder.ins().f32const(val))
            }
            PatternExpr::Binary { op, a, b } => {
                let va = eval_pattern_expr(
                    builder, module, a, outer_load_vals, axis_vars, recovered,
                    layout, ptr_table, math_funcs, vars, buf_ptrs,
                )?;
                let vb = eval_pattern_expr(
                    builder, module, b, outer_load_vals, axis_vars, recovered,
                    layout, ptr_table, math_funcs, vars, buf_ptrs,
                )?;
                Ok(emit_binop(builder, *op, va, vb))
            }
            PatternExpr::Unary { op, input } => {
                let vi = eval_pattern_expr(
                    builder, module, input, outer_load_vals, axis_vars, recovered,
                    layout, ptr_table, math_funcs, vars, buf_ptrs,
                )?;
                emit_unaryop(builder, module, *op, vi, math_funcs)
            }
            PatternExpr::Reduce {
                accum_op,
                identity_bits,
                body,
                reduce_idx,
            } => {
                let red = &recovered.reductions[*reduce_idx];

                let acc_var = vars.next();
                builder.declare_var(acc_var, types::F32);
                let k_var = vars.next();
                builder.declare_var(k_var, types::I64);

                // Initialize accumulator
                let identity = f64::from_bits(*identity_bits) as f32;
                let id_val = builder.ins().f32const(identity);
                builder.def_var(acc_var, id_val);

                // k = 0
                let zero_k = builder.ins().iconst(types::I64, 0);
                builder.def_var(k_var, zero_k);

                let k_header = builder.create_block();
                let k_body = builder.create_block();
                let k_exit = builder.create_block();

                builder.ins().jump(k_header, &[]);
                builder.switch_to_block(k_header);

                let k = builder.use_var(k_var);
                let k_limit = builder.ins().iconst(types::I64, red.depth as i64);
                let k_cmp = builder.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::SignedLessThan,
                    k,
                    k_limit,
                );
                builder.ins().brif(k_cmp, k_body, &[], k_exit, &[]);

                builder.switch_to_block(k_body);
                builder.seal_block(k_body);

                // Load reduction values for this k
                let body_vals = load_reduction_values(
                    builder,
                    &red.accesses,
                    &red.reduction_coeffs,
                    layout,
                    axis_vars,
                    k_var,
                    buf_ptrs,
                );

                // Evaluate body (no Reduce nodes expected in body for now)
                let term = eval_body_expr(builder, module, body, &body_vals, math_funcs)?;

                // Accumulate
                let acc = builder.use_var(acc_var);
                let new_acc = emit_binop(builder, *accum_op, acc, term);
                builder.def_var(acc_var, new_acc);

                // k++
                let k = builder.use_var(k_var);
                let k_next = builder.ins().iadd_imm(k, 1);
                builder.def_var(k_var, k_next);
                builder.ins().jump(k_header, &[]);

                builder.switch_to_block(k_exit);
                builder.seal_block(k_header);
                builder.seal_block(k_exit);

                Ok(builder.use_var(acc_var))
            }
        }
    }

    /// Evaluate a reduction body expression (no Reduce node support).
    fn eval_body_expr(
        builder: &mut FunctionBuilder,
        module: &mut JITModule,
        pattern: &PatternExpr,
        load_vals: &[Value],
        math_funcs: &MathFuncs,
    ) -> Result<Value, V9Error> {
        match pattern {
            PatternExpr::Load { load_ordinal, .. } => Ok(load_vals[*load_ordinal]),
            PatternExpr::Literal { value_bits } => {
                let val = f64::from_bits(*value_bits) as f32;
                Ok(builder.ins().f32const(val))
            }
            PatternExpr::Binary { op, a, b } => {
                let va = eval_body_expr(builder, module, a, load_vals, math_funcs)?;
                let vb = eval_body_expr(builder, module, b, load_vals, math_funcs)?;
                Ok(emit_binop(builder, *op, va, vb))
            }
            PatternExpr::Unary { op, input } => {
                let vi = eval_body_expr(builder, module, input, load_vals, math_funcs)?;
                emit_unaryop(builder, module, *op, vi, math_funcs)
            }
            PatternExpr::Reduce { .. } => {
                Err(V9Error::Other("Nested reductions not yet supported".into()))
            }
        }
    }

    /// Vector version of `eval_body_expr`. All load values are F32X4.
    /// Literals are splatted to F32X4. Only Neg is supported as a unary op
    /// (callers must verify with `body_has_func_calls` before using this).
    fn eval_body_expr_vec(
        builder: &mut FunctionBuilder,
        module: &mut JITModule,
        pattern: &PatternExpr,
        load_vals: &[Value],
        math_funcs: &MathFuncs,
    ) -> Result<Value, V9Error> {
        match pattern {
            PatternExpr::Load { load_ordinal, .. } => Ok(load_vals[*load_ordinal]),
            PatternExpr::Literal { value_bits } => {
                let val = f64::from_bits(*value_bits) as f32;
                let scalar = builder.ins().f32const(val);
                Ok(builder.ins().splat(types::F32X4, scalar))
            }
            PatternExpr::Binary { op, a, b } => {
                let va = eval_body_expr_vec(builder, module, a, load_vals, math_funcs)?;
                let vb = eval_body_expr_vec(builder, module, b, load_vals, math_funcs)?;
                Ok(emit_binop(builder, *op, va, vb))
            }
            PatternExpr::Unary { op, input } => {
                let vi = eval_body_expr_vec(builder, module, input, load_vals, math_funcs)?;
                match op {
                    ScalarUnaryOp::Neg => Ok(builder.ins().fneg(vi)),
                    _ => Err(V9Error::Other(
                        "SIMD body contains unsupported unary op".into(),
                    )),
                }
            }
            PatternExpr::Reduce { .. } => {
                Err(V9Error::Other("Nested reductions not yet supported".into()))
            }
        }
    }

    fn emit_binop(builder: &mut FunctionBuilder, op: ScalarBinOp, a: Value, b: Value) -> Value {
        match op {
            ScalarBinOp::Add => builder.ins().fadd(a, b),
            ScalarBinOp::Sub => builder.ins().fsub(a, b),
            ScalarBinOp::Mul => builder.ins().fmul(a, b),
            ScalarBinOp::Div => builder.ins().fdiv(a, b),
            ScalarBinOp::Max => builder.ins().fmax(a, b),
            ScalarBinOp::Min => builder.ins().fmin(a, b),
        }
    }

    fn emit_unaryop(
        builder: &mut FunctionBuilder,
        module: &mut JITModule,
        op: ScalarUnaryOp,
        input: Value,
        math_funcs: &MathFuncs,
    ) -> Result<Value, V9Error> {
        let mut call_math = |builder: &mut FunctionBuilder, func_id: cranelift_module::FuncId| {
            let func_ref = module.declare_func_in_func(func_id, builder.func);
            let call = builder.ins().call(func_ref, &[input]);
            builder.inst_results(call)[0]
        };

        Ok(match op {
            ScalarUnaryOp::Neg => builder.ins().fneg(input),
            ScalarUnaryOp::Abs => call_math(builder, math_funcs.fabsf),
            ScalarUnaryOp::Exp => call_math(builder, math_funcs.expf),
            ScalarUnaryOp::Ln => call_math(builder, math_funcs.logf),
            ScalarUnaryOp::Sqrt => call_math(builder, math_funcs.sqrtf),
            ScalarUnaryOp::Reciprocal => {
                let one = builder.ins().f32const(1.0);
                builder.ins().fdiv(one, input)
            }
            ScalarUnaryOp::Tanh => call_math(builder, math_funcs.tanhf),
            ScalarUnaryOp::Floor => call_math(builder, math_funcs.floorf),
            ScalarUnaryOp::Ceil => call_math(builder, math_funcs.ceilf),
            ScalarUnaryOp::Erf => call_math(builder, math_funcs.erff),
        })
    }

    // ------------------------------------------------------------------
    // Math intrinsics
    // ------------------------------------------------------------------

    struct MathFuncs {
        expf: cranelift_module::FuncId,
        sqrtf: cranelift_module::FuncId,
        tanhf: cranelift_module::FuncId,
        logf: cranelift_module::FuncId,
        floorf: cranelift_module::FuncId,
        ceilf: cranelift_module::FuncId,
        fabsf: cranelift_module::FuncId,
        erff: cranelift_module::FuncId,
    }

    fn declare_math_funcs(module: &mut JITModule) -> Result<MathFuncs, V9Error> {
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::F32));
        sig.returns.push(AbiParam::new(types::F32));

        let d = |module: &mut JITModule,
                 name: &str,
                 sig: &cranelift_codegen::ir::Signature|
         -> Result<cranelift_module::FuncId, V9Error> {
            module
                .declare_function(name, Linkage::Import, sig)
                .map_err(|e| V9Error::Other(format!("declare {}: {}", name, e)))
        };

        Ok(MathFuncs {
            expf: d(module, "wt9_expf", &sig)?,
            sqrtf: d(module, "wt9_sqrtf", &sig)?,
            tanhf: d(module, "wt9_tanhf", &sig)?,
            logf: d(module, "wt9_logf", &sig)?,
            floorf: d(module, "wt9_floorf", &sig)?,
            ceilf: d(module, "wt9_ceilf", &sig)?,
            fabsf: d(module, "wt9_fabsf", &sig)?,
            erff: d(module, "wt9_erff", &sig)?,
        })
    }

    extern "C" fn wt9_expf(x: f32) -> f32 {
        x.exp()
    }
    extern "C" fn wt9_sqrtf(x: f32) -> f32 {
        x.sqrt()
    }
    extern "C" fn wt9_tanhf(x: f32) -> f32 {
        x.tanh()
    }
    extern "C" fn wt9_logf(x: f32) -> f32 {
        x.ln()
    }
    extern "C" fn wt9_floorf(x: f32) -> f32 {
        x.floor()
    }
    extern "C" fn wt9_ceilf(x: f32) -> f32 {
        x.ceil()
    }
    extern "C" fn wt9_fabsf(x: f32) -> f32 {
        x.abs()
    }
    extern "C" fn wt9_erff(x: f32) -> f32 {
        let a1: f32 = 0.254829592;
        let a2: f32 = -0.284496736;
        let a3: f32 = 1.421413741;
        let a4: f32 = -1.453152027;
        let a5: f32 = 1.061405429;
        let p: f32 = 0.3275911;
        let sign = if x < 0.0 { -1.0f32 } else { 1.0f32 };
        let x = x.abs();
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
        sign * y
    }
}
