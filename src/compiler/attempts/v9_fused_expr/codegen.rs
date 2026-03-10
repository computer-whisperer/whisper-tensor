//! v9 codegen: single-threaded Cranelift JIT from recovered patterns.
//!
//! Emits nested loops matching the output shape. Reduce nodes in the pattern
//! tree emit their own inner loops inline. No SIMD, no tiling — correctness first.

#[cfg(feature = "cranelift")]
pub mod jit {
    use crate::compiler::attempts::v1_scalar_crystal::codegen::{CodegenError, TensorLayout};
    use crate::compiler::attempts::v9_fused_expr::pattern::*;
    use crate::compiler::common::v2_frontend::{ScalarBinOp, ScalarUnaryOp};
    use crate::graph::GlobalId;
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
        let ndim = pattern.output_shape.len();
        let mut vars = VarAlloc::new();

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
        let outer_vals = load_outer_values(builder, pattern, layout, ptr_table, &axis_vars);

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
        )?;

        // Store result
        store_result(
            builder,
            layout,
            ptr_table,
            pattern.output_tensor,
            flat_out,
            result,
        );

        // Close nested loops
        close_nested_loops(builder, &axis_vars, &loop_blocks);

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

    fn load_outer_values(
        builder: &mut FunctionBuilder,
        pattern: &RecoveredPattern,
        layout: &TensorLayout,
        ptr_table: Value,
        axis_vars: &[Variable],
    ) -> Vec<Value> {
        pattern
            .accesses
            .iter()
            .map(|access| load_affine_value(builder, access, layout, ptr_table, axis_vars, None, 0))
            .collect()
    }

    fn load_reduction_values(
        builder: &mut FunctionBuilder,
        accesses: &[AffineAccess],
        reduction_coeffs: &[i64],
        layout: &TensorLayout,
        ptr_table: Value,
        axis_vars: &[Variable],
        k_var: Variable,
    ) -> Vec<Value> {
        accesses
            .iter()
            .enumerate()
            .map(|(pos, access)| {
                load_affine_value(
                    builder,
                    access,
                    layout,
                    ptr_table,
                    axis_vars,
                    Some(k_var),
                    reduction_coeffs[pos],
                )
            })
            .collect()
    }

    fn load_affine_value(
        builder: &mut FunctionBuilder,
        access: &AffineAccess,
        layout: &TensorLayout,
        ptr_table: Value,
        axis_vars: &[Variable],
        k_var: Option<Variable>,
        red_coeff: i64,
    ) -> Value {
        let slot = layout.tensor_index[&access.tensor];
        let slot_offset = builder.ins().iconst(types::I64, (slot * 8) as i64);
        let buf_ptr_ptr = builder.ins().iadd(ptr_table, slot_offset);
        let buf_ptr =
            builder
                .ins()
                .load(types::I64, MemFlags::trusted(), buf_ptr_ptr, 0);

        let mut flat_idx = builder.ins().iconst(types::I64, access.base);
        for (d, &coeff) in access.axis_coeffs.iter().enumerate() {
            if coeff != 0 {
                let iv = builder.use_var(axis_vars[d]);
                let c = builder.ins().iconst(types::I64, coeff);
                let contrib = builder.ins().imul(c, iv);
                flat_idx = builder.ins().iadd(flat_idx, contrib);
            }
        }
        if let Some(kv) = k_var {
            if red_coeff != 0 {
                let k = builder.use_var(kv);
                let rc = builder.ins().iconst(types::I64, red_coeff);
                let contrib = builder.ins().imul(rc, k);
                flat_idx = builder.ins().iadd(flat_idx, contrib);
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
        ptr_table: Value,
        tensor: GlobalId,
        flat_out: Value,
        value: Value,
    ) {
        let slot = layout.tensor_index[&tensor];
        let slot_offset = builder.ins().iconst(types::I64, (slot * 8) as i64);
        let ptr_ptr = builder.ins().iadd(ptr_table, slot_offset);
        let ptr = builder
            .ins()
            .load(types::I64, MemFlags::trusted(), ptr_ptr, 0);
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
    ) -> Result<Value, V9Error> {
        match pattern {
            PatternExpr::Load { load_ordinal, .. } => Ok(outer_load_vals[*load_ordinal]),
            PatternExpr::Literal { value_bits } => {
                let val = f64::from_bits(*value_bits) as f32;
                Ok(builder.ins().f32const(val))
            }
            PatternExpr::Binary { op, a, b } => {
                let va = eval_pattern_expr(
                    builder,
                    module,
                    a,
                    outer_load_vals,
                    axis_vars,
                    recovered,
                    layout,
                    ptr_table,
                    math_funcs,
                    vars,
                )?;
                let vb = eval_pattern_expr(
                    builder,
                    module,
                    b,
                    outer_load_vals,
                    axis_vars,
                    recovered,
                    layout,
                    ptr_table,
                    math_funcs,
                    vars,
                )?;
                Ok(emit_binop(builder, *op, va, vb))
            }
            PatternExpr::Unary { op, input } => {
                let vi = eval_pattern_expr(
                    builder,
                    module,
                    input,
                    outer_load_vals,
                    axis_vars,
                    recovered,
                    layout,
                    ptr_table,
                    math_funcs,
                    vars,
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
                    ptr_table,
                    axis_vars,
                    k_var,
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
