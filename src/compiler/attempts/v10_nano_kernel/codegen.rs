#![allow(clippy::all, dead_code, unreachable_patterns, unused)]
//! Cranelift JIT codegen for v10 kernel plans.
//!
//! Each KernelPlan becomes a native function:
//!   fn(ptr_table: *const *mut f32, start: i64, end: i64)
//!
//! The function iterates `i` from `start` to `end`, executing the kernel's
//! ops for each element and storing the result.

#[cfg(feature = "cranelift")]
pub mod jit {
    use crate::compiler::attempts::v10_nano_kernel::plan::*;
    use crate::dtype::DType;
    use crate::nano_graph::{ScalarBinOp, ScalarUnaryOp};
    use cranelift_codegen::ir::types;
    use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Value};
    use cranelift_codegen::settings::{self, Configurable};
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::{Linkage, Module};

    // ---- Math function wrappers ----
    extern "C" fn wt10_expf(x: f32) -> f32 {
        x.exp()
    }
    extern "C" fn wt10_logf(x: f32) -> f32 {
        x.ln()
    }
    extern "C" fn wt10_tanhf(x: f32) -> f32 {
        x.tanh()
    }
    extern "C" fn wt10_sqrtf(x: f32) -> f32 {
        x.sqrt()
    }
    extern "C" fn wt10_floorf(x: f32) -> f32 {
        x.floor()
    }
    extern "C" fn wt10_ceilf(x: f32) -> f32 {
        x.ceil()
    }
    extern "C" fn wt10_fabsf(x: f32) -> f32 {
        x.abs()
    }
    extern "C" fn wt10_powf(x: f32, y: f32) -> f32 {
        x.powf(y)
    }
    extern "C" fn wt10_fmodf(x: f32, y: f32) -> f32 {
        x % y
    }

    // ---- Error type ----

    #[derive(Debug, thiserror::Error)]
    pub enum V10Error {
        #[error("v10 codegen: {0}")]
        Codegen(String),
    }

    // ---- Compiled types ----

    pub struct CompiledKernel {
        func_ptr: *const u8,
        pub output_buffer: BufferId,
        pub input_buffers: Vec<BufferId>,
        pub extent: u32,
    }

    impl CompiledKernel {
        /// # Safety
        pub unsafe fn execute(&self, buffers: *const *mut f32) {
            unsafe { self.execute_range(buffers, 0, self.extent as usize) };
        }

        /// # Safety
        pub unsafe fn execute_range(&self, buffers: *const *mut f32, start: usize, end: usize) {
            let func: unsafe extern "C" fn(*const *mut f32, i64, i64) =
                unsafe { std::mem::transmute(self.func_ptr) };
            unsafe { func(buffers, start as i64, end as i64) };
        }
    }

    unsafe impl Send for CompiledKernel {}
    unsafe impl Sync for CompiledKernel {}

    pub struct CompiledGraph {
        _module: JITModule,
        pub kernels: Vec<CompiledKernel>,
        pub num_buffers: usize,
    }

    unsafe impl Send for CompiledGraph {}
    unsafe impl Sync for CompiledGraph {}

    // ---- Math function declarations ----

    struct MathFuncs {
        expf: cranelift_module::FuncId,
        logf: cranelift_module::FuncId,
        tanhf: cranelift_module::FuncId,
        sqrtf: cranelift_module::FuncId,
        floorf: cranelift_module::FuncId,
        ceilf: cranelift_module::FuncId,
        fabsf: cranelift_module::FuncId,
        powf: cranelift_module::FuncId,
        fmodf: cranelift_module::FuncId,
    }

    fn declare_math_funcs(module: &mut JITModule) -> Result<MathFuncs, V10Error> {
        let mut sig1 = module.make_signature();
        sig1.params.push(AbiParam::new(types::F32));
        sig1.returns.push(AbiParam::new(types::F32));

        let mut sig2 = module.make_signature();
        sig2.params.push(AbiParam::new(types::F32));
        sig2.params.push(AbiParam::new(types::F32));
        sig2.returns.push(AbiParam::new(types::F32));

        let decl = |module: &mut JITModule, name: &str, sig: &cranelift_codegen::ir::Signature| {
            module
                .declare_function(name, Linkage::Import, sig)
                .map_err(|e| V10Error::Codegen(format!("declare {}: {}", name, e)))
        };

        Ok(MathFuncs {
            expf: decl(module, "wt10_expf", &sig1)?,
            logf: decl(module, "wt10_logf", &sig1)?,
            tanhf: decl(module, "wt10_tanhf", &sig1)?,
            sqrtf: decl(module, "wt10_sqrtf", &sig1)?,
            floorf: decl(module, "wt10_floorf", &sig1)?,
            ceilf: decl(module, "wt10_ceilf", &sig1)?,
            fabsf: decl(module, "wt10_fabsf", &sig1)?,
            powf: decl(module, "wt10_powf", &sig2)?,
            fmodf: decl(module, "wt10_fmodf", &sig2)?,
        })
    }

    // ---- Compilation ----

    pub fn compile(plan: &CompilationPlan) -> Result<CompiledGraph, V10Error> {
        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed").unwrap();
        let isa_builder = cranelift_native::builder()
            .map_err(|e| V10Error::Codegen(format!("cranelift native ISA: {}", e)))?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| V10Error::Codegen(format!("ISA finish: {}", e)))?;

        let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        jit_builder.symbol("wt10_expf", wt10_expf as *const u8);
        jit_builder.symbol("wt10_logf", wt10_logf as *const u8);
        jit_builder.symbol("wt10_tanhf", wt10_tanhf as *const u8);
        jit_builder.symbol("wt10_sqrtf", wt10_sqrtf as *const u8);
        jit_builder.symbol("wt10_floorf", wt10_floorf as *const u8);
        jit_builder.symbol("wt10_ceilf", wt10_ceilf as *const u8);
        jit_builder.symbol("wt10_fabsf", wt10_fabsf as *const u8);
        jit_builder.symbol("wt10_powf", wt10_powf as *const u8);
        jit_builder.symbol("wt10_fmodf", wt10_fmodf as *const u8);

        let mut module = JITModule::new(jit_builder);
        let mut ctx = module.make_context();
        let mut func_ctx = FunctionBuilderContext::new();

        let mut kernel_infos = Vec::new();
        let mut table_counter: usize = 0;
        let total_kernels = plan.kernels.len();
        let compile_start = std::time::Instant::now();

        for (ki, kplan) in plan.kernels.iter().enumerate() {
            if ki > 0 && ki % 100_000 == 0 {
                let elapsed = compile_start.elapsed().as_secs_f64();
                let rate = ki as f64 / elapsed;
                let eta = (total_kernels - ki) as f64 / rate;
                eprintln!(
                    "[v10] compiling kernel {}/{} ({:.0}/s, ETA {:.0}s)",
                    ki, total_kernels, rate, eta
                );
            }
            let func_name = format!("v10_k{}", ki);

            ctx.func.signature.params.push(AbiParam::new(types::I64)); // ptr_table
            ctx.func.signature.params.push(AbiParam::new(types::I64)); // start
            ctx.func.signature.params.push(AbiParam::new(types::I64)); // end

            let func_id = module
                .declare_function(&func_name, Linkage::Local, &ctx.func.signature)
                .map_err(|e| V10Error::Codegen(format!("declare: {}", e)))?;

            let math_funcs = declare_math_funcs(&mut module)?;

            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);
            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);

            let ptr_table = builder.block_params(entry)[0];
            let start = builder.block_params(entry)[1];
            let end = builder.block_params(entry)[2];

            emit_kernel(
                &mut builder,
                &mut module,
                kplan,
                ptr_table,
                start,
                end,
                &math_funcs,
                &mut table_counter,
            )?;

            builder.ins().return_(&[]);
            builder.finalize();

            module
                .define_function(func_id, &mut ctx)
                .map_err(|e| V10Error::Codegen(format!("define: {}", e)))?;
            ctx.clear();

            kernel_infos.push((
                func_id,
                kplan.output,
                kplan.input_buffers.clone(),
                kplan.extent,
            ));
        }

        module
            .finalize_definitions()
            .map_err(|e| V10Error::Codegen(format!("finalize: {}", e)))?;

        let kernels: Vec<CompiledKernel> = kernel_infos
            .into_iter()
            .map(|(func_id, output, inputs, extent)| {
                let func_ptr = module.get_finalized_function(func_id);
                CompiledKernel {
                    func_ptr,
                    output_buffer: output,
                    input_buffers: inputs,
                    extent,
                }
            })
            .collect();

        Ok(CompiledGraph {
            _module: module,
            kernels,
            num_buffers: plan.buffers.len(),
        })
    }

    // ---- Kernel emission ----

    fn emit_kernel(
        builder: &mut FunctionBuilder,
        module: &mut JITModule,
        kplan: &KernelPlan,
        ptr_table: Value,
        start: Value,
        end: Value,
        math: &MathFuncs,
        table_counter: &mut usize,
    ) -> Result<(), V10Error> {
        // Create the loop: for i in start..end
        let loop_header = builder.create_block();
        let loop_body = builder.create_block();
        let loop_exit = builder.create_block();

        builder.ins().jump(loop_header, &[start]);

        // Loop header: i = phi(start, i+1)
        builder.switch_to_block(loop_header);
        builder.append_block_param(loop_header, types::I64);
        let i_val = builder.block_params(loop_header)[0];

        let cmp = builder.ins().icmp(
            cranelift_codegen::ir::condcodes::IntCC::SignedLessThan,
            i_val,
            end,
        );
        builder.ins().brif(cmp, loop_body, &[], loop_exit, &[]);

        builder.switch_to_block(loop_body);

        // Emit the kernel ops.
        let mut vreg_vals: Vec<Value> = Vec::with_capacity(kplan.ops.len());

        for op in &kplan.ops {
            let val = emit_kop(
                builder,
                module,
                op,
                &vreg_vals,
                &kplan.reductions,
                ptr_table,
                i_val,
                math,
                table_counter,
            )?;
            vreg_vals.push(val);
        }

        // Store result.
        let result_val = vreg_vals[kplan.result.0 as usize];
        let out_buf_ptr = load_buffer_ptr(builder, ptr_table, kplan.output);
        // Store at buf[output_offset + i]
        let store_idx = if kplan.output_offset == 0 {
            i_val
        } else {
            builder.ins().iadd_imm(i_val, kplan.output_offset as i64)
        };
        let byte_offset = builder.ins().imul_imm(store_idx, 4);
        let addr = builder.ins().iadd(out_buf_ptr, byte_offset);
        builder.ins().store(MemFlags::new(), result_val, addr, 0);

        // Increment and loop.
        let i_next = builder.ins().iadd_imm(i_val, 1);
        builder.ins().jump(loop_header, &[i_next]);

        builder.switch_to_block(loop_exit);
        builder.seal_block(loop_header);
        builder.seal_block(loop_body);
        builder.seal_block(loop_exit);

        Ok(())
    }

    fn emit_kop(
        builder: &mut FunctionBuilder,
        module: &mut JITModule,
        op: &KOp,
        vreg_vals: &[Value],
        reductions: &[KernelReduction],
        ptr_table: Value,
        i_val: Value,
        math: &MathFuncs,
        table_counter: &mut usize,
    ) -> Result<Value, V10Error> {
        match op {
            KOp::Load {
                buffer,
                base_offset,
                stride,
            } => {
                let buf_ptr = load_buffer_ptr(builder, ptr_table, *buffer);
                // addr = buf_ptr + (base_offset + stride * i) * 4
                let idx = if *stride == 1 && *base_offset == 0 {
                    i_val
                } else if *stride == 1 {
                    builder.ins().iadd_imm(i_val, *base_offset as i64)
                } else {
                    let si = builder.ins().imul_imm(i_val, *stride as i64);
                    builder.ins().iadd_imm(si, *base_offset as i64)
                };
                let byte_offset = builder.ins().imul_imm(idx, 4);
                let addr = builder.ins().iadd(buf_ptr, byte_offset);
                Ok(builder.ins().load(types::F32, MemFlags::new(), addr, 0))
            }

            KOp::ModLoad { buffer, modulus } => {
                let buf_ptr = load_buffer_ptr(builder, ptr_table, *buffer);
                // idx = i % modulus
                let modulus_val = builder.ins().iconst(types::I64, *modulus as i64);
                // Use unsigned remainder (srem would work too since i >= 0)
                let idx = builder.ins().urem(i_val, modulus_val);
                let byte_offset = builder.ins().imul_imm(idx, 4);
                let addr = builder.ins().iadd(buf_ptr, byte_offset);
                Ok(builder.ins().load(types::F32, MemFlags::new(), addr, 0))
            }

            KOp::TableLoad { buffer, offsets } => {
                let buf_ptr = load_buffer_ptr(builder, ptr_table, *buffer);
                if offsets.is_empty() {
                    return Ok(builder.ins().f32const(0.0));
                }
                if offsets.len() == 1 {
                    let byte_off = offsets[0] as i64 * 4;
                    return Ok(builder.ins().load(
                        types::F32,
                        MemFlags::new(),
                        buf_ptr,
                        byte_off as i32,
                    ));
                }

                // Store offset table as a data section in the JIT module.
                let data_name = format!("v10_tbl_{}", *table_counter);
                *table_counter += 1;
                let data_id = module
                    .declare_data(&data_name, Linkage::Local, false, false)
                    .map_err(|e| V10Error::Codegen(format!("declare table: {}", e)))?;
                let mut data_desc = cranelift_module::DataDescription::new();
                let bytes: Vec<u8> = offsets
                    .iter()
                    .flat_map(|&o| (o as u32).to_le_bytes())
                    .collect();
                data_desc.define(bytes.into_boxed_slice());
                module
                    .define_data(data_id, &data_desc)
                    .map_err(|e| V10Error::Codegen(format!("define table: {}", e)))?;

                let gv = module.declare_data_in_func(data_id, builder.func);
                let table_ptr = builder.ins().global_value(types::I64, gv);

                // Load offset at position i_val.
                let idx_byte_off = builder.ins().imul_imm(i_val, 4);
                let idx_addr = builder.ins().iadd(table_ptr, idx_byte_off);
                let raw_offset = builder.ins().load(types::I32, MemFlags::new(), idx_addr, 0);
                let offset_i64 = builder.ins().uextend(types::I64, raw_offset);

                // Load from data buffer at the offset.
                let data_byte_off = builder.ins().imul_imm(offset_i64, 4);
                let data_addr = builder.ins().iadd(buf_ptr, data_byte_off);
                Ok(builder
                    .ins()
                    .load(types::F32, MemFlags::new(), data_addr, 0))
            }

            KOp::BroadcastLoad { buffer, offset } => {
                let buf_ptr = load_buffer_ptr(builder, ptr_table, *buffer);
                let byte_offset = (*offset as i64) * 4;
                Ok(builder
                    .ins()
                    .load(types::F32, MemFlags::new(), buf_ptr, byte_offset as i32))
            }

            KOp::Literal(val) => Ok(builder.ins().f32const(*val)),

            KOp::Binary {
                op,
                a,
                b,
                compute_dt,
            } => {
                let a_val = vreg_vals[a.0 as usize];
                let b_val = vreg_vals[b.0 as usize];
                emit_binop(builder, module, math, *op, a_val, b_val)
            }

            KOp::Unary { op, x, compute_dt } => {
                let x_val = vreg_vals[x.0 as usize];
                emit_unop(builder, module, math, *op, x_val)
            }

            KOp::Select {
                cond,
                x,
                y,
                compute_dt,
            } => {
                let c = vreg_vals[cond.0 as usize];
                let xv = vreg_vals[x.0 as usize];
                let yv = vreg_vals[y.0 as usize];
                let zero = builder.ins().f32const(0.0);
                let is_nonzero = builder.ins().fcmp(
                    cranelift_codegen::ir::condcodes::FloatCC::NotEqual,
                    c,
                    zero,
                );
                Ok(builder.ins().select(is_nonzero, xv, yv))
            }

            KOp::Cast { x, to } => {
                // For now everything is F32, so cast is identity.
                Ok(vreg_vals[x.0 as usize])
            }

            KOp::ReduceResult(red_idx) => {
                let reduction = &reductions[*red_idx];
                emit_reduction_v2(
                    builder, module, reduction, vreg_vals, ptr_table, i_val, math,
                )
            }
        }
    }

    // ---- Reduction emission ----

    fn emit_reduction_v2(
        builder: &mut FunctionBuilder,
        module: &mut JITModule,
        reduction: &KernelReduction,
        outer_vreg_vals: &[Value],
        ptr_table: Value,
        i_val: Value,
        math: &MathFuncs,
    ) -> Result<Value, V10Error> {
        // Use a Cranelift Variable for the accumulator.
        // This avoids the block-param threading issue.
        let acc_var = Variable::from_u32(1000); // arbitrary unique index
        builder.declare_var(acc_var, types::F32);

        let init_val = if reduction.is_sum {
            builder.ins().f32const(0.0)
        } else {
            builder.ins().f32const(f32::NEG_INFINITY)
        };
        builder.def_var(acc_var, init_val);

        let k_var = Variable::from_u32(1001);
        builder.declare_var(k_var, types::I64);
        let k_init = builder.ins().iconst(types::I64, 0);
        builder.def_var(k_var, k_init);

        let red_header = builder.create_block();
        let red_body = builder.create_block();
        let red_exit = builder.create_block();

        let bound = builder.ins().iconst(types::I64, reduction.bound as i64);

        builder.ins().jump(red_header, &[]);

        builder.switch_to_block(red_header);
        let k_val = builder.use_var(k_var);
        let k_cmp = builder.ins().icmp(
            cranelift_codegen::ir::condcodes::IntCC::SignedLessThan,
            k_val,
            bound,
        );
        builder.ins().brif(k_cmp, red_body, &[], red_exit, &[]);

        builder.switch_to_block(red_body);
        let k_val = builder.use_var(k_var);

        // Emit reduction body.
        let mut body_vals: Vec<Value> = Vec::with_capacity(reduction.body.len());
        for body_op in &reduction.body {
            let val = emit_reduce_body_op(
                builder,
                module,
                body_op,
                &body_vals,
                outer_vreg_vals,
                ptr_table,
                i_val,
                k_val,
                math,
            )?;
            body_vals.push(val);
        }

        let body_result = body_vals[reduction.body_result as usize];
        let acc = builder.use_var(acc_var);
        let new_acc = if reduction.is_sum {
            builder.ins().fadd(acc, body_result)
        } else {
            let cmp = builder.ins().fcmp(
                cranelift_codegen::ir::condcodes::FloatCC::GreaterThan,
                body_result,
                acc,
            );
            builder.ins().select(cmp, body_result, acc)
        };
        builder.def_var(acc_var, new_acc);

        let k_next = builder.ins().iadd_imm(k_val, 1);
        builder.def_var(k_var, k_next);

        builder.ins().jump(red_header, &[]);

        builder.switch_to_block(red_exit);
        builder.seal_block(red_header);
        builder.seal_block(red_body);
        builder.seal_block(red_exit);

        Ok(builder.use_var(acc_var))
    }

    fn emit_reduce_body_op(
        builder: &mut FunctionBuilder,
        module: &mut JITModule,
        op: &KReduceOp,
        body_vals: &[Value],
        outer_vreg_vals: &[Value],
        ptr_table: Value,
        i_val: Value,
        k_val: Value,
        math: &MathFuncs,
    ) -> Result<Value, V10Error> {
        match op {
            KReduceOp::SymLoad {
                buffer,
                base_offset,
                stride_i,
                stride_k,
            } => {
                let buf_ptr = load_buffer_ptr(builder, ptr_table, *buffer);
                // idx = base_offset + stride_i * i + stride_k * k
                let si = builder.ins().imul_imm(i_val, *stride_i as i64);
                let sk = builder.ins().imul_imm(k_val, *stride_k as i64);
                let idx = builder.ins().iadd(si, sk);
                let idx = builder.ins().iadd_imm(idx, *base_offset as i64);
                let byte_offset = builder.ins().imul_imm(idx, 4);
                let addr = builder.ins().iadd(buf_ptr, byte_offset);
                Ok(builder.ins().load(types::F32, MemFlags::new(), addr, 0))
            }

            KReduceOp::BroadcastLoad { buffer, offset } => {
                let buf_ptr = load_buffer_ptr(builder, ptr_table, *buffer);
                Ok(builder.ins().load(
                    types::F32,
                    MemFlags::new(),
                    buf_ptr,
                    (*offset as i64 * 4) as i32,
                ))
            }

            KReduceOp::OuterRef(vreg) => Ok(outer_vreg_vals[vreg.0 as usize]),

            KReduceOp::Literal(val) => Ok(builder.ins().f32const(*val)),

            KReduceOp::Binary {
                op,
                a,
                b,
                compute_dt,
            } => {
                let a_val = body_vals[*a as usize];
                let b_val = body_vals[*b as usize];
                emit_binop(builder, module, math, *op, a_val, b_val)
            }

            KReduceOp::Unary { op, x, compute_dt } => {
                let x_val = body_vals[*x as usize];
                emit_unop(builder, module, math, *op, x_val)
            }

            KReduceOp::Cast { x, to } => {
                // F32 only for now.
                Ok(body_vals[*x as usize])
            }
        }
    }

    // ---- Scalar op emission ----

    fn emit_binop(
        builder: &mut FunctionBuilder,
        module: &mut JITModule,
        math: &MathFuncs,
        op: ScalarBinOp,
        a: Value,
        b: Value,
    ) -> Result<Value, V10Error> {
        Ok(match op {
            ScalarBinOp::Add => builder.ins().fadd(a, b),
            ScalarBinOp::Sub => builder.ins().fsub(a, b),
            ScalarBinOp::Mul => builder.ins().fmul(a, b),
            ScalarBinOp::Div => builder.ins().fdiv(a, b),
            ScalarBinOp::Max => {
                let cmp = builder.ins().fcmp(
                    cranelift_codegen::ir::condcodes::FloatCC::GreaterThan,
                    a,
                    b,
                );
                builder.ins().select(cmp, a, b)
            }
            ScalarBinOp::Min => {
                let cmp =
                    builder
                        .ins()
                        .fcmp(cranelift_codegen::ir::condcodes::FloatCC::LessThan, a, b);
                builder.ins().select(cmp, a, b)
            }
            ScalarBinOp::Pow => {
                let func_ref = module.declare_func_in_func(math.powf, builder.func);
                let call = builder.ins().call(func_ref, &[a, b]);
                builder.inst_results(call)[0]
            }
            ScalarBinOp::Mod => {
                let func_ref = module.declare_func_in_func(math.fmodf, builder.func);
                let call = builder.ins().call(func_ref, &[a, b]);
                builder.inst_results(call)[0]
            }
        })
    }

    fn emit_unop(
        builder: &mut FunctionBuilder,
        module: &mut JITModule,
        math: &MathFuncs,
        op: ScalarUnaryOp,
        x: Value,
    ) -> Result<Value, V10Error> {
        Ok(match op {
            ScalarUnaryOp::Neg => builder.ins().fneg(x),
            ScalarUnaryOp::Abs => {
                let func_ref = module.declare_func_in_func(math.fabsf, builder.func);
                let call = builder.ins().call(func_ref, &[x]);
                builder.inst_results(call)[0]
            }
            ScalarUnaryOp::Exp => {
                let func_ref = module.declare_func_in_func(math.expf, builder.func);
                let call = builder.ins().call(func_ref, &[x]);
                builder.inst_results(call)[0]
            }
            ScalarUnaryOp::Ln => {
                let func_ref = module.declare_func_in_func(math.logf, builder.func);
                let call = builder.ins().call(func_ref, &[x]);
                builder.inst_results(call)[0]
            }
            ScalarUnaryOp::Sqrt => {
                let func_ref = module.declare_func_in_func(math.sqrtf, builder.func);
                let call = builder.ins().call(func_ref, &[x]);
                builder.inst_results(call)[0]
            }
            ScalarUnaryOp::Reciprocal => {
                let one = builder.ins().f32const(1.0);
                builder.ins().fdiv(one, x)
            }
            ScalarUnaryOp::Tanh => {
                let func_ref = module.declare_func_in_func(math.tanhf, builder.func);
                let call = builder.ins().call(func_ref, &[x]);
                builder.inst_results(call)[0]
            }
            ScalarUnaryOp::Floor => {
                let func_ref = module.declare_func_in_func(math.floorf, builder.func);
                let call = builder.ins().call(func_ref, &[x]);
                builder.inst_results(call)[0]
            }
            ScalarUnaryOp::Ceil => {
                let func_ref = module.declare_func_in_func(math.ceilf, builder.func);
                let call = builder.ins().call(func_ref, &[x]);
                builder.inst_results(call)[0]
            }
        })
    }

    // ---- Helpers ----

    fn load_buffer_ptr(builder: &mut FunctionBuilder, ptr_table: Value, buffer: BufferId) -> Value {
        let offset = (buffer.0 as i64) * 8; // each pointer is 8 bytes
        builder
            .ins()
            .load(types::I64, MemFlags::new(), ptr_table, offset as i32)
    }
}
