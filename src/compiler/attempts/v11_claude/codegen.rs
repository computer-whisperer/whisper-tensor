#![allow(clippy::all, dead_code, unreachable_patterns)]
//! Cranelift JIT codegen for v11 kernel plans.
//!
//! Each KernelPlan → native function: fn(ptr_table: *const *mut f32, start: i64, end: i64)
//! The function iterates i from start to end, executing the kernel's ops
//! and storing the result.
//!
//! Math functions (exp, log, tanh, pow, fmod) are called via indirect calls
//! with the function pointer as an immediate constant. This avoids PC-relative
//! relocations that can overflow i32 on x86-64 with ASLR.
//! Simple ops (abs, sqrt, floor, ceil, neg) use native Cranelift IR instructions.

#[cfg(feature = "cranelift")]
pub mod jit {
    use crate::compiler::attempts::v11_claude::plan::*;
    use crate::dtype::DType;
    use crate::nano_graph::{ScalarBinOp, ScalarUnaryOp};
    use cranelift_codegen::ir::types;
    use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Value};
    use cranelift_codegen::settings::{self, Configurable};
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::{Linkage, Module};

    // Math function wrappers — addresses are embedded as immediates in JIT code.
    extern "C" fn wt11_expf(x: f32) -> f32 { x.exp() }
    extern "C" fn wt11_logf(x: f32) -> f32 { x.ln() }
    extern "C" fn wt11_tanhf(x: f32) -> f32 { x.tanh() }
    extern "C" fn wt11_powf(x: f32, y: f32) -> f32 { x.powf(y) }
    extern "C" fn wt11_fmodf(x: f32, y: f32) -> f32 { x % y }

    /// Raw function pointers for indirect calls (avoids relocation issues).
    struct MathPtrs {
        expf: i64,
        logf: i64,
        tanhf: i64,
        powf: i64,
        fmodf: i64,
        sig1: cranelift_codegen::ir::SigRef,
        sig2: cranelift_codegen::ir::SigRef,
    }

    #[derive(Debug, thiserror::Error)]
    pub enum V11Error {
        #[error("v11 codegen: {0}")]
        Codegen(String),
    }

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
        _modules: Vec<JITModule>,
        pub kernels: Vec<CompiledKernel>,
        pub num_buffers: usize,
    }

    unsafe impl Send for CompiledGraph {}
    unsafe impl Sync for CompiledGraph {}

    /// Max kernels per JIT module. Data tables can accumulate, so we split.
    const KERNELS_PER_MODULE: usize = 128;

    fn new_jit_module(isa: std::sync::Arc<dyn cranelift_codegen::isa::TargetIsa>) -> JITModule {
        // No external symbols registered — all math calls are indirect.
        let jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        JITModule::new(jit_builder)
    }

    pub fn compile(plan: &CompilationPlan) -> Result<CompiledGraph, V11Error> {
        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed").unwrap();
        let isa_builder = cranelift_native::builder()
            .map_err(|e| V11Error::Codegen(format!("cranelift native ISA: {}", e)))?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| V11Error::Codegen(format!("ISA finish: {}", e)))?;

        let total_kernels = plan.kernels.len();
        let compile_start = std::time::Instant::now();
        let mut all_modules: Vec<JITModule> = Vec::new();
        let mut all_kernels: Vec<CompiledKernel> = Vec::with_capacity(total_kernels);

        for batch_start in (0..total_kernels).step_by(KERNELS_PER_MODULE) {
            let batch_end = (batch_start + KERNELS_PER_MODULE).min(total_kernels);
            let mut module = new_jit_module(isa.clone());
            let mut ctx = module.make_context();
            let mut func_ctx = FunctionBuilderContext::new();
            let mut batch_infos: Vec<(cranelift_module::FuncId, BufferId, Vec<BufferId>, u32)> = Vec::new();

            for ki in batch_start..batch_end {
                if ki > 0 && ki % 10_000 == 0 {
                    let elapsed = compile_start.elapsed().as_secs_f64();
                    let rate = ki as f64 / elapsed;
                    let eta = (total_kernels - ki) as f64 / rate;
                    eprintln!(
                        "[v11] compiling kernel {}/{} ({:.0}/s, ETA {:.0}s)",
                        ki, total_kernels, rate, eta
                    );
                }
                let kplan = &plan.kernels[ki];
                let func_name = format!("v11_k{}", ki);

                ctx.func.signature.params.push(AbiParam::new(types::I64));
                ctx.func.signature.params.push(AbiParam::new(types::I64));
                ctx.func.signature.params.push(AbiParam::new(types::I64));

                let func_id = module
                    .declare_function(&func_name, Linkage::Local, &ctx.func.signature)
                    .map_err(|e| V11Error::Codegen(format!("declare: {}", e)))?;

                let mut builder = FunctionBuilder::new(&mut ctx.func, &mut func_ctx);

                // Create indirect call signatures for math functions.
                let mut sig1 = module.make_signature();
                sig1.params.push(AbiParam::new(types::F32));
                sig1.returns.push(AbiParam::new(types::F32));
                let sig1_ref = builder.import_signature(sig1);

                let mut sig2 = module.make_signature();
                sig2.params.push(AbiParam::new(types::F32));
                sig2.params.push(AbiParam::new(types::F32));
                sig2.returns.push(AbiParam::new(types::F32));
                let sig2_ref = builder.import_signature(sig2);

                let math = MathPtrs {
                    expf: wt11_expf as *const u8 as i64,
                    logf: wt11_logf as *const u8 as i64,
                    tanhf: wt11_tanhf as *const u8 as i64,
                    powf: wt11_powf as *const u8 as i64,
                    fmodf: wt11_fmodf as *const u8 as i64,
                    sig1: sig1_ref,
                    sig2: sig2_ref,
                };

                let entry = builder.create_block();
                builder.append_block_params_for_function_params(entry);
                builder.switch_to_block(entry);
                builder.seal_block(entry);

                let ptr_table = builder.block_params(entry)[0];
                let start = builder.block_params(entry)[1];
                let end = builder.block_params(entry)[2];

                let mut next_var_id: u32 = 1000;
                emit_kernel(
                    &mut builder, kplan, ptr_table, start, end,
                    &math, &mut next_var_id,
                )?;

                builder.ins().return_(&[]);
                builder.finalize();

                module.define_function(func_id, &mut ctx)
                    .map_err(|e| V11Error::Codegen(format!("define: {}", e)))?;
                ctx.clear();

                batch_infos.push((func_id, kplan.output, kplan.input_buffers.clone(), kplan.extent));
            }

            module.finalize_definitions()
                .map_err(|e| V11Error::Codegen(format!("finalize: {}", e)))?;

            for (func_id, output, inputs, extent) in batch_infos {
                let func_ptr = module.get_finalized_function(func_id);
                all_kernels.push(CompiledKernel {
                    func_ptr, output_buffer: output, input_buffers: inputs, extent,
                });
            }

            all_modules.push(module);
        }

        let elapsed = compile_start.elapsed().as_secs_f64();
        eprintln!(
            "[v11] compiled {} kernels in {:.1}s across {} modules ({:.0} k/s)",
            total_kernels, elapsed, all_modules.len(), total_kernels as f64 / elapsed
        );

        Ok(CompiledGraph { _modules: all_modules, kernels: all_kernels, num_buffers: plan.buffers.len() })
    }

    fn emit_kernel(
        builder: &mut FunctionBuilder,
        kplan: &KernelPlan,
        ptr_table: Value,
        start: Value,
        end: Value,
        math: &MathPtrs,
        next_var_id: &mut u32,
    ) -> Result<(), V11Error> {
        let loop_header = builder.create_block();
        let loop_body = builder.create_block();
        let loop_exit = builder.create_block();

        builder.ins().jump(loop_header, &[start]);

        builder.switch_to_block(loop_header);
        builder.append_block_param(loop_header, types::I64);
        let i_val = builder.block_params(loop_header)[0];

        let cmp = builder.ins().icmp(
            cranelift_codegen::ir::condcodes::IntCC::SignedLessThan, i_val, end,
        );
        builder.ins().brif(cmp, loop_body, &[], loop_exit, &[]);

        builder.switch_to_block(loop_body);

        let mut vreg_vals: Vec<Value> = Vec::with_capacity(kplan.ops.len());

        for op in &kplan.ops {
            let val = emit_kop(
                builder, op, &vreg_vals, &kplan.reductions,
                ptr_table, i_val, math, next_var_id,
            )?;
            vreg_vals.push(val);
        }

        // Store result.
        let result_val = vreg_vals[kplan.result.0 as usize];
        let out_buf_ptr = load_buffer_ptr(builder, ptr_table, kplan.output);
        let store_idx = if kplan.output_offset == 0 {
            i_val
        } else {
            builder.ins().iadd_imm(i_val, kplan.output_offset as i64)
        };
        let byte_offset = builder.ins().imul_imm(store_idx, 4);
        let addr = builder.ins().iadd(out_buf_ptr, byte_offset);
        builder.ins().store(MemFlags::new(), result_val, addr, 0);

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
        op: &KOp,
        vreg_vals: &[Value],
        reductions: &[KernelReduction],
        ptr_table: Value,
        i_val: Value,
        math: &MathPtrs,
        next_var_id: &mut u32,
    ) -> Result<Value, V11Error> {
        match op {
            KOp::Load { buffer, base_offset, stride } => {
                let buf_ptr = load_buffer_ptr(builder, ptr_table, *buffer);
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
                let modulus_val = builder.ins().iconst(types::I64, *modulus as i64);
                let idx = builder.ins().urem(i_val, modulus_val);
                let byte_offset = builder.ins().imul_imm(idx, 4);
                let addr = builder.ins().iadd(buf_ptr, byte_offset);
                Ok(builder.ins().load(types::F32, MemFlags::new(), addr, 0))
            }

            KOp::TableLoad { buffer, table_buffer } => {
                let buf_ptr = load_buffer_ptr(builder, ptr_table, *buffer);
                // Load the offset from the table buffer (stored as u32 bit-cast to f32).
                let tbl_ptr = load_buffer_ptr(builder, ptr_table, *table_buffer);
                let idx_byte_off = builder.ins().imul_imm(i_val, 4);
                let idx_addr = builder.ins().iadd(tbl_ptr, idx_byte_off);
                // Load as i32 (the u32 offset), zero-extend to i64.
                let raw_offset = builder.ins().load(types::I32, MemFlags::new(), idx_addr, 0);
                let offset_i64 = builder.ins().uextend(types::I64, raw_offset);

                let data_byte_off = builder.ins().imul_imm(offset_i64, 4);
                let data_addr = builder.ins().iadd(buf_ptr, data_byte_off);
                Ok(builder.ins().load(types::F32, MemFlags::new(), data_addr, 0))
            }

            KOp::GatherLoad { gather_bufs, table_buffer } => {
                // Table stores (buf_slot: u32, offset: u32) pairs, 2 entries per element.
                let tbl_ptr = load_buffer_ptr(builder, ptr_table, *table_buffer);
                // Index into table: i * 2 entries * 4 bytes each = i * 8
                let tbl_byte_off = builder.ins().imul_imm(i_val, 8);
                let slot_addr = builder.ins().iadd(tbl_ptr, tbl_byte_off);
                // Load buf_slot (u32)
                let buf_slot = builder.ins().load(types::I32, MemFlags::new(), slot_addr, 0);
                // Load offset (u32)
                let offset = builder.ins().load(types::I32, MemFlags::new(), slot_addr, 4);

                // Load the buffer pointer from ptr_table[gather_bufs[buf_slot]]
                // We need to map buf_slot → actual BufferId → ptr_table index.
                // Build a chain of comparisons for each possible slot value.
                let offset_i64 = builder.ins().uextend(types::I64, offset);
                let data_byte_off = builder.ins().imul_imm(offset_i64, 4);

                // Start with the first buffer's pointer as default
                let mut result_ptr = load_buffer_ptr(builder, ptr_table, gather_bufs[0]);
                result_ptr = builder.ins().iadd(result_ptr, data_byte_off);

                // For each additional buffer slot, conditionally select the pointer
                for (si, &gbuf) in gather_bufs.iter().enumerate().skip(1) {
                    let slot_val = builder.ins().iconst(types::I32, si as i64);
                    let is_this_slot = builder.ins().icmp(
                        cranelift_codegen::ir::condcodes::IntCC::Equal,
                        buf_slot, slot_val,
                    );
                    let alt_ptr = load_buffer_ptr(builder, ptr_table, gbuf);
                    let alt_addr = builder.ins().iadd(alt_ptr, data_byte_off);
                    result_ptr = builder.ins().select(is_this_slot, alt_addr, result_ptr);
                }

                Ok(builder.ins().load(types::F32, MemFlags::new(), result_ptr, 0))
            }

            KOp::BroadcastLoad { buffer, offset } => {
                let buf_ptr = load_buffer_ptr(builder, ptr_table, *buffer);
                let byte_offset = (*offset as i64) * 4;
                Ok(builder.ins().load(types::F32, MemFlags::new(), buf_ptr, byte_offset as i32))
            }

            KOp::Literal(val) => Ok(builder.ins().f32const(*val)),

            KOp::Binary { op, a, b, .. } => {
                emit_binop(builder, math, *op, vreg_vals[a.0 as usize], vreg_vals[b.0 as usize])
            }

            KOp::Unary { op, x, .. } => {
                emit_unop(builder, math, *op, vreg_vals[x.0 as usize])
            }

            KOp::Select { cond, x, y, .. } => {
                let c = vreg_vals[cond.0 as usize];
                let xv = vreg_vals[x.0 as usize];
                let yv = vreg_vals[y.0 as usize];
                let zero = builder.ins().f32const(0.0);
                let is_nonzero = builder.ins().fcmp(
                    cranelift_codegen::ir::condcodes::FloatCC::NotEqual, c, zero,
                );
                Ok(builder.ins().select(is_nonzero, xv, yv))
            }

            KOp::Cast { x, to } => {
                // F32-only for now.
                Ok(vreg_vals[x.0 as usize])
            }

            KOp::ReduceResult(red_idx) => {
                let reduction = &reductions[*red_idx];
                emit_reduction(builder, reduction, vreg_vals, ptr_table, i_val, math, next_var_id)
            }
        }
    }

    fn emit_reduction(
        builder: &mut FunctionBuilder,
        reduction: &KernelReduction,
        outer_vreg_vals: &[Value],
        ptr_table: Value,
        i_val: Value,
        math: &MathPtrs,
        next_var_id: &mut u32,
    ) -> Result<Value, V11Error> {
        let acc_var = Variable::from_u32(*next_var_id);
        *next_var_id += 1;
        builder.declare_var(acc_var, types::F32);

        let init_val = if reduction.is_sum {
            builder.ins().f32const(0.0)
        } else {
            builder.ins().f32const(f32::NEG_INFINITY)
        };
        builder.def_var(acc_var, init_val);

        let k_var = Variable::from_u32(*next_var_id);
        *next_var_id += 1;
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
            cranelift_codegen::ir::condcodes::IntCC::SignedLessThan, k_val, bound,
        );
        builder.ins().brif(k_cmp, red_body, &[], red_exit, &[]);

        builder.switch_to_block(red_body);
        let k_val = builder.use_var(k_var);

        let mut body_vals: Vec<Value> = Vec::with_capacity(reduction.body.len());
        for body_op in &reduction.body {
            let val = emit_reduce_body_op(
                builder, body_op, &body_vals, outer_vreg_vals,
                ptr_table, i_val, k_val, math,
            )?;
            body_vals.push(val);
        }

        let body_result = body_vals[reduction.body_result as usize];
        let acc = builder.use_var(acc_var);
        let new_acc = if reduction.is_sum {
            builder.ins().fadd(acc, body_result)
        } else {
            let cmp = builder.ins().fcmp(
                cranelift_codegen::ir::condcodes::FloatCC::GreaterThan, body_result, acc,
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
        op: &KReduceOp,
        body_vals: &[Value],
        outer_vreg_vals: &[Value],
        ptr_table: Value,
        i_val: Value,
        k_val: Value,
        math: &MathPtrs,
    ) -> Result<Value, V11Error> {
        match op {
            KReduceOp::SymLoad { buffer, base_offset, stride_i, stride_k } => {
                let buf_ptr = load_buffer_ptr(builder, ptr_table, *buffer);
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
                    types::F32, MemFlags::new(), buf_ptr, (*offset as i64 * 4) as i32,
                ))
            }
            KReduceOp::OuterRef(vreg) => {
                Ok(outer_vreg_vals[vreg.0 as usize])
            }
            KReduceOp::Literal(val) => {
                Ok(builder.ins().f32const(*val))
            }
            KReduceOp::Binary { op, a, b, .. } => {
                emit_binop(builder, math, *op, body_vals[*a as usize], body_vals[*b as usize])
            }
            KReduceOp::Unary { op, x, .. } => {
                emit_unop(builder, math, *op, body_vals[*x as usize])
            }
            KReduceOp::Cast { x, .. } => {
                Ok(body_vals[*x as usize])
            }
        }
    }

    /// Emit an indirect call to a math function via its raw pointer.
    fn call_indirect_1(builder: &mut FunctionBuilder, math: &MathPtrs, ptr: i64, x: Value) -> Value {
        let addr = builder.ins().iconst(types::I64, ptr);
        let call = builder.ins().call_indirect(math.sig1, addr, &[x]);
        builder.inst_results(call)[0]
    }

    fn call_indirect_2(builder: &mut FunctionBuilder, math: &MathPtrs, ptr: i64, a: Value, b: Value) -> Value {
        let addr = builder.ins().iconst(types::I64, ptr);
        let call = builder.ins().call_indirect(math.sig2, addr, &[a, b]);
        builder.inst_results(call)[0]
    }

    fn emit_binop(
        builder: &mut FunctionBuilder,
        math: &MathPtrs,
        op: ScalarBinOp,
        a: Value,
        b: Value,
    ) -> Result<Value, V11Error> {
        Ok(match op {
            ScalarBinOp::Add => builder.ins().fadd(a, b),
            ScalarBinOp::Sub => builder.ins().fsub(a, b),
            ScalarBinOp::Mul => builder.ins().fmul(a, b),
            ScalarBinOp::Div => builder.ins().fdiv(a, b),
            ScalarBinOp::Max => {
                let cmp = builder.ins().fcmp(
                    cranelift_codegen::ir::condcodes::FloatCC::GreaterThan, a, b,
                );
                builder.ins().select(cmp, a, b)
            }
            ScalarBinOp::Min => {
                let cmp = builder.ins().fcmp(
                    cranelift_codegen::ir::condcodes::FloatCC::LessThan, a, b,
                );
                builder.ins().select(cmp, a, b)
            }
            ScalarBinOp::Pow => call_indirect_2(builder, math, math.powf, a, b),
            ScalarBinOp::Mod => call_indirect_2(builder, math, math.fmodf, a, b),
        })
    }

    fn emit_unop(
        builder: &mut FunctionBuilder,
        math: &MathPtrs,
        op: ScalarUnaryOp,
        x: Value,
    ) -> Result<Value, V11Error> {
        Ok(match op {
            // Native Cranelift IR instructions — no external calls needed.
            ScalarUnaryOp::Neg => builder.ins().fneg(x),
            ScalarUnaryOp::Abs => builder.ins().fabs(x),
            ScalarUnaryOp::Sqrt => builder.ins().sqrt(x),
            ScalarUnaryOp::Floor => builder.ins().floor(x),
            ScalarUnaryOp::Ceil => builder.ins().ceil(x),
            ScalarUnaryOp::Reciprocal => {
                let one = builder.ins().f32const(1.0);
                builder.ins().fdiv(one, x)
            }
            // Indirect calls for transcendentals.
            ScalarUnaryOp::Exp => call_indirect_1(builder, math, math.expf, x),
            ScalarUnaryOp::Ln => call_indirect_1(builder, math, math.logf, x),
            ScalarUnaryOp::Tanh => call_indirect_1(builder, math, math.tanhf, x),
        })
    }

    fn load_buffer_ptr(builder: &mut FunctionBuilder, ptr_table: Value, buffer: BufferId) -> Value {
        let offset = (buffer.0 as i64) * 8;
        builder.ins().load(types::I64, MemFlags::new(), ptr_table, offset as i32)
    }
}
