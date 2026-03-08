//! Cranelift JIT codegen for v2 fusion kernels.

use super::kernel::*;
use crate::graph::GlobalId;
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use std::collections::HashMap;

/// Metadata about the compiled graph's tensor layout.
#[derive(Debug, Clone)]
pub struct TensorLayout {
    pub tensor_index: HashMap<GlobalId, usize>,
    pub tensor_sizes: HashMap<GlobalId, usize>,
    pub num_buffers: usize,
}

impl TensorLayout {
    pub fn from_shapes(shapes: &HashMap<GlobalId, Vec<usize>>) -> Self {
        let mut tensor_index = HashMap::new();
        let mut tensor_sizes = HashMap::new();
        let mut sorted: Vec<_> = shapes.iter().collect();
        sorted.sort_by_key(|(id, _)| **id);
        for (i, (id, shape)) in sorted.iter().enumerate() {
            tensor_index.insert(**id, i);
            tensor_sizes.insert(**id, shape.iter().product::<usize>().max(1));
        }
        TensorLayout {
            num_buffers: sorted.len(),
            tensor_index,
            tensor_sizes,
        }
    }
}

extern "C" fn wt_logf(x: f32) -> f32 {
    x.ln()
}

pub struct NativeCompiledGraph {
    _module: JITModule,
    func_ptr: *const u8,
    pub layout: TensorLayout,
}

unsafe impl Send for NativeCompiledGraph {}

#[derive(Debug, thiserror::Error)]
pub enum CodegenError {
    #[error("Cranelift module error: {0}")]
    ModuleError(#[from] cranelift_module::ModuleError),
    #[error("Cranelift codegen error: {0}")]
    CodegenError(String),
    #[error("Unknown tensor {0} not in layout")]
    UnknownTensor(GlobalId),
}

impl NativeCompiledGraph {
    pub unsafe fn execute(&self, buffers: &mut [*mut f32]) {
        assert!(
            buffers.len() >= self.layout.num_buffers,
            "Expected at least {} buffers, got {}",
            self.layout.num_buffers,
            buffers.len()
        );
        let func: unsafe extern "C" fn(*const *mut f32) =
            unsafe { std::mem::transmute(self.func_ptr) };
        unsafe { func(buffers.as_ptr()) };
    }
}

/// Compile a kernel plan into native code.
pub fn compile(
    kernels: &[KernelOp],
    layout: &TensorLayout,
) -> Result<NativeCompiledGraph, CodegenError> {
    let (mut module, math_func_ids) = setup_jit_module()?;
    let ptr_type = module.isa().pointer_type();

    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(ptr_type));
    let func_id = module.declare_function("wt_compiled_graph", Linkage::Local, &sig)?;

    let mut ctx = module.make_context();
    ctx.func.signature = sig;

    let math_refs: HashMap<&str, cranelift_codegen::ir::FuncRef> = math_func_ids
        .iter()
        .map(|(name, fid)| {
            let fref = module.declare_func_in_func(*fid, &mut ctx.func);
            (*name, fref)
        })
        .collect();

    let mut fn_builder_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);

    let entry = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    builder.seal_block(entry);

    let ptr_table = builder.block_params(entry)[0];

    let mut next_var = 0u32;

    for kernel in kernels {
        match kernel {
            KernelOp::Elementwise(ek) => {
                emit_elementwise(
                    &mut builder, ek, layout, ptr_table, ptr_type, &math_refs,
                    &mut next_var,
                )?;
            }
            KernelOp::Gemm(gk) => {
                emit_gemm(
                    &mut builder, gk, layout, ptr_table, ptr_type, &mut next_var,
                )?;
            }
        }
    }

    builder.ins().return_(&[]);
    builder.finalize();

    module.define_function(func_id, &mut ctx)?;
    module.clear_context(&mut ctx);
    module.finalize_definitions().expect("finalize JIT");

    let code_ptr = module.get_finalized_function(func_id);

    Ok(NativeCompiledGraph {
        _module: module,
        func_ptr: code_ptr,
        layout: layout.clone(),
    })
}

// ---------------------------------------------------------------------------
// Loop infrastructure
// ---------------------------------------------------------------------------

struct LoopInfo {
    iv: cranelift_codegen::ir::Value,
    var: Variable,
    header: cranelift_codegen::ir::Block,
    exit: cranelift_codegen::ir::Block,
    count: usize,
}

fn open_loop(
    builder: &mut FunctionBuilder,
    count: usize,
    next_var: &mut u32,
) -> LoopInfo {
    let var = Variable::from_u32(*next_var);
    *next_var += 1;
    builder.declare_var(var, types::I64);
    let zero = builder.ins().iconst(types::I64, 0);
    builder.def_var(var, zero);

    let header = builder.create_block();
    let exit = builder.create_block();
    builder.ins().jump(header, &[]);
    builder.switch_to_block(header);

    let iv = builder.use_var(var);

    LoopInfo {
        iv,
        var,
        header,
        exit,
        count,
    }
}

fn close_loop(builder: &mut FunctionBuilder, info: &LoopInfo) {
    let iv_next = builder.ins().iadd_imm(info.iv, 1);
    builder.def_var(info.var, iv_next);
    let limit = builder.ins().iconst(types::I64, info.count as i64);
    let cmp = builder.ins().icmp(IntCC::SignedLessThan, iv_next, limit);
    builder.ins().brif(cmp, info.header, &[], info.exit, &[]);
    builder.seal_block(info.header);
    builder.switch_to_block(info.exit);
    builder.seal_block(info.exit);
}

// ---------------------------------------------------------------------------
// Elementwise kernel emission
// ---------------------------------------------------------------------------

fn emit_elementwise(
    builder: &mut FunctionBuilder,
    kernel: &ElementwiseKernel,
    layout: &TensorLayout,
    ptr_table: cranelift_codegen::ir::Value,
    ptr_type: cranelift_codegen::ir::Type,
    math_refs: &HashMap<&str, cranelift_codegen::ir::FuncRef>,
    next_var: &mut u32,
) -> Result<(), CodegenError> {
    // Hoist tensor base pointer loads BEFORE the loops.
    let mut base_ptr_cache: HashMap<GlobalId, cranelift_codegen::ir::Value> = HashMap::new();
    for body_op in &kernel.body {
        let tensor = match body_op {
            BodyOp::Load { tensor, .. } | BodyOp::Store { tensor, .. } => *tensor,
            _ => continue,
        };
        if !base_ptr_cache.contains_key(&tensor) {
            let ptr = load_tensor_base_ptr(builder, layout, ptr_table, ptr_type, tensor)?;
            base_ptr_cache.insert(tensor, ptr);
        }
    }

    // Open nested loops.
    let mut loops: Vec<LoopInfo> = Vec::new();
    for &dim in &kernel.dims {
        loops.push(open_loop(builder, dim, next_var));
    }

    // Emit body.
    let mut body_values: Vec<cranelift_codegen::ir::Value> = Vec::new();

    for body_op in &kernel.body {
        let val = match body_op {
            BodyOp::Load { tensor, strides } => {
                let idx = compute_multi_index(builder, &loops, strides);
                let base = base_ptr_cache[tensor];
                let byte_off = builder.ins().ishl_imm(idx, 2);
                let addr = builder.ins().iadd(base, byte_off);
                builder.ins().load(types::F32, MemFlags::trusted(), addr, 0)
            }
            BodyOp::Store {
                tensor,
                strides,
                value_ref,
            } => {
                let idx = compute_multi_index(builder, &loops, strides);
                let base = base_ptr_cache[tensor];
                let byte_off = builder.ins().ishl_imm(idx, 2);
                let addr = builder.ins().iadd(base, byte_off);
                let val = body_values[*value_ref];
                builder.ins().store(MemFlags::trusted(), val, addr, 0);
                val
            }
            BodyOp::BinOp { op, a_ref, b_ref } => {
                emit_bin_op(builder, *op, body_values[*a_ref], body_values[*b_ref])
            }
            BodyOp::UnaryOp { op, input_ref } => {
                emit_unary_op(builder, *op, body_values[*input_ref], math_refs)
            }
            BodyOp::Literal { value } => builder.ins().f32const(*value as f32),
        };
        body_values.push(val);
    }

    // Close loops innermost first.
    for info in loops.iter().rev() {
        close_loop(builder, info);
    }

    Ok(())
}

/// Compute flat index from loop variables and per-dimension strides.
fn compute_multi_index(
    builder: &mut FunctionBuilder,
    loops: &[LoopInfo],
    strides: &[isize],
) -> cranelift_codegen::ir::Value {
    let mut idx = builder.ins().iconst(types::I64, 0);

    for (d, &stride) in strides.iter().enumerate() {
        if stride == 0 {
            continue;
        }
        let term = if stride == 1 {
            loops[d].iv
        } else {
            let s = builder.ins().iconst(types::I64, stride as i64);
            builder.ins().imul(loops[d].iv, s)
        };
        idx = builder.ins().iadd(idx, term);
    }

    idx
}

// ---------------------------------------------------------------------------
// Gemm kernel emission
// ---------------------------------------------------------------------------

fn emit_gemm(
    builder: &mut FunctionBuilder,
    gk: &GemmKernel,
    layout: &TensorLayout,
    ptr_table: cranelift_codegen::ir::Value,
    ptr_type: cranelift_codegen::ir::Type,
    next_var: &mut u32,
) -> Result<(), CodegenError> {
    // Load tensor base pointers once (hoisted out of all loops).
    let a_base = load_tensor_base_ptr(builder, layout, ptr_table, ptr_type, gk.a)?;
    let b_base = load_tensor_base_ptr(builder, layout, ptr_table, ptr_type, gk.b)?;
    let c_base = load_tensor_base_ptr(builder, layout, ptr_table, ptr_type, gk.c)?;

    // Batch loop.
    let batch_loop = open_loop(builder, gk.batch_size, next_var);

    // Compute batch offsets: batch_iv * batch_stride.
    let a_batch_off = if gk.a_batch_stride == 0 {
        builder.ins().iconst(types::I64, 0)
    } else {
        let stride = builder.ins().iconst(types::I64, gk.a_batch_stride as i64);
        builder.ins().imul(batch_loop.iv, stride)
    };
    let b_batch_off = if gk.b_batch_stride == 0 {
        builder.ins().iconst(types::I64, 0)
    } else {
        let stride = builder.ins().iconst(types::I64, gk.b_batch_stride as i64);
        builder.ins().imul(batch_loop.iv, stride)
    };
    let c_batch_off = if gk.c_batch_stride == 0 {
        builder.ins().iconst(types::I64, 0)
    } else {
        let stride = builder.ins().iconst(types::I64, gk.c_batch_stride as i64);
        builder.ins().imul(batch_loop.iv, stride)
    };

    // M loop.
    let m_loop = open_loop(builder, gk.m, next_var);

    // m_offset = m * a_row_stride (= m * k)
    let m_a_off = if gk.k == 1 {
        m_loop.iv
    } else {
        let row_stride = builder.ins().iconst(types::I64, gk.k as i64);
        builder.ins().imul(m_loop.iv, row_stride)
    };
    // m_c_off = m * c_row_stride (= m * n)
    let m_c_off = if gk.n == 1 {
        m_loop.iv
    } else {
        let row_stride = builder.ins().iconst(types::I64, gk.n as i64);
        builder.ins().imul(m_loop.iv, row_stride)
    };

    // Precompute A row byte address (invariant across N).
    let a_row_byte_ptr = {
        let off = builder.ins().iadd(a_batch_off, m_a_off);
        let byte_off = builder.ins().ishl_imm(off, 2);
        builder.ins().iadd(a_base, byte_off)
    };

    let b_stride_bytes = (gk.n as i64) * 4;
    let n_tiles = gk.n / 4;
    let n_remainder = gk.n % 4;

    // --- Tiled N loop: process 4 columns at a time ---
    if n_tiles > 0 {
        let n_tile_loop = open_loop(builder, n_tiles, next_var);

        // n_elem = n_tile_iv * 4 (element offset for this tile)
        let n_elem = builder.ins().ishl_imm(n_tile_loop.iv, 2);

        // B tile byte address: b_base + (b_batch_off + n_elem) * 4
        let b_tile_ptr = {
            let off = builder.ins().iadd(b_batch_off, n_elem);
            let byte_off = builder.ins().ishl_imm(off, 2);
            builder.ins().iadd(b_base, byte_off)
        };

        // Running pointers for K loop.
        let a_ptr_var = Variable::from_u32(*next_var);
        *next_var += 1;
        builder.declare_var(a_ptr_var, types::I64);
        builder.def_var(a_ptr_var, a_row_byte_ptr);

        let b_ptr_var = Variable::from_u32(*next_var);
        *next_var += 1;
        builder.declare_var(b_ptr_var, types::I64);
        builder.def_var(b_ptr_var, b_tile_ptr);

        // 4 accumulators.
        let zero_f32 = builder.ins().f32const(0.0);
        let mut acc_vars = [Variable::from_u32(0); 4];
        for i in 0..4 {
            acc_vars[i] = Variable::from_u32(*next_var);
            *next_var += 1;
            builder.declare_var(acc_vars[i], types::F32);
            builder.def_var(acc_vars[i], zero_f32);
        }

        // K loop.
        let k_loop = open_loop(builder, gk.k, next_var);

        let a_ptr = builder.use_var(a_ptr_var);
        let b_ptr = builder.use_var(b_ptr_var);
        let a_val = builder.ins().load(types::F32, MemFlags::trusted(), a_ptr, 0);

        // Load 4 consecutive B values (same cache line).
        for i in 0..4 {
            let acc = builder.use_var(acc_vars[i]);
            let b_val = builder.ins().load(types::F32, MemFlags::trusted(), b_ptr, (i * 4) as i32);
            let new_acc = builder.ins().fma(a_val, b_val, acc);
            builder.def_var(acc_vars[i], new_acc);
        }

        let a_next = builder.ins().iadd_imm(a_ptr, 4);
        builder.def_var(a_ptr_var, a_next);
        let b_next = builder.ins().iadd_imm(b_ptr, b_stride_bytes);
        builder.def_var(b_ptr_var, b_next);

        close_loop(builder, &k_loop);

        // Store 4 C values.
        let c_tile_ptr = {
            let c_off = builder.ins().iadd(c_batch_off, m_c_off);
            let c_off = builder.ins().iadd(c_off, n_elem);
            let byte_off = builder.ins().ishl_imm(c_off, 2);
            builder.ins().iadd(c_base, byte_off)
        };
        for i in 0..4 {
            let acc = builder.use_var(acc_vars[i]);
            builder.ins().store(MemFlags::trusted(), acc, c_tile_ptr, (i * 4) as i32);
        }

        close_loop(builder, &n_tile_loop);
    }

    // --- Scalar tail for remaining N columns ---
    if n_remainder > 0 {
        let n_start = (n_tiles * 4) as i64;
        let n_tail_loop = open_loop(builder, n_remainder, next_var);
        let n_actual = builder.ins().iadd_imm(n_tail_loop.iv, n_start);

        let b_col_ptr = {
            let off = builder.ins().iadd(b_batch_off, n_actual);
            let byte_off = builder.ins().ishl_imm(off, 2);
            builder.ins().iadd(b_base, byte_off)
        };

        let a_ptr_var = Variable::from_u32(*next_var);
        *next_var += 1;
        builder.declare_var(a_ptr_var, types::I64);
        builder.def_var(a_ptr_var, a_row_byte_ptr);

        let b_ptr_var = Variable::from_u32(*next_var);
        *next_var += 1;
        builder.declare_var(b_ptr_var, types::I64);
        builder.def_var(b_ptr_var, b_col_ptr);

        let acc_var = Variable::from_u32(*next_var);
        *next_var += 1;
        builder.declare_var(acc_var, types::F32);
        let zero = builder.ins().f32const(0.0);
        builder.def_var(acc_var, zero);

        let k_loop = open_loop(builder, gk.k, next_var);
        let a_ptr = builder.use_var(a_ptr_var);
        let b_ptr = builder.use_var(b_ptr_var);
        let acc = builder.use_var(acc_var);
        let a_val = builder.ins().load(types::F32, MemFlags::trusted(), a_ptr, 0);
        let b_val = builder.ins().load(types::F32, MemFlags::trusted(), b_ptr, 0);
        let new_acc = builder.ins().fma(a_val, b_val, acc);
        builder.def_var(acc_var, new_acc);
        let a_next = builder.ins().iadd_imm(a_ptr, 4);
        builder.def_var(a_ptr_var, a_next);
        let b_next = builder.ins().iadd_imm(b_ptr, b_stride_bytes);
        builder.def_var(b_ptr_var, b_next);
        close_loop(builder, &k_loop);

        let c_off = builder.ins().iadd(c_batch_off, m_c_off);
        let c_off = builder.ins().iadd(c_off, n_actual);
        let c_byte = builder.ins().ishl_imm(c_off, 2);
        let c_addr = builder.ins().iadd(c_base, c_byte);
        let final_acc = builder.use_var(acc_var);
        builder.ins().store(MemFlags::trusted(), final_acc, c_addr, 0);

        close_loop(builder, &n_tail_loop);
    }

    close_loop(builder, &m_loop);
    close_loop(builder, &batch_loop);

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn load_tensor_base_ptr(
    builder: &mut FunctionBuilder,
    layout: &TensorLayout,
    ptr_table: cranelift_codegen::ir::Value,
    ptr_type: cranelift_codegen::ir::Type,
    tensor: GlobalId,
) -> Result<cranelift_codegen::ir::Value, CodegenError> {
    let tidx = layout
        .tensor_index
        .get(&tensor)
        .ok_or(CodegenError::UnknownTensor(tensor))?;
    let ptr_offset = (*tidx as i64) * (ptr_type.bytes() as i64);
    let addr = builder.ins().iadd_imm(ptr_table, ptr_offset);
    Ok(builder.ins().load(ptr_type, MemFlags::trusted(), addr, 0))
}


fn emit_bin_op(
    builder: &mut FunctionBuilder,
    op: ScalarBinOp,
    va: cranelift_codegen::ir::Value,
    vb: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    match op {
        ScalarBinOp::Add => builder.ins().fadd(va, vb),
        ScalarBinOp::Sub => builder.ins().fsub(va, vb),
        ScalarBinOp::Mul => builder.ins().fmul(va, vb),
        ScalarBinOp::Div => builder.ins().fdiv(va, vb),
        ScalarBinOp::Min => builder.ins().fmin(va, vb),
        ScalarBinOp::Max => builder.ins().fmax(va, vb),
    }
}

fn emit_unary_op(
    builder: &mut FunctionBuilder,
    op: ScalarUnaryOp,
    vin: cranelift_codegen::ir::Value,
    math_refs: &HashMap<&str, cranelift_codegen::ir::FuncRef>,
) -> cranelift_codegen::ir::Value {
    match op {
        ScalarUnaryOp::Neg => builder.ins().fneg(vin),
        ScalarUnaryOp::Abs => builder.ins().fabs(vin),
        ScalarUnaryOp::Sqrt => builder.ins().sqrt(vin),
        ScalarUnaryOp::Floor => builder.ins().floor(vin),
        ScalarUnaryOp::Ceil => builder.ins().ceil(vin),
        ScalarUnaryOp::Reciprocal => {
            let one = builder.ins().f32const(1.0);
            builder.ins().fdiv(one, vin)
        }
        ScalarUnaryOp::Exp => emit_inline_exp(builder, vin),
        ScalarUnaryOp::Ln => {
            let fref = math_refs["wt_logf"];
            let call = builder.ins().call(fref, &[vin]);
            builder.inst_results(call)[0]
        }
        ScalarUnaryOp::Tanh => emit_inline_tanh(builder, vin),
    }
}

/// Inline tanh via Padé [7,6] rational approximation.
///
/// tanh(x) ≈ x·(135135 + x²(17325 + x²(378 + x²)))
///            / (135135 + x²(62370 + x²(3150 + 28x²)))
///
/// Full f32 precision for |x| < ~4.5. Input clamped to [-4.97, 4.97]
/// and output clamped to [-1, 1] so no function call is needed.
fn emit_inline_tanh(
    builder: &mut FunctionBuilder,
    x: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    let neg_bound = builder.ins().f32const(-4.97);
    let pos_bound = builder.ins().f32const(4.97);
    let xc = builder.ins().fmax(x, neg_bound);
    let xc = builder.ins().fmin(xc, pos_bound);

    let x2 = builder.ins().fmul(xc, xc);

    // Numerator: x * (135135 + x²(17325 + x²(378 + x²)))
    let c135135 = builder.ins().f32const(135135.0);
    let c378 = builder.ins().f32const(378.0);
    let ni = builder.ins().fadd(c378, x2);
    let c17325 = builder.ins().f32const(17325.0);
    let ni = builder.ins().fmul(x2, ni);
    let ni = builder.ins().fadd(c17325, ni);
    let ni = builder.ins().fmul(x2, ni);
    let ni = builder.ins().fadd(c135135, ni);
    let num = builder.ins().fmul(xc, ni);

    // Denominator: 135135 + x²(62370 + x²(3150 + 28x²))
    let c28 = builder.ins().f32const(28.0);
    let di = builder.ins().fmul(c28, x2);
    let c3150 = builder.ins().f32const(3150.0);
    let di = builder.ins().fadd(c3150, di);
    let di = builder.ins().fmul(x2, di);
    let c62370 = builder.ins().f32const(62370.0);
    let di = builder.ins().fadd(c62370, di);
    let di = builder.ins().fmul(x2, di);
    let den = builder.ins().fadd(c135135, di);

    let result = builder.ins().fdiv(num, den);

    let neg_one = builder.ins().f32const(-1.0);
    let pos_one = builder.ins().f32const(1.0);
    let result = builder.ins().fmax(result, neg_one);
    builder.ins().fmin(result, pos_one)
}

/// Inline exp via argument reduction + degree-5 polynomial.
///
/// exp(x) = 2^n · P(r), where n = round(x·log₂e), r = x − n·ln2.
/// No function call needed — uses bitcast for the 2^n construction.
fn emit_inline_exp(
    builder: &mut FunctionBuilder,
    x: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    // Clamp to prevent overflow/underflow.
    let lo = builder.ins().f32const(-87.3);
    let hi = builder.ins().f32const(88.7);
    let xc = builder.ins().fmax(x, lo);
    let xc = builder.ins().fmin(xc, hi);

    // n = round(x * log2(e))
    let log2e = builder.ins().f32const(std::f32::consts::LOG2_E);
    let xlog2e = builder.ins().fmul(xc, log2e);
    let n_float = builder.ins().nearest(xlog2e);

    // Cody-Waite argument reduction: r = x - n*ln2_hi - n*ln2_lo
    let ln2_hi = builder.ins().f32const(0.693_359_375_f32);
    let ln2_lo = builder.ins().f32const(-2.121_944_4e-4_f32);
    let t = builder.ins().fmul(n_float, ln2_hi);
    let r = builder.ins().fsub(xc, t);
    let t = builder.ins().fmul(n_float, ln2_lo);
    let r = builder.ins().fsub(r, t);

    // exp(r) ≈ 1 + r(1 + r(1/2 + r(1/6 + r(1/24 + r/120))))
    let c5 = builder.ins().f32const(1.0 / 120.0);
    let c4 = builder.ins().f32const(1.0 / 24.0);
    let c3 = builder.ins().f32const(1.0 / 6.0);
    let c2 = builder.ins().f32const(0.5);
    let one = builder.ins().f32const(1.0);

    let p = builder.ins().fmul(c5, r);
    let p = builder.ins().fadd(p, c4);
    let p = builder.ins().fmul(p, r);
    let p = builder.ins().fadd(p, c3);
    let p = builder.ins().fmul(p, r);
    let p = builder.ins().fadd(p, c2);
    let p = builder.ins().fmul(p, r);
    let p = builder.ins().fadd(p, one);
    let p = builder.ins().fmul(p, r);
    let exp_r = builder.ins().fadd(p, one);

    // 2^n via integer bit manipulation: float_bits = (n_int + 127) << 23
    let n_int = builder.ins().fcvt_to_sint_sat(types::I32, n_float);
    let biased = builder.ins().iadd_imm(n_int, 127);
    let bits = builder.ins().ishl_imm(biased, 23);
    let two_n = builder.ins().bitcast(types::F32, MemFlags::new(), bits);

    builder.ins().fmul(exp_r, two_n)
}

fn setup_jit_module(
) -> Result<(JITModule, HashMap<&'static str, cranelift_module::FuncId>), CodegenError> {
    let mut flag_builder = settings::builder();
    flag_builder
        .set("use_colocated_libcalls", "false")
        .map_err(|e| CodegenError::CodegenError(e.to_string()))?;
    flag_builder
        .set("is_pic", "false")
        .map_err(|e| CodegenError::CodegenError(e.to_string()))?;
    flag_builder
        .set("opt_level", "speed")
        .map_err(|e| CodegenError::CodegenError(e.to_string()))?;
    let flags = settings::Flags::new(flag_builder);
    let isa = cranelift_native::builder()
        .map_err(|msg| CodegenError::CodegenError(msg.to_string()))?
        .finish(flags)
        .map_err(|e| CodegenError::CodegenError(e.to_string()))?;

    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    jit_builder.symbol("wt_logf", wt_logf as *const u8);

    let mut module = JITModule::new(jit_builder);
    let math_func_ids = declare_math_functions(&mut module)?;
    Ok((module, math_func_ids))
}

fn declare_math_functions(
    module: &mut JITModule,
) -> Result<HashMap<&'static str, cranelift_module::FuncId>, CodegenError> {
    let mut ids = HashMap::new();
    for name in &["wt_logf"] {
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::F32));
        sig.returns.push(AbiParam::new(types::F32));
        let fid = module.declare_function(name, Linkage::Import, &sig)?;
        ids.insert(*name, fid);
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::attempts::v2_fusion::planner;
    use crate::graph::GlobalId;
    use crate::milli_graph::MilliOpGraph;
    use crate::milli_graph::ops::{MatMul, SimpleBinary, SimpleUnaryOp};

    /// Helper: build buffers, run compiled, return output.
    unsafe fn run_compiled_graph(
        compiled: &NativeCompiledGraph,
        tensor_data: &mut HashMap<GlobalId, Vec<f32>>,
        output_id: GlobalId,
    ) -> Vec<f32> {
        let layout = &compiled.layout;
        let mut buffers_storage: Vec<Vec<f32>> = (0..layout.num_buffers)
            .map(|_| Vec::new())
            .collect();

        for (id, data) in tensor_data.iter() {
            if let Some(&idx) = layout.tensor_index.get(id) {
                buffers_storage[idx] = data.clone();
            }
        }
        for (id, &size) in &layout.tensor_sizes {
            let idx = layout.tensor_index[id];
            if buffers_storage[idx].is_empty() {
                buffers_storage[idx] = vec![0.0f32; size];
            }
        }

        let mut ptrs: Vec<*mut f32> = buffers_storage
            .iter_mut()
            .map(|v| v.as_mut_ptr())
            .collect();

        unsafe { compiled.execute(&mut ptrs) };

        let out_idx = layout.tensor_index[&output_id];
        buffers_storage[out_idx].clone()
    }

    #[test]
    fn test_fused_elementwise() {
        let mut rng = wyrand::WyRand::new(42);
        let ext_a = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let a = input_map[&ext_a];
        let b = input_map[&ext_b];
        // out = neg(a + b)
        let add = SimpleBinary::add(&mut graph, a, b, &mut rng);
        let neg = SimpleUnaryOp::neg(&mut graph, add, &mut rng);
        let ext_out = GlobalId::new(&mut rng);
        graph.set_output_map([(neg, ext_out)]);

        let mut shapes = HashMap::new();
        shapes.insert(a, vec![8]);
        shapes.insert(b, vec![8]);
        shapes.insert(add, vec![8]);
        shapes.insert(neg, vec![8]);

        let kernels = planner::plan(&graph, &shapes).unwrap();
        assert_eq!(kernels.len(), 1); // fused

        let layout = TensorLayout::from_shapes(&shapes);
        let compiled = compile(&kernels, &layout).unwrap();

        let a_data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let b_data: Vec<f32> = (10..=80).step_by(10).map(|x| x as f32).collect();
        let mut data = HashMap::new();
        data.insert(a, a_data.clone());
        data.insert(b, b_data.clone());

        let result = unsafe { run_compiled_graph(&compiled, &mut data, neg) };
        let expected: Vec<f32> = a_data
            .iter()
            .zip(&b_data)
            .map(|(a, b)| -(a + b))
            .collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matmul_2x3_times_3x2() {
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
        shapes.insert(a, vec![2, 3]);
        shapes.insert(b, vec![3, 2]);
        shapes.insert(c, vec![2, 2]);

        let kernels = planner::plan(&graph, &shapes).unwrap();
        let layout = TensorLayout::from_shapes(&shapes);
        let compiled = compile(&kernels, &layout).unwrap();

        // A = [[1,2,3],[4,5,6]], B = [[7,8],[9,10],[11,12]]
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut data = HashMap::new();
        data.insert(a, a_data);
        data.insert(b, b_data);

        let result = unsafe { run_compiled_graph(&compiled, &mut data, c) };
        // C[0,0] = 1*7 + 2*9 + 3*11 = 58
        // C[0,1] = 1*8 + 2*10 + 3*12 = 64
        // C[1,0] = 4*7 + 5*9 + 6*11 = 139
        // C[1,1] = 4*8 + 5*10 + 6*12 = 154
        assert_eq!(result, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_matmul_then_bias_tanh() {
        // y = tanh(x @ w + b)
        let mut rng = wyrand::WyRand::new(42);
        let ext_x = GlobalId::new(&mut rng);
        let ext_w = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_x, ext_w, ext_b], &mut rng);
        let x = input_map[&ext_x];
        let w = input_map[&ext_w];
        let b = input_map[&ext_b];

        let mm = MatMul::push_new(&mut graph, x, w, &mut rng);
        let add = SimpleBinary::add(&mut graph, mm, b, &mut rng);
        let tanh = SimpleUnaryOp::trig(
            &mut graph,
            add,
            crate::TrigOp::Tanh,
            &mut rng,
        );
        let ext_out = GlobalId::new(&mut rng);
        graph.set_output_map([(tanh, ext_out)]);

        // x=[2,3], w=[3,4], b=[1,4]
        let mut shapes = HashMap::new();
        shapes.insert(x, vec![2, 3]);
        shapes.insert(w, vec![3, 4]);
        shapes.insert(b, vec![1, 4]);
        shapes.insert(mm, vec![2, 4]);
        shapes.insert(add, vec![2, 4]);
        shapes.insert(tanh, vec![2, 4]);

        let kernels = planner::plan(&graph, &shapes).unwrap();
        // Should be: 1 Gemm + 1 fused Elementwise (add+tanh)
        let s = super::super::kernel::stats(&kernels);
        assert_eq!(s.num_gemm, 1);
        assert_eq!(s.num_elementwise, 1);
        assert_eq!(s.total_fused_ops, 2);

        let layout = TensorLayout::from_shapes(&shapes);
        let compiled = compile(&kernels, &layout).unwrap();

        // Simple values for manual verification.
        let x_data = vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0]; // identity-like rows
        let w_data = vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
        ];
        let b_data = vec![0.0f32, 0.0, 0.0, 0.0];
        let mut data = HashMap::new();
        data.insert(x, x_data);
        data.insert(w, w_data);
        data.insert(b, b_data);

        let result = unsafe { run_compiled_graph(&compiled, &mut data, tanh) };
        // x @ w: row 0 = w row 0 = [0.1, 0.2, 0.3, 0.4]
        //        row 1 = w row 1 = [0.5, 0.6, 0.7, 0.8]
        // + b (zeros) → same
        // tanh of those values
        let expected: Vec<f32> = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            .iter()
            .map(|v: &f32| v.tanh())
            .collect();

        for (i, (r, e)) in result.iter().zip(&expected).enumerate() {
            assert!(
                (r - e).abs() < 1e-6,
                "Mismatch at {i}: got {r} expected {e}"
            );
        }
    }
}
