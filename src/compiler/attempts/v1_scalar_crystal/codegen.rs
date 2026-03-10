//! Cranelift JIT compilation from nano ops.
//!
//! Takes a stream of NanoOps and compiles them into native machine code
//! via Cranelift. The compiled function operates on a table of tensor
//! buffer pointers.

use super::crystal::{CrystalLoop, CrystalOp, LoopBodyOp};
use super::nano_op::{NanoOp, ScalarBinOp, ScalarUnaryOp};
use crate::graph::GlobalId;
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
    /// Maps each tensor GlobalId to its index in the buffer pointer table.
    pub tensor_index: HashMap<GlobalId, usize>,
    /// Size in elements for each tensor (by GlobalId).
    pub tensor_sizes: HashMap<GlobalId, usize>,
    /// Total number of buffer slots.
    pub num_buffers: usize,
    /// DType per tensor. If absent, defaults to F32.
    pub tensor_dtypes: HashMap<GlobalId, crate::dtype::DType>,
    /// Shape per tensor. Needed for interpreted ops to reconstruct typed tensors.
    pub tensor_shapes: Option<HashMap<GlobalId, Vec<usize>>>,
}

impl TensorLayout {
    /// Build a tensor layout from shapes. Assigns sequential indices.
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
            tensor_dtypes: HashMap::new(),
            tensor_shapes: Some(shapes.clone()),
        }
    }

    /// Build a tensor layout from shapes and dtypes.
    pub fn from_shapes_and_dtypes(
        shapes: &HashMap<GlobalId, Vec<usize>>,
        dtypes: &HashMap<GlobalId, crate::dtype::DType>,
    ) -> Self {
        let mut layout = Self::from_shapes(shapes);
        layout.tensor_dtypes = dtypes.clone();
        layout
    }

    /// Get the dtype for a tensor, defaulting to F32.
    pub fn dtype_of(&self, tensor: &GlobalId) -> crate::dtype::DType {
        self.tensor_dtypes
            .get(tensor)
            .copied()
            .unwrap_or(crate::dtype::DType::F32)
    }
}

// Wrappers for math functions callable from JIT code
extern "C" fn wt_expf(x: f32) -> f32 {
    x.exp()
}
extern "C" fn wt_logf(x: f32) -> f32 {
    x.ln()
}
extern "C" fn wt_tanhf(x: f32) -> f32 {
    x.tanh()
}

/// A compiled graph ready for execution.
pub struct NativeCompiledGraph {
    /// The JIT module (must stay alive while function pointer is in use).
    _module: JITModule,
    /// Function pointer: fn(*const *mut f32)
    func_ptr: *const u8,
    /// Tensor layout info for the caller.
    pub layout: TensorLayout,
}

// Safety: The JIT module and function pointer are self-contained.
// The compiled code only accesses memory through the buffer table passed by the caller.
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
    /// Execute the compiled graph.
    ///
    /// `buffers` must be a slice of mutable f32 pointers, one per tensor,
    /// indexed according to `self.layout.tensor_index`. The caller is
    /// responsible for allocating and filling input/constant buffers
    /// and reading output buffers after execution.
    ///
    /// # Safety
    /// The caller must ensure all buffer pointers are valid and correctly sized.
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

/// Compile a nano op stream into native code.
pub fn compile(
    nano_ops: &[NanoOp],
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

    // Map NanoValue -> cranelift Value
    let mut value_map: HashMap<u64, cranelift_codegen::ir::Value> = HashMap::new();

    for nano_op in nano_ops {
        match nano_op {
            NanoOp::Load {
                dst,
                tensor,
                flat_index,
            } => {
                let tidx = layout
                    .tensor_index
                    .get(tensor)
                    .ok_or(CodegenError::UnknownTensor(*tensor))?;
                // Load tensor base pointer from the table
                let ptr_offset = (*tidx as i64) * (ptr_type.bytes() as i64);
                let base_ptr_addr = builder.ins().iadd_imm(ptr_table, ptr_offset);
                let base_ptr = builder
                    .ins()
                    .load(ptr_type, MemFlags::trusted(), base_ptr_addr, 0);
                // Load f32 element
                let elem_offset = (*flat_index as i64) * 4;
                let elem_addr = builder.ins().iadd_imm(base_ptr, elem_offset);
                let val = builder
                    .ins()
                    .load(types::F32, MemFlags::trusted(), elem_addr, 0);
                value_map.insert(dst.0, val);
            }

            NanoOp::Store {
                tensor,
                flat_index,
                src,
            } => {
                let tidx = layout
                    .tensor_index
                    .get(tensor)
                    .ok_or(CodegenError::UnknownTensor(*tensor))?;
                let ptr_offset = (*tidx as i64) * (ptr_type.bytes() as i64);
                let base_ptr_addr = builder.ins().iadd_imm(ptr_table, ptr_offset);
                let base_ptr = builder
                    .ins()
                    .load(ptr_type, MemFlags::trusted(), base_ptr_addr, 0);
                let elem_offset = (*flat_index as i64) * 4;
                let elem_addr = builder.ins().iadd_imm(base_ptr, elem_offset);
                let val = value_map[&src.0];
                builder.ins().store(MemFlags::trusted(), val, elem_addr, 0);
            }

            NanoOp::Literal { dst, value } => {
                let val = builder.ins().f32const(*value as f32);
                value_map.insert(dst.0, val);
            }

            NanoOp::BinOp { dst, op, a, b } => {
                let va = value_map[&a.0];
                let vb = value_map[&b.0];
                let result = match op {
                    ScalarBinOp::Add => builder.ins().fadd(va, vb),
                    ScalarBinOp::Sub => builder.ins().fsub(va, vb),
                    ScalarBinOp::Mul => builder.ins().fmul(va, vb),
                    ScalarBinOp::Div => builder.ins().fdiv(va, vb),
                    ScalarBinOp::Min => builder.ins().fmin(va, vb),
                    ScalarBinOp::Max => builder.ins().fmax(va, vb),
                };
                value_map.insert(dst.0, result);
            }

            NanoOp::UnaryOp { dst, op, input } => {
                let vin = value_map[&input.0];
                let result = match op {
                    ScalarUnaryOp::Neg => builder.ins().fneg(vin),
                    ScalarUnaryOp::Abs => builder.ins().fabs(vin),
                    ScalarUnaryOp::Sqrt => builder.ins().sqrt(vin),
                    ScalarUnaryOp::Floor => builder.ins().floor(vin),
                    ScalarUnaryOp::Ceil => builder.ins().ceil(vin),
                    ScalarUnaryOp::Reciprocal => {
                        let one = builder.ins().f32const(1.0);
                        builder.ins().fdiv(one, vin)
                    }
                    ScalarUnaryOp::Exp => {
                        let fref = math_refs["wt_expf"];
                        let call = builder.ins().call(fref, &[vin]);
                        builder.inst_results(call)[0]
                    }
                    ScalarUnaryOp::Ln => {
                        let fref = math_refs["wt_logf"];
                        let call = builder.ins().call(fref, &[vin]);
                        builder.inst_results(call)[0]
                    }
                    ScalarUnaryOp::Tanh => {
                        let fref = math_refs["wt_tanhf"];
                        let call = builder.ins().call(fref, &[vin]);
                        builder.inst_results(call)[0]
                    }
                    _ => panic!("unsupported unary op {:?} in v1 codegen", op),
                };
                value_map.insert(dst.0, result);
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

/// Compile a crystallized op stream into native code.
///
/// Crystal loops become actual loops in the generated code, with an induction
/// variable controlling index computation. Scalar ops are emitted inline.
pub fn compile_crystallized(
    crystal_ops: &[CrystalOp],
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

    // We use cranelift Variables for the loop induction variable.
    // Each crystal loop gets its own Variable index.
    let mut next_var_index = 0u32;

    for crystal_op in crystal_ops {
        match crystal_op {
            CrystalOp::Scalar(nano_op) => {
                emit_scalar_nano_op(
                    &mut builder,
                    nano_op,
                    layout,
                    ptr_table,
                    ptr_type,
                    &math_refs,
                    &mut HashMap::new(), // scalar ops are self-contained after crystallization
                )?;
            }
            CrystalOp::Loop(crystal_loop) => {
                emit_crystal_loop(
                    &mut builder,
                    crystal_loop,
                    layout,
                    ptr_table,
                    ptr_type,
                    &math_refs,
                    &mut next_var_index,
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

/// Emit a single scalar nano op into the current block.
fn emit_scalar_nano_op(
    builder: &mut FunctionBuilder,
    nano_op: &NanoOp,
    layout: &TensorLayout,
    ptr_table: cranelift_codegen::ir::Value,
    ptr_type: cranelift_codegen::ir::Type,
    math_refs: &HashMap<&str, cranelift_codegen::ir::FuncRef>,
    value_map: &mut HashMap<u64, cranelift_codegen::ir::Value>,
) -> Result<(), CodegenError> {
    match nano_op {
        NanoOp::Load {
            dst,
            tensor,
            flat_index,
        } => {
            let val = emit_load(builder, layout, ptr_table, ptr_type, *tensor, *flat_index)?;
            value_map.insert(dst.0, val);
        }
        NanoOp::Store {
            tensor,
            flat_index,
            src,
        } => {
            let val = value_map[&src.0];
            emit_store(
                builder,
                layout,
                ptr_table,
                ptr_type,
                *tensor,
                *flat_index,
                val,
            )?;
        }
        NanoOp::Literal { dst, value } => {
            let val = builder.ins().f32const(*value as f32);
            value_map.insert(dst.0, val);
        }
        NanoOp::BinOp { dst, op, a, b } => {
            let va = value_map[&a.0];
            let vb = value_map[&b.0];
            let result = emit_bin_op(builder, *op, va, vb);
            value_map.insert(dst.0, result);
        }
        NanoOp::UnaryOp { dst, op, input } => {
            let vin = value_map[&input.0];
            let result = emit_unary_op(builder, *op, vin, math_refs);
            value_map.insert(dst.0, result);
        }
    }
    Ok(())
}

/// Emit a crystal loop as a cranelift loop construct.
///
/// Structure:
///   loop_header(i: i64):
///     ... body ops using base_index + i * step ...
///     i_next = i + 1
///     if i_next < count: jump loop_header(i_next)
///     else: jump loop_exit
///   loop_exit:
///     (continue)
fn emit_crystal_loop(
    builder: &mut FunctionBuilder,
    crystal_loop: &CrystalLoop,
    layout: &TensorLayout,
    ptr_table: cranelift_codegen::ir::Value,
    ptr_type: cranelift_codegen::ir::Type,
    math_refs: &HashMap<&str, cranelift_codegen::ir::FuncRef>,
    next_var_index: &mut u32,
) -> Result<(), CodegenError> {
    // Declare a cranelift Variable for the induction variable
    let iv_var = Variable::from_u32(*next_var_index);
    *next_var_index += 1;
    builder.declare_var(iv_var, types::I64);

    // Initialize i = 0 in the current block
    let zero = builder.ins().iconst(types::I64, 0);
    builder.def_var(iv_var, zero);

    let loop_header = builder.create_block();
    let loop_exit = builder.create_block();

    // Jump from current block to loop header
    builder.ins().jump(loop_header, &[]);

    // --- Loop header ---
    builder.switch_to_block(loop_header);

    let i = builder.use_var(iv_var);

    // Body op values: index into body -> cranelift Value
    let mut body_values: Vec<cranelift_codegen::ir::Value> =
        Vec::with_capacity(crystal_loop.body.len());

    for body_op in &crystal_loop.body {
        let val = match body_op {
            LoopBodyOp::Load {
                tensor,
                base_index,
                step,
            } => {
                let tidx = layout
                    .tensor_index
                    .get(tensor)
                    .ok_or(CodegenError::UnknownTensor(*tensor))?;
                // Load base pointer from table
                let ptr_offset = (*tidx as i64) * (ptr_type.bytes() as i64);
                let base_ptr_addr = builder.ins().iadd_imm(ptr_table, ptr_offset);
                let base_ptr = builder
                    .ins()
                    .load(ptr_type, MemFlags::trusted(), base_ptr_addr, 0);
                // Compute element index: base_index + i * step
                let elem_idx = if *step == 0 {
                    // Broadcast: always same index
                    builder.ins().iconst(types::I64, *base_index as i64)
                } else if *step == 1 {
                    // Common case: contiguous
                    builder.ins().iadd_imm(i, *base_index as i64)
                } else {
                    let step_val = builder.ins().iconst(types::I64, *step as i64);
                    let offset = builder.ins().imul(i, step_val);
                    builder.ins().iadd_imm(offset, *base_index as i64)
                };
                // byte offset = elem_idx * 4
                let byte_offset = builder.ins().ishl_imm(elem_idx, 2);
                let elem_addr = builder.ins().iadd(base_ptr, byte_offset);
                builder
                    .ins()
                    .load(types::F32, MemFlags::trusted(), elem_addr, 0)
            }
            LoopBodyOp::Store {
                tensor,
                base_index,
                step,
                value_ref,
            } => {
                let tidx = layout
                    .tensor_index
                    .get(tensor)
                    .ok_or(CodegenError::UnknownTensor(*tensor))?;
                let ptr_offset = (*tidx as i64) * (ptr_type.bytes() as i64);
                let base_ptr_addr = builder.ins().iadd_imm(ptr_table, ptr_offset);
                let base_ptr = builder
                    .ins()
                    .load(ptr_type, MemFlags::trusted(), base_ptr_addr, 0);
                let elem_idx = if *step == 0 {
                    builder.ins().iconst(types::I64, *base_index as i64)
                } else if *step == 1 {
                    builder.ins().iadd_imm(i, *base_index as i64)
                } else {
                    let step_val = builder.ins().iconst(types::I64, *step as i64);
                    let offset = builder.ins().imul(i, step_val);
                    builder.ins().iadd_imm(offset, *base_index as i64)
                };
                let byte_offset = builder.ins().ishl_imm(elem_idx, 2);
                let elem_addr = builder.ins().iadd(base_ptr, byte_offset);
                let val = body_values[*value_ref];
                builder.ins().store(MemFlags::trusted(), val, elem_addr, 0);
                // Store doesn't produce a value, but we need a placeholder
                val // reuse the stored value as placeholder
            }
            LoopBodyOp::BinOp { op, a_ref, b_ref } => {
                let va = body_values[*a_ref];
                let vb = body_values[*b_ref];
                emit_bin_op(builder, *op, va, vb)
            }
            LoopBodyOp::UnaryOp { op, input_ref } => {
                let vin = body_values[*input_ref];
                emit_unary_op(builder, *op, vin, math_refs)
            }
            LoopBodyOp::Literal { value } => builder.ins().f32const(*value as f32),
        };
        body_values.push(val);
    }

    // Increment induction variable: i_next = i + 1
    let i_next = builder.ins().iadd_imm(i, 1);
    builder.def_var(iv_var, i_next);

    // Branch: if i_next < count, loop back; else exit
    let count = builder.ins().iconst(types::I64, crystal_loop.count as i64);
    let cmp = builder.ins().icmp(
        cranelift_codegen::ir::condcodes::IntCC::SignedLessThan,
        i_next,
        count,
    );
    builder.ins().brif(cmp, loop_header, &[], loop_exit, &[]);

    // Seal the loop header now that all predecessors are known (entry jump + back-edge)
    builder.seal_block(loop_header);

    // --- Loop exit ---
    builder.switch_to_block(loop_exit);
    builder.seal_block(loop_exit);

    Ok(())
}

/// Emit a binary f32 operation.
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

/// Emit a unary f32 operation.
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
        ScalarUnaryOp::Exp => {
            let fref = math_refs["wt_expf"];
            let call = builder.ins().call(fref, &[vin]);
            builder.inst_results(call)[0]
        }
        ScalarUnaryOp::Ln => {
            let fref = math_refs["wt_logf"];
            let call = builder.ins().call(fref, &[vin]);
            builder.inst_results(call)[0]
        }
        ScalarUnaryOp::Tanh => {
            let fref = math_refs["wt_tanhf"];
            let call = builder.ins().call(fref, &[vin]);
            builder.inst_results(call)[0]
        }
        _ => panic!("unsupported unary op {:?} in v1 emit_scalar_unary", op),
    }
}

/// Emit a load from a tensor buffer.
fn emit_load(
    builder: &mut FunctionBuilder,
    layout: &TensorLayout,
    ptr_table: cranelift_codegen::ir::Value,
    ptr_type: cranelift_codegen::ir::Type,
    tensor: GlobalId,
    flat_index: usize,
) -> Result<cranelift_codegen::ir::Value, CodegenError> {
    let tidx = layout
        .tensor_index
        .get(&tensor)
        .ok_or(CodegenError::UnknownTensor(tensor))?;
    let ptr_offset = (*tidx as i64) * (ptr_type.bytes() as i64);
    let base_ptr_addr = builder.ins().iadd_imm(ptr_table, ptr_offset);
    let base_ptr = builder
        .ins()
        .load(ptr_type, MemFlags::trusted(), base_ptr_addr, 0);
    let elem_offset = (flat_index as i64) * 4;
    let elem_addr = builder.ins().iadd_imm(base_ptr, elem_offset);
    Ok(builder
        .ins()
        .load(types::F32, MemFlags::trusted(), elem_addr, 0))
}

/// Emit a store to a tensor buffer.
fn emit_store(
    builder: &mut FunctionBuilder,
    layout: &TensorLayout,
    ptr_table: cranelift_codegen::ir::Value,
    ptr_type: cranelift_codegen::ir::Type,
    tensor: GlobalId,
    flat_index: usize,
    val: cranelift_codegen::ir::Value,
) -> Result<(), CodegenError> {
    let tidx = layout
        .tensor_index
        .get(&tensor)
        .ok_or(CodegenError::UnknownTensor(tensor))?;
    let ptr_offset = (*tidx as i64) * (ptr_type.bytes() as i64);
    let base_ptr_addr = builder.ins().iadd_imm(ptr_table, ptr_offset);
    let base_ptr = builder
        .ins()
        .load(ptr_type, MemFlags::trusted(), base_ptr_addr, 0);
    let elem_offset = (flat_index as i64) * 4;
    let elem_addr = builder.ins().iadd_imm(base_ptr, elem_offset);
    builder.ins().store(MemFlags::trusted(), val, elem_addr, 0);
    Ok(())
}

/// Set up the JIT module with math function symbols and ISA.
fn setup_jit_module()
-> Result<(JITModule, HashMap<&'static str, cranelift_module::FuncId>), CodegenError> {
    let mut flag_builder = settings::builder();
    flag_builder
        .set("use_colocated_libcalls", "false")
        .map_err(|e| CodegenError::CodegenError(e.to_string()))?;
    flag_builder
        .set("is_pic", "false")
        .map_err(|e| CodegenError::CodegenError(e.to_string()))?;
    let flags = settings::Flags::new(flag_builder);
    let isa = cranelift_native::builder()
        .map_err(|msg| CodegenError::CodegenError(msg.to_string()))?
        .finish(flags)
        .map_err(|e| CodegenError::CodegenError(e.to_string()))?;

    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    jit_builder.symbol("wt_expf", wt_expf as *const u8);
    jit_builder.symbol("wt_logf", wt_logf as *const u8);
    jit_builder.symbol("wt_tanhf", wt_tanhf as *const u8);

    let mut module = JITModule::new(jit_builder);
    let math_func_ids = declare_math_functions(&mut module)?;

    Ok((module, math_func_ids))
}

/// Declare external math functions in the JIT module.
fn declare_math_functions(
    module: &mut JITModule,
) -> Result<HashMap<&'static str, cranelift_module::FuncId>, CodegenError> {
    let mut ids = HashMap::new();

    for name in &["wt_expf", "wt_logf", "wt_tanhf"] {
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
    use crate::compiler::nano_op::NanoOpExpander;
    use crate::graph::GlobalId;
    use crate::milli_graph::MilliOpGraph;
    use crate::milli_graph::ops::SimpleBinary;

    #[test]
    fn test_compile_and_run_add() {
        let mut rng = wyrand::WyRand::new(42);

        // Build graph: out = a + b, shapes [2, 2]
        let ext_a = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let int_a = input_map[&ext_a];
        let int_b = input_map[&ext_b];
        let int_out = SimpleBinary::add(&mut graph, int_a, int_b, &mut rng);

        let mut shapes = HashMap::new();
        shapes.insert(int_a, vec![2, 2]);
        shapes.insert(int_b, vec![2, 2]);
        shapes.insert(int_out, vec![2, 2]);

        // Expand to nano ops
        let mut expander = NanoOpExpander::new(shapes.clone());
        let nano_ops = expander.expand(&graph).unwrap();

        // Build layout and compile
        let layout = TensorLayout::from_shapes(&shapes);
        let compiled = compile(&nano_ops, &layout).unwrap();

        // Set up buffers
        let mut a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut b_data = vec![10.0f32, 20.0, 30.0, 40.0];
        let mut out_data = vec![0.0f32; 4];

        let a_idx = layout.tensor_index[&int_a];
        let b_idx = layout.tensor_index[&int_b];
        let out_idx = layout.tensor_index[&int_out];

        let mut buffers = vec![std::ptr::null_mut::<f32>(); layout.num_buffers];
        buffers[a_idx] = a_data.as_mut_ptr();
        buffers[b_idx] = b_data.as_mut_ptr();
        buffers[out_idx] = out_data.as_mut_ptr();

        unsafe {
            compiled.execute(&mut buffers);
        }

        assert_eq!(out_data, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_compile_and_run_mul_broadcast() {
        let mut rng = wyrand::WyRand::new(99);

        // out = a * b, a=[2,3], b=[1,3] (broadcast)
        let ext_a = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let int_a = input_map[&ext_a];
        let int_b = input_map[&ext_b];
        let int_out = SimpleBinary::mul(&mut graph, int_a, int_b, &mut rng);

        let mut shapes = HashMap::new();
        shapes.insert(int_a, vec![2, 3]);
        shapes.insert(int_b, vec![1, 3]);
        shapes.insert(int_out, vec![2, 3]);

        let mut expander = NanoOpExpander::new(shapes.clone());
        let nano_ops = expander.expand(&graph).unwrap();

        let layout = TensorLayout::from_shapes(&shapes);
        let compiled = compile(&nano_ops, &layout).unwrap();

        let mut a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut b_data = vec![10.0f32, 100.0, 1000.0];
        let mut out_data = vec![0.0f32; 6];

        let mut buffers = vec![std::ptr::null_mut::<f32>(); layout.num_buffers];
        buffers[layout.tensor_index[&int_a]] = a_data.as_mut_ptr();
        buffers[layout.tensor_index[&int_b]] = b_data.as_mut_ptr();
        buffers[layout.tensor_index[&int_out]] = out_data.as_mut_ptr();

        unsafe {
            compiled.execute(&mut buffers);
        }

        assert_eq!(out_data, vec![10.0, 200.0, 3000.0, 40.0, 500.0, 6000.0]);
    }

    #[test]
    fn test_crystal_compile_add() {
        use crate::compiler::crystal::crystallize;

        let mut rng = wyrand::WyRand::new(42);
        let ext_a = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let int_a = input_map[&ext_a];
        let int_b = input_map[&ext_b];
        let int_out = SimpleBinary::add(&mut graph, int_a, int_b, &mut rng);

        let mut shapes = HashMap::new();
        shapes.insert(int_a, vec![8]);
        shapes.insert(int_b, vec![8]);
        shapes.insert(int_out, vec![8]);

        let mut expander = NanoOpExpander::new(shapes.clone());
        let nano_ops = expander.expand(&graph).unwrap();
        let crystal_ops = crystallize(&nano_ops);

        let layout = TensorLayout::from_shapes(&shapes);
        let compiled = compile_crystallized(&crystal_ops, &layout).unwrap();

        let mut a_data: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let mut b_data: Vec<f32> = (10..=80).step_by(10).map(|x| x as f32).collect();
        let mut out_data = vec![0.0f32; 8];

        let mut buffers = vec![std::ptr::null_mut::<f32>(); layout.num_buffers];
        buffers[layout.tensor_index[&int_a]] = a_data.as_mut_ptr();
        buffers[layout.tensor_index[&int_b]] = b_data.as_mut_ptr();
        buffers[layout.tensor_index[&int_out]] = out_data.as_mut_ptr();

        unsafe {
            compiled.execute(&mut buffers);
        }

        let expected: Vec<f32> = (0..8)
            .map(|i| (i + 1) as f32 + ((i + 1) * 10) as f32)
            .collect();
        assert_eq!(out_data, expected);
    }

    #[test]
    fn test_crystal_compile_broadcast() {
        use crate::compiler::crystal::crystallize;

        let mut rng = wyrand::WyRand::new(99);
        let ext_a = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let int_a = input_map[&ext_a];
        let int_b = input_map[&ext_b];
        let int_out = SimpleBinary::mul(&mut graph, int_a, int_b, &mut rng);

        // a=[2,3], b=[1,3] — broadcast b's dim 0
        let mut shapes = HashMap::new();
        shapes.insert(int_a, vec![2, 3]);
        shapes.insert(int_b, vec![1, 3]);
        shapes.insert(int_out, vec![2, 3]);

        let mut expander = NanoOpExpander::new(shapes.clone());
        let nano_ops = expander.expand(&graph).unwrap();
        let crystal_ops = crystallize(&nano_ops);

        let layout = TensorLayout::from_shapes(&shapes);
        let compiled = compile_crystallized(&crystal_ops, &layout).unwrap();

        let mut a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut b_data = vec![10.0f32, 100.0, 1000.0];
        let mut out_data = vec![0.0f32; 6];

        let mut buffers = vec![std::ptr::null_mut::<f32>(); layout.num_buffers];
        buffers[layout.tensor_index[&int_a]] = a_data.as_mut_ptr();
        buffers[layout.tensor_index[&int_b]] = b_data.as_mut_ptr();
        buffers[layout.tensor_index[&int_out]] = out_data.as_mut_ptr();

        unsafe {
            compiled.execute(&mut buffers);
        }

        assert_eq!(out_data, vec![10.0, 200.0, 3000.0, 40.0, 500.0, 6000.0]);
    }

    #[test]
    fn test_crystal_compile_chain() {
        use crate::compiler::crystal::crystallize;

        let mut rng = wyrand::WyRand::new(42);
        let ext_a = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let int_a = input_map[&ext_a];
        let int_b = input_map[&ext_b];
        let mul_out = SimpleBinary::mul(&mut graph, int_a, int_b, &mut rng);
        let neg_out = crate::milli_graph::ops::SimpleUnaryOp::neg(&mut graph, mul_out, &mut rng);

        let mut shapes = HashMap::new();
        shapes.insert(int_a, vec![16]);
        shapes.insert(int_b, vec![16]);
        shapes.insert(mul_out, vec![16]);
        shapes.insert(neg_out, vec![16]);

        let mut expander = NanoOpExpander::new(shapes.clone());
        let nano_ops = expander.expand(&graph).unwrap();
        let crystal_ops = crystallize(&nano_ops);

        let layout = TensorLayout::from_shapes(&shapes);
        let compiled = compile_crystallized(&crystal_ops, &layout).unwrap();

        let mut a_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let mut b_data: Vec<f32> = (1..=16).map(|x| x as f32 * 0.5).collect();
        let mut mul_data = vec![0.0f32; 16];
        let mut neg_data = vec![0.0f32; 16];

        let mut buffers = vec![std::ptr::null_mut::<f32>(); layout.num_buffers];
        buffers[layout.tensor_index[&int_a]] = a_data.as_mut_ptr();
        buffers[layout.tensor_index[&int_b]] = b_data.as_mut_ptr();
        buffers[layout.tensor_index[&mul_out]] = mul_data.as_mut_ptr();
        buffers[layout.tensor_index[&neg_out]] = neg_data.as_mut_ptr();

        unsafe {
            compiled.execute(&mut buffers);
        }

        for i in 0..16 {
            let expected = -((i + 1) as f32 * ((i + 1) as f32 * 0.5));
            assert!(
                (neg_data[i] - expected).abs() < 1e-5,
                "Mismatch at {i}: got {} expected {}",
                neg_data[i],
                expected
            );
        }
    }
}
