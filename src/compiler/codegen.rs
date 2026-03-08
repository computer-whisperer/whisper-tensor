//! Cranelift JIT compilation from nano ops.
//!
//! Takes a stream of NanoOps and compiles them into native machine code
//! via Cranelift. The compiled function operates on a table of tensor
//! buffer pointers.

use super::nano_op::{NanoOp, ScalarBinOp, ScalarUnaryOp};
use crate::graph::GlobalId;
use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
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
        }
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
    // Set up Cranelift
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

    let ptr_type = isa.pointer_type();

    let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

    // Register math function symbols
    jit_builder.symbol("wt_expf", wt_expf as *const u8);
    jit_builder.symbol("wt_logf", wt_logf as *const u8);
    jit_builder.symbol("wt_tanhf", wt_tanhf as *const u8);

    let mut module = JITModule::new(jit_builder);

    // Declare math function signatures
    let math_func_ids = declare_math_functions(&mut module)?;

    // Create the main function signature: fn(ptr_table: *const *mut f32) -> void
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(ptr_type));
    let func_id = module.declare_function("wt_compiled_graph", Linkage::Local, &sig)?;

    let mut ctx = module.make_context();
    ctx.func.signature = sig;

    // Import math function refs into this function
    let math_refs: HashMap<&str, cranelift_codegen::ir::FuncRef> = math_func_ids
        .iter()
        .map(|(name, fid)| {
            let fref = module.declare_func_in_func(*fid, &mut ctx.func);
            (*name, fref)
        })
        .collect();

    // Build function body
    let mut fn_builder_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);

    let entry = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    builder.seal_block(entry);

    let ptr_table = builder.block_params(entry)[0];

    // Map NanoValue -> cranelift Value
    let mut value_map: HashMap<u32, cranelift_codegen::ir::Value> = HashMap::new();

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
                let base_ptr =
                    builder
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
                let base_ptr =
                    builder
                        .ins()
                        .load(ptr_type, MemFlags::trusted(), base_ptr_addr, 0);
                let elem_offset = (*flat_index as i64) * 4;
                let elem_addr = builder.ins().iadd_imm(base_ptr, elem_offset);
                let val = value_map[&src.0];
                builder
                    .ins()
                    .store(MemFlags::trusted(), val, elem_addr, 0);
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

        assert_eq!(
            out_data,
            vec![10.0, 200.0, 3000.0, 40.0, 500.0, 6000.0]
        );
    }
}
