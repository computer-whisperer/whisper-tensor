#![allow(clippy::all, dead_code, unreachable_patterns)]
//! v8 generic kernel codegen.
//!
//! Derives register blocking and loop ordering from v6's recovered access
//! patterns. No operation is pattern-matched by name — the same codegen
//! handles matmul, fused matmul+activation, and any other additive reduction
//! or pointwise operation.
//!
//! Key insight: for an additive reduction `out[i,j] = sum_k(term(loads...))`,
//! register blocking on output axis `a` is beneficial when at least one input
//! access has `output_coeffs[a] == 0` (it doesn't vary across lanes on that
//! axis). That access gets loaded once and broadcast to NR accumulators. This
//! is exactly what a matmul micro-kernel does — but derived generically from
//! the affine access analysis, not from recognizing "this is a matmul".

use crate::compiler::attempts::v1_scalar_crystal::codegen::{CodegenError, TensorLayout};
use crate::compiler::attempts::v6_schedule_synthesis::synthesis::{
    AccessDimRole, LoopIntent, PipelineArtifacts, PointwiseIntent, RecoveredLoop,
    RecoveredTensorAccess, ReductionBinOp, ReductionIntent, ReductionTermPattern, ReductionUnaryOp,
    build_from_graph_sampled,
};
use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraph;
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use std::collections::{BTreeSet, HashMap, HashSet};

/// Maximum register-block width. When SIMD is active, NR/4 vector accumulators
/// fit comfortably in 16 xmm registers. The scalar fallback may spill at NR>8
/// but Cranelift handles that. Higher NR means more work per k-step (more
/// j-lanes share each invariant A load).
const MAX_NR: usize = 32;

/// SIMD lane width. F32X4 (SSE) is baseline x86-64.
const SIMD_F32_LANES: usize = 4;

// ---- Error types ----

#[derive(Debug, thiserror::Error)]
pub enum V8Error {
    #[error(transparent)]
    Synthesis(#[from] crate::compiler::attempts::v6_schedule_synthesis::synthesis::PipelineError),
    #[error(transparent)]
    Codegen(#[from] CodegenError),
    #[error("v8: {0}")]
    Unsupported(String),
}

// ---- Compiled graph ----

/// A single JIT-compiled kernel function.
///
/// Each kernel takes `(ptr_table, i_start, i_end)` where i_start..i_end
/// bounds the outermost parallelizable axis.  Calling with (0, parallel_extent)
/// executes the full range.
pub struct CompiledKernel {
    func_ptr: *const u8,
    /// Extent of the parallelizable axis (the outermost non-blocked output axis
    /// for reductions, axis 0 for pointwise).  A tile covers [i_start, i_end)
    /// within 0..parallel_extent.
    pub parallel_extent: usize,
    /// Output tensor this kernel writes.
    pub output_tensor: GlobalId,
    /// Input tensors this kernel reads (deduplicated).
    pub input_tensors: Vec<GlobalId>,
}

impl CompiledKernel {
    /// Execute this kernel for the given row range.
    ///
    /// # Safety
    /// `buffers` must contain valid pointers for all tensors in the layout.
    pub unsafe fn execute_range(&self, buffers: *const *mut f32, i_start: usize, i_end: usize) {
        let func: unsafe extern "C" fn(*const *mut f32, i64, i64) =
            unsafe { std::mem::transmute(self.func_ptr) };
        unsafe { func(buffers, i_start as i64, i_end as i64) };
    }
}

pub struct NativeCompiledGraph {
    _module: JITModule,
    pub layout: TensorLayout,
    pub kernels: Vec<CompiledKernel>,
    /// Scratch buffer for B-panel packing.  The raw pointer is embedded in
    /// the JIT'd code as an iconst, so this Vec must live as long as the
    /// compiled graph.
    _scratch: Vec<f32>,
}

// Safety: JIT function pointers and scratch buffer are valid for the
// lifetime of NativeCompiledGraph.  Concurrent calls with non-overlapping
// tile ranges are safe (each writes to disjoint output rows).
unsafe impl Send for CompiledKernel {}
unsafe impl Sync for CompiledKernel {}
unsafe impl Send for NativeCompiledGraph {}
unsafe impl Sync for NativeCompiledGraph {}

impl NativeCompiledGraph {
    /// Execute all kernels serially, each with its full range.
    ///
    /// # Safety
    /// `buffers` must contain valid pointers for all tensors in the layout.
    pub unsafe fn execute(&self, buffers: &mut [*mut f32]) {
        assert!(
            buffers.len() >= self.layout.num_buffers,
            "Expected at least {} buffers, got {}",
            self.layout.num_buffers,
            buffers.len()
        );
        let ptr = buffers.as_ptr();
        for kernel in &self.kernels {
            unsafe { kernel.execute_range(ptr, 0, kernel.parallel_extent) };
        }
    }
}

// ---- Internal types ----

#[derive(Debug, Clone)]
struct CompiledAccess {
    tensor: GlobalId,
    /// Coefficient per output axis: flat += output_coeffs[a] * axis_var[a]
    output_coeffs: Vec<i64>,
    /// Coefficient for the reduction variable: flat += reduction_coeff * k
    reduction_coeff: i64,
    /// Constant offset
    base: i64,
    /// Total elements in the tensor (for bounds validation)
    len: usize,
}

#[derive(Debug, Clone)]
struct BlockingStrategy {
    /// Which output axis to register-block on (placed innermost in loop order).
    block_axis: Option<usize>,
    /// Number of accumulators (register tile width).
    nr: usize,
    /// Per-access: true if invariant on block_axis (output_coeffs[block_axis] == 0).
    /// Invariant accesses are loaded once per k-iteration and reused across NR lanes.
    invariant: Vec<bool>,
}

/// Cache tiling parameters for reduction kernels.
///
/// For a matmul C[M,N] = sum_k(A[M,K]*B[K,N]) with NR register blocking:
///   - `kc`: reduction tile — limits the K-depth per pass so that the B panel
///     B[kc:kc+KC, jc:jc+NC] fits in L2.
///   - `nc`: blocked-axis tile — limits the N-width per pass so that the B
///     panel and accumulators stay cache-resident.
///   - When kc < K, each (i,j) output is accumulated across multiple kc tiles.
///     On the first tile accumulators start at 0; on subsequent tiles they
///     reload partial sums from the output buffer.
#[derive(Debug, Clone)]
struct CacheTiling {
    /// Reduction tile size (KC).  `None` means no tiling (full K in one pass).
    kc: Option<usize>,
    /// Blocked-axis tile size (NC).  `None` means no tiling.
    nc: Option<usize>,
}

#[derive(Debug, Clone)]
struct ReductionKernelSpec {
    output_tensor: GlobalId,
    output_shape: Vec<usize>,
    reduction_terms: usize,
    canonical_term: ReductionTermPattern,
    accesses: Vec<CompiledAccess>,
    blocking: BlockingStrategy,
    /// Non-blocked axes first, blocked axis last.
    axis_order: Vec<usize>,
    /// Cache tiling parameters.
    tiling: CacheTiling,
    /// Indices of accesses eligible for B-panel packing: varying accesses
    /// whose only output dependency is on block_axis and whose reduction
    /// stride is > 1 (strided in k).
    packable: Vec<usize>,
}

/// Runtime context for B-panel packing within a KC×NC tile.
///
/// Created by `emit_kc_and_inner_axes` after emitting the packing loops,
/// then threaded through to the inner kernel functions.
struct TilePackingCtx {
    /// Base pointer of the scratch buffer (Cranelift Value, ptr type).
    packed_ptr: cranelift_codegen::ir::Value,
    /// Current tile's k_start (needed for packed index: k_offset = k - k_start).
    k_start: cranelift_codegen::ir::Value,
    /// Current tile's jc_start (needed for packed index: j_offset = j - jc_start).
    jc_start: cranelift_codegen::ir::Value,
    /// Width of packed panel: jc_end - jc_start (Cranelift Value).
    nc_extent: cranelift_codegen::ir::Value,
    /// Per-access: true if this access is packed (indexed by access position).
    is_packed: Vec<bool>,
}

#[derive(Debug, Clone)]
struct PointwiseKernelSpec {
    output_tensor: GlobalId,
    output_shape: Vec<usize>,
    canonical_term: ReductionTermPattern,
    accesses: Vec<CompiledAccess>,
}

#[derive(Debug, Clone)]
enum KernelSpec {
    Reduction(ReductionKernelSpec),
    Pointwise(PointwiseKernelSpec),
}

// ---- Public API ----

pub fn compile_graph(
    graph: &MilliOpGraph,
    shapes: &HashMap<GlobalId, Vec<usize>>,
) -> Result<(NativeCompiledGraph, PipelineArtifacts), V8Error> {
    let artifacts = build_from_graph_sampled(graph, shapes)?;
    let specs = build_kernel_specs(&artifacts)?;
    let layout = TensorLayout::from_shapes(shapes);
    let compiled = compile_kernels(&specs, &layout)?;
    Ok((compiled, artifacts))
}

// ---- Spec building ----

fn build_kernel_specs(artifacts: &PipelineArtifacts) -> Result<Vec<KernelSpec>, V8Error> {
    let order = derive_execution_order(artifacts)?;
    let mut specs = Vec::with_capacity(order.len());
    for loop_index in order {
        let recovered = &artifacts.schedule.loops[loop_index];
        let spec = match &recovered.intent {
            LoopIntent::AdditiveReduction(reduction) => {
                KernelSpec::Reduction(build_reduction_spec(recovered, reduction)?)
            }
            LoopIntent::Pointwise(pointwise) => {
                KernelSpec::Pointwise(build_pointwise_spec(recovered, pointwise)?)
            }
            LoopIntent::Unknown => {
                return Err(V8Error::Unsupported(format!(
                    "loop {loop_index} has unknown intent"
                )));
            }
        };
        specs.push(spec);
    }
    Ok(specs)
}

fn build_reduction_spec(
    recovered: &RecoveredLoop,
    reduction: &ReductionIntent,
) -> Result<ReductionKernelSpec, V8Error> {
    let output_rank = recovered.output_shape.len();
    let accesses: Vec<CompiledAccess> = reduction
        .accesses
        .iter()
        .map(|a| compile_access(a, output_rank))
        .collect::<Result<_, _>>()?;
    let blocking = analyze_blocking(&recovered.output_shape, &accesses);
    let axis_order = compute_axis_order(output_rank, blocking.block_axis);
    let packable = find_packable_accesses(&accesses, &blocking);
    let tiling = compute_tiling(
        &recovered.output_shape,
        reduction.terms,
        &blocking,
        !packable.is_empty(),
    );

    Ok(ReductionKernelSpec {
        output_tensor: recovered.output_tensor,
        output_shape: recovered.output_shape.clone(),
        reduction_terms: reduction.terms,
        canonical_term: reduction.canonical_term.clone(),
        accesses,
        blocking,
        axis_order,
        tiling,
        packable,
    })
}

fn build_pointwise_spec(
    recovered: &RecoveredLoop,
    pointwise: &PointwiseIntent,
) -> Result<PointwiseKernelSpec, V8Error> {
    let output_rank = recovered.output_shape.len();
    let accesses: Vec<CompiledAccess> = pointwise
        .accesses
        .iter()
        .map(|a| compile_access(a, output_rank))
        .collect::<Result<_, _>>()?;
    Ok(PointwiseKernelSpec {
        output_tensor: recovered.output_tensor,
        output_shape: recovered.output_shape.clone(),
        canonical_term: pointwise.canonical_term.clone(),
        accesses,
    })
}

fn derive_execution_order(artifacts: &PipelineArtifacts) -> Result<Vec<usize>, V8Error> {
    let loop_count = artifacts.schedule.loops.len();
    let mut output_to_loop = HashMap::<GlobalId, usize>::new();
    for (i, recovered) in artifacts.schedule.loops.iter().enumerate() {
        output_to_loop.insert(recovered.output_tensor, i);
    }

    let mut indegree = vec![0usize; loop_count];
    let mut consumers = vec![Vec::<usize>::new(); loop_count];
    for (i, recovered) in artifacts.schedule.loops.iter().enumerate() {
        let mut deps = HashSet::new();
        for tensor in &recovered.load_tensors {
            if let Some(&producer) = output_to_loop.get(tensor) {
                if producer != i && deps.insert(producer) {
                    indegree[i] += 1;
                    consumers[producer].push(i);
                }
            }
        }
    }

    let mut ready = BTreeSet::new();
    for (i, &deg) in indegree.iter().enumerate() {
        if deg == 0 {
            ready.insert(i);
        }
    }

    let mut order = Vec::with_capacity(loop_count);
    while let Some(&i) = ready.first() {
        ready.remove(&i);
        order.push(i);
        for &consumer in &consumers[i] {
            indegree[consumer] -= 1;
            if indegree[consumer] == 0 {
                ready.insert(consumer);
            }
        }
    }

    if order.len() != loop_count {
        return Err(V8Error::Unsupported(
            "dependency cycle in recovered loops".to_string(),
        ));
    }
    Ok(order)
}

// ---- Access analysis ----

fn compile_access(
    access: &RecoveredTensorAccess,
    output_rank: usize,
) -> Result<CompiledAccess, V8Error> {
    // Compute physical strides for row-major layout.
    let mut dim_strides = vec![0i64; access.tensor_shape.len()];
    let mut running = 1i64;
    for axis in (0..access.tensor_shape.len()).rev() {
        dim_strides[axis] = running;
        running = running.saturating_mul(access.tensor_shape[axis] as i64);
    }

    let mut output_coeffs = vec![0i64; output_rank];
    let mut reduction_coeff = 0i64;
    let mut base = 0i64;

    for (dim, role) in access.dim_roles.iter().enumerate() {
        let ds = dim_strides.get(dim).copied().unwrap_or(0);
        match role {
            AccessDimRole::Constant { value } => {
                base += (*value as i64) * ds;
            }
            AccessDimRole::OutputAxis {
                axis,
                stride,
                offset,
            } => {
                if *axis < output_coeffs.len() {
                    output_coeffs[*axis] += (*stride as i64) * ds;
                }
                base += (*offset as i64) * ds;
            }
            AccessDimRole::ReductionAxis { stride, offset } => {
                reduction_coeff += (*stride as i64) * ds;
                base += (*offset as i64) * ds;
            }
            AccessDimRole::AffineMixed {
                output_strides,
                reduction_stride,
                offset,
            } => {
                for (axis, stride) in output_strides {
                    if *axis < output_coeffs.len() {
                        output_coeffs[*axis] += (*stride as i64) * ds;
                    }
                }
                reduction_coeff += (*reduction_stride as i64) * ds;
                base += (*offset as i64) * ds;
            }
            AccessDimRole::Unknown => {
                return Err(V8Error::Unsupported(format!(
                    "unknown access role for tensor {}",
                    access.tensor
                )));
            }
        }
    }

    Ok(CompiledAccess {
        tensor: access.tensor,
        output_coeffs,
        reduction_coeff,
        base,
        len: access
            .tensor_shape
            .iter()
            .copied()
            .fold(1usize, usize::saturating_mul),
    })
}

/// Determine which output axis to register-block on.
///
/// We scan output axes from innermost to outermost and pick the first axis
/// where at least one input access is *invariant* (its `output_coeffs` entry
/// for that axis is zero). Invariant means the load address doesn't change
/// across lanes — load once, broadcast to NR accumulators.
///
/// For a standard matmul C[i,j] = sum_k(A[i,k]*B[k,j]):
///   A has output_coeffs = [K, 0] → invariant on axis 1
///   B has output_coeffs = [0, 1] → varies on axis 1
/// So we block on axis 1: A[i,k] is loaded once, broadcast across NR j-lanes.
fn analyze_blocking(output_shape: &[usize], accesses: &[CompiledAccess]) -> BlockingStrategy {
    if accesses.is_empty() {
        return BlockingStrategy {
            block_axis: None,
            nr: 1,
            invariant: Vec::new(),
        };
    }
    for axis in (0..output_shape.len()).rev() {
        let extent = output_shape[axis];
        if extent < 2 {
            continue;
        }
        let invariant: Vec<bool> = accesses
            .iter()
            .map(|a| a.output_coeffs.get(axis).copied().unwrap_or(0) == 0)
            .collect();
        if invariant.iter().any(|&v| v) {
            let nr = choose_nr(extent);
            return BlockingStrategy {
                block_axis: Some(axis),
                nr,
                invariant,
            };
        }
    }
    BlockingStrategy {
        block_axis: None,
        nr: 1,
        invariant: vec![false; accesses.len()],
    }
}

/// Find accesses eligible for B-panel packing.
///
/// An access is packable when:
/// 1. It's varying (not invariant — invariant accesses are splatted)
/// 2. Its only output dependency is on block_axis (so packing is done once
///    per KC×NC tile, not per outer-axis iteration)
/// 3. Its reduction_coeff is > 1 (strided k-access; stride=1 is already sequential)
fn find_packable_accesses(accesses: &[CompiledAccess], blocking: &BlockingStrategy) -> Vec<usize> {
    let block_axis = match blocking.block_axis {
        Some(a) => a,
        None => return Vec::new(),
    };
    let mut packable = Vec::new();
    for (idx, access) in accesses.iter().enumerate() {
        if blocking.invariant[idx] {
            continue;
        }
        // Only pack if non-blocked output coefficients are all zero.
        let only_block_axis = access
            .output_coeffs
            .iter()
            .enumerate()
            .all(|(a, &c)| a == block_axis || c == 0);
        if only_block_axis && access.reduction_coeff.unsigned_abs() > 1 {
            packable.push(idx);
        }
    }
    packable
}

fn choose_nr(extent: usize) -> usize {
    // Prefer NR that divides extent evenly (avoids tail loop).
    for nr in (2..=MAX_NR).rev() {
        if extent % nr == 0 {
            return nr;
        }
    }
    MAX_NR.min(extent)
}

/// Place non-blocked axes first (natural order), blocked axis last.
fn compute_axis_order(rank: usize, block_axis: Option<usize>) -> Vec<usize> {
    let mut order: Vec<usize> = (0..rank).filter(|&a| Some(a) != block_axis).collect();
    if let Some(ba) = block_axis {
        order.push(ba);
    }
    order
}

/// Choose cache tile sizes for a reduction kernel.
///
/// The goal is to keep the working set within L2 (~256KB–1MB).  For the
/// inner kernel body we touch:
///   - B panel:  KC × NC × 4 bytes   (should fit in L2)
///   - A strip:  1 × KC × 4 bytes    (fits easily)
///   - accumulators:  NR × 4 bytes   (in registers)
///
/// We target KC × NC × 4 ≤ L2_TARGET, rounding KC and NC to multiples of
/// NR for alignment.
fn compute_tiling(
    output_shape: &[usize],
    reduction_terms: usize,
    blocking: &BlockingStrategy,
    has_packable_access: bool,
) -> CacheTiling {
    // Tiling + packing is currently net-negative: JIT packing loop overhead
    // exceeds the cache benefit. Disabled by default; enable with WT_V8_TILE=1.
    if std::env::var("WT_V8_TILE").is_err() {
        return CacheTiling { kc: None, nc: None };
    }
    let _ = has_packable_access;
    const L2_TARGET: usize = 256 * 1024; // 256 KB target footprint
    const MIN_KC: usize = 64;
    const MIN_NC: usize = 64;

    let nr = blocking.nr;
    let k = reduction_terms;
    let n = blocking
        .block_axis
        .and_then(|a| output_shape.get(a).copied())
        .unwrap_or(1);

    // Don't bother tiling tiny problems.
    if k <= MIN_KC && n <= MIN_NC {
        return CacheTiling { kc: None, nc: None };
    }

    // Work out KC: how deep we can go in the reduction per pass.
    // We want KC * NC * 4 ≤ L2_TARGET.  Start with NC = min(n, 256), solve for KC.
    let nc_candidate = if n > MIN_NC { n.min(256) } else { n };
    let kc_from_l2 = L2_TARGET / (nc_candidate * 4);
    let kc = if kc_from_l2 >= k {
        None // full K fits
    } else {
        // Round down to multiple of 8 for alignment.
        Some((kc_from_l2.max(MIN_KC)) & !7)
    };

    let nc = if n > nc_candidate {
        // Round down to multiple of NR.
        Some((nc_candidate / nr) * nr)
    } else {
        None // full N fits in tile
    };

    CacheTiling { kc, nc }
}

/// Check if the blocked reduction can use SIMD (F32X4) vector operations.
///
/// Requirements:
/// - NR is a multiple of SIMD_F32_LANES (4)
/// - block_axis is the last output axis (so consecutive j values = contiguous memory)
/// - All varying accesses have output_coeffs[block_axis] == 1 (contiguous loads)
/// - The term expression contains no transcendentals (Exp/Ln/Tanh need scalar calls)
fn can_use_simd(spec: &ReductionKernelSpec) -> bool {
    let block_axis = match spec.blocking.block_axis {
        Some(a) => a,
        None => return false,
    };
    let nr = spec.blocking.nr;
    if nr < SIMD_F32_LANES || nr % SIMD_F32_LANES != 0 {
        return false;
    }
    // Block axis must be the last output axis (stride 1 in row-major output).
    if block_axis + 1 != spec.output_shape.len() {
        return false;
    }
    // All varying accesses must have stride 1 on the block axis.
    for (idx, access) in spec.accesses.iter().enumerate() {
        if !spec.blocking.invariant[idx] {
            let coeff = access.output_coeffs.get(block_axis).copied().unwrap_or(0);
            if coeff != 1 {
                return false;
            }
        }
    }
    !term_has_transcendentals(&spec.canonical_term)
}

fn term_has_transcendentals(term: &ReductionTermPattern) -> bool {
    match term {
        ReductionTermPattern::Load(_) | ReductionTermPattern::Literal(_) => false,
        ReductionTermPattern::Bin { a, b, .. } => {
            term_has_transcendentals(a) || term_has_transcendentals(b)
        }
        ReductionTermPattern::Unary { op, input } => {
            matches!(
                op,
                ReductionUnaryOp::Exp | ReductionUnaryOp::Ln | ReductionUnaryOp::Tanh
            ) || term_has_transcendentals(input)
        }
    }
}

// ---- Kernel compilation ----

/// Determine the parallelizable axis for a reduction kernel.
///
/// Returns `Some(axis)` if there's a non-blocked output axis that can be
/// partitioned across threads, or `None` for single-tile kernels.
fn reduction_parallel_axis(spec: &ReductionKernelSpec) -> Option<usize> {
    // The outermost non-blocked axis is axis_order[0], but only if there's
    // more than one axis (otherwise axis_order[0] is the blocked axis itself).
    if spec.axis_order.len() > 1 {
        let axis = spec.axis_order[0];
        if Some(axis) != spec.blocking.block_axis {
            return Some(axis);
        }
    }
    None
}

fn compile_kernels(
    specs: &[KernelSpec],
    layout: &TensorLayout,
) -> Result<NativeCompiledGraph, V8Error> {
    // Compute max packed panel size across all reduction specs.
    let max_panel_elements = specs
        .iter()
        .filter_map(|s| match s {
            KernelSpec::Reduction(r) if !r.packable.is_empty() => {
                let kc = r.tiling.kc.unwrap_or(r.reduction_terms);
                let nc = r.tiling.nc.unwrap_or(
                    r.blocking
                        .block_axis
                        .map(|a| r.output_shape[a])
                        .unwrap_or(1),
                );
                Some(kc * nc * r.packable.len())
            }
            _ => None,
        })
        .max()
        .unwrap_or(0);

    let mut scratch = vec![0.0f32; max_panel_elements.max(1)];
    let scratch_raw_ptr = scratch.as_mut_ptr() as i64;

    let (mut module, math_func_ids) = setup_jit_module()?;
    let ptr_type = module.isa().pointer_type();

    // Declare one function per kernel, each with signature:
    //   fn(ptr_table: ptr, i_start: i64, i_end: i64)
    let mut func_ids = Vec::with_capacity(specs.len());
    for idx in 0..specs.len() {
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(ptr_type)); // ptr_table
        sig.params.push(AbiParam::new(types::I64)); // i_start
        sig.params.push(AbiParam::new(types::I64)); // i_end
        let name = format!("wt_v8_k{idx}");
        let func_id = module
            .declare_function(&name, Linkage::Local, &sig)
            .map_err(CodegenError::from)?;
        func_ids.push((func_id, sig));
    }

    let mut fn_builder_ctx = FunctionBuilderContext::new();

    // Compile each kernel into its own function.
    for (idx, spec) in specs.iter().enumerate() {
        let (fid, ref sig) = func_ids[idx];
        let mut ctx = module.make_context();
        ctx.func.signature = sig.clone();

        let math_refs: HashMap<&str, cranelift_codegen::ir::FuncRef> = math_func_ids
            .iter()
            .map(|(name, mid)| {
                let fref = module.declare_func_in_func(*mid, &mut ctx.func);
                (*name, fref)
            })
            .collect();

        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);
        let entry = builder.create_block();
        builder.append_block_params_for_function_params(entry);
        builder.switch_to_block(entry);
        builder.seal_block(entry);

        let params = builder.block_params(entry);
        let ptr_table = params[0];
        let par_start = params[1];
        let par_end = params[2];

        let scratch_ptr_val = if max_panel_elements > 0 {
            Some(builder.ins().iconst(ptr_type, scratch_raw_ptr))
        } else {
            None
        };

        let mut next_var = 0u32;

        match spec {
            KernelSpec::Reduction(rspec) => {
                let output_ptr = load_tensor_base_ptr(
                    &mut builder,
                    layout,
                    ptr_table,
                    ptr_type,
                    rspec.output_tensor,
                )?;
                let mut tensor_ptrs: HashMap<GlobalId, cranelift_codegen::ir::Value> =
                    HashMap::new();
                for access in &rspec.accesses {
                    if !tensor_ptrs.contains_key(&access.tensor) {
                        let ptr = load_tensor_base_ptr(
                            &mut builder,
                            layout,
                            ptr_table,
                            ptr_type,
                            access.tensor,
                        )?;
                        tensor_ptrs.insert(access.tensor, ptr);
                    }
                }
                let input_ptrs: Vec<cranelift_codegen::ir::Value> = rspec
                    .accesses
                    .iter()
                    .map(|a| tensor_ptrs[&a.tensor])
                    .collect();

                let order = rspec.axis_order.clone();
                let mut axis_vals = vec![None; rspec.output_shape.len()];
                let par_axis = reduction_parallel_axis(rspec);
                emit_reduction_output_loops(
                    &mut builder,
                    &mut next_var,
                    rspec,
                    &order,
                    &mut axis_vals,
                    output_ptr,
                    &input_ptrs,
                    &math_refs,
                    scratch_ptr_val,
                    par_axis,
                    par_start,
                    par_end,
                )?;
            }
            KernelSpec::Pointwise(pspec) => {
                let output_ptr = load_tensor_base_ptr(
                    &mut builder,
                    layout,
                    ptr_table,
                    ptr_type,
                    pspec.output_tensor,
                )?;
                let mut tensor_ptrs: HashMap<GlobalId, cranelift_codegen::ir::Value> =
                    HashMap::new();
                for access in &pspec.accesses {
                    if !tensor_ptrs.contains_key(&access.tensor) {
                        let ptr = load_tensor_base_ptr(
                            &mut builder,
                            layout,
                            ptr_table,
                            ptr_type,
                            access.tensor,
                        )?;
                        tensor_ptrs.insert(access.tensor, ptr);
                    }
                }
                let input_ptrs: Vec<cranelift_codegen::ir::Value> = pspec
                    .accesses
                    .iter()
                    .map(|a| tensor_ptrs[&a.tensor])
                    .collect();

                let par_axis = if !pspec.output_shape.is_empty() {
                    Some(0)
                } else {
                    None
                };
                let mut axis_vals = vec![None; pspec.output_shape.len()];
                emit_pointwise_loops(
                    &mut builder,
                    &mut next_var,
                    pspec,
                    0,
                    &mut axis_vals,
                    output_ptr,
                    &input_ptrs,
                    &math_refs,
                    par_axis,
                    par_start,
                    par_end,
                )?;
            }
        }

        builder.ins().return_(&[]);
        builder.finalize();

        module
            .define_function(fid, &mut ctx)
            .map_err(CodegenError::from)?;
        module.clear_context(&mut ctx);
    }

    module.finalize_definitions().expect("finalize JIT");

    // Resolve function pointers and build CompiledKernel metadata.
    let mut compiled_kernels = Vec::with_capacity(specs.len());
    for (idx, spec) in specs.iter().enumerate() {
        let (fid, _) = func_ids[idx];
        let func_ptr = module.get_finalized_function(fid);
        let (parallel_extent, output_tensor, input_tensors) = match spec {
            KernelSpec::Reduction(r) => {
                let par_axis = reduction_parallel_axis(r);
                let extent = par_axis.map(|a| r.output_shape[a]).unwrap_or(1);
                let mut inputs: Vec<GlobalId> = r
                    .accesses
                    .iter()
                    .map(|a| a.tensor)
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .collect();
                inputs.sort();
                (extent, r.output_tensor, inputs)
            }
            KernelSpec::Pointwise(p) => {
                let extent = p.output_shape.first().copied().unwrap_or(1);
                let mut inputs: Vec<GlobalId> = p
                    .accesses
                    .iter()
                    .map(|a| a.tensor)
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .collect();
                inputs.sort();
                (extent, p.output_tensor, inputs)
            }
        };
        compiled_kernels.push(CompiledKernel {
            func_ptr,
            parallel_extent,
            output_tensor,
            input_tensors,
        });
    }

    Ok(NativeCompiledGraph {
        _module: module,
        layout: layout.clone(),
        kernels: compiled_kernels,
        _scratch: scratch,
    })
}

// ---- Reduction emission ----

/// Emit the full loop nest for a reduction kernel, including cache tiling.
///
/// With tiling, the structure becomes (for a matmul):
/// ```text
/// for jc in 0..N step NC:        // blocked-axis tile
///   for kc in 0..K step KC:      // reduction tile
///     for i in 0..M:             // non-blocked output axes
///       for j in jc..jc+NC step NR:  // blocked axis within tile
///         inner_kernel(k_start=kc, k_end=min(kc+KC,K), first_kc=(kc==0))
/// ```
/// On the first kc pass, accumulators start at 0.  On subsequent passes
/// they reload partial sums from the output buffer and accumulate further.
fn emit_reduction_output_loops(
    builder: &mut FunctionBuilder,
    next_var: &mut u32,
    spec: &ReductionKernelSpec,
    axis_order: &[usize],
    axis_vals: &mut [Option<cranelift_codegen::ir::Value>],
    output_ptr: cranelift_codegen::ir::Value,
    input_ptrs: &[cranelift_codegen::ir::Value],
    math_refs: &HashMap<&str, cranelift_codegen::ir::FuncRef>,
    scratch_ptr: Option<cranelift_codegen::ir::Value>,
    par_axis: Option<usize>,
    par_start: cranelift_codegen::ir::Value,
    par_end: cranelift_codegen::ir::Value,
) -> Result<(), V8Error> {
    let k_total = spec.reduction_terms as i64;
    let kc_size = spec.tiling.kc.map(|v| v as i64);
    let block_axis = spec.blocking.block_axis;
    let nc_size = spec.tiling.nc.map(|v| v as i64);
    let blocked_extent = block_axis.map(|a| spec.output_shape[a] as i64).unwrap_or(0);

    let use_nc_tiling = matches!((nc_size, block_axis), (Some(nc), Some(_)) if nc < blocked_extent);

    if use_nc_tiling {
        let nc_step = nc_size.unwrap();
        let jc_iv = alloc_loop_var(next_var);
        emit_for_loop(
            builder,
            jc_iv,
            0,
            blocked_extent,
            nc_step,
            |builder, jc_val| {
                let jc_end_raw = builder.ins().iadd_imm(jc_val, nc_step);
                let ext_v = builder.ins().iconst(types::I64, blocked_extent);
                let cmp = builder.ins().icmp(IntCC::SignedLessThan, jc_end_raw, ext_v);
                let jc_end = builder.ins().select(cmp, jc_end_raw, ext_v);
                emit_kc_and_inner_axes(
                    builder,
                    next_var,
                    spec,
                    axis_order,
                    axis_vals,
                    output_ptr,
                    input_ptrs,
                    math_refs,
                    k_total,
                    kc_size,
                    Some(jc_val),
                    Some(jc_end),
                    scratch_ptr,
                    par_axis,
                    par_start,
                    par_end,
                )
            },
        )
    } else {
        emit_kc_and_inner_axes(
            builder,
            next_var,
            spec,
            axis_order,
            axis_vals,
            output_ptr,
            input_ptrs,
            math_refs,
            k_total,
            kc_size,
            None,
            None,
            scratch_ptr,
            par_axis,
            par_start,
            par_end,
        )
    }
}

/// Emit the KC tiling loop (if needed), B-panel packing, and inner axes.
#[allow(clippy::too_many_arguments)]
fn emit_kc_and_inner_axes(
    builder: &mut FunctionBuilder,
    next_var: &mut u32,
    spec: &ReductionKernelSpec,
    axis_order: &[usize],
    axis_vals: &mut [Option<cranelift_codegen::ir::Value>],
    output_ptr: cranelift_codegen::ir::Value,
    input_ptrs: &[cranelift_codegen::ir::Value],
    math_refs: &HashMap<&str, cranelift_codegen::ir::FuncRef>,
    k_total: i64,
    kc_size: Option<i64>,
    jc_start: Option<cranelift_codegen::ir::Value>,
    jc_end: Option<cranelift_codegen::ir::Value>,
    scratch_ptr: Option<cranelift_codegen::ir::Value>,
    par_axis: Option<usize>,
    par_start: cranelift_codegen::ir::Value,
    par_end: cranelift_codegen::ir::Value,
) -> Result<(), V8Error> {
    let do_packing = !spec.packable.is_empty()
        && scratch_ptr.is_some()
        && jc_start.is_some()
        && kc_size.is_some();

    if let Some(kc_step) = kc_size {
        let kc_iv = alloc_loop_var(next_var);
        emit_for_loop(builder, kc_iv, 0, k_total, kc_step, |builder, kc_val| {
            let k_end_raw = builder.ins().iadd_imm(kc_val, kc_step);
            let k_total_v = builder.ins().iconst(types::I64, k_total);
            let cmp = builder
                .ins()
                .icmp(IntCC::SignedLessThan, k_end_raw, k_total_v);
            let k_end = builder.ins().select(cmp, k_end_raw, k_total_v);
            let is_first = builder.ins().icmp_imm(IntCC::Equal, kc_val, 0);

            // Emit packing loops and create packing context.
            let packing = if do_packing {
                let js = jc_start.unwrap();
                let je = jc_end.unwrap();
                let nc_ext = builder.ins().isub(je, js);
                emit_packing_loops(
                    builder,
                    next_var,
                    spec,
                    input_ptrs,
                    scratch_ptr.unwrap(),
                    kc_val,
                    k_end,
                    js,
                    nc_ext,
                )?;
                let mut is_packed = vec![false; spec.accesses.len()];
                for &idx in &spec.packable {
                    is_packed[idx] = true;
                }
                Some(TilePackingCtx {
                    packed_ptr: scratch_ptr.unwrap(),
                    k_start: kc_val,
                    jc_start: js,
                    nc_extent: nc_ext,
                    is_packed,
                })
            } else {
                None
            };

            emit_reduction_inner_axes(
                builder,
                next_var,
                spec,
                axis_order,
                0,
                axis_vals,
                output_ptr,
                input_ptrs,
                math_refs,
                kc_val,
                k_end,
                is_first,
                jc_start,
                jc_end,
                packing.as_ref(),
                par_axis,
                par_start,
                par_end,
            )
        })
    } else {
        // No kc tiling — full reduction in one pass, no packing.
        let k_start = builder.ins().iconst(types::I64, 0);
        let k_end = builder.ins().iconst(types::I64, k_total);
        let is_first = builder.ins().iconst(types::I8, 1);
        emit_reduction_inner_axes(
            builder, next_var, spec, axis_order, 0, axis_vals, output_ptr, input_ptrs, math_refs,
            k_start, k_end, is_first, jc_start, jc_end, None, par_axis, par_start, par_end,
        )
    }
}

/// Emit the packing loops that copy varying accesses into a contiguous
/// scratch buffer for the current KC×NC tile.
///
/// Packed layout: `packed[(k-k_start)*nc_extent + (j-jc_start)] = src[flat(k,j)]`
#[allow(clippy::too_many_arguments)]
fn emit_packing_loops(
    builder: &mut FunctionBuilder,
    next_var: &mut u32,
    spec: &ReductionKernelSpec,
    input_ptrs: &[cranelift_codegen::ir::Value],
    packed_ptr: cranelift_codegen::ir::Value,
    k_start: cranelift_codegen::ir::Value,
    k_end: cranelift_codegen::ir::Value,
    jc_start: cranelift_codegen::ir::Value,
    nc_extent: cranelift_codegen::ir::Value,
) -> Result<(), V8Error> {
    let block_axis = spec.blocking.block_axis.unwrap();
    let pack_flags = MemFlags::new().with_notrap();

    for (pack_idx, &access_idx) in spec.packable.iter().enumerate() {
        let access = &spec.accesses[access_idx];
        let src_ptr = input_ptrs[access_idx];
        let panel_offset = if pack_idx == 0 {
            0i64
        } else {
            // Multiple packed accesses: offset by kc*nc per panel.
            // We compute this dynamically since kc_extent varies for last tile.
            // For simplicity, use the max tile size (from spec.tiling).
            let kc_max = spec.tiling.kc.unwrap_or(spec.reduction_terms) as i64;
            let nc_max = spec.tiling.nc.unwrap_or(
                spec.blocking
                    .block_axis
                    .map(|a| spec.output_shape[a])
                    .unwrap_or(1),
            ) as i64;
            (pack_idx as i64) * kc_max * nc_max
        };

        let block_coeff = access.output_coeffs[block_axis];
        let red_coeff = access.reduction_coeff;
        let base = access.base;

        // for pk in 0..(k_end - k_start):
        let kc_extent = builder.ins().isub(k_end, k_start);
        let pk_iv = alloc_loop_var(next_var);
        let zero_pk = builder.ins().iconst(types::I64, 0);
        emit_for_loop_dynamic(builder, pk_iv, zero_pk, kc_extent, 1, |builder, pk| {
            let k_global = builder.ins().iadd(k_start, pk);
            // for pj in 0..nc_extent:
            let pj_iv = alloc_loop_var(next_var);
            let zero = builder.ins().iconst(types::I64, 0);
            emit_for_loop_dynamic(builder, pj_iv, zero, nc_extent, 1, |builder, pj| {
                let j_global = builder.ins().iadd(jc_start, pj);
                // src_flat = base + block_coeff * j_global + red_coeff * k_global
                let mut src_flat = builder.ins().iconst(types::I64, base);
                src_flat = add_scaled_i64(builder, src_flat, j_global, block_coeff);
                src_flat = add_scaled_i64(builder, src_flat, k_global, red_coeff);
                let src_val = load_f32_at_flat(builder, src_ptr, src_flat);
                // dst_idx = panel_offset + pk * nc_extent + pj
                let dst_idx = builder.ins().imul(pk, nc_extent);
                let dst_idx = builder.ins().iadd(dst_idx, pj);
                let dst_idx = if panel_offset != 0 {
                    builder.ins().iadd_imm(dst_idx, panel_offset)
                } else {
                    dst_idx
                };
                let byte_off = builder.ins().ishl_imm(dst_idx, 2);
                let addr = builder.ins().iadd(packed_ptr, byte_off);
                builder.ins().store(pack_flags, src_val, addr, 0);
                Ok::<(), V8Error>(())
            })
        })?;
    }
    Ok(())
}

/// Emit the inner output-axis loops within a tiling context.
///
/// `k_start`/`k_end`: reduction range for this tile.
/// `is_first_kc`: i8 value, 1 if this is the first kc tile (init to 0), 0 if
///   accumulators should be loaded from output.
/// `jc_start`/`jc_end`: blocked-axis tile bounds (None = full extent).
#[allow(clippy::too_many_arguments)]
fn emit_reduction_inner_axes(
    builder: &mut FunctionBuilder,
    next_var: &mut u32,
    spec: &ReductionKernelSpec,
    axis_order: &[usize],
    depth: usize,
    axis_vals: &mut [Option<cranelift_codegen::ir::Value>],
    output_ptr: cranelift_codegen::ir::Value,
    input_ptrs: &[cranelift_codegen::ir::Value],
    math_refs: &HashMap<&str, cranelift_codegen::ir::FuncRef>,
    k_start: cranelift_codegen::ir::Value,
    k_end: cranelift_codegen::ir::Value,
    is_first_kc: cranelift_codegen::ir::Value,
    jc_start: Option<cranelift_codegen::ir::Value>,
    jc_end: Option<cranelift_codegen::ir::Value>,
    packing: Option<&TilePackingCtx>,
    par_axis: Option<usize>,
    par_start: cranelift_codegen::ir::Value,
    par_end: cranelift_codegen::ir::Value,
) -> Result<(), V8Error> {
    if depth == axis_order.len() {
        return emit_scalar_reduction(
            builder,
            next_var,
            spec,
            axis_vals,
            output_ptr,
            input_ptrs,
            math_refs,
            k_start,
            k_end,
            is_first_kc,
            packing,
        );
    }

    let axis = axis_order[depth];
    let full_extent = spec.output_shape[axis] as i64;
    let is_blocked = spec.blocking.block_axis == Some(axis) && spec.blocking.nr > 1;
    let is_parallel = par_axis == Some(axis);

    if is_blocked {
        let nr = spec.blocking.nr;

        let (loop_start, loop_end_val) = if let (Some(js), Some(je)) = (jc_start, jc_end) {
            (js, je)
        } else {
            let s = builder.ins().iconst(types::I64, 0);
            let e = builder.ins().iconst(types::I64, full_extent);
            (s, e)
        };

        if jc_start.is_none() {
            let main_end = (full_extent / nr as i64) * nr as i64;

            if main_end > 0 {
                let iv = alloc_loop_var(next_var);
                emit_for_loop(builder, iv, 0, main_end, nr as i64, |builder, j0| {
                    emit_blocked_reduction(
                        builder,
                        next_var,
                        spec,
                        axis,
                        j0,
                        axis_vals,
                        output_ptr,
                        input_ptrs,
                        math_refs,
                        k_start,
                        k_end,
                        is_first_kc,
                        packing,
                    )
                })?;
            }

            if main_end < full_extent {
                let iv = alloc_loop_var(next_var);
                emit_for_loop(builder, iv, main_end, full_extent, 1, |builder, j| {
                    axis_vals[axis] = Some(j);
                    emit_scalar_reduction(
                        builder,
                        next_var,
                        spec,
                        axis_vals,
                        output_ptr,
                        input_ptrs,
                        math_refs,
                        k_start,
                        k_end,
                        is_first_kc,
                        packing,
                    )
                })?;
            }
        } else {
            let range = builder.ins().isub(loop_end_val, loop_start);
            let nr_v = builder.ins().iconst(types::I64, nr as i64);
            let main_count_raw = builder.ins().sdiv(range, nr_v);
            let main_count = builder.ins().imul(main_count_raw, nr_v);
            let main_end = builder.ins().iadd(loop_start, main_count);

            let iv = alloc_loop_var(next_var);
            emit_for_loop_dynamic(
                builder,
                iv,
                loop_start,
                main_end,
                nr as i64,
                |builder, j0| {
                    emit_blocked_reduction(
                        builder,
                        next_var,
                        spec,
                        axis,
                        j0,
                        axis_vals,
                        output_ptr,
                        input_ptrs,
                        math_refs,
                        k_start,
                        k_end,
                        is_first_kc,
                        packing,
                    )
                },
            )?;

            let iv = alloc_loop_var(next_var);
            emit_for_loop_dynamic(builder, iv, main_end, loop_end_val, 1, |builder, j| {
                axis_vals[axis] = Some(j);
                emit_scalar_reduction(
                    builder,
                    next_var,
                    spec,
                    axis_vals,
                    output_ptr,
                    input_ptrs,
                    math_refs,
                    k_start,
                    k_end,
                    is_first_kc,
                    packing,
                )
            })?;
        }

        Ok(())
    } else if is_parallel {
        // Use the function's par_start..par_end range for this axis.
        let iv = alloc_loop_var(next_var);
        emit_for_loop_dynamic(builder, iv, par_start, par_end, 1, |builder, val| {
            axis_vals[axis] = Some(val);
            emit_reduction_inner_axes(
                builder,
                next_var,
                spec,
                axis_order,
                depth + 1,
                axis_vals,
                output_ptr,
                input_ptrs,
                math_refs,
                k_start,
                k_end,
                is_first_kc,
                jc_start,
                jc_end,
                packing,
                par_axis,
                par_start,
                par_end,
            )
        })
    } else {
        let iv = alloc_loop_var(next_var);
        emit_for_loop(builder, iv, 0, full_extent, 1, |builder, val| {
            axis_vals[axis] = Some(val);
            emit_reduction_inner_axes(
                builder,
                next_var,
                spec,
                axis_order,
                depth + 1,
                axis_vals,
                output_ptr,
                input_ptrs,
                math_refs,
                k_start,
                k_end,
                is_first_kc,
                jc_start,
                jc_end,
                packing,
                par_axis,
                par_start,
                par_end,
            )
        })
    }
}

/// Emit the NR-way blocked reduction inner body.
///
/// For each k-iteration:
///   1. Load invariant accesses once (their output_coeffs[block_axis] == 0).
///   2. For each lane 0..NR (unrolled at emit time):
///      a. Load varying accesses at block_axis = j0 + lane.
///      b. Evaluate canonical_term with all loaded values.
///      c. Accumulate into acc[lane].
/// After the k-loop, store NR results.
///
/// `k_start`/`k_end`: reduction loop bounds (for cache tiling).
/// `is_first_kc`: i8, 1 = init accumulators to 0, 0 = reload from output.
#[allow(clippy::too_many_arguments)]
fn emit_blocked_reduction(
    builder: &mut FunctionBuilder,
    next_var: &mut u32,
    spec: &ReductionKernelSpec,
    block_axis: usize,
    j0: cranelift_codegen::ir::Value,
    axis_vals: &mut [Option<cranelift_codegen::ir::Value>],
    output_ptr: cranelift_codegen::ir::Value,
    input_ptrs: &[cranelift_codegen::ir::Value],
    math_refs: &HashMap<&str, cranelift_codegen::ir::FuncRef>,
    k_start: cranelift_codegen::ir::Value,
    k_end: cranelift_codegen::ir::Value,
    is_first_kc: cranelift_codegen::ir::Value,
    packing: Option<&TilePackingCtx>,
) -> Result<(), V8Error> {
    if can_use_simd(spec) {
        return emit_blocked_reduction_simd(
            builder,
            next_var,
            spec,
            block_axis,
            j0,
            axis_vals,
            output_ptr,
            input_ptrs,
            math_refs,
            k_start,
            k_end,
            is_first_kc,
            packing,
        );
    }

    let nr = spec.blocking.nr;

    // Allocate NR accumulator variables.
    let acc_vars: Vec<Variable> = (0..nr)
        .map(|_| {
            let v = Variable::from_u32(alloc_loop_var(next_var));
            builder.declare_var(v, types::F32);
            v
        })
        .collect();

    // Init accumulators: if is_first_kc, zero; otherwise reload from output.
    let zero = builder.ins().f32const(0.0);
    for (lane, &acc_v) in acc_vars.iter().enumerate() {
        let j_lane = if lane == 0 {
            j0
        } else {
            builder.ins().iadd_imm(j0, lane as i64)
        };
        axis_vals[block_axis] = Some(j_lane);
        let out_flat = emit_output_flat_index(builder, &spec.output_shape, axis_vals)?;
        let prev = load_f32_at_flat(builder, output_ptr, out_flat);
        // select: is_first_kc ? 0.0 : prev
        let init = builder.ins().select(is_first_kc, zero, prev);
        builder.def_var(acc_v, init);
    }

    // Reduction loop over k_start..k_end.
    let k_iv = alloc_loop_var(next_var);
    emit_for_loop_dynamic(builder, k_iv, k_start, k_end, 1, |builder, k_val| {
        // Load invariant accesses once (they don't depend on block_axis).
        let mut invariant_vals: Vec<Option<cranelift_codegen::ir::Value>> =
            vec![None; spec.accesses.len()];
        for (idx, access) in spec.accesses.iter().enumerate() {
            if spec.blocking.invariant[idx] {
                // block_axis coeff is 0, so the value we set here is irrelevant.
                axis_vals[block_axis] = Some(j0);
                let flat = emit_access_index(builder, access, axis_vals, Some(k_val));
                invariant_vals[idx] = Some(load_f32_at_flat(builder, input_ptrs[idx], flat));
            }
        }

        // Unrolled lane loop (NR iterations at Cranelift-emit time).
        for lane in 0..nr {
            let j_lane = if lane == 0 {
                j0
            } else {
                builder.ins().iadd_imm(j0, lane as i64)
            };
            axis_vals[block_axis] = Some(j_lane);

            // Build load_vals for this lane.
            let mut load_vals = Vec::with_capacity(spec.accesses.len());
            for (idx, access) in spec.accesses.iter().enumerate() {
                if spec.blocking.invariant[idx] {
                    load_vals.push(invariant_vals[idx].unwrap());
                } else {
                    let flat = emit_access_index(builder, access, axis_vals, Some(k_val));
                    load_vals.push(load_f32_at_flat(builder, input_ptrs[idx], flat));
                }
            }

            // Evaluate canonical term generically.
            let result =
                emit_term_expr(builder, &spec.canonical_term, &load_vals, math_refs, None)?;

            // Accumulate.
            let cur = builder.use_var(acc_vars[lane]);
            let next = builder.ins().fadd(cur, result);
            builder.def_var(acc_vars[lane], next);
        }

        Ok::<(), V8Error>(())
    })?;

    // Store NR results.
    for lane in 0..nr {
        let j_lane = if lane == 0 {
            j0
        } else {
            builder.ins().iadd_imm(j0, lane as i64)
        };
        axis_vals[block_axis] = Some(j_lane);
        let out_flat = emit_output_flat_index(builder, &spec.output_shape, axis_vals)?;
        let acc = builder.use_var(acc_vars[lane]);
        store_f32_at_flat(builder, output_ptr, out_flat, acc);
    }

    Ok(())
}

/// SIMD (F32X4) version of the blocked reduction.
///
/// Instead of NR scalar accumulators, uses NR/4 vector accumulators.
/// Invariant accesses are loaded as scalars and splatted.  Varying accesses
/// are loaded as 4-wide contiguous vector loads.  The canonical term is
/// evaluated on F32X4 values (Cranelift arithmetic is type-polymorphic).
#[allow(clippy::too_many_arguments)]
fn emit_blocked_reduction_simd(
    builder: &mut FunctionBuilder,
    next_var: &mut u32,
    spec: &ReductionKernelSpec,
    block_axis: usize,
    j0: cranelift_codegen::ir::Value,
    axis_vals: &mut [Option<cranelift_codegen::ir::Value>],
    output_ptr: cranelift_codegen::ir::Value,
    input_ptrs: &[cranelift_codegen::ir::Value],
    math_refs: &HashMap<&str, cranelift_codegen::ir::FuncRef>,
    k_start: cranelift_codegen::ir::Value,
    k_end: cranelift_codegen::ir::Value,
    is_first_kc: cranelift_codegen::ir::Value,
    packing: Option<&TilePackingCtx>,
) -> Result<(), V8Error> {
    let vlen = SIMD_F32_LANES;
    let vty = types::F32.by(vlen as u32).unwrap();
    let nr = spec.blocking.nr;
    let nvec = nr / vlen;
    // MemFlags without alignment assertion (SIMD loads may not be 16-byte aligned).
    let simd_flags = MemFlags::new().with_notrap();

    // Allocate nvec vector accumulator variables.
    let acc_vars: Vec<Variable> = (0..nvec)
        .map(|_| {
            let v = Variable::from_u32(alloc_loop_var(next_var));
            builder.declare_var(v, vty);
            v
        })
        .collect();

    // Init accumulators: if is_first_kc, zero vector; otherwise vector-load from output.
    let f32_zero = builder.ins().f32const(0.0);
    let vec_zero = builder.ins().splat(vty, f32_zero);
    for (vi, &acc_v) in acc_vars.iter().enumerate() {
        let j_group = if vi == 0 {
            j0
        } else {
            builder.ins().iadd_imm(j0, (vi * vlen) as i64)
        };
        axis_vals[block_axis] = Some(j_group);
        let out_flat = emit_output_flat_index(builder, &spec.output_shape, axis_vals)?;
        let byte_off = builder.ins().ishl_imm(out_flat, 2);
        let addr = builder.ins().iadd(output_ptr, byte_off);
        let prev = builder.ins().load(vty, simd_flags, addr, 0);
        let init = builder.ins().select(is_first_kc, vec_zero, prev);
        builder.def_var(acc_v, init);
    }

    // Reduction loop over k_start..k_end.
    let k_iv = alloc_loop_var(next_var);
    emit_for_loop_dynamic(builder, k_iv, k_start, k_end, 1, |builder, k_val| {
        // Load invariant accesses (scalar) and splat to vector.
        let mut invariant_vecs: Vec<Option<cranelift_codegen::ir::Value>> =
            vec![None; spec.accesses.len()];
        for (idx, access) in spec.accesses.iter().enumerate() {
            if spec.blocking.invariant[idx] {
                axis_vals[block_axis] = Some(j0);
                let flat = emit_access_index(builder, access, axis_vals, Some(k_val));
                let scalar = load_f32_at_flat(builder, input_ptrs[idx], flat);
                invariant_vecs[idx] = Some(builder.ins().splat(vty, scalar));
            }
        }

        // Process nvec vector groups.
        for vi in 0..nvec {
            let j_group = if vi == 0 {
                j0
            } else {
                builder.ins().iadd_imm(j0, (vi * vlen) as i64)
            };
            axis_vals[block_axis] = Some(j_group);

            // Build vector load_vals.
            let mut load_vals = Vec::with_capacity(spec.accesses.len());
            for (idx, access) in spec.accesses.iter().enumerate() {
                if spec.blocking.invariant[idx] {
                    load_vals.push(invariant_vecs[idx].unwrap());
                } else if let Some(pack) = packing.filter(|p| p.is_packed[idx]) {
                    // Packed vector load: 4 contiguous elements from packed buffer.
                    // packed_flat = (k - k_start) * nc_extent + (j - jc_start)
                    let k_off = builder.ins().isub(k_val, pack.k_start);
                    let j_off = builder.ins().isub(j_group, pack.jc_start);
                    let packed_flat = builder.ins().imul(k_off, pack.nc_extent);
                    let packed_flat = builder.ins().iadd(packed_flat, j_off);
                    let byte_off = builder.ins().ishl_imm(packed_flat, 2);
                    let addr = builder.ins().iadd(pack.packed_ptr, byte_off);
                    load_vals.push(builder.ins().load(vty, simd_flags, addr, 0));
                } else {
                    // Unpacked vector load: 4 contiguous elements from original.
                    let flat = emit_access_index(builder, access, axis_vals, Some(k_val));
                    let byte_off = builder.ins().ishl_imm(flat, 2);
                    let addr = builder.ins().iadd(input_ptrs[idx], byte_off);
                    load_vals.push(builder.ins().load(vty, simd_flags, addr, 0));
                }
            }

            // Evaluate canonical term on vector values.
            let result = emit_term_expr(
                builder,
                &spec.canonical_term,
                &load_vals,
                math_refs,
                Some(vty),
            )?;

            // Accumulate.
            let cur = builder.use_var(acc_vars[vi]);
            let next = builder.ins().fadd(cur, result);
            builder.def_var(acc_vars[vi], next);
        }

        Ok::<(), V8Error>(())
    })?;

    // Vector-store results.
    for (vi, &acc_v) in acc_vars.iter().enumerate() {
        let j_group = if vi == 0 {
            j0
        } else {
            builder.ins().iadd_imm(j0, (vi * vlen) as i64)
        };
        axis_vals[block_axis] = Some(j_group);
        let out_flat = emit_output_flat_index(builder, &spec.output_shape, axis_vals)?;
        let byte_off = builder.ins().ishl_imm(out_flat, 2);
        let addr = builder.ins().iadd(output_ptr, byte_off);
        let acc = builder.use_var(acc_v);
        builder.ins().store(simd_flags, acc, addr, 0);
    }

    Ok(())
}

/// Emit a scalar (NR=1) reduction: single accumulator, no blocking.
///
/// `k_start`/`k_end`: reduction loop bounds (for cache tiling).
/// `is_first_kc`: i8, 1 = init accumulator to 0, 0 = reload from output.
#[allow(clippy::too_many_arguments)]
fn emit_scalar_reduction(
    builder: &mut FunctionBuilder,
    next_var: &mut u32,
    spec: &ReductionKernelSpec,
    axis_vals: &[Option<cranelift_codegen::ir::Value>],
    output_ptr: cranelift_codegen::ir::Value,
    input_ptrs: &[cranelift_codegen::ir::Value],
    math_refs: &HashMap<&str, cranelift_codegen::ir::FuncRef>,
    k_start: cranelift_codegen::ir::Value,
    k_end: cranelift_codegen::ir::Value,
    is_first_kc: cranelift_codegen::ir::Value,
    packing: Option<&TilePackingCtx>,
) -> Result<(), V8Error> {
    let acc_var = Variable::from_u32(alloc_loop_var(next_var));
    builder.declare_var(acc_var, types::F32);

    let zero = builder.ins().f32const(0.0);
    let out_flat = emit_output_flat_index(builder, &spec.output_shape, axis_vals)?;
    let prev = load_f32_at_flat(builder, output_ptr, out_flat);
    let init = builder.ins().select(is_first_kc, zero, prev);
    builder.def_var(acc_var, init);

    let block_axis = spec.blocking.block_axis;

    let k_iv = alloc_loop_var(next_var);
    emit_for_loop_dynamic(builder, k_iv, k_start, k_end, 1, |builder, k_val| {
        let mut load_vals = Vec::with_capacity(spec.accesses.len());
        for (idx, access) in spec.accesses.iter().enumerate() {
            if let Some(pack) = packing.filter(|p| p.is_packed[idx]) {
                // Packed scalar load.
                let k_off = builder.ins().isub(k_val, pack.k_start);
                let j_val = block_axis
                    .and_then(|a| axis_vals[a])
                    .unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
                let j_off = builder.ins().isub(j_val, pack.jc_start);
                let packed_flat = builder.ins().imul(k_off, pack.nc_extent);
                let packed_flat = builder.ins().iadd(packed_flat, j_off);
                load_vals.push(load_f32_at_flat(builder, pack.packed_ptr, packed_flat));
            } else {
                let flat = emit_access_index(builder, access, axis_vals, Some(k_val));
                load_vals.push(load_f32_at_flat(builder, input_ptrs[idx], flat));
            }
        }
        let result = emit_term_expr(builder, &spec.canonical_term, &load_vals, math_refs, None)?;
        let cur = builder.use_var(acc_var);
        let next = builder.ins().fadd(cur, result);
        builder.def_var(acc_var, next);
        Ok::<(), V8Error>(())
    })?;

    let acc = builder.use_var(acc_var);
    store_f32_at_flat(builder, output_ptr, out_flat, acc);
    Ok(())
}

// ---- Pointwise emission ----

fn emit_pointwise_loops(
    builder: &mut FunctionBuilder,
    next_var: &mut u32,
    spec: &PointwiseKernelSpec,
    depth: usize,
    axis_vals: &mut [Option<cranelift_codegen::ir::Value>],
    output_ptr: cranelift_codegen::ir::Value,
    input_ptrs: &[cranelift_codegen::ir::Value],
    math_refs: &HashMap<&str, cranelift_codegen::ir::FuncRef>,
    par_axis: Option<usize>,
    par_start: cranelift_codegen::ir::Value,
    par_end: cranelift_codegen::ir::Value,
) -> Result<(), V8Error> {
    if depth == spec.output_shape.len() {
        let mut load_vals = Vec::with_capacity(spec.accesses.len());
        for (idx, access) in spec.accesses.iter().enumerate() {
            let flat = emit_access_index(builder, access, axis_vals, None);
            load_vals.push(load_f32_at_flat(builder, input_ptrs[idx], flat));
        }
        let out_flat = emit_output_flat_index(builder, &spec.output_shape, axis_vals)?;
        let result = emit_term_expr(builder, &spec.canonical_term, &load_vals, math_refs, None)?;
        store_f32_at_flat(builder, output_ptr, out_flat, result);
        return Ok(());
    }

    let extent = spec.output_shape[depth] as i64;
    let is_parallel = par_axis == Some(depth);
    let iv = alloc_loop_var(next_var);
    if is_parallel {
        emit_for_loop_dynamic(builder, iv, par_start, par_end, 1, |builder, val| {
            axis_vals[depth] = Some(val);
            emit_pointwise_loops(
                builder,
                next_var,
                spec,
                depth + 1,
                axis_vals,
                output_ptr,
                input_ptrs,
                math_refs,
                par_axis,
                par_start,
                par_end,
            )
        })
    } else {
        emit_for_loop(builder, iv, 0, extent, 1, |builder, val| {
            axis_vals[depth] = Some(val);
            emit_pointwise_loops(
                builder,
                next_var,
                spec,
                depth + 1,
                axis_vals,
                output_ptr,
                input_ptrs,
                math_refs,
                par_axis,
                par_start,
                par_end,
            )
        })
    }
}

// ---- Term expression emission ----

/// Emit the expression tree for a reduction term.
///
/// When `vector_type` is `Some(ty)`, all operations produce SIMD vectors of
/// that type (e.g. `F32X4`).  Literals are splatted and `load_vals` must
/// already be vector values.  Transcendentals (Exp/Ln/Tanh) are not supported
/// in vector mode — the caller must check `term_has_transcendentals` first.
fn emit_term_expr(
    builder: &mut FunctionBuilder,
    term: &ReductionTermPattern,
    load_vals: &[cranelift_codegen::ir::Value],
    math_refs: &HashMap<&str, cranelift_codegen::ir::FuncRef>,
    vector_type: Option<cranelift_codegen::ir::Type>,
) -> Result<cranelift_codegen::ir::Value, V8Error> {
    Ok(match term {
        ReductionTermPattern::Load(i) => *load_vals
            .get(*i)
            .ok_or_else(|| V8Error::Unsupported(format!("term load index {i} out of range")))?,
        ReductionTermPattern::Literal(bits) => {
            let v = f64::from_bits(*bits) as f32;
            let scalar = builder.ins().f32const(v);
            if let Some(vty) = vector_type {
                builder.ins().splat(vty, scalar)
            } else {
                scalar
            }
        }
        ReductionTermPattern::Bin { op, a, b } => {
            let va = emit_term_expr(builder, a, load_vals, math_refs, vector_type)?;
            let vb = emit_term_expr(builder, b, load_vals, math_refs, vector_type)?;
            match op {
                ReductionBinOp::Add => builder.ins().fadd(va, vb),
                ReductionBinOp::Sub => builder.ins().fsub(va, vb),
                ReductionBinOp::Mul => builder.ins().fmul(va, vb),
                ReductionBinOp::Div => builder.ins().fdiv(va, vb),
                ReductionBinOp::Max => builder.ins().fmax(va, vb),
                ReductionBinOp::Min => builder.ins().fmin(va, vb),
            }
        }
        ReductionTermPattern::Unary { op, input } => {
            let vin = emit_term_expr(builder, input, load_vals, math_refs, vector_type)?;
            match op {
                ReductionUnaryOp::Neg => builder.ins().fneg(vin),
                ReductionUnaryOp::Abs => builder.ins().fabs(vin),
                ReductionUnaryOp::Sqrt => builder.ins().sqrt(vin),
                ReductionUnaryOp::Floor => builder.ins().floor(vin),
                ReductionUnaryOp::Ceil => builder.ins().ceil(vin),
                ReductionUnaryOp::Reciprocal => {
                    let one = builder.ins().f32const(1.0);
                    let one = if let Some(vty) = vector_type {
                        builder.ins().splat(vty, one)
                    } else {
                        one
                    };
                    builder.ins().fdiv(one, vin)
                }
                ReductionUnaryOp::Exp => {
                    let fref = math_refs["wt_expf"];
                    let call = builder.ins().call(fref, &[vin]);
                    builder.inst_results(call)[0]
                }
                ReductionUnaryOp::Ln => {
                    let fref = math_refs["wt_logf"];
                    let call = builder.ins().call(fref, &[vin]);
                    builder.inst_results(call)[0]
                }
                ReductionUnaryOp::Tanh => {
                    let fref = math_refs["wt_tanhf"];
                    let call = builder.ins().call(fref, &[vin]);
                    builder.inst_results(call)[0]
                }
            }
        }
    })
}

// ---- Cranelift helpers ----

fn emit_access_index(
    builder: &mut FunctionBuilder,
    access: &CompiledAccess,
    axis_vals: &[Option<cranelift_codegen::ir::Value>],
    k_val: Option<cranelift_codegen::ir::Value>,
) -> cranelift_codegen::ir::Value {
    let mut flat = builder.ins().iconst(types::I64, access.base);
    for (axis, &coeff) in access.output_coeffs.iter().enumerate() {
        if coeff != 0 {
            if let Some(val) = axis_vals.get(axis).and_then(|v| *v) {
                flat = add_scaled_i64(builder, flat, val, coeff);
            }
        }
    }
    if let Some(k) = k_val {
        if access.reduction_coeff != 0 {
            flat = add_scaled_i64(builder, flat, k, access.reduction_coeff);
        }
    }
    flat
}

fn emit_output_flat_index(
    builder: &mut FunctionBuilder,
    shape: &[usize],
    axis_vals: &[Option<cranelift_codegen::ir::Value>],
) -> Result<cranelift_codegen::ir::Value, V8Error> {
    let mut flat = builder.ins().iconst(types::I64, 0);
    for (axis, &dim) in shape.iter().enumerate() {
        let idx = axis_vals
            .get(axis)
            .and_then(|v| *v)
            .ok_or_else(|| V8Error::Unsupported(format!("missing output axis value {axis}")))?;
        flat = if dim == 1 {
            flat
        } else {
            builder.ins().imul_imm(flat, dim as i64)
        };
        flat = builder.ins().iadd(flat, idx);
    }
    Ok(flat)
}

fn add_scaled_i64(
    builder: &mut FunctionBuilder,
    base: cranelift_codegen::ir::Value,
    iv: cranelift_codegen::ir::Value,
    stride: i64,
) -> cranelift_codegen::ir::Value {
    match stride {
        0 => base,
        1 => builder.ins().iadd(base, iv),
        -1 => {
            let neg = builder.ins().ineg(iv);
            builder.ins().iadd(base, neg)
        }
        _ => {
            let scaled = builder.ins().imul_imm(iv, stride);
            builder.ins().iadd(base, scaled)
        }
    }
}

fn emit_for_loop<F, E>(
    builder: &mut FunctionBuilder,
    iv_index: u32,
    start: i64,
    end: i64,
    step: i64,
    mut body: F,
) -> Result<(), E>
where
    F: FnMut(&mut FunctionBuilder, cranelift_codegen::ir::Value) -> Result<(), E>,
{
    if start >= end {
        return Ok(());
    }

    let iv = Variable::from_u32(iv_index);
    builder.declare_var(iv, types::I64);
    let start_val = builder.ins().iconst(types::I64, start);
    builder.def_var(iv, start_val);

    let loop_header = builder.create_block();
    let loop_body = builder.create_block();
    let loop_exit = builder.create_block();

    builder.ins().jump(loop_header, &[]);
    builder.switch_to_block(loop_header);

    let i = builder.use_var(iv);
    let end_val = builder.ins().iconst(types::I64, end);
    let cmp = builder.ins().icmp(IntCC::SignedLessThan, i, end_val);
    builder.ins().brif(cmp, loop_body, &[], loop_exit, &[]);

    builder.switch_to_block(loop_body);
    body(builder, i)?;
    let i_next = builder.ins().iadd_imm(i, step);
    builder.def_var(iv, i_next);
    builder.ins().jump(loop_header, &[]);
    builder.seal_block(loop_body);
    builder.seal_block(loop_header);

    builder.switch_to_block(loop_exit);
    builder.seal_block(loop_exit);
    Ok(())
}

/// Like `emit_for_loop` but accepts Cranelift Values for start/end bounds
/// (used when tiling produces dynamic loop bounds).
fn emit_for_loop_dynamic<F, E>(
    builder: &mut FunctionBuilder,
    iv_index: u32,
    start_val: cranelift_codegen::ir::Value,
    end_val: cranelift_codegen::ir::Value,
    step: i64,
    mut body: F,
) -> Result<(), E>
where
    F: FnMut(&mut FunctionBuilder, cranelift_codegen::ir::Value) -> Result<(), E>,
{
    let iv = Variable::from_u32(iv_index);
    builder.declare_var(iv, types::I64);
    builder.def_var(iv, start_val);

    let loop_header = builder.create_block();
    let loop_body = builder.create_block();
    let loop_exit = builder.create_block();

    builder.ins().jump(loop_header, &[]);
    builder.switch_to_block(loop_header);

    let i = builder.use_var(iv);
    let cmp = builder.ins().icmp(IntCC::SignedLessThan, i, end_val);
    builder.ins().brif(cmp, loop_body, &[], loop_exit, &[]);

    builder.switch_to_block(loop_body);
    body(builder, i)?;
    let i_next = builder.ins().iadd_imm(i, step);
    builder.def_var(iv, i_next);
    builder.ins().jump(loop_header, &[]);
    builder.seal_block(loop_body);
    builder.seal_block(loop_header);

    builder.switch_to_block(loop_exit);
    builder.seal_block(loop_exit);
    Ok(())
}

fn load_tensor_base_ptr(
    builder: &mut FunctionBuilder,
    layout: &TensorLayout,
    ptr_table: cranelift_codegen::ir::Value,
    ptr_type: cranelift_codegen::ir::Type,
    tensor: GlobalId,
) -> Result<cranelift_codegen::ir::Value, V8Error> {
    let tidx = layout
        .tensor_index
        .get(&tensor)
        .ok_or(V8Error::Codegen(CodegenError::UnknownTensor(tensor)))?;
    let ptr_offset = (*tidx as i64) * (ptr_type.bytes() as i64);
    let addr = builder.ins().iadd_imm(ptr_table, ptr_offset);
    Ok(builder.ins().load(ptr_type, MemFlags::trusted(), addr, 0))
}

fn load_f32_at_flat(
    builder: &mut FunctionBuilder,
    base_ptr: cranelift_codegen::ir::Value,
    flat_idx: cranelift_codegen::ir::Value,
) -> cranelift_codegen::ir::Value {
    let byte_offset = builder.ins().ishl_imm(flat_idx, 2);
    let addr = builder.ins().iadd(base_ptr, byte_offset);
    builder.ins().load(types::F32, MemFlags::trusted(), addr, 0)
}

fn store_f32_at_flat(
    builder: &mut FunctionBuilder,
    base_ptr: cranelift_codegen::ir::Value,
    flat_idx: cranelift_codegen::ir::Value,
    val: cranelift_codegen::ir::Value,
) {
    let byte_offset = builder.ins().ishl_imm(flat_idx, 2);
    let addr = builder.ins().iadd(base_ptr, byte_offset);
    builder.ins().store(MemFlags::trusted(), val, addr, 0);
}

fn alloc_loop_var(next_var: &mut u32) -> u32 {
    let idx = *next_var;
    *next_var += 1;
    idx
}

extern "C" fn wt_expf(x: f32) -> f32 {
    x.exp()
}
extern "C" fn wt_logf(x: f32) -> f32 {
    x.ln()
}
extern "C" fn wt_tanhf(x: f32) -> f32 {
    x.tanh()
}

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

// ---- Tests ----

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::milli_graph::ops::{MatMul, SimpleBinary, SimpleUnaryOp};
    use rand::RngCore;

    fn fill_random(rng: &mut wyrand::WyRand, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| (rng.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0)
            .collect()
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    // -- Matmul --

    fn run_matmul(m: usize, k: usize, n: usize) -> Result<(Vec<f32>, Vec<f32>), V8Error> {
        let mut rng = wyrand::WyRand::new(8100 + m as u64 + n as u64);
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

        let (compiled, _) = compile_graph(&graph, &shapes)?;
        let layout = &compiled.layout;
        let mut bufs: Vec<Vec<f32>> = (0..layout.num_buffers).map(|_| Vec::new()).collect();
        let mut rng_a = wyrand::WyRand::new(8200 + m as u64 + k as u64);
        let mut rng_b = wyrand::WyRand::new(8300 + k as u64 + n as u64);

        for (id, &size) in &layout.tensor_sizes {
            let idx = layout.tensor_index[id];
            bufs[idx] = if *id == a {
                fill_random(&mut rng_a, size)
            } else if *id == b {
                fill_random(&mut rng_b, size)
            } else {
                vec![0.0f32; size]
            };
        }

        let a_data = bufs[layout.tensor_index[&a]].clone();
        let b_data = bufs[layout.tensor_index[&b]].clone();
        let mut ptrs: Vec<*mut f32> = bufs.iter_mut().map(|v| v.as_mut_ptr()).collect();
        unsafe { compiled.execute(&mut ptrs) };
        let got = bufs[layout.tensor_index[&c]].clone();

        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a_data[i * k + kk] * b_data[kk * n + j];
                }
                expected[i * n + j] = acc;
            }
        }
        Ok((got, expected))
    }

    #[test]
    fn test_matmul_small() {
        let (got, expected) = run_matmul(4, 8, 3).expect("v8 compile");
        assert!(max_diff(&got, &expected) < 1e-4, "max diff too large");
    }

    #[test]
    fn test_matmul_medium() {
        let (got, expected) = run_matmul(64, 96, 80).expect("v8 compile");
        assert!(max_diff(&got, &expected) < 2e-3, "max diff too large");
    }

    #[test]
    fn test_matmul_non_aligned_nr() {
        // N=7 is not divisible by any NR in 2..=8, testing tail handling.
        let (got, expected) = run_matmul(8, 12, 7).expect("v8 compile");
        assert!(max_diff(&got, &expected) < 1e-4, "max diff too large");
    }

    // -- Elementwise --

    #[test]
    fn test_elementwise_add() {
        let mut rng = wyrand::WyRand::new(8001);
        let ext_a = GlobalId::new(&mut rng);
        let ext_b = GlobalId::new(&mut rng);
        let (mut graph, input_map) = MilliOpGraph::new([ext_a, ext_b], &mut rng);
        let a = input_map[&ext_a];
        let b = input_map[&ext_b];
        let c = SimpleBinary::add(&mut graph, a, b, &mut rng);

        let mut shapes = HashMap::new();
        shapes.insert(a, vec![32]);
        shapes.insert(b, vec![32]);
        shapes.insert(c, vec![32]);

        let (compiled, _) = compile_graph(&graph, &shapes).expect("v8 compile");
        let layout = &compiled.layout;
        let mut bufs: Vec<Vec<f32>> = (0..layout.num_buffers).map(|_| Vec::new()).collect();
        let mut rng_a = wyrand::WyRand::new(8011);
        let mut rng_b = wyrand::WyRand::new(8012);

        for (id, &size) in &layout.tensor_sizes {
            let idx = layout.tensor_index[id];
            bufs[idx] = if *id == a {
                fill_random(&mut rng_a, size)
            } else if *id == b {
                fill_random(&mut rng_b, size)
            } else {
                vec![0.0f32; size]
            };
        }

        let a_data = bufs[layout.tensor_index[&a]].clone();
        let b_data = bufs[layout.tensor_index[&b]].clone();
        let mut ptrs: Vec<*mut f32> = bufs.iter_mut().map(|v| v.as_mut_ptr()).collect();
        unsafe { compiled.execute(&mut ptrs) };
        let got = bufs[layout.tensor_index[&c]].clone();

        let expected: Vec<f32> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(x, y)| x + y)
            .collect();
        assert!(max_diff(&got, &expected) < 1e-6, "max diff too large");
    }

    // -- Matmul + sigmoid + bias (multi-kernel: reduction then pointwise) --

    #[test]
    fn test_matmul_sigmoid_bias() {
        let mut rng = wyrand::WyRand::new(8900);
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
        let ext_out = GlobalId::new(&mut rng);
        graph.set_output_map([(out, ext_out)]);

        let (m, k, n) = (16, 24, 16);
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

        let (compiled, _) = compile_graph(&graph, &shapes).expect("v8 compile");
        let layout = &compiled.layout;
        let mut bufs: Vec<Vec<f32>> = (0..layout.num_buffers).map(|_| Vec::new()).collect();
        let mut rng_a = wyrand::WyRand::new(8911);
        let mut rng_b = wyrand::WyRand::new(8921);
        let mut rng_bias = wyrand::WyRand::new(8931);

        for (id, &size) in &layout.tensor_sizes {
            let idx = layout.tensor_index[id];
            bufs[idx] = if *id == a {
                fill_random(&mut rng_a, size)
            } else if *id == b {
                fill_random(&mut rng_b, size)
            } else if *id == bias {
                fill_random(&mut rng_bias, size)
            } else if *id == ones {
                vec![1.0f32; size]
            } else {
                vec![0.0f32; size]
            };
        }

        let a_data = bufs[layout.tensor_index[&a]].clone();
        let b_data = bufs[layout.tensor_index[&b]].clone();
        let bias_data = bufs[layout.tensor_index[&bias]].clone();
        let mut ptrs: Vec<*mut f32> = bufs.iter_mut().map(|v| v.as_mut_ptr()).collect();
        unsafe { compiled.execute(&mut ptrs) };
        let got = bufs[layout.tensor_index[&out]].clone();

        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += a_data[i * k + kk] * b_data[kk * n + j];
                }
                let z = acc + bias_data[j];
                expected[i * n + j] = 1.0 / (1.0 + (-z).exp());
            }
        }

        assert!(
            max_diff(&got, &expected) < 2e-3,
            "max diff: {}",
            max_diff(&got, &expected)
        );
    }

    // -- Large matmul (would blow up without sampled synthesis) --

    #[test]
    #[ignore] // ~20s on CI — run with `cargo test -- --ignored`
    fn test_matmul_1024() {
        // 1024^3 = 1 billion nano-ops in the full expansion.
        // Sampled synthesis emits ~256 output elements × 4K reduction steps = ~1M nano-ops.
        let (got, expected) = run_matmul(1024, 1024, 1024).expect("v8 compile 1024^3");
        assert!(
            max_diff(&got, &expected) < 0.1,
            "1024^3 max diff: {}",
            max_diff(&got, &expected)
        );
    }

    #[test]
    #[ignore] // ~6min on CI — run with `cargo test -- --ignored`
    fn test_matmul_realistic_ffn() {
        // Approximate a real model FFN: [256, 4096] × [4096, 8192]
        // Full expansion: 256 × 4096 × 8192 ≈ 8.6 billion nano-ops.
        let (got, expected) = run_matmul(256, 4096, 8192).expect("v8 compile FFN-scale");
        assert!(
            max_diff(&got, &expected) < 0.5,
            "FFN-scale max diff: {}",
            max_diff(&got, &expected)
        );
    }
}
