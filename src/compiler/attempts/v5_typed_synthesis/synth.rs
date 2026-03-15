#![allow(clippy::all, dead_code, unreachable_patterns)]
//! Typed schedule synthesis and software bf16 matmul execution.
//!
//! This module is intentionally BLAS-independent. It consumes recovered
//! matmul crystals and emits a heuristic blocked schedule with explicit
//! bf16->fp32 conversion in the compute loop.

use crate::compiler::attempts::v4_pool_growth::codegen::whitewash_pool_order;
use crate::compiler::attempts::v4_pool_growth::growth::{
    MatMulCrystal, PoolGrowthPlan, grow_from_pool,
};
use crate::compiler::common::v1_frontend::nano_op::{NanoExpandError, NanoOpExpander};
use crate::dtype::DType;
use crate::graph::GlobalId;
use crate::milli_graph::MilliOpGraph;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy)]
pub struct HardwareProfile {
    pub l1_bytes: usize,
    pub l2_bytes: usize,
    pub simd_width_f32: usize,
    pub threads: usize,
}

impl Default for HardwareProfile {
    fn default() -> Self {
        Self {
            l1_bytes: 32 * 1024,
            l2_bytes: 512 * 1024,
            simd_width_f32: 8,
            threads: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bf16KernelStrategy {
    /// Convert bf16 operands to fp32 each inner-loop iteration.
    OnTheFlyConvert,
    /// Pack current B panel as fp32 once per tile, then reuse.
    PackBPanelF32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Blocking {
    pub mc: usize,
    pub nc: usize,
    pub kc: usize,
    pub mr: usize,
    pub nr: usize,
}

#[derive(Debug, Clone)]
pub struct MatMulKernelPlan {
    pub crystal: MatMulCrystal,
    pub a_dtype: DType,
    pub b_dtype: DType,
    pub c_dtype: DType,
    pub accumulate_dtype: DType,
    pub blocking: Blocking,
    pub bf16_strategy: Option<Bf16KernelStrategy>,
}

#[derive(Debug, Clone, Default)]
pub struct TypedSynthesisPlan {
    pub matmul_kernels: Vec<MatMulKernelPlan>,
    pub skipped_matmul_crystals: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum SynthesisError {
    #[error(transparent)]
    Expand(#[from] NanoExpandError),
    #[error(transparent)]
    Growth(#[from] crate::compiler::attempts::v4_pool_growth::growth::GrowthError),
    #[error("Missing dtype for tensor {0}")]
    MissingTensorDType(GlobalId),
    #[error("Unsupported matmul dtype tuple: A={a:?}, B={b:?}, C={c:?}")]
    UnsupportedDTypeTuple { a: DType, b: DType, c: DType },
}

#[derive(Debug, thiserror::Error)]
pub enum KernelExecError {
    #[error("Kernel execution currently supports only BF16xBF16->F32")]
    UnsupportedKernelPrecision,
    #[error("Input/output buffer length mismatch")]
    BufferSizeMismatch,
}

/// Lightweight counters used to inspect execution quality and diagnose
/// conversion/packing overheads.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KernelExecStats {
    pub bf16_to_f32_a: usize,
    pub bf16_to_f32_b: usize,
    pub fma_ops: usize,
    pub packed_b_panels: usize,
    pub micro_tiles: usize,
}

const MAX_MR: usize = 16;
const MAX_NR: usize = 16;

/// Build a typed synthesis plan directly from graph+shapes+dtypes by going
/// through the nano-op pool recovery pipeline.
pub fn build_from_graph(
    graph: &MilliOpGraph,
    shapes: &HashMap<GlobalId, Vec<usize>>,
    dtypes: &HashMap<GlobalId, DType>,
    hw: HardwareProfile,
) -> Result<(TypedSynthesisPlan, PoolGrowthPlan), SynthesisError> {
    let mut expander = NanoOpExpander::new(shapes.clone());
    let ordered_nano: Vec<_> = expander.expand_iter(graph).collect::<Result<Vec<_>, _>>()?;
    let whitewashed = whitewash_pool_order(ordered_nano);
    let pool_plan = grow_from_pool(&whitewashed, shapes)?;
    let typed = synthesize_from_pool(&pool_plan, dtypes, hw)?;
    Ok((typed, pool_plan))
}

pub fn synthesize_from_pool(
    pool_plan: &PoolGrowthPlan,
    dtypes: &HashMap<GlobalId, DType>,
    hw: HardwareProfile,
) -> Result<TypedSynthesisPlan, SynthesisError> {
    let mut out = TypedSynthesisPlan::default();
    for mm in &pool_plan.matmul_crystals {
        let a_dtype = *dtypes
            .get(&mm.a)
            .ok_or(SynthesisError::MissingTensorDType(mm.a))?;
        let b_dtype = *dtypes
            .get(&mm.b)
            .ok_or(SynthesisError::MissingTensorDType(mm.b))?;
        let c_dtype = *dtypes
            .get(&mm.c)
            .ok_or(SynthesisError::MissingTensorDType(mm.c))?;

        let accumulate_dtype = choose_accumulate_dtype(a_dtype, b_dtype, c_dtype)?;
        let blocking = choose_blocking(mm, hw, a_dtype, b_dtype);
        let bf16_strategy = choose_bf16_strategy(mm, a_dtype, b_dtype, accumulate_dtype);

        out.matmul_kernels.push(MatMulKernelPlan {
            crystal: mm.clone(),
            a_dtype,
            b_dtype,
            c_dtype,
            accumulate_dtype,
            blocking,
            bf16_strategy,
        });
    }
    Ok(out)
}

fn choose_accumulate_dtype(
    a_dtype: DType,
    b_dtype: DType,
    c_dtype: DType,
) -> Result<DType, SynthesisError> {
    match (a_dtype, b_dtype, c_dtype) {
        (DType::BF16, DType::BF16, DType::F32) => Ok(DType::F32),
        (DType::F32, DType::F32, DType::F32) => Ok(DType::F32),
        (a, b, c) => Err(SynthesisError::UnsupportedDTypeTuple { a, b, c }),
    }
}

fn choose_blocking(
    mm: &MatMulCrystal,
    hw: HardwareProfile,
    a_dtype: DType,
    b_dtype: DType,
) -> Blocking {
    let simd = hw.simd_width_f32.max(4);
    let mr = if simd >= 16 {
        16
    } else if simd >= 8 {
        8
    } else {
        4
    }
    .min(MAX_MR);
    let nr = mr.min(MAX_NR);

    // Rough L1-guided kc estimate: account for one A micro-panel and one B panel.
    let a_bytes = a_dtype.size().unwrap_or(4);
    let b_bytes = if b_dtype == DType::BF16 {
        4
    } else {
        b_dtype.size().unwrap_or(4)
    };
    let bytes_per_k = mr * a_bytes + nr * b_bytes;
    let target_l1 = hw.l1_bytes.saturating_div(2).max(1024);
    let mut kc = (target_l1 / bytes_per_k.max(1)).clamp(32, 512);
    kc = kc.min(mm.k.max(1));
    kc = (kc / 8).max(1) * 8;

    let mut mc = if mm.m >= 256 { 128 } else { 64 };
    let mut nc = if mm.n >= 256 { 128 } else { 64 };
    mc = mc.min(mm.m.max(1));
    nc = nc.min(mm.n.max(1));

    // Keep tile panels roughly under L2.
    let panel_bytes = (mc * kc * a_bytes) + (kc * nc * b_bytes);
    if panel_bytes > hw.l2_bytes && nc > 32 {
        nc = 32.min(mm.n.max(1));
    }

    Blocking { mc, nc, kc, mr, nr }
}

fn choose_bf16_strategy(
    mm: &MatMulCrystal,
    a_dtype: DType,
    b_dtype: DType,
    accumulate_dtype: DType,
) -> Option<Bf16KernelStrategy> {
    if !(a_dtype == DType::BF16 && b_dtype == DType::BF16 && accumulate_dtype == DType::F32) {
        return None;
    }

    // Heuristic: pack-B when K*N tile work is large enough to amortize pack cost.
    let work = mm.k * mm.n;
    if work >= 8_192 || mm.k >= 192 {
        Some(Bf16KernelStrategy::PackBPanelF32)
    } else {
        Some(Bf16KernelStrategy::OnTheFlyConvert)
    }
}

#[inline]
fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Execute a synthesized BF16xBF16->F32 matmul kernel.
///
/// Layout is row-major:
/// - `a_bf16`: [m, k]
/// - `b_bf16`: [k, n]
/// - `c_f32`: [m, n]
pub fn execute_matmul_bf16_f32(
    plan: &MatMulKernelPlan,
    a_bf16: &[u16],
    b_bf16: &[u16],
    c_f32: &mut [f32],
) -> Result<(), KernelExecError> {
    let _ = execute_matmul_bf16_f32_with_stats(plan, a_bf16, b_bf16, c_f32)?;
    Ok(())
}

/// Same as `execute_matmul_bf16_f32`, but returns execution counters useful
/// for comparing synthesis strategies.
pub fn execute_matmul_bf16_f32_with_stats(
    plan: &MatMulKernelPlan,
    a_bf16: &[u16],
    b_bf16: &[u16],
    c_f32: &mut [f32],
) -> Result<KernelExecStats, KernelExecError> {
    let mm = &plan.crystal;
    if !(plan.a_dtype == DType::BF16
        && plan.b_dtype == DType::BF16
        && plan.c_dtype == DType::F32
        && plan.accumulate_dtype == DType::F32)
    {
        return Err(KernelExecError::UnsupportedKernelPrecision);
    }
    if a_bf16.len() != mm.m * mm.k || b_bf16.len() != mm.k * mm.n || c_f32.len() != mm.m * mm.n {
        return Err(KernelExecError::BufferSizeMismatch);
    }

    c_f32.fill(0.0);

    let Blocking { mc, nc, kc, mr, nr } = sanitize_blocking(plan.blocking, mm);
    let strategy = plan
        .bf16_strategy
        .unwrap_or(Bf16KernelStrategy::OnTheFlyConvert);
    let mut stats = KernelExecStats::default();

    match strategy {
        Bf16KernelStrategy::OnTheFlyConvert => {
            for m0 in (0..mm.m).step_by(mc) {
                let m_tile = (mm.m - m0).min(mc);
                for n0 in (0..mm.n).step_by(nc) {
                    let n_tile = (mm.n - n0).min(nc);
                    for k0 in (0..mm.k).step_by(kc) {
                        let k_tile = (mm.k - k0).min(kc);
                        kernel_block_onthefly(
                            mm, m0, n0, k0, m_tile, n_tile, k_tile, mr, nr, a_bf16, b_bf16, c_f32,
                            &mut stats,
                        );
                    }
                }
            }
        }
        Bf16KernelStrategy::PackBPanelF32 => {
            let mut packed_b = Vec::new();
            for n0 in (0..mm.n).step_by(nc) {
                let n_tile = (mm.n - n0).min(nc);
                for k0 in (0..mm.k).step_by(kc) {
                    let k_tile = (mm.k - k0).min(kc);
                    pack_b_panel(
                        mm,
                        n0,
                        k0,
                        n_tile,
                        k_tile,
                        b_bf16,
                        &mut packed_b,
                        &mut stats,
                    );
                    for m0 in (0..mm.m).step_by(mc) {
                        let m_tile = (mm.m - m0).min(mc);
                        kernel_block_packed_b(
                            mm, m0, n0, k0, m_tile, n_tile, k_tile, mr, nr, a_bf16, &packed_b,
                            c_f32, &mut stats,
                        );
                    }
                }
            }
        }
    }

    Ok(stats)
}

fn sanitize_blocking(blocking: Blocking, mm: &MatMulCrystal) -> Blocking {
    let mut mr = blocking.mr.max(1).min(MAX_MR).min(mm.m.max(1));
    let mut nr = blocking.nr.max(1).min(MAX_NR).min(mm.n.max(1));
    if mr == 0 {
        mr = 1;
    }
    if nr == 0 {
        nr = 1;
    }

    let mut kc = blocking.kc.max(1).min(mm.k.max(1));
    if kc >= 8 {
        kc = (kc / 8).max(1) * 8;
    }
    kc = kc.max(1).min(mm.k.max(1));

    Blocking {
        mc: blocking.mc.max(mr).min(mm.m.max(1)),
        nc: blocking.nc.max(nr).min(mm.n.max(1)),
        kc,
        mr,
        nr,
    }
}

#[allow(clippy::too_many_arguments)]
fn kernel_block_onthefly(
    mm: &MatMulCrystal,
    m0: usize,
    n0: usize,
    k0: usize,
    m_tile: usize,
    n_tile: usize,
    k_tile: usize,
    mr: usize,
    nr: usize,
    a_bf16: &[u16],
    b_bf16: &[u16],
    c_f32: &mut [f32],
    stats: &mut KernelExecStats,
) {
    for mi in (0..m_tile).step_by(mr) {
        let mr_tile = (m_tile - mi).min(mr);
        for nj in (0..n_tile).step_by(nr) {
            let nr_tile = (n_tile - nj).min(nr);
            let mut acc = [0.0f32; MAX_MR * MAX_NR];
            stats.micro_tiles += 1;

            for kk in 0..k_tile {
                let k = k0 + kk;
                let mut b_vals = [0.0f32; MAX_NR];
                for jj in 0..nr_tile {
                    let col = n0 + nj + jj;
                    b_vals[jj] = bf16_bits_to_f32(b_bf16[k * mm.n + col]);
                    stats.bf16_to_f32_b += 1;
                }

                for ii in 0..mr_tile {
                    let row = m0 + mi + ii;
                    let a = bf16_bits_to_f32(a_bf16[row * mm.k + k]);
                    stats.bf16_to_f32_a += 1;
                    let acc_row = ii * MAX_NR;
                    for jj in 0..nr_tile {
                        acc[acc_row + jj] += a * b_vals[jj];
                        stats.fma_ops += 1;
                    }
                }
            }

            for ii in 0..mr_tile {
                let row = m0 + mi + ii;
                let out_row = row * mm.n;
                let acc_row = ii * MAX_NR;
                for jj in 0..nr_tile {
                    c_f32[out_row + (n0 + nj + jj)] += acc[acc_row + jj];
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn kernel_block_packed_b(
    mm: &MatMulCrystal,
    m0: usize,
    n0: usize,
    k0: usize,
    m_tile: usize,
    n_tile: usize,
    k_tile: usize,
    mr: usize,
    nr: usize,
    a_bf16: &[u16],
    b_panel_f32: &[f32],
    c_f32: &mut [f32],
    stats: &mut KernelExecStats,
) {
    for mi in (0..m_tile).step_by(mr) {
        let mr_tile = (m_tile - mi).min(mr);
        for nj in (0..n_tile).step_by(nr) {
            let nr_tile = (n_tile - nj).min(nr);
            let mut acc = [0.0f32; MAX_MR * MAX_NR];
            stats.micro_tiles += 1;

            for kk in 0..k_tile {
                let mut b_vals = [0.0f32; MAX_NR];
                let b_row = kk * n_tile;
                for jj in 0..nr_tile {
                    b_vals[jj] = b_panel_f32[b_row + (nj + jj)];
                }

                for ii in 0..mr_tile {
                    let row = m0 + mi + ii;
                    let a = bf16_bits_to_f32(a_bf16[row * mm.k + (k0 + kk)]);
                    stats.bf16_to_f32_a += 1;
                    let acc_row = ii * MAX_NR;
                    for jj in 0..nr_tile {
                        acc[acc_row + jj] += a * b_vals[jj];
                        stats.fma_ops += 1;
                    }
                }
            }

            for ii in 0..mr_tile {
                let row = m0 + mi + ii;
                let out_row = row * mm.n;
                let acc_row = ii * MAX_NR;
                for jj in 0..nr_tile {
                    c_f32[out_row + (n0 + nj + jj)] += acc[acc_row + jj];
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn pack_b_panel(
    mm: &MatMulCrystal,
    n0: usize,
    k0: usize,
    n_tile: usize,
    k_tile: usize,
    b_bf16: &[u16],
    out_panel: &mut Vec<f32>,
    stats: &mut KernelExecStats,
) {
    let panel_len = k_tile * n_tile;
    out_panel.clear();
    out_panel.resize(panel_len, 0.0);

    for kk in 0..k_tile {
        let k = k0 + kk;
        for j in 0..n_tile {
            let col = n0 + j;
            out_panel[kk * n_tile + j] = bf16_bits_to_f32(b_bf16[k * mm.n + col]);
            stats.bf16_to_f32_b += 1;
        }
    }
    stats.packed_b_panels += 1;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::attempts::v1_scalar_crystal::nano_op::{NanoOp, NanoValue, ScalarBinOp};
    use crate::compiler::attempts::v4_pool_growth::codegen::whitewash_pool_order;
    use half::bf16;
    use rand::Rng;

    fn make_matmul_nano_pool(
        m: usize,
        n: usize,
        k: usize,
        a: GlobalId,
        b: GlobalId,
        c: GlobalId,
    ) -> Vec<NanoOp> {
        let mut next_value = 0u64;
        let mut ops = Vec::new();
        for row in 0..m {
            for col in 0..n {
                let mut acc = NanoValue(next_value);
                next_value += 1;
                ops.push(NanoOp::Literal {
                    dst: acc,
                    value: 0.0,
                });
                for kk in 0..k {
                    let va = NanoValue(next_value);
                    next_value += 1;
                    ops.push(NanoOp::Load {
                        dst: va,
                        tensor: a,
                        flat_index: row * k + kk,
                    });
                    let vb = NanoValue(next_value);
                    next_value += 1;
                    ops.push(NanoOp::Load {
                        dst: vb,
                        tensor: b,
                        flat_index: kk * n + col,
                    });
                    let vm = NanoValue(next_value);
                    next_value += 1;
                    ops.push(NanoOp::BinOp {
                        dst: vm,
                        op: ScalarBinOp::Mul,
                        a: va,
                        b: vb,
                    });
                    let vs = NanoValue(next_value);
                    next_value += 1;
                    ops.push(NanoOp::BinOp {
                        dst: vs,
                        op: ScalarBinOp::Add,
                        a: acc,
                        b: vm,
                    });
                    acc = vs;
                }
                ops.push(NanoOp::Store {
                    tensor: c,
                    flat_index: row * n + col,
                    src: acc,
                });
            }
        }
        whitewash_pool_order(ops)
    }

    #[test]
    fn test_synthesize_bf16_matmul_plan_from_pool() {
        let m = 4usize;
        let n = 6usize;
        let k = 8usize;
        let a = GlobalId(501);
        let b = GlobalId(502);
        let c = GlobalId(503);
        let pool = make_matmul_nano_pool(m, n, k, a, b, c);

        let mut shapes = HashMap::new();
        shapes.insert(a, vec![m, k]);
        shapes.insert(b, vec![k, n]);
        shapes.insert(c, vec![m, n]);
        let pool_plan = grow_from_pool(&pool, &shapes).unwrap();
        assert_eq!(pool_plan.matmul_crystals.len(), 1);

        let mut dtypes = HashMap::new();
        dtypes.insert(a, DType::BF16);
        dtypes.insert(b, DType::BF16);
        dtypes.insert(c, DType::F32);

        let typed = synthesize_from_pool(&pool_plan, &dtypes, HardwareProfile::default()).unwrap();
        assert_eq!(typed.matmul_kernels.len(), 1);
        let k0 = &typed.matmul_kernels[0];
        assert_eq!(k0.accumulate_dtype, DType::F32);
        assert!(k0.bf16_strategy.is_some());
    }

    fn run_reference_bf16_matmul(m: usize, n: usize, k: usize, a: &[u16], b: &[u16]) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    let av = bf16::from_bits(a[row * k + kk]).to_f32();
                    let bv = bf16::from_bits(b[kk * n + col]).to_f32();
                    acc += av * bv;
                }
                out[row * n + col] = acc;
            }
        }
        out
    }

    #[test]
    fn test_execute_bf16_f32_matches_reference() {
        let (m, n, k) = (17usize, 19usize, 23usize);
        let crystal = MatMulCrystal {
            a: GlobalId(1),
            b: GlobalId(2),
            c: GlobalId(3),
            m,
            n,
            k,
            covered_outputs: (0..m * n).collect(),
        };

        let mut rng = wyrand::WyRand::new(44);
        let mut a_f = vec![0.0f32; m * k];
        let mut b_f = vec![0.0f32; k * n];
        for v in &mut a_f {
            let bits = rng.next_u32();
            *v = (bits as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }
        for v in &mut b_f {
            let bits = rng.next_u32();
            *v = (bits as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }
        let a_bf16: Vec<u16> = a_f.iter().map(|x| bf16::from_f32(*x).to_bits()).collect();
        let b_bf16: Vec<u16> = b_f.iter().map(|x| bf16::from_f32(*x).to_bits()).collect();
        let expected = run_reference_bf16_matmul(m, n, k, &a_bf16, &b_bf16);

        for strategy in [
            Bf16KernelStrategy::OnTheFlyConvert,
            Bf16KernelStrategy::PackBPanelF32,
        ] {
            let plan = MatMulKernelPlan {
                crystal: crystal.clone(),
                a_dtype: DType::BF16,
                b_dtype: DType::BF16,
                c_dtype: DType::F32,
                accumulate_dtype: DType::F32,
                blocking: Blocking {
                    mc: 8,
                    nc: 8,
                    kc: 16,
                    mr: 8,
                    nr: 8,
                },
                bf16_strategy: Some(strategy),
            };

            let mut out = vec![0.0f32; m * n];
            execute_matmul_bf16_f32(&plan, &a_bf16, &b_bf16, &mut out).unwrap();
            for (i, (got, exp)) in out.iter().zip(&expected).enumerate() {
                assert!(
                    (got - exp).abs() < 5e-3,
                    "strategy {:?} mismatch at {i}: got {got} expected {exp}",
                    strategy
                );
            }
        }
    }

    #[test]
    fn test_pack_b_reduces_bf16_b_conversions() {
        let (m, n, k) = (48usize, 32usize, 40usize);
        let crystal = MatMulCrystal {
            a: GlobalId(11),
            b: GlobalId(12),
            c: GlobalId(13),
            m,
            n,
            k,
            covered_outputs: (0..m * n).collect(),
        };

        let mut rng = wyrand::WyRand::new(777);
        let a_bf16: Vec<u16> = (0..m * k)
            .map(|_| {
                let bits = rng.next_u32();
                let x = (bits as f32 / u32::MAX as f32) * 2.0 - 1.0;
                bf16::from_f32(x).to_bits()
            })
            .collect();
        let b_bf16: Vec<u16> = (0..k * n)
            .map(|_| {
                let bits = rng.next_u32();
                let x = (bits as f32 / u32::MAX as f32) * 2.0 - 1.0;
                bf16::from_f32(x).to_bits()
            })
            .collect();

        let blocking = Blocking {
            mc: 16,
            nc: 16,
            kc: 16,
            mr: 8,
            nr: 8,
        };
        let mut plan = MatMulKernelPlan {
            crystal: crystal.clone(),
            a_dtype: DType::BF16,
            b_dtype: DType::BF16,
            c_dtype: DType::F32,
            accumulate_dtype: DType::F32,
            blocking,
            bf16_strategy: Some(Bf16KernelStrategy::OnTheFlyConvert),
        };

        let mut out_onthefly = vec![0.0f32; m * n];
        let onthefly_stats =
            execute_matmul_bf16_f32_with_stats(&plan, &a_bf16, &b_bf16, &mut out_onthefly).unwrap();

        plan.bf16_strategy = Some(Bf16KernelStrategy::PackBPanelF32);
        let mut out_packed = vec![0.0f32; m * n];
        let packed_stats =
            execute_matmul_bf16_f32_with_stats(&plan, &a_bf16, &b_bf16, &mut out_packed).unwrap();

        let expected_fma = m * n * k;
        assert_eq!(onthefly_stats.fma_ops, expected_fma);
        assert_eq!(packed_stats.fma_ops, expected_fma);
        assert_eq!(onthefly_stats.packed_b_panels, 0);
        assert!(packed_stats.packed_b_panels > 0);
        assert!(packed_stats.bf16_to_f32_b < onthefly_stats.bf16_to_f32_b);

        for (i, (lhs, rhs)) in out_onthefly.iter().zip(&out_packed).enumerate() {
            assert!(
                (lhs - rhs).abs() < 5e-3,
                "strategy outputs diverged at {i}: onthefly={lhs}, packed={rhs}"
            );
        }
    }
}
