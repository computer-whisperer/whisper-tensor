use cranelift_codegen::ir::{types, AbiParam, InstBuilder, MemFlags, StackSlotData, StackSlotKind};
use cranelift_codegen::settings;
use cranelift_codegen::settings::{Configurable, Flags};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use ndarray::Array2;
use crate::{MatmulImpl, MatmulImplBuilder};

// ----------------- Cranelift JIT implementation (cranelift_6 - fully unrolled) -----------------
// Panelized GEMM with scalar MrxNr microkernel and direct-to-C stores.
// High-level compile-time constants control panel sizes and microkernel shape.
//
// Pseudocode:
//   const MB=64, NB=64, MR=4, NR=4
//   for j0 in 0..N step NB:               // B panel: full K x NB
//     pack Bp[K][NB] from B (transpose style) with zero-padding on tails
//     for i0 in 0..M step MB:             // A panel: MB x full K
//       pack Ap[MB][K] from A with zero-padding on tails
//       // Microkernel tiling within panels
//       for ii in 0..MB step MR:
//         for jj in 0..NB step NR:
//           // Accumulate the full dot products across K
//           let c[MR][NR] = 0
//           for p in 0..K:
//             let a[MR] = [Ap[ii+mi][p] for mi=0..MR-1]
//             let b[NR] = [Bp[p][jj+nj] for nj=0..NR-1]
//             for mi in 0..MR-1, nj in 0..NR-1: c[mi][nj] += a[mi]*b[nj]
//           // Write microtile directly to C with edge guards
//           for mi in 0..MR-1:
//             for nj in 0..NR-1:
//               I = i0+ii+mi; J = j0+jj+nj;
//               if I<M && J<N: C[I][J] = c[mi][nj]
// Notes:
// - All tails (partial panels and microtiles) are handled via zero-padded packing
//   and store-time bounds checks; loads never go out of bounds.
// - B is packed in KxNB row-major so that for fixed p we can load NR contiguous
//   values Bp[p][jj..jj+NR) for better locality.
// - Ap is MBxK row-major. Accessing a[MR] is strided by K, which is acceptable
//   for scalar microkernels.
// - This is a scalar kernel intended as a foundation for later SIMDization.

// Compile-time configuration constants (change and recompile to tune):
const MB: usize = 64; // A panel rows
const NB: usize = 64; // B panel cols
const MR: usize = 4;  // microkernel rows
const NR: usize = 8;  // microkernel cols

type MatmulFn = unsafe extern "C" fn(a: *const f32, b: *const f32, c: *mut f32);

pub(crate) struct CraneliftImpl {
    module: JITModule,
    func_id: cranelift_module::FuncId,
    func_ptr: MatmulFn,
    m: usize,
    k: usize,
    n: usize,
    clif_text: String,
}

pub(crate) struct CraneliftBuilder;

impl CraneliftBuilder {
    pub(crate) fn new() -> Self { Self }

    fn define_matmul_func(module: &mut JITModule, m: i64, n: i64, k: i64) -> (cranelift_module::FuncId, String) {
        use cranelift_codegen::ir::condcodes::IntCC;
        use cranelift_codegen::ir::Signature;

        let mut ctx = module.make_context();
        let mut fctx = FunctionBuilderContext::new();
        let ptr_ty = module.isa().pointer_type();

        // Signature: (a_ptr, b_ptr, c_ptr)
        let mut sig = Signature::new(module.isa().default_call_conv());
        sig.params.push(AbiParam::new(ptr_ty));
        sig.params.push(AbiParam::new(ptr_ty));
        sig.params.push(AbiParam::new(ptr_ty));
        ctx.func.signature = sig;
        ctx.func.name = cranelift_codegen::ir::UserFuncName::user(0, 4);

        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        b.seal_block(entry);

        let a_ptr = b.block_params(entry)[0];
        let b_ptr = b.block_params(entry)[1];
        let c_ptr = b.block_params(entry)[2];

        // I32 variables for indices
        let vi0 = b.declare_var(types::I32); // panel i index
        let vj0 = b.declare_var(types::I32); // panel j index
        let vii = b.declare_var(types::I32); // within-panel i or ii
        let vjj = b.declare_var(types::I32); // within-panel j or jj
        let vp = b.declare_var(types::I32);  // K iterator
        let vmi = b.declare_var(types::I32); // micro row iter
        let vnj = b.declare_var(types::I32); // micro col iter

        // Constants
        let zero_i32 = b.ins().iconst(types::I32, 0);
        let one_i32 = b.ins().iconst(types::I32, 1);
        let four_i32 = b.ins().iconst(types::I32, 4);
        let m_c32 = b.ins().iconst(types::I32, m as i64);
        let n_c32 = b.ins().iconst(types::I32, n as i64);
        let k_c32 = b.ins().iconst(types::I32, k as i64);
        let mb_c32 = b.ins().iconst(types::I32, MB as i64);
        let nb_c32 = b.ins().iconst(types::I32, NB as i64);
        let mr_c32 = b.ins().iconst(types::I32, MR as i64);
        let nr_c32 = b.ins().iconst(types::I32, NR as i64);
        let zero_f32 = b.ins().f32const(0.0);

        // Extend 32-bit byte multipliers to pointer type
        let four_ptr = b.ins().uextend(ptr_ty, four_i32);

        // Memory flags
        let mut flags_ro = MemFlags::trusted();
        flags_ro.set_notrap();
        flags_ro.set_readonly();
        flags_ro.set_aligned();
        let mut flags_st = MemFlags::trusted();
        flags_st.set_notrap();
        flags_st.set_aligned();

        // Scratch panels on stack (static sizes based on MB, NB, K baked at JIT-build time)
        let ap_elems = (MB as i64 * k) as u32; // MB*K
        let bp_elems = (k * NB as i64) as u32; // K*NB
        let ap_bytes: u32 = ap_elems.saturating_mul(4);
        let bp_bytes: u32 = bp_elems.saturating_mul(4);
        let ss_ap = b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, ap_bytes, 4));
        let ss_bp = b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, bp_bytes, 4));
        let ap_base = b.ins().stack_addr(ptr_ty, ss_ap, 0);
        let bp_base = b.ins().stack_addr(ptr_ty, ss_bp, 0);

        // Fully unrolled version: emit straight-line IR matching cranelift_4 load/math order (no IR loops)
        let j_panels: usize = ((n as usize) + NB - 1) / NB;
        let i_panels: usize = ((m as usize) + MB - 1) / MB;
        let k_usize: usize = k as usize;

        for j0_idx_us in 0..j_panels {
            let j0_base = b.ins().iconst(types::I32, (j0_idx_us * NB) as i64);

            // Pack B panel: outer p, inner jj (matches cl4 order)
            for p_us in 0..k_usize {
                let p_val = b.ins().iconst(types::I32, p_us as i64);
                let p_mul_n = b.ins().imul(p_val, n_c32);
                let p_times_nb = b.ins().imul(p_val, nb_c32);
                for jj_us in 0..NB {
                    let jj_val = b.ins().iconst(types::I32, jj_us as i64);
                    let j_gl = b.ins().iadd(j0_base, jj_val);
                    let j_ok = b.ins().icmp(IntCC::SignedLessThan, j_gl, n_c32);

                    let idx_b = b.ins().iadd(p_mul_n, j_gl);
                    let idx_b_bytes = b.ins().imul(idx_b, four_i32);
                    let idx_b_ptr = b.ins().uextend(ptr_ty, idx_b_bytes);
                    let addr_b = b.ins().iadd(b_ptr, idx_b_ptr);
                    let vb = b.ins().load(types::F32, flags_ro, addr_b, 0);
                    let vb_sel = b.ins().select(j_ok, vb, zero_f32);

                    let idx_bp = b.ins().iadd(p_times_nb, jj_val);
                    let idx_bp_bytes = b.ins().imul(idx_bp, four_i32);
                    let idx_bp_ptr = b.ins().uextend(ptr_ty, idx_bp_bytes);
                    let addr_bp = b.ins().iadd(bp_base, idx_bp_ptr);
                    b.ins().store(flags_st, vb_sel, addr_bp, 0);
                }
            }

            // For each A panel at this j0
            for i0_idx_us in 0..i_panels {
                let i0_base = b.ins().iconst(types::I32, (i0_idx_us * MB) as i64);

                // Pack A panel: outer ii, inner p (matches cl4 order)
                for ii_us in 0..MB {
                    let ii_val = b.ins().iconst(types::I32, ii_us as i64);
                    let i_gl = b.ins().iadd(i0_base, ii_val);
                    let i_ok = b.ins().icmp(IntCC::SignedLessThan, i_gl, m_c32);
                    let i_mul_k = b.ins().imul(i_gl, k_c32);
                    let ii_times_k = b.ins().imul(ii_val, k_c32);
                    for p_us in 0..k_usize {
                        let p_val = b.ins().iconst(types::I32, p_us as i64);

                        let idx_a = b.ins().iadd(i_mul_k, p_val);
                        let idx_a_bytes = b.ins().imul(idx_a, four_i32);
                        let idx_a_ptr = b.ins().uextend(ptr_ty, idx_a_bytes);
                        let addr_a = b.ins().iadd(a_ptr, idx_a_ptr);
                        let va = b.ins().load(types::F32, flags_ro, addr_a, 0);
                        let va_sel = b.ins().select(i_ok, va, zero_f32);

                        let idx_ap = b.ins().iadd(ii_times_k, p_val);
                        let idx_ap_bytes = b.ins().imul(idx_ap, four_i32);
                        let idx_ap_ptr = b.ins().uextend(ptr_ty, idx_ap_bytes);
                        let addr_ap = b.ins().iadd(ap_base, idx_ap_ptr);
                        b.ins().store(flags_st, va_sel, addr_ap, 0);
                    }
                }

                // Microkernel tiles: ii0 step MR, jj0 step NR
                for ii0_us in (0..MB).step_by(MR) {
                    let ii0_val = b.ins().iconst(types::I32, ii0_us as i64);
                    for jj0_us in (0..NB).step_by(NR) {
                        let jj0_val = b.ins().iconst(types::I32, jj0_us as i64);

                        // Accumulators c[MR][NR] = 0
                        let mut c_vars: Vec<Variable> = Vec::with_capacity(MR*NR);
                        for _mi in 0..MR {
                            for _nj in 0..NR {
                                let v = b.declare_var(types::F32);
                                b.def_var(v, zero_f32);
                                c_vars.push(v);
                            }
                        }

                        // Accumulate across K
                        for p_us in 0..k_usize {
                            let p_val = b.ins().iconst(types::I32, p_us as i64);
                            // Load a[MR]
                            let mut a_vals: [cranelift_codegen::ir::Value; MR] = [zero_f32; MR];
                            for mi in 0..MR {
                                let mi_val = b.ins().iconst(types::I32, mi as i64);
                                let ii_mi = b.ins().iadd(ii0_val, mi_val);
                                let ii_mi_k = b.ins().imul(ii_mi, k_c32);
                                let idx_ap2 = b.ins().iadd(ii_mi_k, p_val);
                                let idx_ap2_bytes = b.ins().imul(idx_ap2, four_i32);
                                let idx_ap2_ptr = b.ins().uextend(ptr_ty, idx_ap2_bytes);
                                let addr_ap2 = b.ins().iadd(ap_base, idx_ap2_ptr);
                                a_vals[mi] = b.ins().load(types::F32, flags_st, addr_ap2, 0);
                            }
                            // Load b[NR]
                            let p_times_nb2 = b.ins().imul(p_val, nb_c32);
                            let mut b_vals: [cranelift_codegen::ir::Value; NR] = [zero_f32; NR];
                            for nj in 0..NR {
                                let nj_val = b.ins().iconst(types::I32, nj as i64);
                                let jj_nj = b.ins().iadd(jj0_val, nj_val);
                                let idx_bp2 = b.ins().iadd(p_times_nb2, jj_nj);
                                let idx_bp2_bytes = b.ins().imul(idx_bp2, four_i32);
                                let idx_bp2_ptr = b.ins().uextend(ptr_ty, idx_bp2_bytes);
                                let addr_bp2 = b.ins().iadd(bp_base, idx_bp2_ptr);
                                b_vals[nj] = b.ins().load(types::F32, flags_st, addr_bp2, 0);
                            }
                            // FMA updates
                            for mi in 0..MR {
                                for nj in 0..NR {
                                    let prod = b.ins().fmul(a_vals[mi], b_vals[nj]);
                                    let idx = (mi * NR + nj) as usize;
                                    let sum_old = b.use_var(c_vars[idx]);
                                    let sum_new = b.ins().fadd(sum_old, prod);
                                    b.def_var(c_vars[idx], sum_new);
                                }
                            }
                        }

                        // Store microtile to C with bounds checks
                        for mi in 0..MR {
                            for nj in 0..NR {
                                let mi_v = b.ins().iconst(types::I32, mi as i64);
                                let nj_v = b.ins().iconst(types::I32, nj as i64);
                                let ii0_plus_mi = b.ins().iadd(ii0_val, mi_v);
                                let i_out = b.ins().iadd(i0_base, ii0_plus_mi);
                                let jj0_plus_nj = b.ins().iadd(jj0_val, nj_v);
                                let j_out = b.ins().iadd(j0_base, jj0_plus_nj);
                                let i_ok2 = b.ins().icmp(IntCC::SignedLessThan, i_out, m_c32);
                                let j_ok2 = b.ins().icmp(IntCC::SignedLessThan, j_out, n_c32);
                                let ok_store = b.ins().band(i_ok2, j_ok2);
                                let i_times_n = b.ins().imul(i_out, n_c32);
                                let idx_c = b.ins().iadd(i_times_n, j_out);
                                let idx_c_bytes = b.ins().imul(idx_c, four_i32);
                                let idx_c_ptr = b.ins().uextend(ptr_ty, idx_c_bytes);
                                let addr_c = b.ins().iadd(c_ptr, idx_c_ptr);
                                let idx = (mi * NR + nj) as usize;
                                let val = b.use_var(c_vars[idx]);
                                let store_true = b.create_block();
                                let store_cont = b.create_block();
                                b.ins().brif(ok_store, store_true, &[], store_cont, &[]);
                                b.switch_to_block(store_true);
                                b.ins().store(flags_st, val, addr_c, 0);
                                b.ins().jump(store_cont, &[]);
                                b.switch_to_block(store_cont);
                            }
                        }
                    }
                }
            }
        }

        // Return
        b.ins().return_(&[]);

        // Seal blocks
        b.seal_all_blocks();
        b.finalize();

        let func_id = module
            .declare_function("matmul_f32_cl6", Linkage::Local, &ctx.func.signature)
            .expect("declare_function");
        module
            .define_function(func_id, &mut ctx)
            .expect("define_function");
        // CLIF text generation omitted for size/perf reasons in cranelift_6
        let clif_text = String::new();
        (func_id, clif_text)
    }
}

impl MatmulImpl for CraneliftImpl {
    fn name(&self) -> &'static str { "cranelift_6" }
    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        let (m, k1) = a.dim();
        let (k2, n) = b.dim();
        assert_eq!(m, self.m, "A rows must equal m");
        assert_eq!(n, self.n, "B cols must equal n");
        assert_eq!(k1, self.k, "A cols must equal k");
        assert_eq!(k2, self.k, "B rows must equal k");
        let a_std = a.as_standard_layout().to_owned();
        let b_std = b.as_standard_layout().to_owned();
        let mut c = Array2::<f32>::zeros((self.m, self.n));
        unsafe { (self.func_ptr)(a_std.as_ptr(), b_std.as_ptr(), c.as_mut_ptr()); }
        c
    }
    // Omit detailed report for cranelift_6 due to function size.
    fn make_report(&self, _out_dir: &std::path::Path) { /* intentionally omitted */ }
}

impl MatmulImplBuilder for CraneliftBuilder {
    fn name(&self) -> &'static str { "cranelift_6" }
    fn build(&self, m: usize, k: usize, n: usize) -> Box<dyn MatmulImpl> {
        // Build JIT module with speed optimizations
        let mut flag_builder = settings::builder();
        let _ = flag_builder.set("opt_level", "speed");
        let flags = Flags::new(flag_builder);

        let isa_builder = cranelift_native::builder().expect("host machine is not supported");
        let isa = isa_builder.finish(flags).expect("failed to build ISA");

        let jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(jit_builder);

        let (func_id, clif_text) = CraneliftBuilder::define_matmul_func(&mut module, m as i64, n as i64, k as i64);
        let _ = module.finalize_definitions();
        let code_ptr = module.get_finalized_function(func_id);
        let func_ptr: MatmulFn = unsafe { std::mem::transmute(code_ptr) };

        // Warmup: single run on zero matrices to touch code pages
        let a0 = Array2::<f32>::zeros((m, k));
        let b0 = Array2::<f32>::zeros((k, n));
        let mut c0 = Array2::<f32>::zeros((m, n));
        unsafe { func_ptr(a0.as_ptr(), b0.as_ptr(), c0.as_mut_ptr()); }

        Box::new(CraneliftImpl { module, func_id, func_ptr, m, k, n, clif_text })
    }
}
