use cranelift_codegen::ir::{types, AbiParam, InstBuilder, MemFlags, StackSlotData, StackSlotKind};
use cranelift_codegen::settings;
use cranelift_codegen::settings::{Configurable, Flags};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use ndarray::Array2;
use crate::{MatmulImpl, MatmulImplBuilder};

// ----------------- Cranelift JIT implementation (cranelift_4) -----------------
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

    fn define_matmul_func(module: &mut JITModule, m: i64, n: i64, k: i64, lanes: usize, use_fma: bool) -> (cranelift_module::FuncId, String) {
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
        // NR (vector lanes) derived from host features
        let nr_c32 = b.ins().iconst(types::I32, lanes as i64);
        let nr_minus1_c32 = b.ins().iconst(types::I32, (lanes as i64) - 1);
        // Select vector type for this kernel: use 128-bit vectors everywhere and process in 4-lane chunks
        let vty = types::F32X4;
        let chunks: usize = (lanes + 3) / 4; // number of F32X4 chunks to cover the NR lanes
        // Memflags for vector ops: allow unaligned to be safe on stack slots
        let mut flags_vec = MemFlags::trusted();
        flags_vec.set_notrap();
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

        // Helper lambdas (as closures) cannot be used; inline address arithmetic with IR below.
        // Loop structure:
        // for j0 in 0.. while j0*NB < n
        b.def_var(vj0, zero_i32);
        let lj0_head = b.create_block();
        let lj0_body = b.create_block();
        let lj0_after = b.create_block();
        b.ins().jump(lj0_head, &[]);

        b.switch_to_block(lj0_head);
        let j0_idx = b.use_var(vj0);
        let j0_base = b.ins().imul(j0_idx, nb_c32); // j0*NB
        let j0_cond = b.ins().icmp(IntCC::SignedLessThan, j0_base, n_c32);
        b.ins().brif(j0_cond, lj0_body, &[], lj0_after, &[]);

        // lj0_body: pack B panel (K x NB row-major: idx = p*NB + jj)
        b.switch_to_block(lj0_body);
        b.def_var(vp, zero_i32);
        let lb_p_head = b.create_block();
        let lb_p_body = b.create_block();
        let lb_j_head = b.create_block();
        let lb_j_body = b.create_block();
        let lb_after = b.create_block();
        let lb_j_inc = b.create_block();

        b.ins().jump(lb_p_head, &[]);
        b.switch_to_block(lb_p_head);
        let p_b = b.use_var(vp);
        let p_b_cond = b.ins().icmp(IntCC::SignedLessThan, p_b, k_c32);
        b.ins().brif(p_b_cond, lb_p_body, &[], lb_after, &[]);

        b.switch_to_block(lb_p_body);
        b.def_var(vjj, zero_i32);
        b.ins().jump(lb_j_head, &[]);

        b.switch_to_block(lb_j_head);
        let jj_b = b.use_var(vjj);
        let jj_b_cond = b.ins().icmp(IntCC::SignedLessThan, jj_b, nb_c32);
        b.ins().brif(jj_b_cond, lb_j_body, &[], lb_j_inc, &[]);

        b.switch_to_block(lb_j_body);
        // Global j = j0_base + jj_b; Bounds check j<n
        let j_gl = b.ins().iadd(j0_base, jj_b);
        let j_ok = b.ins().icmp(IntCC::SignedLessThan, j_gl, n_c32);
        // B index: (p*n + j)
        let p_mul_n = b.ins().imul(p_b, n_c32);
        let idx_b = b.ins().iadd(p_mul_n, j_gl);
        let idx_b_bytes = b.ins().imul(idx_b, four_i32);
        let idx_b_ptr = b.ins().uextend(ptr_ty, idx_b_bytes);
        let addr_b = b.ins().iadd(b_ptr, idx_b_ptr);
        let vb = b.ins().load(types::F32, flags_ro, addr_b, 0);
        let vb_sel = b.ins().select(j_ok, vb, zero_f32);
        // Store into Bp at [p][jj] => idx = p*NB + jj
        let p_times_nb = b.ins().imul(p_b, nb_c32);
        let idx_bp = b.ins().iadd(p_times_nb, jj_b);
        let idx_bp_bytes = b.ins().imul(idx_bp, four_i32);
        let idx_bp_ptr = b.ins().uextend(ptr_ty, idx_bp_bytes);
        let addr_bp = b.ins().iadd(bp_base, idx_bp_ptr);
        b.ins().store(flags_st, vb_sel, addr_bp, 0);
        // jj++
        let jj_next = b.ins().iadd(jj_b, one_i32);
        b.def_var(vjj, jj_next);
        b.ins().jump(lb_j_head, &[]);

        // after each p row of Bp, continue p
        b.switch_to_block(lb_j_inc);
        let p_next_b = b.ins().iadd(p_b, one_i32);
        b.def_var(vp, p_next_b);
        b.ins().jump(lb_p_head, &[]);

        // lb_after: proceed to A panels
        b.switch_to_block(lb_after);

        // i0 loop over A panels (i0_base = i0*MB, while i0_base < m)
        b.def_var(vi0, zero_i32);
        let li0_head = b.create_block();
        let li0_body = b.create_block();
        let li0_after = b.create_block();
        b.ins().jump(li0_head, &[]);

        b.switch_to_block(li0_head);
        let i0_idx = b.use_var(vi0);
        let i0_base = b.ins().imul(i0_idx, mb_c32);
        let i0_cond = b.ins().icmp(IntCC::SignedLessThan, i0_base, m_c32);
        b.ins().brif(i0_cond, li0_body, &[], li0_after, &[]);

        // li0_body: pack A panel Ap[MB][K] from A
        b.switch_to_block(li0_body);
        b.def_var(vii, zero_i32);
        let la_i_head = b.create_block();
        let la_i_body = b.create_block();
        let la_p_head = b.create_block();
        let la_p_body = b.create_block();
        let la_after = b.create_block();
        let la_p_inc = b.create_block();

        b.ins().jump(la_i_head, &[]);
        b.switch_to_block(la_i_head);
        let ii_a = b.use_var(vii);
        let ii_a_cond = b.ins().icmp(IntCC::SignedLessThan, ii_a, mb_c32);
        b.ins().brif(ii_a_cond, la_i_body, &[], la_after, &[]);

        b.switch_to_block(la_i_body);
        b.def_var(vp, zero_i32);
        b.ins().jump(la_p_head, &[]);

        b.switch_to_block(la_p_head);
        let p_a = b.use_var(vp);
        let p_a_cond = b.ins().icmp(IntCC::SignedLessThan, p_a, k_c32);
        b.ins().brif(p_a_cond, la_p_body, &[], la_p_inc, &[]);

        b.switch_to_block(la_p_body);
        // Global i = i0_base + ii; Bounds i<m
        let i_gl = b.ins().iadd(i0_base, ii_a);
        let i_ok = b.ins().icmp(IntCC::SignedLessThan, i_gl, m_c32);
        // A index: (i*k + p)
        let i_mul_k = b.ins().imul(i_gl, k_c32);
        let idx_a = b.ins().iadd(i_mul_k, p_a);
        let idx_a_bytes = b.ins().imul(idx_a, four_i32);
        let idx_a_ptr = b.ins().uextend(ptr_ty, idx_a_bytes);
        let addr_a = b.ins().iadd(a_ptr, idx_a_ptr);
        let va = b.ins().load(types::F32, flags_ro, addr_a, 0);
        let va_sel = b.ins().select(i_ok, va, zero_f32);
        // Store into Ap at [ii][p] => idx = ii*K + p
        let ii_times_k = b.ins().imul(ii_a, k_c32);
        let idx_ap = b.ins().iadd(ii_times_k, p_a);
        let idx_ap_bytes = b.ins().imul(idx_ap, four_i32);
        let idx_ap_ptr = b.ins().uextend(ptr_ty, idx_ap_bytes);
        let addr_ap = b.ins().iadd(ap_base, idx_ap_ptr);
        b.ins().store(flags_st, va_sel, addr_ap, 0);

        // p++
        let p_next_a = b.ins().iadd(p_a, one_i32);
        b.def_var(vp, p_next_a);
        b.ins().jump(la_p_head, &[]);

        // after finishing p for current ii: ii++
        b.switch_to_block(la_p_inc);
        let ii_next_a = b.ins().iadd(ii_a, one_i32);
        b.def_var(vii, ii_next_a);
        b.ins().jump(la_i_head, &[]);

        // la_after: compute microkernel tiles for this Ap and current Bp
        b.switch_to_block(la_after);

        // ii loop over micro-rows: for ii in 0..MB step MR
        b.def_var(vii, zero_i32);
        let lmk_i_head = b.create_block();
        let lmk_i_body = b.create_block();
        let lmk_i_after = b.create_block();
        b.ins().jump(lmk_i_head, &[]);

        b.switch_to_block(lmk_i_head);
        let ii0 = b.use_var(vii);
        let ii0_cond = b.ins().icmp(IntCC::SignedLessThan, ii0, mb_c32);
        b.ins().brif(ii0_cond, lmk_i_body, &[], lmk_i_after, &[]);

        b.switch_to_block(lmk_i_body);
        // jj loop over micro-cols: for jj in 0..NB step NR
        b.def_var(vjj, zero_i32);
        let lmk_j_head = b.create_block();
        let lmk_j_body = b.create_block();
        let lmk_j_after = b.create_block();
        b.ins().jump(lmk_j_head, &[]);

        b.switch_to_block(lmk_j_head);
        let jj0 = b.use_var(vjj);
        let jj0_cond = b.ins().icmp(IntCC::SignedLessThan, jj0, nb_c32);
        b.ins().brif(jj0_cond, lmk_j_body, &[], lmk_j_after, &[]);

        // Microkernel body for tile (ii0..ii0+MR, jj0..jj0+NR)
        b.switch_to_block(lmk_j_body);

        // Vector accumulators: one vector per micro-row (MR)
        // Note: avoid creating wide vector constants (workaround verifier issue) by initializing
        // accumulators on the first K-iteration instead of splatting 0.0.
        let mut c_vec_vars: Vec<Variable> = Vec::new();
        for _mi in 0..MR {
            for _c in 0..chunks {
                let v = b.declare_var(vty);
                c_vec_vars.push(v);
            }
        }

        // p loop over K
        b.def_var(vp, zero_i32);
        let lmk_p_head = b.create_block();
        let lmk_p_body = b.create_block();
        let lmk_p_after = b.create_block();
        b.ins().jump(lmk_p_head, &[]);

        b.switch_to_block(lmk_p_head);
        let p_val = b.use_var(vp);
        let p_cond = b.ins().icmp(IntCC::SignedLessThan, p_val, k_c32);
        b.ins().brif(p_cond, lmk_p_body, &[], lmk_p_after, &[]);

        b.switch_to_block(lmk_p_body);
        // Load a[MR] scalars and B vector
        let mut a_vals: [cranelift_codegen::ir::Value; MR] = [zero_f32; MR];
        for mi in 0..MR {
            // idx_ap = (ii0+mi)*K + p
            let mi_val = b.ins().iconst(types::I32, mi as i64);
            let ii_mi = b.ins().iadd(ii0, mi_val);
            let ii_mi_k = b.ins().imul(ii_mi, k_c32);
            let idx_ap2 = b.ins().iadd(ii_mi_k, p_val);
            let idx_ap2_bytes = b.ins().imul(idx_ap2, four_i32);
            let idx_ap2_ptr = b.ins().uextend(ptr_ty, idx_ap2_bytes);
            let addr_ap2 = b.ins().iadd(ap_base, idx_ap2_ptr);
            a_vals[mi] = b.ins().load(types::F32, flags_st, addr_ap2, 0);
        }
        // B vector from packed panel: base = p*NB + jj0
        let p_times_nb2 = b.ins().imul(p_val, nb_c32);
        let idx_bp2 = b.ins().iadd(p_times_nb2, jj0);
        let idx_bp2_bytes = b.ins().imul(idx_bp2, four_i32);
        let idx_bp2_ptr = b.ins().uextend(ptr_ty, idx_bp2_bytes);
        let addr_bp2 = b.ins().iadd(bp_base, idx_bp2_ptr);
        let _b_vec_dummy = b.ins().load(vty, flags_vec, addr_bp2, 0); // initial chunk base pointer computed; actual loads happen per chunk

        // Branch on first K-iteration to initialize accumulators without vector constants
        let p_is_zero = b.ins().icmp(IntCC::Equal, p_val, zero_i32);
        let p_first = b.create_block();
        let p_gen = b.create_block();
        let p_merge = b.create_block();
        b.ins().brif(p_is_zero, p_first, &[], p_gen, &[]);

        // First K iteration: acc = a*b for each 4-lane chunk
        b.switch_to_block(p_first);
        for c in 0..chunks {
            let c_off_elems = b.ins().iconst(types::I32, (c as i64) * 4);
            let idx_bp2c = b.ins().iadd(idx_bp2, c_off_elems);
            let idx_bp2c_bytes = b.ins().imul(idx_bp2c, four_i32);
            let idx_bp2c_ptr = b.ins().uextend(ptr_ty, idx_bp2c_bytes);
            let addr_bp2c = b.ins().iadd(bp_base, idx_bp2c_ptr);
            let b_vec_c = b.ins().load(vty, flags_vec, addr_bp2c, 0);
            for mi in 0..MR {
                let a_bcast = b.ins().splat(vty, a_vals[mi]);
                let prod = b.ins().fmul(a_bcast, b_vec_c);
                let idx = mi * chunks + c;
                b.def_var(c_vec_vars[idx], prod);
            }
        }
        b.ins().jump(p_merge, &[]);

        // General K iteration: acc = fma(a,b,acc) or acc += prod for each 4-lane chunk
        b.switch_to_block(p_gen);
        for c in 0..chunks {
            let c_off_elems = b.ins().iconst(types::I32, (c as i64) * 4);
            let idx_bp2c = b.ins().iadd(idx_bp2, c_off_elems);
            let idx_bp2c_bytes = b.ins().imul(idx_bp2c, four_i32);
            let idx_bp2c_ptr = b.ins().uextend(ptr_ty, idx_bp2c_bytes);
            let addr_bp2c = b.ins().iadd(bp_base, idx_bp2c_ptr);
            let b_vec_c = b.ins().load(vty, flags_vec, addr_bp2c, 0);
            for mi in 0..MR {
                let a_bcast = b.ins().splat(vty, a_vals[mi]);
                let idx = mi * chunks + c;
                let acc_old = b.use_var(c_vec_vars[idx]);
                let acc_new = if use_fma {
                    b.ins().fma(a_bcast, b_vec_c, acc_old)
                } else {
                    let prod = b.ins().fmul(a_bcast, b_vec_c);
                    b.ins().fadd(acc_old, prod)
                };
                b.def_var(c_vec_vars[idx], acc_new);
            }
        }
        b.ins().jump(p_merge, &[]);

        b.switch_to_block(p_merge);
        // p++
        let p_next = b.ins().iadd(p_val, one_i32);
        b.def_var(vp, p_next);
        b.ins().jump(lmk_p_head, &[]);

        // After K: store microtile (vector store if in-bounds, else per-lane)
        b.switch_to_block(lmk_p_after);
        let j_base = b.ins().iadd(j0_base, jj0);
        for mi in 0..MR {
            // Compute output row index and bounds
            let mi_v = b.ins().iconst(types::I32, mi as i64);
            let ii0_plus_mi = b.ins().iadd(ii0, mi_v);
            let i_out = b.ins().iadd(i0_base, ii0_plus_mi);
            let i_ok2 = b.ins().icmp(IntCC::SignedLessThan, i_out, m_c32);
            // Full-width check: j_base + (lanes-1) < n
            let j_end = b.ins().iadd(j_base, nr_minus1_c32);
            let j_full_ok = b.ins().icmp(IntCC::SignedLessThan, j_end, n_c32);
            let ok_full = b.ins().band(i_ok2, j_full_ok);

            // Base pointer for C row start
            let i_times_n = b.ins().imul(i_out, n_c32);
            let idx_c0 = b.ins().iadd(i_times_n, j_base);
            let idx_c0_bytes = b.ins().imul(idx_c0, four_i32);
            let idx_c0_ptr = b.ins().uextend(ptr_ty, idx_c0_bytes);
            let addr_c0 = b.ins().iadd(c_ptr, idx_c0_ptr);

            let vec_store_true = b.create_block();
            let vec_store_fallback = b.create_block();
            let vec_store_cont = b.create_block();
            b.ins().brif(ok_full, vec_store_true, &[], vec_store_fallback, &[]);

            // Vector store path (store each 4-lane chunk)
            b.switch_to_block(vec_store_true);
            for c in 0..chunks {
                let idx = mi * chunks + c;
                let v_acc = b.use_var(c_vec_vars[idx]);
                let c_off_bytes = b.ins().iconst(types::I32, (c as i64) * 16); // 4 lanes * 4 bytes
                let c_off_ptr = b.ins().uextend(ptr_ty, c_off_bytes);
                let addr_c_chunk = b.ins().iadd(addr_c0, c_off_ptr);
                b.ins().store(flags_vec, v_acc, addr_c_chunk, 0);
            }
            b.ins().jump(vec_store_cont, &[]);

            // Fallback: per-lane guarded stores across chunks
            b.switch_to_block(vec_store_fallback);
            for c in 0..chunks {
                let idx = mi * chunks + c;
                let v_acc2 = b.use_var(c_vec_vars[idx]);
                for lane in 0..4 {
                    let lane_global = b.ins().iconst(types::I32, (c as i64) * 4 + lane as i64);
                    let j_out = b.ins().iadd(j_base, lane_global);
                    let j_ok2 = b.ins().icmp(IntCC::SignedLessThan, j_out, n_c32);
                    let ok_store = b.ins().band(i_ok2, j_ok2);
                    let idx_c = b.ins().iadd(i_times_n, j_out);
                    let idx_c_bytes = b.ins().imul(idx_c, four_i32);
                    let idx_c_ptr = b.ins().uextend(ptr_ty, idx_c_bytes);
                    let addr_c = b.ins().iadd(c_ptr, idx_c_ptr);
                    let lane_val = b.ins().extractlane(v_acc2, lane as u8);
                    let store_true = b.create_block();
                    let store_cont = b.create_block();
                    b.ins().brif(ok_store, store_true, &[], store_cont, &[]);
                    b.switch_to_block(store_true);
                    b.ins().store(flags_st, lane_val, addr_c, 0);
                    b.ins().jump(store_cont, &[]);
                    b.switch_to_block(store_cont);
                }
            }
            b.ins().jump(vec_store_cont, &[]);
            b.switch_to_block(vec_store_cont);
        }
        // Advance jj by NR
        let jj_next0 = b.ins().iadd(jj0, nr_c32);
        b.def_var(vjj, jj_next0);
        b.ins().jump(lmk_j_head, &[]);

        // After all jj: advance ii by MR
        b.switch_to_block(lmk_j_after);
        let ii_next0 = b.ins().iadd(ii0, mr_c32);
        b.def_var(vii, ii_next0);
        b.ins().jump(lmk_i_head, &[]);

        // After ii microtiles: proceed to next i0 panel
        b.switch_to_block(lmk_i_after);
        let i0_next = b.ins().iadd(i0_idx, one_i32);
        b.def_var(vi0, i0_next);
        b.ins().jump(li0_head, &[]);

        // After i-panels for this B panel: advance j0
        b.switch_to_block(li0_after);
        let j0_next = b.ins().iadd(j0_idx, one_i32);
        b.def_var(vj0, j0_next);
        b.ins().jump(lj0_head, &[]);

        // After all panels: return
        b.switch_to_block(lj0_after);
        b.ins().return_(&[]);

        // Seal blocks
        b.seal_all_blocks();
        b.finalize();

        let func_id = module
            .declare_function("matmul_f32_cl5", Linkage::Local, &ctx.func.signature)
            .expect("declare_function");
        module
            .define_function(func_id, &mut ctx)
            .expect("define_function");
        let clif_text = format!("{}", ctx.func.display());
        (func_id, clif_text)
    }
}

impl MatmulImpl for CraneliftImpl {
    fn name(&self) -> &'static str { "cranelift_5" }
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
    fn make_report(&self, out_dir: &std::path::Path) {
        crate::reports::write_cranelift_report(
            out_dir,
            self.name(),
            self.m,
            self.k,
            self.n,
            &self.clif_text,
            &self.module,
            self.func_id,
        );
    }
}

impl MatmulImplBuilder for CraneliftBuilder {
    fn name(&self) -> &'static str { "cranelift_5" }
    fn build(&self, m: usize, k: usize, n: usize) -> Box<dyn MatmulImpl> {
        // Build JIT module with speed optimizations
        let mut flag_builder = settings::builder();
        let _ = flag_builder.set("opt_level", "speed");
        let _ = flag_builder.set("enable_simd", "true");
        let flags = Flags::new(flag_builder);

        let isa_builder = cranelift_native::builder().expect("host machine is not supported");
        let isa = isa_builder.finish(flags).expect("failed to build ISA");

        let jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(jit_builder);

        // Detect SIMD vector width and FMA support on the host
        // Prefer widest safe vector on this CPU: 512-bit (if avx512f), else 256-bit (if avx/avx2), else 128-bit.
        let (lanes, use_fma) = {
            #[cfg(target_arch = "x86_64")]
            {
                let has_avx512 = std::arch::is_x86_feature_detected!("avx512f");
                let has_avx = std::arch::is_x86_feature_detected!("avx");
                let has_fma = std::arch::is_x86_feature_detected!("fma");
                let lanes = if has_avx512 { 16 } else if has_avx { 8 } else { 4 };
                (lanes, has_fma)
            }
            #[cfg(target_arch = "aarch64")]
            {
                // NEON is 128-bit (4 lanes of f32); AArch64 generally has FMA in ASIMD
                (4usize, true)
            }
            #[cfg(all(not(target_arch = "x86_64"), not(target_arch = "aarch64")))]
            {
                (4usize, false)
            }
        };

        let (func_id, clif_text) = CraneliftBuilder::define_matmul_func(&mut module, m as i64, n as i64, k as i64, lanes, use_fma);
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
