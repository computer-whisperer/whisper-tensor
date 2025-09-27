use cranelift_codegen::ir::{types, AbiParam, InstBuilder, MemFlags, StackSlotData, StackSlotKind};
use cranelift_codegen::settings;
use cranelift_codegen::settings::{Configurable, Flags};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use ndarray::Array2;
use crate::{MatmulImpl, MatmulImplBuilder};

// ----------------- Cranelift JIT implementation (cranelift_3) -----------------
// Introduces 32x32 tiling: reorders A and B into tiles on the stack (B tile transposed),
// then accumulates into a 32x32 C tile which is finally written back with edge guards.
// Safety: raw pointers and no bounds checking; caller must ensure valid buffers and sizes.

type MatmulFn = unsafe extern "C" fn(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
);

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
        ctx.func.name = cranelift_codegen::ir::UserFuncName::user(0, 3);

        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        b.seal_block(entry);

        // Params
        let a_ptr = b.block_params(entry)[0];
        let b_ptr = b.block_params(entry)[1];
        let c_ptr = b.block_params(entry)[2];

        // Tile size constants
        let tile = 32i64;

        // Variables (induction variables in I32)
        let vi0 = b.declare_var(types::I32); // tile i
        let vj0 = b.declare_var(types::I32); // tile j
        let vp0 = b.declare_var(types::I32); // tile p
        let vii = b.declare_var(types::I32); // in-tile i
        let vjj = b.declare_var(types::I32); // in-tile j
        let vpp = b.declare_var(types::I32); // in-tile p

        // Needed for address calcs
        let vtmpf = b.declare_var(types::F32); // temp float
        let vcacc = b.declare_var(types::F32); // accumulator for CT[ii][jj]

        // Constants
        let zero_i32 = b.ins().iconst(types::I32, 0);
        let one_i32 = b.ins().iconst(types::I32, 1);
        let four_i32 = b.ins().iconst(types::I32, 4);
        let m_c32 = b.ins().iconst(types::I32, m as i64);
        let n_c32 = b.ins().iconst(types::I32, n as i64);
        let k_c32 = b.ins().iconst(types::I32, k as i64);
        let tile_c32 = b.ins().iconst(types::I32, tile);
        let zero_f32 = b.ins().f32const(0.0);

        // Precompute strides (I32)
        let k_bytes_i32 = b.ins().imul(k_c32, four_i32); // k*4
        let n_bytes_i32 = b.ins().imul(n_c32, four_i32); // n*4
        let four_ptr = b.ins().uextend(ptr_ty, four_i32);

        // MemFlags
        let mut flags_ro = MemFlags::trusted();
        flags_ro.set_notrap();
        flags_ro.set_readonly();
        flags_ro.set_aligned();
        let mut flags_st = MemFlags::trusted();
        flags_st.set_notrap();
        flags_st.set_aligned();

        // Stack tiles: A[32x32], BT[32x32], CT[32x32]
        let tile_bytes: u32 = (tile * tile * 4) as u32;
        let ss_a = b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, tile_bytes, 4));
        let ss_bt = b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, tile_bytes, 4));
        let ss_ct = b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, tile_bytes, 4));
        let a_tile_base = b.ins().stack_addr(ptr_ty, ss_a, 0);
        let btile_base = b.ins().stack_addr(ptr_ty, ss_bt, 0);
        let ctile_base = b.ins().stack_addr(ptr_ty, ss_ct, 0);

        // Helper: compute base row pointers for global A/B/C given i/j indices
        // We compute when needed inside loops to reduce variables.

        // Outer tile loops: for i0 in 0..ceil_div(m,32), for j0 in 0..ceil_div(n,32)
        b.def_var(vi0, zero_i32);
        let li0_head = b.create_block();
        let li0_body = b.create_block();
        let lj0_head = b.create_block();
        let lj0_body = b.create_block();
        let after_j0 = b.create_block();
        let after_i0 = b.create_block();
        let after_i0_inc = b.create_block();

        b.ins().jump(li0_head, &[]);
        b.switch_to_block(li0_head);
        let i0v = b.use_var(vi0);
        // i0 < ceil_div(m,32) => i0*32 < m
        let i0_mul = b.ins().imul(i0v, tile_c32);
        let i0_cond = b.ins().icmp(IntCC::SignedLessThan, i0_mul, m_c32);
        b.ins().brif(i0_cond, li0_body, &[], after_i0, &[]);

        b.switch_to_block(li0_body);
        // j0 loop init
        b.def_var(vj0, zero_i32);
        b.ins().jump(lj0_head, &[]);

        b.switch_to_block(lj0_head);
        let j0v = b.use_var(vj0);
        let j0_mul = b.ins().imul(j0v, tile_c32);
        let j0_cond = b.ins().icmp(IntCC::SignedLessThan, j0_mul, n_c32);
        b.ins().brif(j0_cond, lj0_body, &[], after_i0_inc, &[]);

        // lj0_body: zero-initialize C tile
        b.switch_to_block(lj0_body);
        // Zero CT tile: for ii in 0..32 { for jj in 0..32 { CT[ii][jj] = 0; } }
        b.def_var(vii, zero_i32);
        let lz_i_head = b.create_block();
        let lz_i_body = b.create_block();
        let lz_j_head = b.create_block();
        let lz_j_body = b.create_block();
        let lz_after = b.create_block();
        let lz_done = b.create_block();

        b.ins().jump(lz_i_head, &[]);
        b.switch_to_block(lz_i_head);
        let ii_z = b.use_var(vii);
        let ii_z_cond = b.ins().icmp(IntCC::SignedLessThan, ii_z, tile_c32);
        b.ins().brif(ii_z_cond, lz_i_body, &[], lz_done, &[]);

        b.switch_to_block(lz_i_body);
        b.def_var(vjj, zero_i32);
        b.ins().jump(lz_j_head, &[]);

        b.switch_to_block(lz_j_head);
        let jj_z = b.use_var(vjj);
        let jj_z_cond = b.ins().icmp(IntCC::SignedLessThan, jj_z, tile_c32);
        b.ins().brif(jj_z_cond, lz_j_body, &[], lz_after, &[]);

        b.switch_to_block(lz_j_body);
        // addr = ctile_base + ((ii*32 + jj)*4)
        let ii_times_tile = b.ins().imul(ii_z, tile_c32);
        let ii_tile_plus_jj = b.ins().iadd(ii_times_tile, jj_z);
        let elem_off_i32 = b.ins().imul(ii_tile_plus_jj, four_i32);
        let elem_off_ptr = b.ins().uextend(ptr_ty, elem_off_i32);
        let addr = b.ins().iadd(ctile_base, elem_off_ptr);
        b.ins().store(flags_st, zero_f32, addr, 0);
        let jj_next = b.ins().iadd(jj_z, one_i32);
        b.def_var(vjj, jj_next);
        b.ins().jump(lz_j_head, &[]);

        b.switch_to_block(lz_after);
        // after zeroing rows, advance ii
        let ii_next = b.ins().iadd(ii_z, one_i32);
        b.def_var(vii, ii_next);
        // Go back to lz_i_head to check condition again
        b.ins().jump(lz_i_head, &[]);
        // We will seal blocks later

        b.switch_to_block(lz_done);
        // After zero CT, begin p0 tiles
        let lp0_head = b.create_block();
        let lp0_body = b.create_block();
        let after_p0 = b.create_block();
        b.def_var(vp0, zero_i32);
        b.ins().jump(lp0_head, &[]);

        b.switch_to_block(lp0_head);
        let p0v = b.use_var(vp0);
        let p0_mul = b.ins().imul(p0v, tile_c32);
        let p0_cond = b.ins().icmp(IntCC::SignedLessThan, p0_mul, k_c32);
        b.ins().brif(p0_cond, lp0_body, &[], after_p0, &[]);

        b.switch_to_block(lp0_body);
        // 1) Build A tile for this (i0, p0)
        b.def_var(vii, zero_i32);
        let la_i_head = b.create_block();
        let la_i_body = b.create_block();
        let la_p_head = b.create_block();
        let la_p_body = b.create_block();
        let la_after = b.create_block();
        let la_inc = b.create_block();
        b.ins().jump(la_i_head, &[]);

        b.switch_to_block(la_i_head);
        let ii_a = b.use_var(vii);
        let ii_a_cond = b.ins().icmp(IntCC::SignedLessThan, ii_a, tile_c32);
        b.ins().brif(ii_a_cond, la_i_body, &[], la_after, &[]);

        b.switch_to_block(la_i_body);
        b.def_var(vpp, zero_i32);
        b.ins().jump(la_p_head, &[]);

        b.switch_to_block(la_p_head);
        let pp_a = b.use_var(vpp);
        let pp_a_cond = b.ins().icmp(IntCC::SignedLessThan, pp_a, tile_c32);
        b.ins().brif(pp_a_cond, la_p_body, &[], la_inc, &[]);

        b.switch_to_block(la_p_body);
        // Global indices: i = i0*32 + ii; p = p0*32 + pp
        let i_gl = b.ins().iadd(i0_mul, ii_a);
        let p_gl = b.ins().iadd(p0_mul, pp_a);
        // Bounds check: if i<m && p<k then load else 0
        let icond = b.ins().icmp(IntCC::SignedLessThan, i_gl, m_c32);
        let pcond = b.ins().icmp(IntCC::SignedLessThan, p_gl, k_c32);
        let both = b.ins().band(icond, pcond);
        // Compute A address: a_ptr + (i*k + p)*4
        let i_times_k = b.ins().imul(i_gl, k_c32);
        let idx_a = b.ins().iadd(i_times_k, p_gl);
        let idx_a_bytes = b.ins().imul(idx_a, four_i32);
        let idx_a_ptr = b.ins().uextend(ptr_ty, idx_a_bytes);
        let addr_a = b.ins().iadd(a_ptr, idx_a_ptr);
        // Guarded load: if in-bounds, load; else, use 0.0
        let load_a_ok = b.create_block();
        let load_a_else = b.create_block();
        let load_a_join = b.create_block();
        b.ins().brif(both, load_a_ok, &[], load_a_else, &[]);
        // in-bounds path
        b.switch_to_block(load_a_ok);
        let addr_a_val = b.ins().load(types::F32, flags_ro, addr_a, 0);
        b.def_var(vtmpf, addr_a_val);
        b.ins().jump(load_a_join, &[]);
        // out-of-bounds path
        b.switch_to_block(load_a_else);
        b.def_var(vtmpf, zero_f32);
        b.ins().jump(load_a_join, &[]);
        // join
        b.switch_to_block(load_a_join);
        let val = b.use_var(vtmpf);
        // Store into A tile at [ii][pp]
        let ii_times_tile2 = b.ins().imul(ii_a, tile_c32);
        let ii_pp = b.ins().iadd(ii_times_tile2, pp_a);
        let off_ap_i32 = b.ins().imul(ii_pp, four_i32);
        let off_ap = b.ins().uextend(ptr_ty, off_ap_i32);
        let addr_ap = b.ins().iadd(a_tile_base, off_ap);
        b.ins().store(flags_st, val, addr_ap, 0);
        // pp++
        let pp_next = b.ins().iadd(pp_a, one_i32);
        b.def_var(vpp, pp_next);
        b.ins().jump(la_p_head, &[]);

        // 2) Build B tile (transposed) for this (p0, j0)
        // After finishing p loop for current ii, increment ii and continue A tile build
        b.switch_to_block(la_inc);
        let ii_cur_a = b.use_var(vii);
        let ii_next_a = b.ins().iadd(ii_cur_a, one_i32);
        b.def_var(vii, ii_next_a);
        b.ins().jump(la_i_head, &[]);

        b.switch_to_block(la_after);
        b.def_var(vpp, zero_i32);
        let lb_p_head = b.create_block();
        let lb_p_body = b.create_block();
        let lb_j_head = b.create_block();
        let lb_j_body = b.create_block();
        let lb_after = b.create_block();
        let lb_inc = b.create_block();
        b.ins().jump(lb_p_head, &[]);

        b.switch_to_block(lb_p_head);
        let pp_b = b.use_var(vpp);
        let pp_b_cond = b.ins().icmp(IntCC::SignedLessThan, pp_b, tile_c32);
        b.ins().brif(pp_b_cond, lb_p_body, &[], lb_after, &[]);

        b.switch_to_block(lb_p_body);
        b.def_var(vjj, zero_i32);
        b.ins().jump(lb_j_head, &[]);

        b.switch_to_block(lb_j_head);
        let jj_b = b.use_var(vjj);
        let jj_b_cond = b.ins().icmp(IntCC::SignedLessThan, jj_b, tile_c32);
        b.ins().brif(jj_b_cond, lb_j_body, &[], lb_inc, &[]);

        b.switch_to_block(lb_j_body);
        // Global indices: p = p0*32 + pp; j = j0*32 + jj
        let p_gl2 = b.ins().iadd(p0_mul, pp_b);
        let j_gl = b.ins().iadd(j0_mul, jj_b);
        // Bounds: p<k && j<n
        let p_ok = b.ins().icmp(IntCC::SignedLessThan, p_gl2, k_c32);
        let j_ok = b.ins().icmp(IntCC::SignedLessThan, j_gl, n_c32);
        let both2 = b.ins().band(p_ok, j_ok);
        // Address in B: b_ptr + (p*n + j)*4
        let p_times_n = b.ins().imul(p_gl2, n_c32);
        let idx_b = b.ins().iadd(p_times_n, j_gl);
        let idx_b_bytes = b.ins().imul(idx_b, four_i32);
        let idx_b_ptr = b.ins().uextend(ptr_ty, idx_b_bytes);
        let addr_b = b.ins().iadd(b_ptr, idx_b_ptr);
        // Guarded load from B: if in-bounds, load; else, 0.0
        let load_b_ok = b.create_block();
        let load_b_else = b.create_block();
        let load_b_join = b.create_block();
        b.ins().brif(both2, load_b_ok, &[], load_b_else, &[]);
        b.switch_to_block(load_b_ok);
        let vb = b.ins().load(types::F32, flags_ro, addr_b, 0);
        b.def_var(vtmpf, vb);
        b.ins().jump(load_b_join, &[]);
        b.switch_to_block(load_b_else);
        b.def_var(vtmpf, zero_f32);
        b.ins().jump(load_b_join, &[]);
        b.switch_to_block(load_b_join);
        let vb_sel = b.use_var(vtmpf);
        // Store transposed into BT at [jj][pp]
        let jj_times_tile = b.ins().imul(jj_b, tile_c32);
        let jj_pp = b.ins().iadd(jj_times_tile, pp_b);
        let off_bt_i32 = b.ins().imul(jj_pp, four_i32);
        let off_bt = b.ins().uextend(ptr_ty, off_bt_i32);
        let addr_bt = b.ins().iadd(btile_base, off_bt);
        b.ins().store(flags_st, vb_sel, addr_bt, 0);
        let jj_next2 = b.ins().iadd(jj_b, one_i32);
        b.def_var(vjj, jj_next2);
        b.ins().jump(lb_j_head, &[]);

        // 3) Compute Ctile += A_tile x BT
        // After finishing j loop for current pp, increment pp and continue B-tile build
        b.switch_to_block(lb_inc);
        let pp_cur_b = b.use_var(vpp);
        let pp_next_b = b.ins().iadd(pp_cur_b, one_i32);
        b.def_var(vpp, pp_next_b);
        b.ins().jump(lb_p_head, &[]);

        b.switch_to_block(lb_after);
        // for ii in 0..32
        b.def_var(vii, zero_i32);
        let lc_i_head = b.create_block();
        let lc_i_body = b.create_block();
        let lc_j_head = b.create_block();
        let lc_j_body = b.create_block();
        let lc_p_head = b.create_block();
        let lc_p_body = b.create_block();
        let lc_p_after = b.create_block();
        let lc_after = b.create_block();
        let lc_i_inc = b.create_block();
        b.ins().jump(lc_i_head, &[]);

        b.switch_to_block(lc_i_head);
        let ii_c = b.use_var(vii);
        let ii_c_cond = b.ins().icmp(IntCC::SignedLessThan, ii_c, tile_c32);
        b.ins().brif(ii_c_cond, lc_i_body, &[], lc_after, &[]);

        b.switch_to_block(lc_i_body);
        b.def_var(vjj, zero_i32);
        b.ins().jump(lc_j_head, &[]);

        b.switch_to_block(lc_j_head);
        let jj_c = b.use_var(vjj);
        let jj_c_cond = b.ins().icmp(IntCC::SignedLessThan, jj_c, tile_c32);
        b.ins().brif(jj_c_cond, lc_j_body, &[], lc_i_inc, &[]);

        b.switch_to_block(lc_j_body);
        // Load current cacc = CT[ii][jj]
        let ii_times_tile3 = b.ins().imul(ii_c, tile_c32);
        let ii_jj = b.ins().iadd(ii_times_tile3, jj_c);
        let off_ct_i32 = b.ins().imul(ii_jj, four_i32);
        let off_ct = b.ins().uextend(ptr_ty, off_ct_i32);
        let addr_ct = b.ins().iadd(ctile_base, off_ct);
        let cacc0 = b.ins().load(types::F32, flags_st, addr_ct, 0);
        b.def_var(vcacc, cacc0);
        // p loop
        b.def_var(vpp, zero_i32);
        b.ins().jump(lc_p_head, &[]);

        b.switch_to_block(lc_p_head);
        let pp_c = b.use_var(vpp);
        let pp_c_cond = b.ins().icmp(IntCC::SignedLessThan, pp_c, tile_c32);
        b.ins().brif(pp_c_cond, lc_p_body, &[], lc_p_after, &[]);

        b.switch_to_block(lc_p_body);
        // a = A_tile[ii][pp]; bt = BT[jj][pp]
        let ii_mul_tile = b.ins().imul(ii_c, tile_c32);
        let ii_pp2 = b.ins().iadd(ii_mul_tile, pp_c);
        let off_ap2_i32 = b.ins().imul(ii_pp2, four_i32);
        let off_ap2 = b.ins().uextend(ptr_ty, off_ap2_i32);
        let addr_ap2 = b.ins().iadd(a_tile_base, off_ap2);
        let va = b.ins().load(types::F32, flags_st, addr_ap2, 0);

        let jj_mul_tile = b.ins().imul(jj_c, tile_c32);
        let jj_pp2 = b.ins().iadd(jj_mul_tile, pp_c);
        let off_bt2_i32 = b.ins().imul(jj_pp2, four_i32);
        let off_bt2 = b.ins().uextend(ptr_ty, off_bt2_i32);
        let addr_bt2 = b.ins().iadd(btile_base, off_bt2);
        let vb = b.ins().load(types::F32, flags_st, addr_bt2, 0);

        let prod = b.ins().fmul(va, vb);
        let sum_old = b.use_var(vcacc);
        let sum_new = b.ins().fadd(sum_old, prod);
        b.def_var(vcacc, sum_new);
        let pp_next2 = b.ins().iadd(pp_c, one_i32);
        b.def_var(vpp, pp_next2);
        b.ins().jump(lc_p_head, &[]);

        b.switch_to_block(lc_p_after);
        // After finishing all p, store vcacc to CT[ii][jj] and advance jj
        let ii_times_tile4 = b.ins().imul(ii_c, tile_c32);
        let ii_jj2 = b.ins().iadd(ii_times_tile4, jj_c);
        let off_ct2_i32 = b.ins().imul(ii_jj2, four_i32);
        let off_ct2 = b.ins().uextend(ptr_ty, off_ct2_i32);
        let addr_ct2 = b.ins().iadd(ctile_base, off_ct2);
        let sum_final = b.use_var(vcacc);
        b.ins().store(flags_st, sum_final, addr_ct2, 0);
        let jj_next3 = b.ins().iadd(jj_c, one_i32);
        b.def_var(vjj, jj_next3);
        b.ins().jump(lc_j_head, &[]);

        // Increment ii after finishing jj loop for this ii
        b.switch_to_block(lc_i_inc);
        let ii_cur_c = b.use_var(vii);
        let ii_next_c = b.ins().iadd(ii_cur_c, one_i32);
        b.def_var(vii, ii_next_c);
        b.ins().jump(lc_i_head, &[]);

        // After computing this p0 block, advance p0
        b.switch_to_block(lc_after);
        let p0_next = b.ins().iadd(p0v, one_i32);
        b.def_var(vp0, p0_next);
        b.ins().jump(lp0_head, &[]);

        // After all p0, write CT back to C with bounds
        b.switch_to_block(after_p0);
        b.def_var(vii, zero_i32);
        let lw_i_head = b.create_block();
        let lw_i_body = b.create_block();
        let lw_j_head = b.create_block();
        let lw_j_body = b.create_block();
        let lw_after = b.create_block();
        b.ins().jump(lw_i_head, &[]);

        b.switch_to_block(lw_i_head);
        let ii_w = b.use_var(vii);
        let ii_w_cond = b.ins().icmp(IntCC::SignedLessThan, ii_w, tile_c32);
        b.ins().brif(ii_w_cond, lw_i_body, &[], after_j0, &[]);

        b.switch_to_block(lw_i_body);
        b.def_var(vjj, zero_i32);
        b.ins().jump(lw_j_head, &[]);

        b.switch_to_block(lw_j_head);
        let jj_w = b.use_var(vjj);
        let jj_w_cond = b.ins().icmp(IntCC::SignedLessThan, jj_w, tile_c32);
        b.ins().brif(jj_w_cond, lw_j_body, &[], lw_i_head, &[]);

        b.switch_to_block(lw_j_body);
        // Global indices for output
        let i_out = b.ins().iadd(i0_mul, ii_w);
        let j_out = b.ins().iadd(j0_mul, jj_w);
        let i_ok = b.ins().icmp(IntCC::SignedLessThan, i_out, m_c32);
        let j_ok2 = b.ins().icmp(IntCC::SignedLessThan, j_out, n_c32);
        let ok_store = b.ins().band(i_ok, j_ok2);
        // Load from CT
        let ii_times_tile5 = b.ins().imul(ii_w, tile_c32);
        let ii_jj3 = b.ins().iadd(ii_times_tile5, jj_w);
        let off_ct3_i32 = b.ins().imul(ii_jj3, four_i32);
        let off_ct3 = b.ins().uextend(ptr_ty, off_ct3_i32);
        let addr_ct3 = b.ins().iadd(ctile_base, off_ct3);
        let val_ct = b.ins().load(types::F32, flags_st, addr_ct3, 0);
        // Addr in C
        let i_times_n = b.ins().imul(i_out, n_c32);
        let idx_c = b.ins().iadd(i_times_n, j_out);
        let idx_c_bytes = b.ins().imul(idx_c, four_i32);
        let idx_c_ptr = b.ins().uextend(ptr_ty, idx_c_bytes);
        let addr_c = b.ins().iadd(c_ptr, idx_c_ptr);
        // Conditional store: emulate with select + store; we can't guard store directly, so select value and still store is ok because address might be OOB. So use guarded path: branch if ok
        let store_true = b.create_block();
        let store_cont = b.create_block();
        b.ins().brif(ok_store, store_true, &[], store_cont, &[]);
        b.switch_to_block(store_true);
        b.ins().store(flags_st, val_ct, addr_c, 0);
        b.ins().jump(store_cont, &[]);
        b.switch_to_block(store_cont);
        let jj_next4 = b.ins().iadd(jj_w, one_i32);
        b.def_var(vjj, jj_next4);
        b.ins().jump(lw_j_head, &[]);

        // After J0 body: advance j0
        b.switch_to_block(after_j0);
        let j0_next = b.ins().iadd(j0v, one_i32);
        b.def_var(vj0, j0_next);
        b.ins().jump(lj0_head, &[]);

        // After finishing J0 tiles for this I0, increment I0 and continue
        b.switch_to_block(after_i0_inc);
        let i0_cur = b.use_var(vi0);
        let i0_next2 = b.ins().iadd(i0_cur, one_i32);
        b.def_var(vi0, i0_next2);
        b.ins().jump(li0_head, &[]);

        // After I0: function return
        b.switch_to_block(after_i0);
        b.ins().return_(&[]);

        // Seal blocks (best-effort; some may be sealed implicitly)
        // Note: Cranelift frontend requires sealing all blocks that have all predecessors known.
        // We attempt to seal most created blocks; missing seals may still validate because we created forward edges before sealing.
        b.seal_all_blocks();

        b.finalize();

        let func_id = module
            .declare_function("matmul_f32_cl3", Linkage::Local, &ctx.func.signature)
            .expect("declare_function");
        module
            .define_function(func_id, &mut ctx)
            .expect("define_function");
        let clif_text = format!("{}", ctx.func.display());
        (func_id, clif_text)
    }
}

impl MatmulImpl for CraneliftImpl {
    fn name(&self) -> &'static str { "cranelift_3" }
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
        unsafe {
            (self.func_ptr)(
                a_std.as_ptr(),
                b_std.as_ptr(),
                c.as_mut_ptr(),
            );
        }
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
    fn name(&self) -> &'static str { "cranelift_3" }
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
