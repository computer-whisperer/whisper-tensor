use cranelift_codegen::ir::{types, AbiParam, InstBuilder, MemFlags};
use cranelift_codegen::settings;
use cranelift_codegen::settings::{Configurable, Flags};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use ndarray::Array2;
use crate::{MatmulImpl, MatmulImplBuilder};

// ----------------- Cranelift JIT implementation (cranelift_2) -----------------
// Adds up to 16x unrolling in the innermost (p) loop, with pointer bumping and LICM from _1.
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
        ctx.func.name = cranelift_codegen::ir::UserFuncName::user(0, 2);

        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        b.seal_block(entry);

        // Params
        let a_ptr = b.block_params(entry)[0];
        let b_ptr = b.block_params(entry)[1];
        let c_ptr = b.block_params(entry)[2];

        // Declare variables (use I32 for induction vars)
        let vi = b.declare_var(types::I32);
        let vj = b.declare_var(types::I32);
        let vp = b.declare_var(types::I32);
        let vsum = b.declare_var(types::F32);
        let va_cur = b.declare_var(ptr_ty);
        let vb_cur = b.declare_var(ptr_ty);


        // Constants
        let zero_i32 = b.ins().iconst(types::I32, 0);
        let one_i32 = b.ins().iconst(types::I32, 1);
        let four_i32 = b.ins().iconst(types::I32, 4);
        let sixteen_i32 = b.ins().iconst(types::I32, 16);
        let zero_f32 = b.ins().f32const(0.0);
        let m_c32 = b.ins().iconst(types::I32, m as i64);
        let n_c32 = b.ins().iconst(types::I32, n as i64);
        let k_c32 = b.ins().iconst(types::I32, k as i64);
        // Precompute byte strides (I32), then extend to pointer when used
        let k_bytes_i32 = b.ins().imul(k_c32, four_i32); // k*4
        let n_bytes_i32 = b.ins().imul(n_c32, four_i32); // n*4
        let n_bytes_ptr = b.ins().uextend(ptr_ty, n_bytes_i32);
        let four_i32_ptr = b.ins().uextend(ptr_ty, four_i32);

        // Compute unrolled bound: p16_end = k - (k & 15)
        let fifteen_i32 = b.ins().iconst(types::I32, 15);
        let k_and_15 = b.ins().band(k_c32, fifteen_i32);
        let p16_end = b.ins().isub(k_c32, k_and_15);

        // MemFlags: loads from A and B are readonly, aligned (at least 4B), and notrap
        let mut flags_ro = MemFlags::trusted();
        flags_ro.set_notrap();
        flags_ro.set_readonly();
        flags_ro.set_aligned();
        // Stores to C: aligned and notrap
        let mut flags_st = MemFlags::trusted();
        flags_st.set_notrap();
        flags_st.set_aligned();

        b.def_var(vi, zero_i32);

        let li_head = b.create_block();
        let li_body = b.create_block();
        let lj_head = b.create_block();
        let lj_body = b.create_block();
        let lp16_head = b.create_block();
        let lp16_body = b.create_block();
        let lp_rem_head = b.create_block();
        let lp_rem_body = b.create_block();
        let after_p = b.create_block();
        let after_j = b.create_block();
        let after_i = b.create_block();

        // i loop
        b.ins().jump(li_head, &[]);
        b.switch_to_block(li_head);
        let i_val = b.use_var(vi);
        let i_cond = b.ins().icmp(IntCC::SignedLessThan, i_val, m_c32);
        b.ins().brif(i_cond, li_body, &[], after_i, &[]);

        // li_body: initialize j and compute row base pointers for A and C
        b.switch_to_block(li_body);
        // a_row_ptr = a_ptr + (i * (k*4))
        let i_times_kbytes_i32 = b.ins().imul(i_val, k_bytes_i32);
        let i_times_kbytes_ptr = b.ins().uextend(ptr_ty, i_times_kbytes_i32);
        let a_row_ptr = b.ins().iadd(a_ptr, i_times_kbytes_ptr);
        // c_row_ptr = c_ptr + (i * (n*4))
        let i_times_nbytes_i32 = b.ins().imul(i_val, n_bytes_i32);
        let i_times_nbytes_ptr = b.ins().uextend(ptr_ty, i_times_nbytes_i32);
        let c_row_ptr = b.ins().iadd(c_ptr, i_times_nbytes_ptr);

        b.def_var(vj, zero_i32);
        b.ins().jump(lj_head, &[]);

        // j loop
        b.switch_to_block(lj_head);
        let j_val = b.use_var(vj);
        let j_cond = b.ins().icmp(IntCC::SignedLessThan, j_val, n_c32);
        b.ins().brif(j_cond, lj_body, &[], after_j, &[]);

        // lj_body: initialize sum, p, and set up pointer bumping for A and B
        b.switch_to_block(lj_body);
        b.def_var(vsum, zero_f32);
        b.def_var(vp, zero_i32);
        // a_cur = a_row_ptr
        b.def_var(va_cur, a_row_ptr);
        // b_cur = b_ptr + j*4
        let j_bytes_i32 = b.ins().imul(j_val, four_i32);
        let j_bytes_ptr = b.ins().uextend(ptr_ty, j_bytes_i32);
        let b_col_base = b.ins().iadd(b_ptr, j_bytes_ptr);
        b.def_var(vb_cur, b_col_base);
        b.ins().jump(lp16_head, &[]);

        // p chunk loop (16x unrolled)
        b.switch_to_block(lp16_head);
        let p_val16 = b.use_var(vp);
        let cond16 = b.ins().icmp(IntCC::SignedLessThan, p_val16, p16_end);
        b.ins().brif(cond16, lp16_body, &[], lp_rem_head, &[]);

        b.switch_to_block(lp16_body);
        let mut a_addr = b.use_var(va_cur);
        let mut b_addr = b.use_var(vb_cur);
        let mut sum = b.use_var(vsum);
        // Unroll 16 iterations
        for t in 0..16 {
            let va = b.ins().load(types::F32, flags_ro, a_addr, 0);
            let vb = b.ins().load(types::F32, flags_ro, b_addr, 0);
            let prod = b.ins().fmul(va, vb);
            sum = b.ins().fadd(sum, prod);
            if t != 15 {
                a_addr = b.ins().iadd(a_addr, four_i32_ptr);
                b_addr = b.ins().iadd(b_addr, n_bytes_ptr);
            }
        }
        // Advance once more to point to element after the 16th step
        a_addr = b.ins().iadd(a_addr, four_i32_ptr);
        b_addr = b.ins().iadd(b_addr, n_bytes_ptr);
        b.def_var(vsum, sum);
        b.def_var(va_cur, a_addr);
        b.def_var(vb_cur, b_addr);
        let p_next16 = b.ins().iadd(p_val16, sixteen_i32);
        b.def_var(vp, p_next16);
        b.ins().jump(lp16_head, &[]);

        // Remainder scalar loop
        b.switch_to_block(lp_rem_head);
        let p_val = b.use_var(vp);
        let p_cond = b.ins().icmp(IntCC::SignedLessThan, p_val, k_c32);
        b.ins().brif(p_cond, lp_rem_body, &[], after_p, &[]);

        b.switch_to_block(lp_rem_body);
        // Load A[p] and B[p*n + j] using current pointers
        let a_addr_cur = b.use_var(va_cur);
        let b_addr_cur = b.use_var(vb_cur);
        let va = b.ins().load(types::F32, flags_ro, a_addr_cur, 0);
        let vb = b.ins().load(types::F32, flags_ro, b_addr_cur, 0);
        let prod = b.ins().fmul(va, vb);
        let sum_old = b.use_var(vsum);
        let sum_new = b.ins().fadd(sum_old, prod);
        b.def_var(vsum, sum_new);

        // Bump pointers: a_cur += 4; b_cur += n*4
        let a_next = b.ins().iadd(a_addr_cur, four_i32_ptr);
        let b_next = b.ins().iadd(b_addr_cur, n_bytes_ptr);
        b.def_var(va_cur, a_next);
        b.def_var(vb_cur, b_next);

        // p++
        let p_next = b.ins().iadd(p_val, one_i32);
        b.def_var(vp, p_next);
        b.ins().jump(lp_rem_head, &[]);

        // after p: store C[i*n + j] at c_row_ptr + j*4
        b.switch_to_block(after_p);
        let j_bytes_i32_2 = b.ins().imul(j_val, four_i32);
        let j_bytes_ptr_2 = b.ins().uextend(ptr_ty, j_bytes_i32_2);
        let addr_c = b.ins().iadd(c_row_ptr, j_bytes_ptr_2);
        let sum_final = b.use_var(vsum);
        b.ins().store(flags_st, sum_final, addr_c, 0);

        // j++ and loop
        let j_next = b.ins().iadd(j_val, one_i32);
        b.def_var(vj, j_next);
        b.ins().jump(lj_head, &[]);

        // after j: i++ and loop
        b.switch_to_block(after_j);
        let i_next = b.ins().iadd(i_val, one_i32);
        b.def_var(vi, i_next);
        b.ins().jump(li_head, &[]);

        // after i: return
        b.switch_to_block(after_i);
        b.ins().return_(&[]);

        // Seal all blocks now that all predecessors are known
        b.seal_block(lp16_head);
        b.seal_block(lp16_body);
        b.seal_block(lp_rem_head);
        b.seal_block(lp_rem_body);
        b.seal_block(lj_head);
        b.seal_block(lj_body);
        b.seal_block(li_head);
        b.seal_block(li_body);
        b.seal_block(after_p);
        b.seal_block(after_j);
        b.seal_block(after_i);

        b.finalize();

        let func_id = module
            .declare_function("matmul_f32_cl2", Linkage::Local, &ctx.func.signature)
            .expect("declare_function");
        module
            .define_function(func_id, &mut ctx)
            .expect("define_function");
        let clif_text = format!("{}", ctx.func.display());
        (func_id, clif_text)
    }
}

impl MatmulImpl for CraneliftImpl {
    fn name(&self) -> &'static str { "cranelift_2" }
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
    fn name(&self) -> &'static str { "cranelift_2" }
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
