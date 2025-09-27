use cranelift_codegen::ir::{types, AbiParam, InstBuilder, MemFlags};
use cranelift_codegen::settings;
use cranelift_codegen::settings::{Configurable, Flags};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use ndarray::Array2;
use crate::{MatmulImpl, MatmulImplBuilder};

// ----------------- Cranelift JIT implementation -----------------
// Type of the generated JIT function (dims baked into code)
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
        ctx.func.name = cranelift_codegen::ir::UserFuncName::user(0, 0);

        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        b.seal_block(entry);

        // Params
        let a_ptr = b.block_params(entry)[0];
        let b_ptr = b.block_params(entry)[1];
        let c_ptr = b.block_params(entry)[2];

        // Declare variables
        let vi = b.declare_var(types::I64);
        let vj = b.declare_var(types::I64);
        let vp = b.declare_var(types::I64);
        let vsum = b.declare_var(types::F32);

        // Constants
        let zero_i64 = b.ins().iconst(types::I64, 0);
        let one_i64 = b.ins().iconst(types::I64, 1);
        let four_i64 = b.ins().iconst(types::I64, 4);
        let zero_f32 = b.ins().f32const(0.0);
        let m_c = b.ins().iconst(types::I64, m as i64);
        let n_c = b.ins().iconst(types::I64, n as i64);
        let k_c = b.ins().iconst(types::I64, k as i64);

        b.def_var(vi, zero_i64);

        let li_head = b.create_block();
        let li_body = b.create_block();
        let lj_head = b.create_block();
        let lj_body = b.create_block();
        let lp_head = b.create_block();
        let lp_body = b.create_block();
        let after_p = b.create_block();
        let after_j = b.create_block();
        let after_i = b.create_block();

        // i loop
        b.ins().jump(li_head, &[]);
        b.switch_to_block(li_head);
        let i_val = b.use_var(vi);
        let i_cond = b.ins().icmp(IntCC::SignedLessThan, i_val, m_c);
        b.ins().brif(i_cond, li_body, &[], after_i, &[]);

        b.switch_to_block(li_body);
        b.def_var(vj, zero_i64);
        b.ins().jump(lj_head, &[]);

        // j loop
        b.switch_to_block(lj_head);
        let j_val = b.use_var(vj);
        let j_cond = b.ins().icmp(IntCC::SignedLessThan, j_val, n_c);
        b.ins().brif(j_cond, lj_body, &[], after_j, &[]);

        b.switch_to_block(lj_body);
        b.def_var(vsum, zero_f32);
        b.def_var(vp, zero_i64);
        b.ins().jump(lp_head, &[]);

        // p loop
        b.switch_to_block(lp_head);
        let p_val = b.use_var(vp);
        let p_cond = b.ins().icmp(IntCC::SignedLessThan, p_val, k_c);
        b.ins().brif(p_cond, lp_body, &[], after_p, &[]);

        b.switch_to_block(lp_body);
        // Compute idx_a = (i*k + p) * 4
        let i_mul_k = b.ins().imul(i_val, k_c);
        let idx_a = b.ins().iadd(i_mul_k, p_val);
        let off_a_bytes = b.ins().imul(idx_a, four_i64);
        let addr_a = b.ins().iadd(a_ptr, off_a_bytes);
        // Compute idx_b = (p*n + j) * 4
        let p_mul_n = b.ins().imul(p_val, n_c);
        let idx_b = b.ins().iadd(p_mul_n, j_val);
        let off_b_bytes = b.ins().imul(idx_b, four_i64);
        let addr_b = b.ins().iadd(b_ptr, off_b_bytes);

        let flags = MemFlags::trusted();
        let va = b.ins().load(types::F32, flags, addr_a, 0);
        let vb = b.ins().load(types::F32, flags, addr_b, 0);
        let prod = b.ins().fmul(va, vb);
        let sum_old = b.use_var(vsum);
        let sum_new = b.ins().fadd(sum_old, prod);
        b.def_var(vsum, sum_new);

        // p++
        let p_next = b.ins().iadd(p_val, one_i64);
        b.def_var(vp, p_next);
        b.ins().jump(lp_head, &[]);

        // after p: store C[i*n + j]
        b.switch_to_block(after_p);
        let i_mul_n = b.ins().imul(i_val, n_c);
        let idx_c = b.ins().iadd(i_mul_n, j_val);
        let off_c_bytes = b.ins().imul(idx_c, four_i64);
        let addr_c = b.ins().iadd(c_ptr, off_c_bytes);
        let sum_final = b.use_var(vsum);
        b.ins().store(flags, sum_final, addr_c, 0);

        // j++ and loop
        let j_next = b.ins().iadd(j_val, one_i64);
        b.def_var(vj, j_next);
        b.ins().jump(lj_head, &[]);

        // after j: i++ and loop
        b.switch_to_block(after_j);
        let i_next = b.ins().iadd(i_val, one_i64);
        b.def_var(vi, i_next);
        b.ins().jump(li_head, &[]);

        // after i: return
        b.switch_to_block(after_i);
        b.ins().return_(&[]);

        // Seal all blocks now that all predecessors are known
        b.seal_block(lp_head);
        b.seal_block(lp_body);
        b.seal_block(lj_head);
        b.seal_block(lj_body);
        b.seal_block(li_head);
        b.seal_block(li_body);
        b.seal_block(after_p);
        b.seal_block(after_j);
        b.seal_block(after_i);

        b.finalize();

        let func_id = module
            .declare_function("matmul_f32", Linkage::Local, &ctx.func.signature)
            .expect("declare_function");
        module
            .define_function(func_id, &mut ctx)
            .expect("define_function");
        let clif_text = format!("{}", ctx.func.display());
        (func_id, clif_text)
    }
}

impl MatmulImpl for CraneliftImpl {
    fn name(&self) -> &'static str { "cranelift" }
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
    fn name(&self) -> &'static str { "cranelift" }
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
