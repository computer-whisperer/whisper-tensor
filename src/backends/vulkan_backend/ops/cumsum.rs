use crate::backends::vulkan_backend::spirv_helpers::{
    cast_bf16_to_f32, cast_f32_to_bf16, get_spirv_datatype,
};
use crate::backends::vulkan_backend::tensor::VulkanTensor;
use crate::backends::vulkan_backend::{VulkanError, VulkanImmediateExecutor};
use crate::dtype::DType;
use crate::tensor_rank::{DimContainer, DimProduct, Rank};
use rspirv::binary::Assemble;
use rspirv::dr::Operand;
use rspirv::dr::Operand::LiteralBit32;
use rspirv::spirv;
use rspirv::spirv::{
    BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel, LoopControl, SelectionControl,
    StorageClass,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::{PipelineLayoutCreateInfo, PushConstantRange};
use vulkano::pipeline::{
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo, ShaderStages};
use vulkano::sync::GpuFuture;

#[allow(clippy::needless_range_loop)]
fn build_cumsum_pipeline(
    vulkan_immediate_executor: &mut VulkanImmediateExecutor,
    dtype: DType,
    rank: usize,
) -> Result<(Arc<PipelineLayout>, Arc<ComputePipeline>), VulkanError> {
    let mut b = rspirv::dr::Builder::new();
    b.capability(Capability::Shader);
    b.capability(Capability::Float64);
    b.capability(Capability::Float16);
    b.capability(Capability::Int16);
    b.capability(Capability::Int8);
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();

    let data_type = match dtype {
        DType::BF16 => b.type_int(16, 0),
        DType::BOOL => b.type_int(8, 0),
        _ => get_spirv_datatype(&mut b, dtype)?,
    };

    let input_data_type_array = b.type_runtime_array(data_type);
    b.decorate(
        input_data_type_array,
        Decoration::ArrayStride,
        [Operand::LiteralBit32(dtype.size().unwrap() as u32)],
    );
    let input_data_type_array_struct = b.type_struct([input_data_type_array]);
    b.decorate(input_data_type_array_struct, Decoration::Block, []);
    b.member_decorate(
        input_data_type_array_struct,
        0,
        Decoration::Offset,
        [Operand::LiteralBit32(0)],
    );
    let input_sb_ptr = b.type_pointer(
        None,
        StorageClass::StorageBuffer,
        input_data_type_array_struct,
    );
    let input_0_var = b.variable(input_sb_ptr, None, StorageClass::StorageBuffer, None);
    b.decorate(input_0_var, Decoration::DescriptorSet, [LiteralBit32(0)]);
    b.decorate(input_0_var, Decoration::Binding, [LiteralBit32(0)]);

    let output_data_type_array = b.type_runtime_array(data_type);
    if output_data_type_array != input_data_type_array {
        b.decorate(
            output_data_type_array,
            Decoration::ArrayStride,
            [Operand::LiteralBit32(dtype.size().unwrap() as u32)],
        );
    }
    let output_data_type_array_struct = b.type_struct([output_data_type_array]);
    if output_data_type_array_struct != input_data_type_array_struct {
        b.decorate(output_data_type_array_struct, Decoration::Block, []);
        b.member_decorate(
            output_data_type_array_struct,
            0,
            Decoration::Offset,
            [Operand::LiteralBit32(0)],
        );
    }
    let output_sb_ptr = b.type_pointer(
        None,
        StorageClass::StorageBuffer,
        output_data_type_array_struct,
    );
    let output_0_var = b.variable(output_sb_ptr, None, StorageClass::StorageBuffer, None);
    b.decorate(output_0_var, Decoration::DescriptorSet, [LiteralBit32(0)]);
    b.decorate(output_0_var, Decoration::Binding, [LiteralBit32(1)]);

    let u32_t = b.type_int(32, 0);
    // Push constants per-element:
    // 0: input_offset
    // 1: output_offset
    // 2: num_elements
    // 3: axis
    // 4: axis_len
    // 5: flags (bit0 exclusive, bit1 reverse)
    // 6..6+rank: shape
    // 6+rank..6+2*rank: in_strides
    // 6+2*rank..6+3*rank: out_strides
    let metadata_value_count = 6 + 3 * rank;

    let pc_struct = {
        let mut v = Vec::new();
        v.resize(metadata_value_count, u32_t);
        b.type_struct(v)
    };
    b.decorate(pc_struct, Decoration::Block, []);
    for i in 0..metadata_value_count {
        b.member_decorate(
            pc_struct,
            i as u32,
            Decoration::Offset,
            [Operand::LiteralBit32((i * 4) as u32)],
        );
    }

    let pc_ptr = b.type_pointer(None, StorageClass::PushConstant, pc_struct);
    let metadata_var = b.variable(pc_ptr, None, StorageClass::PushConstant, None);

    let vec3u32_t = b.type_vector(u32_t, 3);
    let in_ptr = b.type_pointer(None, StorageClass::Input, vec3u32_t);
    let gid = b.variable(in_ptr, None, StorageClass::Input, None);
    b.decorate(
        gid,
        Decoration::BuiltIn,
        [Operand::BuiltIn(BuiltIn::GlobalInvocationId)],
    );

    let voidf = b.type_function(void, []);
    let main_fn = b
        .begin_function(void, None, spirv::FunctionControl::DONT_INLINE, voidf)
        .unwrap();
    b.entry_point(
        ExecutionModel::GLCompute,
        main_fn,
        "main",
        [gid, metadata_var, output_0_var, input_0_var],
    );
    b.execution_mode(main_fn, ExecutionMode::LocalSize, [64, 1, 1]);
    b.begin_block(None).unwrap();

    // Predeclare common types and function-scope variables in entry block
    let f32_t = b.type_float(32);
    let f16_t = b.type_float(16);
    let acc_ptr_t = b.type_pointer(None, StorageClass::Function, f32_t);
    let acc_var = b.variable(acc_ptr_t, None, StorageClass::Function, None);
    let k_ptr_t = b.type_pointer(None, StorageClass::Function, u32_t);
    let k_var = b.variable(k_ptr_t, None, StorageClass::Function, None);

    /* Constants */
    let c0 = b.constant_bit32(u32_t, 0);
    let c1 = b.constant_bit32(u32_t, 1);

    /* idx = gl_GlobalInvocationID.x */
    let gidv = b.load(vec3u32_t, None, gid, None, []).unwrap();
    let idx = b.composite_extract(u32_t, None, gidv, [0u32]).unwrap();

    /* Load push constants */
    let push_vals = {
        let mut vals = Vec::new();
        let u32_pc_ptr_t = b.type_pointer(None, StorageClass::PushConstant, u32_t);
        for i in 0..metadata_value_count {
            let c_i = b.constant_bit32(u32_t, i as u32);
            let ptr = b
                .access_chain(u32_pc_ptr_t, None, metadata_var, [c_i])
                .unwrap();
            vals.push(b.load(u32_t, None, ptr, None, []).unwrap());
        }
        vals
    };

    let input_offset = push_vals[0];
    let output_offset = push_vals[1];
    let num_elems = push_vals[2];
    let axis = push_vals[3];
    let axis_len = push_vals[4];
    let flags = push_vals[5];

    let shape = &push_vals[6..6 + rank];
    let in_strides = &push_vals[6 + rank..6 + 2 * rank];
    let out_strides = &push_vals[6 + 2 * rank..6 + 3 * rank];

    /* bounds check */
    let bool_type = b.type_bool();
    let in_range = b.u_less_than(bool_type, None, idx, num_elems).unwrap();
    let merge_blk = b.id();
    let then_blk = b.id();
    b.selection_merge(merge_blk, SelectionControl::NONE)
        .unwrap();
    b.branch_conditional(in_range, then_blk, merge_blk, None)
        .unwrap();

    /* ---- THEN block ---- */
    b.begin_block(Some(then_blk)).unwrap();

    // Decode idx into multi-index
    let mut remaining_idx = idx;
    let index: Vec<rspirv::spirv::Word> = shape
        .iter()
        .map(|&dim| {
            let rem = b.u_mod(u32_t, None, remaining_idx, dim).unwrap();
            let div = b.u_div(u32_t, None, remaining_idx, dim).unwrap();
            remaining_idx = div;
            rem
        })
        .collect();

    // Compute base indices excluding axis, and current axis position p
    let mut base_in = input_offset;
    let mut base_out = output_offset;
    let mut p = c0;
    for i in 0..rank {
        let i_u32 = b.constant_bit32(u32_t, i as u32);
        let is_axis = b.i_equal(bool_type, None, i_u32, axis).unwrap();
        // p = index[i] if i == axis
        let p_new = b.select(u32_t, None, is_axis, index[i], p).unwrap();
        p = p_new;
        // if i != axis, add to bases
        let add_in = b.i_mul(u32_t, None, index[i], in_strides[i]).unwrap();
        let add_out = b.i_mul(u32_t, None, index[i], out_strides[i]).unwrap();
        let add_in_sum = b.i_add(u32_t, None, base_in, add_in).unwrap();
        let base_in_new = b.select(u32_t, None, is_axis, base_in, add_in_sum).unwrap();
        let add_out_sum = b.i_add(u32_t, None, base_out, add_out).unwrap();
        let base_out_new = b
            .select(u32_t, None, is_axis, base_out, add_out_sum)
            .unwrap();
        base_in = base_in_new;
        base_out = base_out_new;
    }

    // Flags
    let is_exclusive = {
        let mask = b.constant_bit32(u32_t, 1);
        let andv = b.bitwise_and(u32_t, None, flags, mask).unwrap();
        b.i_not_equal(bool_type, None, andv, c0).unwrap()
    };
    let is_reverse = {
        let mask = b.constant_bit32(u32_t, 2);
        let andv = b.bitwise_and(u32_t, None, flags, mask).unwrap();
        b.i_not_equal(bool_type, None, andv, c0).unwrap()
    };

    // Initialize function variables declared at entry
    let acc_zero = b.constant_bit32(f32_t, 0.0f32.to_bits());
    b.store(acc_var, acc_zero, None, []).unwrap();
    b.store(k_var, c0, None, []).unwrap();

    // Precompute axis stride for input
    let axis_stride_in = {
        let mut stride = c0;
        for i in 0..rank {
            let i_u32 = b.constant_bit32(u32_t, i as u32);
            let is_axis = b.i_equal(bool_type, None, i_u32, axis).unwrap();
            stride = b
                .select(u32_t, None, is_axis, in_strides[i], stride)
                .unwrap();
        }
        stride
    };

    // Loop k = 0..axis_len
    let hdr = b.id();
    let check = b.id();
    let doblk = b.id();
    let cont = b.id();
    let merge = b.id();
    b.branch(hdr).unwrap();
    b.begin_block(Some(hdr)).unwrap();
    b.loop_merge(merge, cont, LoopControl::NONE, []).unwrap();
    b.branch(check).unwrap();

    // check: evaluate condition and branch to body or merge
    b.begin_block(Some(check)).unwrap();
    let k_cur = b.load(u32_t, None, k_var, None, []).unwrap();
    let k_lt_len = b.u_less_than(bool_type, None, k_cur, axis_len).unwrap();
    b.branch_conditional(k_lt_len, doblk, merge, None).unwrap();

    // doblk: loop body
    b.begin_block(Some(doblk)).unwrap();
    let acc_cur = b.load(f32_t, None, acc_var, None, []).unwrap();

    // Decide contribution (no underflow/overflow tricks). For forward: exclusive k<p, inclusive k<=p. For reverse: exclusive k>p, inclusive k>=p.
    let k_lt_p = b.u_less_than(bool_type, None, k_cur, p).unwrap();
    let k_le_p = b.u_less_than_equal(bool_type, None, k_cur, p).unwrap();
    let k_gt_p = b.u_greater_than(bool_type, None, k_cur, p).unwrap();
    let k_ge_p = b.u_greater_than_equal(bool_type, None, k_cur, p).unwrap();
    let fwd = b
        .select(bool_type, None, is_exclusive, k_lt_p, k_le_p)
        .unwrap();
    let rev = b
        .select(bool_type, None, is_exclusive, k_gt_p, k_ge_p)
        .unwrap();
    let contrib = b.select(bool_type, None, is_reverse, rev, fwd).unwrap();

    // Load input at k
    let data_i_ptr_t = b.type_pointer(None, StorageClass::StorageBuffer, data_type);
    let k_mul_in = b.i_mul(u32_t, None, k_cur, axis_stride_in).unwrap();
    let in_index = b.i_add(u32_t, None, base_in, k_mul_in).unwrap();
    let in_ptr = b
        .access_chain(data_i_ptr_t, None, input_0_var, [c0, in_index])
        .unwrap();
    let in_val_raw = b.load(data_type, None, in_ptr, None, []).unwrap();

    let in_f32 = match dtype {
        DType::BF16 => cast_bf16_to_f32(&mut b, in_val_raw),
        DType::F32 => in_val_raw,
        DType::F16 => b.f_convert(f32_t, None, in_val_raw).unwrap(),
        _ => in_val_raw,
    };

    let added = b.f_add(f32_t, None, acc_cur, in_f32).unwrap();
    let acc_new = b.select(f32_t, None, contrib, added, acc_cur).unwrap();
    b.store(acc_var, acc_new, None, []).unwrap();
    b.branch(cont).unwrap();

    // cont: increment and continue
    b.begin_block(Some(cont)).unwrap();
    let k_next = b.i_add(u32_t, None, k_cur, c1).unwrap();
    b.store(k_var, k_next, None, []).unwrap();
    b.branch(hdr).unwrap();

    // merge: exit loop
    b.begin_block(Some(merge)).unwrap();

    // Store result at output index (axis uses p)
    let axis_stride_out = {
        let mut stride = c0;
        for i in 0..rank {
            let i_u32 = b.constant_bit32(u32_t, i as u32);
            let is_axis = b.i_equal(bool_type, None, i_u32, axis).unwrap();
            stride = b
                .select(u32_t, None, is_axis, out_strides[i], stride)
                .unwrap();
        }
        stride
    };

    let data_o_ptr_t = b.type_pointer(None, StorageClass::StorageBuffer, data_type);
    let p_mul_out = b.i_mul(u32_t, None, p, axis_stride_out).unwrap();
    let out_index = b.i_add(u32_t, None, base_out, p_mul_out).unwrap();
    let out_ptr = b
        .access_chain(data_o_ptr_t, None, output_0_var, [c0, out_index])
        .unwrap();

    let acc_final = b.load(f32_t, None, acc_var, None, []).unwrap();
    let out_val = match dtype {
        DType::BF16 => cast_f32_to_bf16(&mut b, acc_final),
        DType::F16 => b.f_convert(f16_t, None, acc_final).unwrap(),
        DType::F32 => acc_final,
        _ => acc_final,
    };
    b.store(out_ptr, out_val, None, []).unwrap();

    b.branch(merge_blk).unwrap();

    b.begin_block(Some(merge_blk)).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();

    let module = b.module();
    let spirv = module.assemble();

    // Debug dump for validation if needed

    let shader = unsafe {
        ShaderModule::new(
            vulkan_immediate_executor.context.device.clone(),
            ShaderModuleCreateInfo::new(&spirv),
        )
    }?;

    let cs = shader.entry_point("main").unwrap();
    let stage = PipelineShaderStageCreateInfo::new(cs);

    let descriptor_set_layout =
        vulkan_immediate_executor.get_descriptor_set_layout(BTreeSet::from([0, 1]))?;

    let layout = PipelineLayout::new(
        vulkan_immediate_executor.context.device.clone(),
        PipelineLayoutCreateInfo {
            set_layouts: vec![descriptor_set_layout],
            push_constant_ranges: vec![PushConstantRange {
                stages: ShaderStages::COMPUTE,
                offset: 0,
                size: (metadata_value_count * 4) as u32,
            }],
            ..Default::default()
        },
    )?;

    let compute_pipeline = ComputePipeline::new(
        vulkan_immediate_executor.context.device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout.clone()),
    )?;

    Ok((layout, compute_pipeline))
}

#[derive(Clone, Hash, Eq, PartialEq, Debug)]
pub(crate) struct CumSumCacheKey {
    dtype: DType,
    rank: u32,
}

impl<R: Rank> VulkanTensor<R> {
    pub fn cumsum(
        &self,
        axis: Option<isize>,
        exclusive: bool,
        reverse: bool,
        vulkan_immediate_executor: &mut VulkanImmediateExecutor,
    ) -> Result<VulkanTensor<R>, VulkanError> {
        match self.dtype() {
            DType::F32 | DType::BF16 | DType::F16 => {}
            _ => return Err(VulkanError::UnsupportedByBackendError),
        }

        let rank = self.rank();
        let mut axis_norm: isize = axis.unwrap_or(0);
        if axis_norm < 0 {
            axis_norm += rank as isize;
        }
        let axis_u = axis_norm as usize;
        assert!(axis_u < rank);

        let num_elems = self.shape().dim_product() as u32;
        let axis_len = self.shape()[axis_u] as u32;

        let (pipeline_layout, compute_pipeline) = {
            let key = CumSumCacheKey {
                dtype: self.dtype(),
                rank: rank as u32,
            };
            if let Some(res) = vulkan_immediate_executor.pipeline_cache.cumsum_op.get(&key) {
                res.clone()
            } else {
                let res = build_cumsum_pipeline(vulkan_immediate_executor, self.dtype(), rank)?;
                vulkan_immediate_executor
                    .pipeline_cache
                    .cumsum_op
                    .insert(key, res.clone());
                res
            }
        };

        let output_tensor = unsafe {
            VulkanTensor::new_uninitialized(
                self.shape().clone(),
                self.dtype(),
                vulkan_immediate_executor,
            )
        }?;

        let (descriptor_set, _) =
            vulkan_immediate_executor.get_descriptor_set_and_layout(&BTreeMap::from([
                (0, self.buffer.clone()),
                (1, output_tensor.buffer.clone()),
            ]))?;

        // Push constants
        let flags: u32 = (if exclusive { 1 } else { 0 }) | (if reverse { 2 } else { 0 });
        let shape = self.shape().as_slice().to_vec();
        let in_strides = self.stride.as_slice().to_vec();
        let out_strides = R::KnownDims::as_slice(&output_tensor.stride).to_vec();

        let op_metadata_vec = {
            let mut v = vec![
                (self.offset as u32 + self.suballocation.offset as u32)
                    / self.dtype().size().unwrap() as u32,
                (output_tensor.offset as u32 + output_tensor.suballocation.offset as u32)
                    / output_tensor.dtype().size().unwrap() as u32,
                num_elems,
                axis_u as u32,
                axis_len,
                flags,
            ];
            for dim in &shape {
                v.push(*dim as u32);
            }
            for dim in &in_strides {
                v.push(*dim as u32);
            }
            for dim in &out_strides {
                v.push(*dim as u32);
            }
            v
        };

        let mut builder = AutoCommandBufferBuilder::primary(
            vulkan_immediate_executor
                .context
                .command_buffer_allocator
                .clone(),
            vulkan_immediate_executor.context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        for (a, b) in op_metadata_vec.iter().enumerate() {
            builder
                .push_constants(pipeline_layout.clone(), (a * 4) as u32, *b)
                .unwrap();
        }

        builder
            .bind_pipeline_compute(compute_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                compute_pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .unwrap();

        unsafe { builder.dispatch([num_elems.div_ceil(64).max(1), 1, 1]) }.unwrap();

        let command_buffer = builder.build()?;
        let future = vulkano::sync::now(vulkan_immediate_executor.context.device.clone())
            .then_execute(
                vulkan_immediate_executor.context.queue.clone(),
                command_buffer,
            )
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();
        future.wait(None).unwrap();

        Ok(output_tensor)
    }
}
