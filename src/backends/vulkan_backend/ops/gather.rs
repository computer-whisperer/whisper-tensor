use crate::backends::vulkan_backend::spirv_helpers::{get_spirv_datatype, spirv_standard_cast};
use crate::backends::vulkan_backend::tensor::VulkanTensor;
use crate::backends::vulkan_backend::{VulkanError, VulkanImmediateExecutor};
use crate::dtype::DType;
use crate::tensor_rank::{DimContainer, DimProduct, Rank};
use rspirv::binary::Assemble;
use rspirv::dr::Operand;
use rspirv::dr::Operand::LiteralBit32;
use rspirv::spirv;
use rspirv::spirv::{
    BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel, SelectionControl, StorageClass,
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

#[allow(clippy::too_many_arguments)]
fn build_gather_pipeline(
    vulkan_immediate_executor: &mut VulkanImmediateExecutor,
    data_dtype: DType,
    indices_dtype: DType,
    data_rank: usize,
    indices_rank: usize,
    axis: usize,
) -> Result<(Arc<PipelineLayout>, Arc<ComputePipeline>), VulkanError> {
    assert!(axis < data_rank);
    let mut b = rspirv::dr::Builder::new();
    b.capability(Capability::Shader);
    b.capability(Capability::Float64);
    b.capability(Capability::Float16);
    b.capability(Capability::Int16);
    b.capability(Capability::Int8);
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();

    let output_rank = indices_rank + (data_rank - 1);
    assert!((axis + indices_rank) <= output_rank);

    let data_data_type = match data_dtype {
        DType::BF16 => b.type_int(16, 0),
        DType::BOOL => b.type_int(8, 0),
        _ => get_spirv_datatype(&mut b, data_dtype)?,
    };

    let indices_data_type = match indices_dtype {
        DType::BF16 => b.type_int(16, 0),
        DType::BOOL => b.type_int(8, 0),
        _ => get_spirv_datatype(&mut b, indices_dtype)?,
    };

    let input_0_data_type_array = b.type_runtime_array(data_data_type);
    b.decorate(
        input_0_data_type_array,
        Decoration::ArrayStride,
        [Operand::LiteralBit32(data_dtype.size().unwrap() as u32)],
    );
    let input_0_data_type_array_struct = b.type_struct([input_0_data_type_array]);
    b.decorate(input_0_data_type_array_struct, Decoration::Block, []);
    b.member_decorate(
        input_0_data_type_array_struct,
        0,
        Decoration::Offset,
        [Operand::LiteralBit32(0)],
    );

    let input_0_sb_ptr = b.type_pointer(
        None,
        StorageClass::StorageBuffer,
        input_0_data_type_array_struct,
    );
    let input_0_var = b.variable(input_0_sb_ptr, None, StorageClass::StorageBuffer, None);
    b.decorate(input_0_var, Decoration::DescriptorSet, [LiteralBit32(0)]);
    b.decorate(input_0_var, Decoration::Binding, [LiteralBit32(0)]);

    let input_1_data_type_array = b.type_runtime_array(indices_data_type);
    if input_1_data_type_array != input_0_data_type_array {
        b.decorate(
            input_1_data_type_array,
            Decoration::ArrayStride,
            [Operand::LiteralBit32(indices_dtype.size().unwrap() as u32)],
        );
    }
    let input_1_data_type_array_struct = b.type_struct([input_1_data_type_array]);
    if input_1_data_type_array_struct != input_0_data_type_array_struct {
        b.decorate(input_1_data_type_array_struct, Decoration::Block, []);
        b.member_decorate(
            input_1_data_type_array_struct,
            0,
            Decoration::Offset,
            [Operand::LiteralBit32(0)],
        );
    }

    let input_1_sb_ptr = b.type_pointer(
        None,
        StorageClass::StorageBuffer,
        input_1_data_type_array_struct,
    );
    let input_1_var = b.variable(input_1_sb_ptr, None, StorageClass::StorageBuffer, None);
    b.decorate(input_1_var, Decoration::DescriptorSet, [LiteralBit32(0)]);
    b.decorate(input_1_var, Decoration::Binding, [LiteralBit32(1)]);

    let output_data_type_array = b.type_runtime_array(data_data_type);
    if output_data_type_array != input_0_data_type_array
        && output_data_type_array != input_1_data_type_array
    {
        b.decorate(
            output_data_type_array,
            Decoration::ArrayStride,
            [Operand::LiteralBit32(data_dtype.size().unwrap() as u32)],
        );
    }
    let output_data_type_array_struct = b.type_struct([output_data_type_array]);
    if output_data_type_array_struct != input_0_data_type_array_struct
        && output_data_type_array_struct != input_1_data_type_array_struct
    {
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
    b.decorate(output_0_var, Decoration::Binding, [LiteralBit32(2)]);

    let u32_t = b.type_int(32, 0);
    // Data:
    // data_offset,
    // indices_offset,
    // output_offset,
    // data_dim_at_axis,
    // output_shape (RANK u32s)
    // data_stride (RANK u32s)
    // indices_stride (RANK u32s)
    // output_stride (RANK u32s)
    let metadata_value_count = data_rank + indices_rank + output_rank * 2 + 4;
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
        [gid, metadata_var, output_0_var, input_0_var, input_1_var],
    );
    b.execution_mode(main_fn, ExecutionMode::LocalSize, [64, 1, 1]);
    b.begin_block(None).unwrap();

    /* Constants */
    let c0 = b.constant_bit32(u32_t, 0);

    /* idx = gl_GlobalInvocationID.x */
    let gid = b.load(vec3u32_t, None, gid, None, []).unwrap();
    let idx = b.composite_extract(u32_t, None, gid, [0u32]).unwrap();

    /* size  = metadata.size */
    let push_constant_vals = {
        let mut vals = Vec::new();
        let u32_pc_ptr_t = b.type_pointer(None, StorageClass::PushConstant, u32_t);
        for i in 0..metadata_value_count {
            let c_i = b.constant_bit32(u32_t, i as u32);
            let size_ptr = b
                .access_chain(u32_pc_ptr_t, None, metadata_var, [c_i])
                .unwrap();
            vals.push(b.load(u32_t, None, size_ptr, None, []).unwrap());
        }
        vals
    };

    let mut offset = 0;
    let data_offset = push_constant_vals[offset];
    offset += 1;
    let indices_offset = push_constant_vals[offset];
    offset += 1;
    let output_offset = push_constant_vals[offset];
    offset += 1;
    let data_dim_at_axis = push_constant_vals[offset];
    offset += 1;
    let output_shape = &push_constant_vals[offset..offset + output_rank];
    offset += output_rank;
    let data_strides = &push_constant_vals[offset..offset + data_rank];
    offset += data_rank;
    let indices_strides = &push_constant_vals[offset..offset + indices_rank];
    offset += indices_rank;
    let output_strides = &push_constant_vals[offset..offset + output_rank];

    let num_elements = {
        let mut num_elements = None;
        for dim in output_shape {
            if let Some(value) = num_elements {
                num_elements = Some(b.i_mul(u32_t, None, value, *dim).unwrap());
            } else {
                num_elements = Some(*dim);
            }
        }
        num_elements.unwrap_or(b.constant_bit32(u32_t, 1))
    };
    /* cmp & branch */

    let bool_type = b.type_bool();
    let cmp = b.u_less_than(bool_type, None, idx, num_elements).unwrap();
    let merge_blk = b.id();
    let then_blk = b.id();

    b.selection_merge(merge_blk, SelectionControl::NONE)
        .unwrap();
    b.branch_conditional(cmp, then_blk, merge_blk, None)
        .unwrap();

    /* ---- THEN block ---- */
    b.begin_block(Some(then_blk)).unwrap();

    // Calculate index
    let output_index = {
        let mut v = Vec::with_capacity(output_rank);
        for i in 0..output_rank {
            // idx_i = (idx / stride[i]) % shape[i]
            let q = b.u_div(u32_t, None, idx, output_strides[i]).unwrap();
            let rem = b.u_mod(u32_t, None, q, output_shape[i]).unwrap();
            v.push(rem);
        }
        v
    };

    let index_value = {
        let mut v = indices_offset;
        for i in 0..indices_rank {
            let p = b
                .i_mul(u32_t, None, output_index[i + axis], indices_strides[i])
                .unwrap();
            v = b.i_add(u32_t, None, v, p).unwrap()
        }
        let index_offset = v;
        let i_ptr_type = b.type_pointer(None, StorageClass::StorageBuffer, indices_data_type);
        let i_ptr = b
            .access_chain(i_ptr_type, None, input_1_var, [c0, index_offset])
            .unwrap();
        let deref = b.load(indices_data_type, None, i_ptr, None, []).unwrap();

        let s32_t = b.type_int(32, 1);
        let idx_s = spirv_standard_cast(&mut b, deref, indices_dtype, DType::I32)?;
        let c = b.constant_bit32(s32_t, 0);
        let is_neg = b.s_less_than(bool_type, None, idx_s, c).unwrap();
        let idx_u = spirv_standard_cast(&mut b, idx_s, DType::I32, DType::U32)?;
        {
            // (idx < 0) ? (idx + axis_dim) : idx
            let idx_plus = {
                let idx_u_from_s = spirv_standard_cast(&mut b, idx_s, DType::I32, DType::U32)?;
                b.i_add(u32_t, None, idx_u_from_s, data_dim_at_axis)
                    .unwrap()
            };
            b.select(u32_t, None, is_neg, idx_plus, idx_u).unwrap()
        }
    };

    let data_value = {
        let mut v = data_offset;
        for i in 0..data_rank {
            let x = if i < axis {
                output_index[i]
            } else if i > axis {
                output_index[output_rank - (data_rank - i)]
            } else {
                index_value
            };
            let p = b.i_mul(u32_t, None, x, data_strides[i]).unwrap();
            v = b.i_add(u32_t, None, v, p).unwrap()
        }
        let data_offset = v;
        let d_ptr_type = b.type_pointer(None, StorageClass::StorageBuffer, data_data_type);
        let d_ptr = b
            .access_chain(d_ptr_type, None, input_0_var, [c0, data_offset])
            .unwrap();
        b.load(data_data_type, None, d_ptr, None, []).unwrap()
    };

    let output_offset = {
        let mut v = output_offset;
        for i in 0..output_rank {
            let mul = b
                .i_mul(u32_t, None, output_index[i], output_strides[i])
                .unwrap();
            v = b.i_add(u32_t, None, v, mul).unwrap();
        }
        v
    };

    /* store */
    let data_o_ptr_t = b.type_pointer(None, StorageClass::StorageBuffer, data_data_type);
    let out_ptr = b
        .access_chain(data_o_ptr_t, None, output_0_var, [c0, output_offset])
        .unwrap();
    b.store(out_ptr, data_value, None, []).unwrap();

    /* branch to merge */
    b.branch(merge_blk).unwrap();

    /* ---- MERGE block ---- */
    b.begin_block(Some(merge_blk)).unwrap();
    b.ret().unwrap(); // OpReturn
    b.end_function().unwrap(); // OpFunction

    let module = b.module();
    let code = module.assemble();

    //VulkanImmediateExecutor::debug_dump_spirv(&code);

    let shader = unsafe {
        ShaderModule::new(
            vulkan_immediate_executor.context.device.clone(),
            ShaderModuleCreateInfo::new(&code),
        )
    }?;

    let cs = shader.entry_point("main").unwrap();

    let stage = PipelineShaderStageCreateInfo::new(cs);

    let descriptor_set_layout =
        vulkan_immediate_executor.get_descriptor_set_layout(BTreeSet::from([0, 1, 2]))?;

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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct GatherCacheKey {
    data_dtype: DType,
    indices_dtype: DType,
    data_rank: usize,
    indices_rank: usize,
    axis: usize,
}

impl<R: Rank> VulkanTensor<R> {
    pub fn gather(
        data: &Self,
        indices: &Self,
        axis: i64,
        vulkan_immediate_executor: &mut VulkanImmediateExecutor,
    ) -> Result<Self, VulkanError> {
        let axis = if axis < 0 {
            data.rank() - axis as usize
        } else {
            axis as usize
        };

        let (pipeline_layout, compute_pipeline) = {
            let key = GatherCacheKey {
                data_dtype: data.dtype(),
                indices_dtype: indices.dtype(),
                data_rank: data.rank(),
                indices_rank: indices.rank(),
                axis,
            };
            let res = vulkan_immediate_executor.pipeline_cache.gather_op.get(&key);
            if let Some(res) = res {
                res.clone()
            } else {
                let res = build_gather_pipeline(
                    vulkan_immediate_executor,
                    data.dtype(),
                    indices.dtype(),
                    data.rank(),
                    indices.rank(),
                    axis,
                )?;
                vulkan_immediate_executor
                    .pipeline_cache
                    .gather_op
                    .insert(key, res.clone());
                res
            }
        };

        let output_shape = {
            let mut output_shape = data.shape().as_slice()[0..axis].to_vec();
            output_shape.extend(indices.shape().as_slice().iter());
            output_shape.extend(data.shape().as_slice()[axis + 1..].iter());
            R::KnownDims::try_from_slice(&output_shape).unwrap()
        };

        let output_tensor = unsafe {
            VulkanTensor::new_uninitialized(
                output_shape.clone(),
                data.dtype(),
                vulkan_immediate_executor,
            )
        }?;

        let (descriptor_set, _) =
            vulkan_immediate_executor.get_descriptor_set_and_layout(&BTreeMap::from([
                (0, data.buffer.clone()),
                (1, indices.buffer.clone()),
                (2, output_tensor.buffer.clone()),
            ]))?;

        let op_metadata_vec = {
            let mut v = vec![
                (data.offset as u32 + data.suballocation.offset as u32)
                    / data.dtype().size().unwrap() as u32,
                (indices.offset as u32 + indices.suballocation.offset as u32)
                    / indices.dtype().size().unwrap() as u32,
                (output_tensor.offset as u32 + output_tensor.suballocation.offset as u32)
                    / output_tensor.dtype().size().unwrap() as u32,
                data.shape().as_slice()[axis] as u32,
            ];
            for dim in output_shape.as_slice() {
                v.push(*dim as u32);
            }
            for dim in data.stride.as_slice() {
                v.push(*dim as u32);
            }
            for dim in indices.stride.as_slice() {
                v.push(*dim as u32);
            }
            for dim in R::KnownDims::as_slice(&output_tensor.stride) {
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

        // Note that we clone the pipeline and the set. Since they are both wrapped in an `Arc`,
        // this only clones the `Arc` and not the whole pipeline or set (which aren't cloneable
        // anyway). In this example we would avoid cloning them since this is the last time we use
        // them, but in real code you would probably need to clone them.
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

        let num_elements = output_shape.dim_product();

        // The command buffer only does one thing: execute the compute pipeline. This is called a
        // *dispatch* operation.
        unsafe { builder.dispatch([num_elements.div_ceil(64) as u32, 1, 1]) }.unwrap();

        // Finish building the command buffer by calling `build`.
        let command_buffer = builder.build()?;

        // Let's execute this command buffer now.
        let future = vulkano::sync::now(vulkan_immediate_executor.context.device.clone())
            .then_execute(
                vulkan_immediate_executor.context.queue.clone(),
                command_buffer,
            )
            .unwrap()
            // This line instructs the GPU to signal a *fence* once the command buffer has finished
            // execution. A fence is a Vulkan object that allows the CPU to know when the GPU has
            // reached a certain point. We need to signal a fence here because below we want to block
            // the CPU until the GPU has reached that point in the execution.
            .then_signal_fence_and_flush()
            .unwrap();

        // Blocks execution until the GPU has finished the operation. This method only exists on the
        // future that corresponds to a signalled fence. In other words, this method wouldn't be
        // available if we didn't call `.then_signal_fence_and_flush()` earlier. The `None` parameter
        // is an optional timeout.
        //
        // Note however that dropping the `future` variable (with `drop(future)` for example) would
        // block execution as well, and this would be the case even if we didn't call
        // `.then_signal_fence_and_flush()`. Therefore the actual point of calling
        // `.then_signal_fence_and_flush()` and `.wait()` is to make things more explicit. In the
        // future, if the Rust language gets linear types vulkano may get modified so that only
        // fence-signalled futures can get destroyed like this.
        future.wait(None).unwrap();

        Ok(output_tensor)
    }
}
