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
    BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel, GLOp, SelectionControl,
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

fn build_binary_pipeline(
    vulkan_immediate_executor: &mut VulkanImmediateExecutor,
    input_0_dtype: DType,
    input_1_dtype: DType,
    output_dtype: DType,
    rank: usize,
    op: fn(
        &mut rspirv::dr::Builder,
        rspirv::spirv::Word,
        rspirv::spirv::Word,
        DType,
        DType,
        DType,
    ) -> Result<rspirv::spirv::Word, VulkanError>,
) -> Result<(Arc<PipelineLayout>, Arc<ComputePipeline>), VulkanError> {
    let mut b = rspirv::dr::Builder::new();
    b.capability(Capability::Shader);
    b.capability(Capability::Float64);
    b.capability(Capability::Float16);
    b.capability(Capability::Int16);
    b.capability(Capability::Int8);
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
    let void = b.type_void();

    let input_0_data_type = match input_0_dtype {
        DType::BF16 => b.type_int(16, 0),
        DType::BOOL => b.type_int(8, 0),
        _ => get_spirv_datatype(&mut b, input_0_dtype)?,
    };

    let input_1_data_type = match input_1_dtype {
        DType::BF16 => b.type_int(16, 0),
        DType::BOOL => b.type_int(8, 0),
        _ => get_spirv_datatype(&mut b, input_1_dtype)?,
    };

    let output_data_type = match output_dtype {
        DType::BF16 => b.type_int(16, 0),
        DType::BOOL => b.type_int(8, 0),
        _ => get_spirv_datatype(&mut b, output_dtype)?,
    };

    let input_0_data_type_array = b.type_runtime_array(input_0_data_type);
    b.decorate(
        input_0_data_type_array,
        Decoration::ArrayStride,
        [Operand::LiteralBit32(input_0_dtype.size().unwrap() as u32)],
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

    let input_1_data_type_array = b.type_runtime_array(input_1_data_type);
    if input_1_data_type_array != input_0_data_type_array {
        b.decorate(
            input_1_data_type_array,
            Decoration::ArrayStride,
            [Operand::LiteralBit32(input_1_dtype.size().unwrap() as u32)],
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

    let output_data_type_array = b.type_runtime_array(output_data_type);
    if output_data_type_array != input_0_data_type_array
        && output_data_type_array != input_1_data_type_array
    {
        b.decorate(
            output_data_type_array,
            Decoration::ArrayStride,
            [Operand::LiteralBit32(output_dtype.size().unwrap() as u32)],
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
    // input_0_offset,
    // input_1_offset,
    // output_offset,
    // num_elements,
    // input_output_shape (RANK u32s)
    // input_0_stride (RANK u32s)
    // input_1_stride (RANK u32s)
    // output_stride (RANK u32s)
    let metadata_value_count = rank * 4 + 4;
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
    let gid = b.load(vec3u32_t, None, gid.clone(), None, []).unwrap();
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

    let input_0_offset = push_constant_vals[0];
    let input_1_offset = push_constant_vals[1];
    let output_offset = push_constant_vals[2];
    let num_elements = push_constant_vals[3];
    let io_shape = &push_constant_vals[4..4 + rank];
    let input_0_strides = &push_constant_vals[4 + rank..4 + rank * 2];
    let input_1_strides = &push_constant_vals[4 + rank * 2..4 + rank * 3];
    let output_strides = &push_constant_vals[4 + rank * 3..4 + rank * 4];

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
    let mut remaining_idx = idx;
    let index = {
        let mut v = vec![];
        for i in 0..rank {
            let shape_val = io_shape[i];
            let rem = b.u_mod(u32_t, None, remaining_idx, shape_val).unwrap();
            let div = b.u_div(u32_t, None, remaining_idx, shape_val).unwrap();
            remaining_idx = div;
            v.push(rem);
        }
        v
    };

    let input_0_index = {
        let mut v = input_0_offset;
        for i in 0..rank {
            let mul = b.i_mul(u32_t, None, index[i], input_0_strides[i]).unwrap();
            v = b.i_add(u32_t, None, v, mul).unwrap();
        }
        v
    };

    let input_1_index = {
        let mut v = input_1_offset;
        for i in 0..rank {
            let mul = b.i_mul(u32_t, None, index[i], input_1_strides[i]).unwrap();
            v = b.i_add(u32_t, None, v, mul).unwrap();
        }
        v
    };

    let output_index = {
        let mut v = output_offset;
        for i in 0..rank {
            let mul = b.i_mul(u32_t, None, index[i], output_strides[i]).unwrap();
            v = b.i_add(u32_t, None, v, mul).unwrap();
        }
        v
    };

    /* value = -input[in_index] */
    let data_0_i_ptr_t = b.type_pointer(None, StorageClass::StorageBuffer, input_0_data_type);
    let data_1_i_ptr_t = b.type_pointer(None, StorageClass::StorageBuffer, input_1_data_type);
    let in0_ptr = b
        .access_chain(data_0_i_ptr_t, None, input_0_var, [c0, input_0_index])
        .unwrap();
    let in0_val = b.load(input_0_data_type, None, in0_ptr, None, []).unwrap();
    let in1_ptr = b
        .access_chain(data_1_i_ptr_t, None, input_1_var, [c0, input_1_index])
        .unwrap();
    let in1_val = b.load(input_1_data_type, None, in1_ptr, None, []).unwrap();

    let (in0_val, in_0_type) = match input_0_dtype {
        DType::BF16 => (cast_bf16_to_f32(&mut b, in0_val), DType::F32),
        DType::BOOL => {
            let bool_type = b.type_bool();
            let u8_type = b.type_int(8, 0);
            let c0 = b.constant_bit32(u8_type, 0);
            (
                b.logical_not_equal(bool_type, None, c0, in0_val).unwrap(),
                DType::BOOL,
            )
        }
        _ => (in0_val, input_0_dtype),
    };

    let (in1_val, in_1_type) = match input_1_dtype {
        DType::BF16 => (cast_bf16_to_f32(&mut b, in1_val), DType::F32),
        DType::BOOL => {
            let bool_type = b.type_bool();
            let u8_type = b.type_int(8, 0);
            let c0 = b.constant_bit32(u8_type, 0);
            (
                b.logical_not_equal(bool_type, None, c0, in1_val).unwrap(),
                DType::BOOL,
            )
        }
        _ => (in1_val, input_1_dtype),
    };

    let out_type = match output_dtype {
        DType::BF16 => DType::F32,
        _ => output_dtype,
    };

    let out_val = op(&mut b, in0_val, in1_val, in_0_type, in_1_type, out_type)?;

    let out_val = match output_dtype {
        DType::BF16 => cast_f32_to_bf16(&mut b, out_val),
        DType::BOOL => {
            let u8_type = b.type_int(8, 0);
            let c0 = b.constant_bit32(u8_type, 0);
            let c1 = b.constant_bit32(u8_type, 1);
            b.select(u8_type, None, out_val, c1, c0).unwrap()
        }
        _ => out_val,
    };

    /* store */
    let data_o_ptr_t = b.type_pointer(None, StorageClass::StorageBuffer, output_data_type);
    let out_ptr = b
        .access_chain(data_o_ptr_t, None, output_0_var, [c0, output_index])
        .unwrap();
    b.store(out_ptr, out_val, None, []).unwrap();

    /* branch to merge */
    b.branch(merge_blk).unwrap();

    /* ---- MERGE block ---- */
    b.begin_block(Some(merge_blk)).unwrap();
    b.ret().unwrap(); // OpReturn
    b.end_function().unwrap(); // OpFunction

    let module = b.module();
    let code = module.assemble();

    VulkanImmediateExecutor::debug_dump_spirv(&code);

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
                size: (rank * 16 + 16) as u32,
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

impl<R: Rank> VulkanTensor<R> {
    fn binary(
        a: &Self,
        b: &Self,
        output_dtype: DType,
        vulkan_immediate_executor: &mut VulkanImmediateExecutor,
        cache_id: u32,
        op: fn(
            &mut rspirv::dr::Builder,
            rspirv::spirv::Word,
            rspirv::spirv::Word,
            DType,
            DType,
            DType,
        ) -> Result<rspirv::spirv::Word, VulkanError>,
    ) -> Result<VulkanTensor<R>, VulkanError> {
        // Start by forcing them to even rank
        let mut a = a.clone();
        let mut b = b.clone();
        while a.rank() < b.rank() {
            a = a.unsqueeze(0)?;
        }
        while b.rank() < a.rank() {
            b = b.unsqueeze(0)?;
        }

        let output_shape = a
            .shape()
            .as_slice()
            .iter()
            .zip(b.shape().as_slice().iter())
            .map(|(a, b)| *a.max(b))
            .collect::<Vec<_>>();
        let output_shape = R::KnownDims::try_from_slice(output_shape.as_slice()).unwrap();

        let a = a.broadcast::<R>(&output_shape).unwrap();
        let b = b.broadcast::<R>(&output_shape).unwrap();

        let rank = output_shape.len();

        let (pipeline_layout, compute_pipeline) = {
            let key = (cache_id, a.dtype(), b.dtype(), output_dtype, rank as u32);
            let res = vulkan_immediate_executor.pipeline_cache.binary_op.get(&key);
            if let Some(res) = res {
                res.clone()
            } else {
                let res = build_binary_pipeline(
                    vulkan_immediate_executor,
                    a.dtype(),
                    b.dtype(),
                    output_dtype,
                    rank,
                    op,
                )?;
                vulkan_immediate_executor
                    .pipeline_cache
                    .binary_op
                    .insert(key, res.clone());
                res
            }
        };

        let output_tensor = unsafe {
            VulkanTensor::new_uninitialized(
                output_shape.clone(),
                output_dtype,
                vulkan_immediate_executor,
            )
        }?;

        let (descriptor_set, _) =
            vulkan_immediate_executor.get_descriptor_set_and_layout(&BTreeMap::from([
                (0, a.buffer.clone()),
                (1, b.buffer.clone()),
                (2, output_tensor.buffer.clone()),
            ]))?;

        let op_metadata_vec = {
            let mut v = vec![];
            v.push(
                (a.offset as u32 + a.suballocation.offset as u32)
                    / a.dtype().size().unwrap() as u32,
            );
            v.push(
                (b.offset as u32 + b.suballocation.offset as u32)
                    / b.dtype().size().unwrap() as u32,
            );
            v.push(
                (output_tensor.offset as u32 + output_tensor.suballocation.offset as u32)
                    / output_dtype.size().unwrap() as u32,
            );
            v.push(output_shape.dim_product() as u32);
            for dim in output_shape.as_slice() {
                v.push(*dim as u32);
            }
            for dim in a.stride.as_slice() {
                v.push(*dim as u32);
            }
            for dim in b.stride.as_slice() {
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

        // The command buffer only does one thing: execute the compute pipeline. This is called a
        // *dispatch* operation.
        unsafe { builder.dispatch([((output_shape.dim_product() / 64) + 1) as u32, 1, 1]) }
            .unwrap();

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

    pub fn add(
        a: &Self,
        b: &Self,
        vulkan_immediate_executor: &mut VulkanImmediateExecutor,
    ) -> Result<VulkanTensor<R>, VulkanError> {
        Self::binary(
            a,
            b,
            a.dtype(),
            vulkan_immediate_executor,
            0,
            |builder, a, b, input_0_dtype, _input_1_dtype, output_dtype| {
                let data_type = get_spirv_datatype(builder, output_dtype)?;
                match input_0_dtype {
                    DType::F16 | DType::F32 | DType::F64 => {
                        Ok(builder.f_add(data_type, None, a, b).unwrap())
                    }
                    DType::I64
                    | DType::I32
                    | DType::I16
                    | DType::I8
                    | DType::U64
                    | DType::U32
                    | DType::U16
                    | DType::U8 => Ok(builder.i_add(data_type, None, a, b).unwrap()),
                    _ => Err(VulkanError::UnsupportedByBackendError),
                }
            },
        )
    }

    pub fn sub(
        a: &Self,
        b: &Self,
        vulkan_immediate_executor: &mut VulkanImmediateExecutor,
    ) -> Result<VulkanTensor<R>, VulkanError> {
        Self::binary(
            a,
            b,
            a.dtype(),
            vulkan_immediate_executor,
            1,
            |builder, a, b, input_0_dtype, _input_1_dtype, output_dtype| {
                let data_type = get_spirv_datatype(builder, output_dtype)?;
                match input_0_dtype {
                    DType::F16 | DType::F32 | DType::F64 => {
                        Ok(builder.f_sub(data_type, None, a, b).unwrap())
                    }
                    DType::I64
                    | DType::I32
                    | DType::I16
                    | DType::I8
                    | DType::U64
                    | DType::U32
                    | DType::U16
                    | DType::U8 => Ok(builder.i_sub(data_type, None, a, b).unwrap()),
                    _ => Err(VulkanError::UnsupportedByBackendError),
                }
            },
        )
    }

    pub fn mul(
        a: &Self,
        b: &Self,
        vulkan_immediate_executor: &mut VulkanImmediateExecutor,
    ) -> Result<VulkanTensor<R>, VulkanError> {
        Self::binary(
            a,
            b,
            a.dtype(),
            vulkan_immediate_executor,
            2,
            |builder, a, b, input_0_dtype, _input_1_dtype, output_dtype| {
                let data_type = get_spirv_datatype(builder, output_dtype)?;
                match input_0_dtype {
                    DType::F16 | DType::F32 | DType::F64 => {
                        Ok(builder.f_mul(data_type, None, a, b).unwrap())
                    }
                    DType::I64
                    | DType::I32
                    | DType::I16
                    | DType::I8
                    | DType::U64
                    | DType::U32
                    | DType::U16
                    | DType::U8 => Ok(builder.i_mul(data_type, None, a, b).unwrap()),
                    _ => Err(VulkanError::UnsupportedByBackendError),
                }
            },
        )
    }

    pub fn div(
        a: &Self,
        b: &Self,
        vulkan_immediate_executor: &mut VulkanImmediateExecutor,
    ) -> Result<VulkanTensor<R>, VulkanError> {
        Self::binary(
            a,
            b,
            a.dtype(),
            vulkan_immediate_executor,
            3,
            |builder, a, b, input_0_dtype, _input_1_dtype, output_dtype| {
                let data_type = get_spirv_datatype(builder, output_dtype)?;
                match input_0_dtype {
                    DType::F16 | DType::F32 | DType::F64 => {
                        Ok(builder.f_div(data_type, None, a, b).unwrap())
                    }
                    DType::I64 | DType::I32 | DType::I16 | DType::I8 => {
                        Ok(builder.s_div(data_type, None, a, b).unwrap())
                    }
                    DType::U64 | DType::U32 | DType::U16 | DType::U8 => {
                        Ok(builder.u_div(data_type, None, a, b).unwrap())
                    }
                    _ => Err(VulkanError::UnsupportedByBackendError),
                }
            },
        )
    }

    pub fn imod(
        a: &Self,
        b: &Self,
        vulkan_immediate_executor: &mut VulkanImmediateExecutor,
    ) -> Result<VulkanTensor<R>, VulkanError> {
        Self::binary(
            a,
            b,
            a.dtype(),
            vulkan_immediate_executor,
            4,
            |builder, a, b, input_0_dtype, _input_1_dtype, output_dtype| {
                let data_type = get_spirv_datatype(builder, output_dtype)?;
                match input_0_dtype {
                    DType::I64 | DType::I32 | DType::I16 | DType::I8 => {
                        Ok(builder.s_mod(data_type, None, a, b).unwrap())
                    }
                    DType::U64 | DType::U32 | DType::U16 | DType::U8 => {
                        Ok(builder.u_mod(data_type, None, a, b).unwrap())
                    }
                    _ => Err(VulkanError::UnsupportedByBackendError),
                }
            },
        )
    }

    pub fn fmod(
        a: &Self,
        b: &Self,
        vulkan_immediate_executor: &mut VulkanImmediateExecutor,
    ) -> Result<VulkanTensor<R>, VulkanError> {
        Self::binary(
            a,
            b,
            a.dtype(),
            vulkan_immediate_executor,
            5,
            |builder, a, b, input_0_dtype, _input_1_dtype, output_dtype| {
                let data_type = get_spirv_datatype(builder, output_dtype)?;
                match input_0_dtype {
                    DType::F16 | DType::F32 | DType::F64 => {
                        Ok(builder.f_rem(data_type, None, a, b).unwrap())
                    }
                    DType::I64 | DType::I32 | DType::I16 | DType::I8 => {
                        Ok(builder.s_rem(data_type, None, a, b).unwrap())
                    }
                    _ => Err(VulkanError::UnsupportedByBackendError),
                }
            },
        )
    }

    pub fn and(
        a: &Self,
        b: &Self,
        vulkan_immediate_executor: &mut VulkanImmediateExecutor,
    ) -> Result<VulkanTensor<R>, VulkanError> {
        Self::binary(
            a,
            b,
            a.dtype(),
            vulkan_immediate_executor,
            6,
            |builder, a, b, input_0_dtype, _input_1_dtype, output_dtype| {
                let data_type = get_spirv_datatype(builder, output_dtype)?;
                match input_0_dtype {
                    DType::BOOL => Ok(builder.logical_and(data_type, None, a, b).unwrap()),
                    _ => Err(VulkanError::UnsupportedByBackendError),
                }
            },
        )
    }

    pub fn or(
        a: &Self,
        b: &Self,
        vulkan_immediate_executor: &mut VulkanImmediateExecutor,
    ) -> Result<VulkanTensor<R>, VulkanError> {
        Self::binary(
            a,
            b,
            a.dtype(),
            vulkan_immediate_executor,
            7,
            |builder, a, b, input_0_dtype, _input_1_dtype, output_dtype| {
                let data_type = get_spirv_datatype(builder, output_dtype)?;
                match input_0_dtype {
                    DType::BOOL => Ok(builder.logical_or(data_type, None, a, b).unwrap()),
                    _ => Err(VulkanError::UnsupportedByBackendError),
                }
            },
        )
    }

    pub fn xor(
        a: &Self,
        b: &Self,
        vulkan_immediate_executor: &mut VulkanImmediateExecutor,
    ) -> Result<VulkanTensor<R>, VulkanError> {
        Self::binary(
            a,
            b,
            a.dtype(),
            vulkan_immediate_executor,
            8,
            |builder, a, b, input_0_dtype, _input_1_dtype, output_dtype| {
                let data_type = get_spirv_datatype(builder, output_dtype)?;
                match input_0_dtype {
                    DType::BOOL => Ok(builder.logical_not_equal(data_type, None, a, b).unwrap()),
                    _ => Err(VulkanError::UnsupportedByBackendError),
                }
            },
        )
    }

    pub fn bitwise_and(
        a: &Self,
        b: &Self,
        vulkan_immediate_executor: &mut VulkanImmediateExecutor,
    ) -> Result<VulkanTensor<R>, VulkanError> {
        Self::binary(
            a,
            b,
            a.dtype(),
            vulkan_immediate_executor,
            9,
            |builder, a, b, input_0_dtype, _input_1_dtype, output_dtype| {
                let data_type = get_spirv_datatype(builder, output_dtype)?;
                match input_0_dtype {
                    DType::I64
                    | DType::I32
                    | DType::I16
                    | DType::I8
                    | DType::U64
                    | DType::U32
                    | DType::U16
                    | DType::U8 => Ok(builder.bitwise_and(data_type, None, a, b).unwrap()),
                    _ => Err(VulkanError::UnsupportedByBackendError),
                }
            },
        )
    }

    pub fn bitwise_or(
        a: &Self,
        b: &Self,
        vulkan_immediate_executor: &mut VulkanImmediateExecutor,
    ) -> Result<VulkanTensor<R>, VulkanError> {
        Self::binary(
            a,
            b,
            a.dtype(),
            vulkan_immediate_executor,
            10,
            |builder, a, b, input_0_dtype, _input_1_dtype, output_dtype| {
                let data_type = get_spirv_datatype(builder, output_dtype)?;
                match input_0_dtype {
                    DType::I64
                    | DType::I32
                    | DType::I16
                    | DType::I8
                    | DType::U64
                    | DType::U32
                    | DType::U16
                    | DType::U8 => Ok(builder.bitwise_or(data_type, None, a, b).unwrap()),
                    _ => Err(VulkanError::UnsupportedByBackendError),
                }
            },
        )
    }

    pub fn bitwise_xor(
        a: &Self,
        b: &Self,
        vulkan_immediate_executor: &mut VulkanImmediateExecutor,
    ) -> Result<VulkanTensor<R>, VulkanError> {
        Self::binary(
            a,
            b,
            a.dtype(),
            vulkan_immediate_executor,
            11,
            |builder, a, b, input_0_dtype, _input_1_dtype, output_dtype| {
                let data_type = get_spirv_datatype(builder, output_dtype)?;
                match input_0_dtype {
                    DType::I64
                    | DType::I32
                    | DType::I16
                    | DType::I8
                    | DType::U64
                    | DType::U32
                    | DType::U16
                    | DType::U8 => Ok(builder.bitwise_xor(data_type, None, a, b).unwrap()),
                    _ => Err(VulkanError::UnsupportedByBackendError),
                }
            },
        )
    }

    pub fn max(
        a: &Self,
        b: &Self,
        vulkan_immediate_executor: &mut VulkanImmediateExecutor,
    ) -> Result<VulkanTensor<R>, VulkanError> {
        Self::binary(
            a,
            b,
            a.dtype(),
            vulkan_immediate_executor,
            12,
            |builder, a, b, input_0_dtype, _input_1_dtype, output_dtype| {
                let data_type = get_spirv_datatype(builder, output_dtype)?;
                let glsl = builder.ext_inst_import("GLSL.std.450");
                match input_0_dtype {
                    DType::F16 | DType::F32 | DType::F64 => Ok(builder
                        .ext_inst(
                            data_type,
                            None,
                            glsl,
                            GLOp::FMax as u32,
                            [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)],
                        )
                        .unwrap()),
                    DType::I64 | DType::I32 | DType::I16 | DType::I8 => Ok(builder
                        .ext_inst(
                            data_type,
                            None,
                            glsl,
                            GLOp::SMax as u32,
                            [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)],
                        )
                        .unwrap()),
                    DType::U64 | DType::U32 | DType::U16 | DType::U8 => Ok(builder
                        .ext_inst(
                            data_type,
                            None,
                            glsl,
                            GLOp::UMax as u32,
                            [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)],
                        )
                        .unwrap()),
                    _ => Err(VulkanError::UnsupportedByBackendError),
                }
            },
        )
    }

    pub fn min(
        a: &Self,
        b: &Self,
        vulkan_immediate_executor: &mut VulkanImmediateExecutor,
    ) -> Result<VulkanTensor<R>, VulkanError> {
        Self::binary(
            a,
            b,
            a.dtype(),
            vulkan_immediate_executor,
            13,
            |builder, a, b, input_0_dtype, _input_1_dtype, output_dtype| {
                let data_type = get_spirv_datatype(builder, output_dtype)?;
                let glsl = builder.ext_inst_import("GLSL.std.450");
                match input_0_dtype {
                    DType::F16 | DType::F32 | DType::F64 => Ok(builder
                        .ext_inst(
                            data_type,
                            None,
                            glsl,
                            GLOp::FMin as u32,
                            [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)],
                        )
                        .unwrap()),
                    DType::I64 | DType::I32 | DType::I16 | DType::I8 => Ok(builder
                        .ext_inst(
                            data_type,
                            None,
                            glsl,
                            GLOp::SMin as u32,
                            [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)],
                        )
                        .unwrap()),
                    DType::U64 | DType::U32 | DType::U16 | DType::U8 => Ok(builder
                        .ext_inst(
                            data_type,
                            None,
                            glsl,
                            GLOp::UMin as u32,
                            [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)],
                        )
                        .unwrap()),
                    _ => Err(VulkanError::UnsupportedByBackendError),
                }
            },
        )
    }

    pub fn equal(
        a: &Self,
        b: &Self,
        vulkan_immediate_executor: &mut VulkanImmediateExecutor,
    ) -> Result<VulkanTensor<R>, VulkanError> {
        Self::binary(
            a,
            b,
            DType::BOOL,
            vulkan_immediate_executor,
            14,
            |builder, a, b, input_0_dtype, _input_1_dtype, output_dtype| {
                let data_type = get_spirv_datatype(builder, output_dtype)?;
                match input_0_dtype {
                    DType::F16 | DType::F32 | DType::F64 => {
                        Ok(builder.f_unord_equal(data_type, None, a, b).unwrap())
                    }
                    DType::I64
                    | DType::I32
                    | DType::I16
                    | DType::I8
                    | DType::U64
                    | DType::U32
                    | DType::U16
                    | DType::U8 => Ok(builder.i_equal(data_type, None, a, b).unwrap()),
                    _ => Err(VulkanError::UnsupportedByBackendError),
                }
            },
        )
    }

    pub fn greater(
        a: &Self,
        b: &Self,
        vulkan_immediate_executor: &mut VulkanImmediateExecutor,
    ) -> Result<VulkanTensor<R>, VulkanError> {
        Self::binary(
            a,
            b,
            DType::BOOL,
            vulkan_immediate_executor,
            15,
            |builder, a, b, input_0_dtype, _input_1_dtype, output_dtype| {
                let data_type = get_spirv_datatype(builder, output_dtype)?;
                match input_0_dtype {
                    DType::F16 | DType::F32 | DType::F64 => {
                        Ok(builder.f_unord_greater_than(data_type, None, a, b).unwrap())
                    }
                    DType::U64 | DType::U32 | DType::U16 | DType::U8 => {
                        Ok(builder.u_greater_than(data_type, None, a, b).unwrap())
                    }
                    DType::I64 | DType::I32 | DType::I16 | DType::I8 => {
                        Ok(builder.s_greater_than(data_type, None, a, b).unwrap())
                    }
                    _ => Err(VulkanError::UnsupportedByBackendError),
                }
            },
        )
    }

    pub fn greater_or_equal(
        a: &Self,
        b: &Self,
        vulkan_immediate_executor: &mut VulkanImmediateExecutor,
    ) -> Result<VulkanTensor<R>, VulkanError> {
        Self::binary(
            a,
            b,
            DType::BOOL,
            vulkan_immediate_executor,
            16,
            |builder, a, b, input_0_dtype, _input_1_dtype, output_dtype| {
                let data_type = get_spirv_datatype(builder, output_dtype)?;
                match input_0_dtype {
                    DType::F16 | DType::F32 | DType::F64 => Ok(builder
                        .f_unord_greater_than_equal(data_type, None, a, b)
                        .unwrap()),
                    DType::U64 | DType::U32 | DType::U16 | DType::U8 => {
                        Ok(builder.u_greater_than_equal(data_type, None, a, b).unwrap())
                    }
                    DType::I64 | DType::I32 | DType::I16 | DType::I8 => {
                        Ok(builder.s_greater_than_equal(data_type, None, a, b).unwrap())
                    }
                    _ => Err(VulkanError::UnsupportedByBackendError),
                }
            },
        )
    }

    pub fn less(
        a: &Self,
        b: &Self,
        vulkan_immediate_executor: &mut VulkanImmediateExecutor,
    ) -> Result<VulkanTensor<R>, VulkanError> {
        Self::binary(
            a,
            b,
            DType::BOOL,
            vulkan_immediate_executor,
            17,
            |builder, a, b, input_0_dtype, _input_1_dtype, output_dtype| {
                let data_type = get_spirv_datatype(builder, output_dtype)?;
                match input_0_dtype {
                    DType::F16 | DType::F32 | DType::F64 => {
                        Ok(builder.f_unord_less_than(data_type, None, a, b).unwrap())
                    }
                    DType::U64 | DType::U32 | DType::U16 | DType::U8 => {
                        Ok(builder.u_less_than(data_type, None, a, b).unwrap())
                    }
                    DType::I64 | DType::I32 | DType::I16 | DType::I8 => {
                        Ok(builder.s_less_than(data_type, None, a, b).unwrap())
                    }
                    _ => Err(VulkanError::UnsupportedByBackendError),
                }
            },
        )
    }

    pub fn less_or_equal(
        a: &Self,
        b: &Self,
        vulkan_immediate_executor: &mut VulkanImmediateExecutor,
    ) -> Result<VulkanTensor<R>, VulkanError> {
        Self::binary(
            a,
            b,
            DType::BOOL,
            vulkan_immediate_executor,
            18,
            |builder, a, b, input_0_dtype, _input_1_dtype, output_dtype| {
                let data_type = get_spirv_datatype(builder, output_dtype)?;
                match input_0_dtype {
                    DType::F16 | DType::F32 | DType::F64 => Ok(builder
                        .f_unord_less_than_equal(data_type, None, a, b)
                        .unwrap()),
                    DType::U64 | DType::U32 | DType::U16 | DType::U8 => {
                        Ok(builder.u_less_than_equal(data_type, None, a, b).unwrap())
                    }
                    DType::I64 | DType::I32 | DType::I16 | DType::I8 => {
                        Ok(builder.s_less_than_equal(data_type, None, a, b).unwrap())
                    }
                    _ => Err(VulkanError::UnsupportedByBackendError),
                }
            },
        )
    }

    pub fn pow(
        &self,
        b: &Self,
        vulkan_immediate_executor: &mut VulkanImmediateExecutor,
    ) -> Result<VulkanTensor<R>, VulkanError> {
        Self::binary(
            self,
            b,
            self.dtype(),
            vulkan_immediate_executor,
            19,
            |builder, a, b, input_0_dtype, input_1_dtype, output_dtype| {
                let output_data_type = get_spirv_datatype(builder, output_dtype)?;
                let glsl = builder.ext_inst_import("GLSL.std.450");
                match input_0_dtype {
                    DType::BF16 | DType::F16 | DType::F32 | DType::F64 => match input_1_dtype {
                        DType::BF16 | DType::F16 | DType::F32 | DType::F64 => Ok(builder
                            .ext_inst(
                                output_data_type,
                                None,
                                glsl,
                                GLOp::Pow as u32,
                                [rspirv::dr::Operand::IdRef(a), rspirv::dr::Operand::IdRef(b)],
                            )
                            .unwrap()),
                        DType::U64 | DType::U32 | DType::U16 | DType::U8 => {
                            let exp = builder.convert_u_to_f(output_data_type, None, b).unwrap();
                            Ok(builder
                                .ext_inst(
                                    output_data_type,
                                    None,
                                    glsl,
                                    GLOp::Pow as u32,
                                    [
                                        rspirv::dr::Operand::IdRef(a),
                                        rspirv::dr::Operand::IdRef(exp),
                                    ],
                                )
                                .unwrap())
                        }
                        DType::I64 | DType::I32 | DType::I16 | DType::I8 => {
                            let exp = builder.convert_s_to_f(output_data_type, None, b).unwrap();
                            Ok(builder
                                .ext_inst(
                                    output_data_type,
                                    None,
                                    glsl,
                                    GLOp::Pow as u32,
                                    [
                                        rspirv::dr::Operand::IdRef(a),
                                        rspirv::dr::Operand::IdRef(exp),
                                    ],
                                )
                                .unwrap())
                        }
                        _ => Err(VulkanError::UnsupportedByBackendError),
                    },
                    DType::I64 | DType::I32 | DType::I16 | DType::I8 => match input_1_dtype {
                        DType::BF16 | DType::F16 | DType::F32 | DType::F64 => {
                            let f32_t = builder.type_float(32);
                            let base = builder.convert_s_to_f(f32_t, None, a).unwrap();
                            let exp = builder.f_convert(f32_t, None, b).unwrap();
                            let result = builder
                                .ext_inst(
                                    f32_t,
                                    None,
                                    glsl,
                                    GLOp::Pow as u32,
                                    [
                                        rspirv::dr::Operand::IdRef(base),
                                        rspirv::dr::Operand::IdRef(exp),
                                    ],
                                )
                                .unwrap();
                            Ok(builder
                                .convert_f_to_s(output_data_type, None, result)
                                .unwrap())
                        }
                        DType::U64 | DType::U32 | DType::U16 | DType::U8 => {
                            let f32_t = builder.type_float(32);
                            let base = builder.convert_s_to_f(f32_t, None, a).unwrap();
                            let exp = builder.convert_u_to_f(f32_t, None, b).unwrap();
                            let result = builder
                                .ext_inst(
                                    f32_t,
                                    None,
                                    glsl,
                                    GLOp::Pow as u32,
                                    [
                                        rspirv::dr::Operand::IdRef(base),
                                        rspirv::dr::Operand::IdRef(exp),
                                    ],
                                )
                                .unwrap();
                            Ok(builder
                                .convert_f_to_s(output_data_type, None, result)
                                .unwrap())
                        }
                        DType::I64 | DType::I32 | DType::I16 | DType::I8 => {
                            let f32_t = builder.type_float(32);
                            let base = builder.convert_s_to_f(f32_t, None, a).unwrap();
                            let exp = builder.convert_s_to_f(f32_t, None, b).unwrap();
                            let result = builder
                                .ext_inst(
                                    f32_t,
                                    None,
                                    glsl,
                                    GLOp::Pow as u32,
                                    [
                                        rspirv::dr::Operand::IdRef(base),
                                        rspirv::dr::Operand::IdRef(exp),
                                    ],
                                )
                                .unwrap();
                            Ok(builder
                                .convert_f_to_s(output_data_type, None, result)
                                .unwrap())
                        }
                        _ => Err(VulkanError::UnsupportedByBackendError),
                    },
                    _ => Err(VulkanError::UnsupportedByBackendError),
                }
            },
        )
    }
}

#[cfg(test)]
mod test {
    use crate::backends::ndarray_backend::NDArrayNumericTensor;
    use crate::backends::vulkan_backend::tensor::VulkanTensor;
    use crate::backends::vulkan_backend::{VulkanContext, VulkanImmediateExecutor};
    use crate::dtype::DType;

    #[test]
    fn test_add() {
        let vulkan_context = VulkanContext::new().unwrap();
        let mut vulkan_runtime = VulkanImmediateExecutor::new(vulkan_context).unwrap();

        let start_data_a = vec![1.0, -2.0, 3.0, -4.0];
        let start_data_b = vec![1.0, -1.0, 3.0, 4.0];
        let expected_data = vec![2.0, -3.0, 6.0, 0.0];
        let end_shape = vec![1, 1, 4];

        let start_tensor_a = NDArrayNumericTensor::from(start_data_a).to_dyn();
        let start_tensor_a = start_tensor_a.reshape(&vec![1, 1, 4]).unwrap();
        let start_tensor_b = NDArrayNumericTensor::from(start_data_b).to_dyn();

        let dtypes_to_test = [
            DType::F64,
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::I64,
            DType::I32,
            DType::I16,
            DType::I8,
        ];

        for dtype in dtypes_to_test {
            let start_tensor_a_cast = start_tensor_a.cast(dtype).unwrap();
            let start_tensor_b_cast = start_tensor_b.cast(dtype).unwrap();
            let start_tensor_a_vk =
                VulkanTensor::from_ndarray(start_tensor_a_cast, &mut vulkan_runtime).unwrap();
            let start_tensor_b_vk =
                VulkanTensor::from_ndarray(start_tensor_b_cast, &mut vulkan_runtime).unwrap();
            let end_tensor_vk =
                VulkanTensor::add(&start_tensor_a_vk, &start_tensor_b_vk, &mut vulkan_runtime)
                    .unwrap();
            let end_tensor = end_tensor_vk.to_ndarray();
            assert_eq!(end_tensor.shape().clone(), end_shape);
            let end_tensor_ranked = end_tensor.cast(DType::F64).unwrap().flatten();
            let end_data: Vec<f64> = end_tensor_ranked.try_to_vec().unwrap();
            assert_eq!(end_data, expected_data);
        }
    }

    #[test]
    fn test_sub() {
        let vulkan_context = VulkanContext::new().unwrap();
        let mut vulkan_runtime = VulkanImmediateExecutor::new(vulkan_context).unwrap();

        let start_data_a = vec![1.0, -2.0, 3.0, -4.0];
        let start_data_b = vec![1.0, -1.0, 3.0, 4.0];
        let expected_data = vec![0.0, -1.0, 0.0, -8.0];
        let end_shape = vec![1, 1, 1, 4];

        let start_tensor_a = NDArrayNumericTensor::from(start_data_a).to_dyn();
        let start_tensor_a = start_tensor_a.reshape(&vec![1, 1, 1, 4]).unwrap();
        let start_tensor_b = NDArrayNumericTensor::from(start_data_b).to_dyn();

        let dtypes_to_test = [
            DType::F64,
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::I64,
            DType::I32,
            DType::I16,
            DType::I8,
        ];

        for dtype in dtypes_to_test {
            let start_tensor_a_cast = start_tensor_a.cast(dtype).unwrap();
            let start_tensor_b_cast = start_tensor_b.cast(dtype).unwrap();
            let start_tensor_a_vk =
                VulkanTensor::from_ndarray(start_tensor_a_cast, &mut vulkan_runtime).unwrap();
            let start_tensor_b_vk =
                VulkanTensor::from_ndarray(start_tensor_b_cast, &mut vulkan_runtime).unwrap();
            let end_tensor_vk =
                VulkanTensor::sub(&start_tensor_a_vk, &start_tensor_b_vk, &mut vulkan_runtime)
                    .unwrap();
            let end_tensor = end_tensor_vk.to_ndarray();
            assert_eq!(end_tensor.shape().clone(), end_shape);
            let end_tensor_ranked = end_tensor.cast(DType::F64).unwrap().flatten();
            let end_data: Vec<f64> = end_tensor_ranked.try_to_vec().unwrap();
            assert_eq!(end_data, expected_data);
        }
    }

    #[test]
    fn test_mul() {
        let vulkan_context = VulkanContext::new().unwrap();
        let mut vulkan_runtime = VulkanImmediateExecutor::new(vulkan_context).unwrap();

        let start_data_a = vec![1.0, -2.0, 3.0, -4.0];
        let start_data_b = vec![1.0, -1.0, 3.0, 4.0];
        let expected_data = vec![1.0, 2.0, 9.0, -16.0];
        let end_shape = vec![1, 1, 4];

        let start_tensor_a = NDArrayNumericTensor::from(start_data_a).to_dyn();
        let start_tensor_a = start_tensor_a.reshape(&vec![1, 1, 4]).unwrap();
        let start_tensor_b = NDArrayNumericTensor::from(start_data_b).to_dyn();

        let dtypes_to_test = [
            DType::F64,
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::I64,
            DType::I32,
            DType::I16,
            DType::I8,
        ];

        for dtype in dtypes_to_test {
            let start_tensor_a_cast = start_tensor_a.cast(dtype).unwrap();
            let start_tensor_b_cast = start_tensor_b.cast(dtype).unwrap();
            let start_tensor_a_vk =
                VulkanTensor::from_ndarray(start_tensor_a_cast, &mut vulkan_runtime).unwrap();
            let start_tensor_b_vk =
                VulkanTensor::from_ndarray(start_tensor_b_cast, &mut vulkan_runtime).unwrap();
            let end_tensor_vk =
                VulkanTensor::mul(&start_tensor_a_vk, &start_tensor_b_vk, &mut vulkan_runtime)
                    .unwrap();
            let end_tensor = end_tensor_vk.to_ndarray();
            assert_eq!(end_tensor.shape().clone(), end_shape);
            let end_tensor_ranked = end_tensor.cast(DType::F64).unwrap().flatten();
            let end_data: Vec<f64> = end_tensor_ranked.try_to_vec().unwrap();
            assert_eq!(end_data, expected_data);
        }
    }

    #[test]
    fn test_div() {
        let vulkan_context = VulkanContext::new().unwrap();
        let mut vulkan_runtime = VulkanImmediateExecutor::new(vulkan_context).unwrap();

        let start_data_a = vec![1.0, -2.0, 3.0, -4.0];
        let start_data_b = vec![1.0, -1.0, 3.0, 4.0];
        let expected_data = vec![1.0, 2.0, 1.0, -1.0];
        let end_shape = vec![1, 1, 4];

        let start_tensor_a = NDArrayNumericTensor::from(start_data_a).to_dyn();
        let start_tensor_b = NDArrayNumericTensor::from(start_data_b).to_dyn();
        let start_tensor_b = start_tensor_b.reshape(&vec![1, 1, 4]).unwrap();

        let dtypes_to_test = [
            DType::F64,
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::I64,
            DType::I32,
            DType::I16,
            DType::I8,
        ];

        for dtype in dtypes_to_test {
            let start_tensor_a_cast = start_tensor_a.cast(dtype).unwrap();
            let start_tensor_b_cast = start_tensor_b.cast(dtype).unwrap();
            let start_tensor_a_vk =
                VulkanTensor::from_ndarray(start_tensor_a_cast, &mut vulkan_runtime).unwrap();
            let start_tensor_b_vk =
                VulkanTensor::from_ndarray(start_tensor_b_cast, &mut vulkan_runtime).unwrap();
            let end_tensor_vk =
                VulkanTensor::div(&start_tensor_a_vk, &start_tensor_b_vk, &mut vulkan_runtime)
                    .unwrap();
            let end_tensor = end_tensor_vk.to_ndarray();
            assert_eq!(end_tensor.shape().clone(), end_shape);
            let end_tensor_ranked = end_tensor.cast(DType::F64).unwrap().flatten();
            let end_data: Vec<f64> = end_tensor_ranked.try_to_vec().unwrap();
            for (a, b) in end_data.iter().zip(expected_data.iter()) {
                assert!((a - b).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn test_and() {
        let vulkan_context = VulkanContext::new().unwrap();
        let mut vulkan_runtime = VulkanImmediateExecutor::new(vulkan_context).unwrap();

        let start_data_a = vec![false, false, true, true, false, false, true, true];
        let start_data_b = vec![true, false, true, false, true, false, true, false];
        let expected_data = vec![false, false, true, false, false, false, true, false];
        let end_shape = vec![1, 1, 8];

        let start_tensor_a = NDArrayNumericTensor::from(start_data_a).to_dyn();
        let start_tensor_b = NDArrayNumericTensor::from(start_data_b).to_dyn();
        let start_tensor_b = start_tensor_b.reshape(&vec![1, 1, 8]).unwrap();

        let start_tensor_a_vk =
            VulkanTensor::from_ndarray(start_tensor_a, &mut vulkan_runtime).unwrap();
        let start_tensor_b_vk =
            VulkanTensor::from_ndarray(start_tensor_b, &mut vulkan_runtime).unwrap();
        let end_tensor_vk =
            VulkanTensor::and(&start_tensor_a_vk, &start_tensor_b_vk, &mut vulkan_runtime).unwrap();
        let end_tensor = end_tensor_vk.to_ndarray();
        assert_eq!(end_tensor.shape().clone(), end_shape);
        let end_tensor_ranked = end_tensor.flatten();
        let end_data: Vec<bool> = end_tensor_ranked.try_to_vec().unwrap();
        assert_eq!(end_data, expected_data)
    }

    #[test]
    fn test_equal() {
        let vulkan_context = VulkanContext::new().unwrap();
        let mut vulkan_runtime = VulkanImmediateExecutor::new(vulkan_context).unwrap();

        let start_data_a = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let start_data_b = vec![1, 1, 4, 4, 3, 5, 7, 3];
        let expected_data = vec![true, false, false, true, false, false, true, false];

        let start_tensor_a = NDArrayNumericTensor::from(start_data_a).to_dyn();
        let start_tensor_b = NDArrayNumericTensor::from(start_data_b).to_dyn();

        let start_tensor_a_vk =
            VulkanTensor::from_ndarray(start_tensor_a, &mut vulkan_runtime).unwrap();
        let start_tensor_b_vk =
            VulkanTensor::from_ndarray(start_tensor_b, &mut vulkan_runtime).unwrap();
        let end_tensor_vk =
            VulkanTensor::equal(&start_tensor_a_vk, &start_tensor_b_vk, &mut vulkan_runtime)
                .unwrap();
        let end_tensor = end_tensor_vk.to_ndarray();
        let end_tensor_ranked = end_tensor.flatten();
        let end_data: Vec<bool> = end_tensor_ranked.try_to_vec().unwrap();
        assert_eq!(end_data, expected_data)
    }
}
