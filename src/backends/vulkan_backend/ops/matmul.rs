use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use rspirv::binary::Assemble;
use rspirv::dr::Operand;
use rspirv::dr::Operand::LiteralBit32;
use rspirv::spirv;
use rspirv::spirv::{BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel, SelectionControl, StorageClass, Word};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::{PipelineLayoutCreateInfo, PushConstantRange};
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo, ShaderStages};
use vulkano::sync::GpuFuture;
use crate::dtype::DType;
use crate::DynRank;
use crate::tensor_rank::{DimContainer, DimProduct, Rank};
use crate::backends::vulkan_backend::tensor::VulkanTensor;
use crate::backends::vulkan_backend::{VulkanError, VulkanImmediateExecutor};
use crate::backends::vulkan_backend::spirv_helpers::{cast_bf16_to_f32, cast_f32_to_bf16, get_spirv_datatype};

fn build_matmul_pipeline(
    vulkan_immediate_executor: &mut VulkanImmediateExecutor,
    base_shape: Vec<u64>,
    m: u64,
    k: u64,
    n: u64,
    a_stride: Vec<u64>,
    b_stride: Vec<u64>,
    stored_dtype: DType
) -> Result<(Arc<PipelineLayout>, Arc<ComputePipeline>), VulkanError>
{
    let output_shape = {
        let mut output_shape = base_shape.clone();
        output_shape.push(m);
        output_shape.push(n);
        output_shape
    };
    let output_stride = VulkanTensor::<DynRank>::get_standard_stride(&output_shape);

    let mut builder = rspirv::dr::Builder::new();
    builder.capability(Capability::Shader);
    builder.capability(Capability::Float64);
    builder.capability(Capability::Float16);
    builder.capability(Capability::Int16);
    builder.capability(Capability::Int8);
    builder.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);

    let stored_spirv_dtype = match stored_dtype {
        DType::BF16 => builder.type_int(16, 0),
        DType::BOOL => builder.type_int(8, 0),
        _ => get_spirv_datatype(&mut builder, stored_dtype)?
    };

    let io_data_type_array = builder.type_runtime_array(stored_spirv_dtype);
    builder.decorate(io_data_type_array, Decoration::ArrayStride, [Operand::LiteralBit32(stored_dtype.size().unwrap() as u32)]);
    let io_data_type_array_struct = builder.type_struct([io_data_type_array]);
    builder.decorate(io_data_type_array_struct, Decoration::Block, []);
    builder.member_decorate(io_data_type_array_struct, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
    let io_sb_ptr  = builder.type_pointer(None, StorageClass::StorageBuffer, io_data_type_array_struct);

    let input_0_var = builder.variable(io_sb_ptr, None, StorageClass::StorageBuffer, None);
    builder.decorate(input_0_var, Decoration::DescriptorSet, [LiteralBit32(0)]);
    builder.decorate(input_0_var, Decoration::Binding, [LiteralBit32(0)]);

    let input_1_var = builder.variable(io_sb_ptr, None, StorageClass::StorageBuffer, None);
    builder.decorate(input_1_var, Decoration::DescriptorSet, [LiteralBit32(0)]);
    builder.decorate(input_1_var, Decoration::Binding, [LiteralBit32(1)]);

    let output_var = builder.variable(io_sb_ptr, None, StorageClass::StorageBuffer, None);
    builder.decorate(output_var, Decoration::DescriptorSet, [LiteralBit32(0)]);
    builder.decorate(output_var, Decoration::Binding, [LiteralBit32(2)]);


    let u32_t = builder.type_int(32, 0);
    // Data:
    // input_0_offset,
    // input_1_offset,
    // output_offset
    let metadata_value_count = 3;
    let pc_struct = {
        let mut v = Vec::new();
        v.resize(metadata_value_count, u32_t);
        builder.type_struct(v)
    };

    builder.decorate(pc_struct, Decoration::Block, []);
    for i in 0..metadata_value_count {
        builder.member_decorate(pc_struct, i as u32, Decoration::Offset,
                                [Operand::LiteralBit32((i*4) as u32)]);
    }

    let pc_ptr = builder.type_pointer(None, StorageClass::PushConstant, pc_struct);
    let metadata_var = builder.variable(pc_ptr, None, StorageClass::PushConstant, None);

    let vec3u32_t  = builder.type_vector(u32_t, 3);
    let in_ptr = builder.type_pointer(None, StorageClass::Input, vec3u32_t);
    let gid    = builder.variable(in_ptr, None, StorageClass::Input, None);
    builder.decorate(gid, Decoration::BuiltIn, [Operand::BuiltIn(BuiltIn::GlobalInvocationId)]);

    let void = builder.type_void();
    let voidf = builder.type_function(void, []);
    let main_fn = builder.begin_function(void, None, (spirv::FunctionControl::DONT_INLINE), voidf).unwrap();
    builder.entry_point(ExecutionModel::GLCompute, main_fn, "main", [gid, metadata_var, output_var, input_0_var, input_1_var]);
    builder.execution_mode(main_fn, ExecutionMode::LocalSize, [8, 8, 1]);
    builder.begin_block(None).unwrap();

    /* Constants */
    let c0 = builder.constant_bit32(u32_t, 0);

    /* idx = gl_GlobalInvocationID.x */
    let gid  = builder.load(vec3u32_t, None, gid.clone(), None, []).unwrap();
    let x_idx  = builder.composite_extract(u32_t, None, gid, [0u32]).unwrap();
    let y_idx  = builder.composite_extract(u32_t, None, gid, [1u32]).unwrap();
    let z_idx  = builder.composite_extract(u32_t, None, gid, [2u32]).unwrap();

    /* size  = metadata.size */
    let push_constant_vals = {
        let mut vals = Vec::new();
        let u32_pc_ptr_t = builder.type_pointer(None, StorageClass::PushConstant, u32_t);
        for i in 0..metadata_value_count {
            let c_i = builder.constant_bit32(u32_t, i as u32);
            let size_ptr = builder.access_chain(u32_pc_ptr_t, None, metadata_var, [c_i]).unwrap();
            vals.push(builder.load(u32_t, None, size_ptr, None, []).unwrap());
        }
        vals
    };

    let input_a_offset = push_constant_vals[0];
    let input_b_offset = push_constant_vals[1];
    let output_offset = push_constant_vals[2];

    let m_const = builder.constant_bit32(u32_t, m as u32);
    //let k_const = builder.constant_bit32(u32_t, k as u32);
    let n_const = builder.constant_bit32(u32_t, n as u32);

    /* cmp & branch */

    let bool_type = builder.type_bool();
    let cmp_a = builder.u_less_than(bool_type, None, x_idx, m_const).unwrap();
    let cmp_b = builder.u_less_than(bool_type, None, y_idx, n_const).unwrap();
    let cmp = builder.logical_and(bool_type, None, cmp_a, cmp_b).unwrap();
    let merge_blk = builder.id();
    let then_blk  = builder.id();

    builder.selection_merge(merge_blk, SelectionControl::NONE).unwrap();
    builder.branch_conditional(cmp, then_blk, merge_blk, None).unwrap();

    /* ---- THEN block ---- */
    builder.begin_block(Some(then_blk)).unwrap();

    // Calculate base index
    let mut remaining_idx = z_idx;
    let index = {
        let mut v = vec![];
        for i in 0..base_shape.len() {
            let shape_val = base_shape[i];
            let shape_val_const = builder.constant_bit32(u32_t, shape_val as u32);
            let rem = builder.u_mod(u32_t, None, remaining_idx, shape_val_const).unwrap();
            let div = builder.u_div(u32_t, None, remaining_idx, shape_val_const).unwrap();
            remaining_idx = div;
            v.push(rem);
        }
        v
    };


    let a_base_offset = {
        let mut v = input_a_offset;
        for i in 0..base_shape.len() {
            let stride_const = builder.constant_bit32(u32_t, a_stride[i] as u32);
            let mul = builder.i_mul(u32_t, None, index[i], stride_const).unwrap();
            v = builder.i_add(u32_t, None, v, mul).unwrap();
        }
        let stride_const = builder.constant_bit32(u32_t, a_stride[base_shape.len()] as u32);
        let mul = builder.i_mul(u32_t, None, x_idx, stride_const).unwrap();
        v = builder.i_add(u32_t, None, v, mul).unwrap();
        v
    };

    let b_base_offset = {
        let mut v = input_b_offset;
        for i in 0..base_shape.len() {
            let stride_const = builder.constant_bit32(u32_t, b_stride[i] as u32);
            let mul = builder.i_mul(u32_t, None, index[i], stride_const).unwrap();
            v = builder.i_add(u32_t, None, v, mul).unwrap();
        }
        let stride_const = builder.constant_bit32(u32_t, b_stride[base_shape.len()+1] as u32);
        let mul = builder.i_mul(u32_t, None, y_idx, stride_const).unwrap();
        v = builder.i_add(u32_t, None, v, mul).unwrap();
        v
    };

    let output_index = {
        let mut v = output_offset;
        for i in 0..base_shape.len() {
            let stride_const = builder.constant_bit32(u32_t, output_stride[i] as u32);
            let mul = builder.i_mul(u32_t, None, index[i], stride_const).unwrap();
            v = builder.i_add(u32_t, None, v, mul).unwrap();
        }
        let stride_const = builder.constant_bit32(u32_t, output_stride[base_shape.len()] as u32);
        let mul = builder.i_mul(u32_t, None, x_idx, stride_const).unwrap();
        v = builder.i_add(u32_t, None, v, mul).unwrap();
        let stride_const = builder.constant_bit32(u32_t, output_stride[base_shape.len()+1] as u32);
        let mul = builder.i_mul(u32_t, None, y_idx, stride_const).unwrap();
        v = builder.i_add(u32_t, None, v, mul).unwrap();
        v
    };


    let mut accumulated_value: Option<Word> = None;
    for i in 0..k {
        let i_const = builder.constant_bit32(u32_t, i as u32);

        let a_index =  {
            let stride_const = builder.constant_bit32(u32_t, a_stride[base_shape.len()+1] as u32);
            let mul = builder.i_mul(u32_t, None, i_const, stride_const).unwrap();
            builder.i_add(u32_t, None, a_base_offset, mul).unwrap()
        };

        let b_index =  {
            let stride_const = builder.constant_bit32(u32_t, b_stride[base_shape.len()] as u32);
            let mul = builder.i_mul(u32_t, None, i_const, stride_const).unwrap();
            builder.i_add(u32_t, None, b_base_offset, mul).unwrap()
        };

        let data_i_ptr_t = builder.type_pointer(None, StorageClass::StorageBuffer, stored_spirv_dtype);
        let in0_ptr = builder.access_chain(data_i_ptr_t, None, input_0_var, [c0, a_index]).unwrap();
        let a_val = builder.load(stored_spirv_dtype, None, in0_ptr, None, []).unwrap();
        let in1_ptr = builder.access_chain(data_i_ptr_t, None, input_1_var, [c0, b_index]).unwrap();
        let b_val = builder.load(stored_spirv_dtype, None, in1_ptr, None, []).unwrap();

        let (a_val, a_dtype) = match stored_dtype {
            DType::BF16 => {
                (cast_bf16_to_f32(&mut builder, a_val), DType::F32)
            }
            _ => {
                (a_val, stored_dtype)
            }
        };

        let (b_val, _b_dtype) = match stored_dtype {
            DType::BF16 => {
                (cast_bf16_to_f32(&mut builder, b_val), DType::F32)
            }
            _ => {
                (b_val, stored_dtype)
            }
        };

        let a_spirv_type = get_spirv_datatype(&mut builder, a_dtype).unwrap();


        let prod = match a_dtype {
            DType::BF16 | DType::F16 | DType::F32 | DType::F64 => builder.f_mul(a_spirv_type, None, a_val, b_val).unwrap(),
            DType::I64 | DType::I32 | DType::I16 | DType::I8 | DType::U64 | DType::U32 | DType::U16 | DType::U8 => builder.i_mul(a_spirv_type, None, a_val, b_val).unwrap(),
            _ => Err(VulkanError::UnsupportedByBackendError)?,
        };

        accumulated_value = match accumulated_value {
            Some(accumulated_value) => {
                Some(match a_dtype {
                    DType::BF16 | DType::F16 | DType::F32 | DType::F64 => builder.f_add(a_spirv_type, None, accumulated_value, prod).unwrap(),
                    DType::I64 | DType::I32 | DType::I16 | DType::I8 | DType::U64 | DType::U32 | DType::U16 | DType::U8 => builder.i_add(a_spirv_type, None, accumulated_value, prod).unwrap(),
                    _ => Err(VulkanError::UnsupportedByBackendError)?,
                })
            }
            None => {
                Some(prod)
            }
        }
    }

    let out_val = match stored_dtype {
        DType::BF16 => {
            cast_f32_to_bf16(&mut builder, accumulated_value.unwrap())
        }
        _ => {
            accumulated_value.unwrap()
        }
    };

    /* store */
    let data_o_ptr_t = builder.type_pointer(None, StorageClass::StorageBuffer, stored_spirv_dtype);
    let out_ptr = builder.access_chain(data_o_ptr_t, None, output_var, [c0, output_index]).unwrap();
    builder.store(out_ptr, out_val, None, []).unwrap();

    /* branch to merge */
    builder.branch(merge_blk).unwrap();

    /* ---- MERGE block ---- */
    builder.begin_block(Some(merge_blk)).unwrap();
    builder.ret().unwrap();                       // OpReturn
    builder.end_function().unwrap();              // OpFunction

    let module = builder.module();
    let code = module.assemble();

    VulkanImmediateExecutor::debug_dump_spirv(&code);

    let shader = unsafe{ ShaderModule::new(
        vulkan_immediate_executor.context.device.clone(),
        ShaderModuleCreateInfo::new(&code)
    )}?;

    let cs = shader.entry_point("main").unwrap();

    let stage = PipelineShaderStageCreateInfo::new(cs);

    let descriptor_set_layout = vulkan_immediate_executor.get_descriptor_set_layout(BTreeSet::from([0, 1, 2]))?;

    let layout = PipelineLayout::new(
        vulkan_immediate_executor.context.device.clone(),
        PipelineLayoutCreateInfo {
            set_layouts: vec![descriptor_set_layout],
            push_constant_ranges: vec![PushConstantRange{
                stages: ShaderStages::COMPUTE,
                offset: 0,
                size: 12u32,
            }],
            .. Default::default()
        }
    )?;

    let compute_pipeline = ComputePipeline::new(
        vulkan_immediate_executor.context.device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout.clone())
    )?;

    Ok((layout, compute_pipeline))
}


impl<R: Rank> VulkanTensor<R> {


    pub fn matmul(a: &Self, b: &Self, vulkan_immediate_executor: &mut VulkanImmediateExecutor) -> Result<Self, VulkanError> {
        assert_eq!(a.dtype(), b.dtype());

        let drop_first_axis_after = a.rank() == 1;
        let mut a = if drop_first_axis_after {
            a.unsqueeze(0)?
        } else {
            a.clone()
        };

        let drop_last_axis_after = b.rank() == 1;
        let mut b = if drop_last_axis_after {
            b.unsqueeze(1)?
        }
        else {
            b.clone()
        };

        // Even up ranks
        while a.rank() < b.rank() {
            a = a.unsqueeze(0)?;
        }
        while b.rank() < a.rank() {
            b = b.unsqueeze(0)?;
        }

        // Collect common shape
        let common_shape = {
            let mut common_shape = vec![];
            for i in 0..a.rank() - 2 {
                common_shape.push(a.shape[i].max(b.shape[i]))
            }
            common_shape
        };

        let m = a.shape[a.rank() - 2];
        let k = a.shape[a.rank() - 1];
        assert_eq!(k, b.shape[b.rank() - 2]);
        let n = b.shape[b.rank() - 1];

        // Broadcast a and b appropriately
        let a: VulkanTensor<R> = {
            let mut a_shape = common_shape.clone();
            a_shape.push(m);
            a_shape.push(k);
            a.broadcast(&R::KnownDims::try_from_slice(&a_shape).unwrap()).unwrap()
        };
        let b: VulkanTensor<R> = {
            let mut b_shape = common_shape.clone();
            b_shape.push(k);
            b_shape.push(n);
            b.broadcast(&R::KnownDims::try_from_slice(&b_shape).unwrap()).unwrap()
        };

        let a_stride: &R::KnownDims = &a.stride;
        let b_stride: &R::KnownDims = &b.stride;
        let a_stride = a_stride.as_slice().to_vec();
        let b_stride = b_stride.as_slice().to_vec();

        let (pipeline_layout, compute_pipeline) = {
            let key = (a.dtype(), common_shape.clone(), m, k, n, a_stride.clone(), b_stride.clone());
            let res = vulkan_immediate_executor.pipeline_cache.matmul_op.get(&key);
            if let Some(res) = res {
                res.clone()
            } else {
                let res = build_matmul_pipeline(
                    vulkan_immediate_executor,
                    common_shape.clone(),
                    m, k, n,
                    a_stride.clone(), b_stride.clone(),
                    a.dtype())?;
                vulkan_immediate_executor.pipeline_cache.matmul_op.insert(key, res.clone());
                res
            }
        };

        let output_shape = {
            let mut output_shape = common_shape.clone();
            output_shape.push(m);
            output_shape.push(n);
            R::KnownDims::try_from_slice(&output_shape).unwrap()
        };

        let output_tensor = unsafe{VulkanTensor::new_uninitialized(output_shape.clone(), a.dtype(), vulkan_immediate_executor)}?;

        let (descriptor_set, _) = vulkan_immediate_executor.get_descriptor_set_and_layout(&BTreeMap::from([
            (0, a.buffer.clone()),
            (1, b.buffer.clone()),
            (2, output_tensor.buffer.clone())
        ]))?;

        let op_metadata_vec = {
            let mut v = vec![];
            v.push((a.offset as u32 + a.suballocation.offset as u32)/a.dtype().size().unwrap() as u32);
            v.push((b.offset as u32 + b.suballocation.offset as u32)/b.dtype().size().unwrap() as u32);
            v.push((output_tensor.offset as u32 + output_tensor.suballocation.offset as u32)/a.dtype().size().unwrap() as u32);
            v
        };


        let mut builder = AutoCommandBufferBuilder::primary(
            vulkan_immediate_executor.context.command_buffer_allocator.clone(),
            vulkan_immediate_executor.context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit
        )?;

        for (a, b) in op_metadata_vec.iter().enumerate() {
            builder.push_constants(pipeline_layout.clone(), (a*4) as u32, *b).unwrap();
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
        unsafe { builder.dispatch([((m+7)/8) as u32, ((n+7)/8) as u32, common_shape.iter().product::<u64>() as u32]) }.unwrap();

        // Finish building the command buffer by calling `build`.
        let command_buffer = builder.build()?;

        // Let's execute this command buffer now.
        let future = vulkano::sync::now(vulkan_immediate_executor.context.device.clone())
            .then_execute(vulkan_immediate_executor.context.queue.clone(), command_buffer)
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

        let output_tensor = if drop_first_axis_after {
            output_tensor.squeeze(0)?
        } else {
            output_tensor
        };

        let output_tensor = if drop_last_axis_after {
            output_tensor.squeeze(output_tensor.rank() - 1)?
        } else {
            output_tensor
        };

        Ok(output_tensor)
    }
}

mod test {
    use crate::dtype::DType;
    use crate::backends::ndarray_backend::NDArrayNumericTensor;
    use crate::backends::vulkan_backend::{VulkanContext, VulkanImmediateExecutor};
    use crate::backends::vulkan_backend::tensor::VulkanTensor;

    #[test]
    fn test_matmul_2d() {
        let vulkan_context = VulkanContext::new().unwrap();
        let mut vulkan_runtime = VulkanImmediateExecutor::new(vulkan_context).unwrap();

        let start_data_a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let start_data_b: Vec<f32> = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let expected_data = vec![40.0, 30.0, 120.0, 94.0];

        let start_tensor_a = NDArrayNumericTensor::from(start_data_a).to_dyn().reshape(&vec![2, 4]).unwrap();
        let start_tensor_b = NDArrayNumericTensor::from(start_data_b).to_dyn().reshape(&vec![4, 2]).unwrap();

        let start_tensor_a_vk = VulkanTensor::from_ndarray(start_tensor_a, &mut vulkan_runtime).unwrap();
        let start_tensor_b_vk = VulkanTensor::from_ndarray(start_tensor_b, &mut vulkan_runtime).unwrap();
        let end_tensor_vk = VulkanTensor::matmul(&start_tensor_a_vk, &start_tensor_b_vk, &mut vulkan_runtime).unwrap();
        let end_tensor = end_tensor_vk.to_ndarray();
        let end_tensor_ranked = end_tensor.flatten();
        let end_data: Vec<f32> = end_tensor_ranked.try_to_vec().unwrap();
        assert_eq!(end_data, expected_data);
    }

    #[test]
    fn test_vec_types() {
        let vulkan_context = VulkanContext::new().unwrap();
        let mut vulkan_runtime = VulkanImmediateExecutor::new(vulkan_context).unwrap();

        let start_data_a = vec![1.0, -2.0, 3.0, -4.0];
        let start_data_b = vec![1.0, -1.0, 3.0, 4.0];
        let expected_data = vec![-4.0];

        let start_tensor_a = NDArrayNumericTensor::from(start_data_a).to_dyn();
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
            let start_tensor_a_vk = VulkanTensor::from_ndarray(start_tensor_a_cast, &mut vulkan_runtime).unwrap();
            let start_tensor_b_vk = VulkanTensor::from_ndarray(start_tensor_b_cast, &mut vulkan_runtime).unwrap();
            let end_tensor_vk = VulkanTensor::matmul(&start_tensor_a_vk, &start_tensor_b_vk, &mut vulkan_runtime).unwrap();
            let end_tensor = end_tensor_vk.to_ndarray();
            let end_tensor_ranked = end_tensor.cast(DType::F64).unwrap().flatten();
            let end_data: Vec<f64> = end_tensor_ranked.try_to_vec().unwrap();
            assert_eq!(end_data, expected_data);
        }
    }
}