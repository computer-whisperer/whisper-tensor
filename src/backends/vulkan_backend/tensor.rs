use crate::backends::ndarray_backend::NDArrayNumericTensor;
use crate::backends::vulkan_backend::{VulkanContext, VulkanError, VulkanImmediateExecutor};
use crate::dtype::{DType, DTypeOfPrimitive};
use crate::tensor_rank::{DimContainer, Rank};
use ndarray::{ArcArray, CowArray};
use std::iter::zip;
use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo};
use vulkano::device::Device;
use vulkano::memory::allocator::Suballocation;
use vulkano::sync::GpuFuture;
use zerocopy::{Immutable, IntoBytes};

#[derive(Debug, Clone)]
pub(crate) struct BufferTransferKit {
    transfer_buffer: Subbuffer<[u8]>,
    context: VulkanContext,
}

#[derive(Debug, Clone)]
pub struct VulkanTensor<R: Rank> {
    pub(crate) dtype: DType,
    pub(crate) shape: R::KnownDims,
    pub(crate) stride: R::KnownDims,
    pub(crate) suballocation: Arc<Suballocation>,
    pub(crate) offset: usize,
    pub(crate) buffer: Subbuffer<[u8]>,
    pub(crate) buffer_transfer_kit: BufferTransferKit,
}

impl<R: Rank> VulkanTensor<R> {
    pub fn get_standard_stride(shape: &R::KnownDims) -> R::KnownDims {
        let mut stride = vec![];
        let mut v = 1;
        for &i in shape.as_slice().iter().rev() {
            stride.push(v);
            v *= i;
        }
        stride.reverse();
        R::KnownDims::try_from_slice(stride.as_slice()).unwrap()
    }

    pub fn get_device(&self) -> &Arc<Device> {
        &self.buffer_transfer_kit.context.device
    }

    /// # Safety
    ///
    /// This function is unsafe because it does not initialize the tensor.
    pub unsafe fn new_uninitialized(
        shape: R::KnownDims,
        dtype: DType,
        executor: &mut VulkanImmediateExecutor,
    ) -> Result<Self, VulkanError> {
        let stride = Self::get_standard_stride(&shape);
        unsafe { Self::new_uninitialized_with_stride(shape, stride, dtype, executor) }
    }

    /// # Safety
    ///
    /// This function is unsafe because it does not initialize the tensor.
    pub unsafe fn new_uninitialized_with_stride(
        shape: R::KnownDims,
        stride: R::KnownDims,
        dtype: DType,
        executor: &mut VulkanImmediateExecutor,
    ) -> Result<Self, VulkanError> {
        // Compute needed space carefully to handle zero-sized dimensions.
        let has_zero_dim = shape.as_slice().contains(&0);
        let needed_elems: u64 = if has_zero_dim {
            0
        } else {
            zip(shape.as_slice(), stride.as_slice())
                .map(|(a, b)| (a - 1) * b)
                .sum::<u64>()
                + 1
        };
        let needed_space = (needed_elems as usize) * dtype.size().unwrap();
        // Ensure non-zero allocations for internal buffers; transfers will be skipped when size is zero.
        let alloc_size = needed_space.max(1);
        let (buffer, suballocation) = executor.alloc_space(alloc_size);

        let transfer_buffer = if let Some(x) = &executor.host_transfer_buffer
            && x.len() >= alloc_size as u64
        {
            x
        } else {
            executor.allocate_host_transfer_buffer(alloc_size * 2)?;
            executor.host_transfer_buffer.as_ref().unwrap()
        }
        .clone();

        let buffer_transfer_kit = BufferTransferKit {
            context: executor.context.clone(),
            transfer_buffer,
        };

        Ok(VulkanTensor {
            dtype,
            shape,
            suballocation: Arc::new(suballocation),
            offset: 0,
            buffer,
            stride,
            buffer_transfer_kit,
        })
    }

    pub fn from_ndarray(
        source: NDArrayNumericTensor<R>,
        executor: &mut VulkanImmediateExecutor,
    ) -> Result<Self, VulkanError> {
        match source {
            NDArrayNumericTensor::F64(x) => Self::from_ndarray_inner(x, executor),
            NDArrayNumericTensor::F32(x) => Self::from_ndarray_inner(x, executor),
            NDArrayNumericTensor::BF16(x) => Self::from_ndarray_inner(x, executor),
            NDArrayNumericTensor::F16(x) => Self::from_ndarray_inner(x, executor),
            NDArrayNumericTensor::U64(x) => Self::from_ndarray_inner(x, executor),
            NDArrayNumericTensor::I64(x) => Self::from_ndarray_inner(x, executor),
            NDArrayNumericTensor::U32(x) => Self::from_ndarray_inner(x, executor),
            NDArrayNumericTensor::I32(x) => Self::from_ndarray_inner(x, executor),
            NDArrayNumericTensor::U16(x) => Self::from_ndarray_inner(x, executor),
            NDArrayNumericTensor::I16(x) => Self::from_ndarray_inner(x, executor),
            NDArrayNumericTensor::U8(x) => Self::from_ndarray_inner(x, executor),
            NDArrayNumericTensor::I8(x) => Self::from_ndarray_inner(x, executor),
            NDArrayNumericTensor::BOOL(x) => Self::from_ndarray_inner_bool(x, executor),
            _ => {
                unimplemented!()
            }
        }
    }

    fn from_ndarray_inner<T: DTypeOfPrimitive + IntoBytes + Clone + Immutable>(
        source: ArcArray<T, R::NDArrayDim>,
        executor: &mut VulkanImmediateExecutor,
    ) -> Result<Self, VulkanError> {
        let source = if source.as_slice_memory_order().is_none() {
            source.as_standard_layout()
        } else {
            CowArray::from(source.view())
        };
        let source_dtype = T::DTYPE;
        let source_data = source.as_slice_memory_order().unwrap();
        let source_stride = source.strides();

        let source_shape = source.shape().iter().map(|x| *x as u64).collect::<Vec<_>>();
        let source_stride = source_stride.iter().map(|x| *x as u64).collect::<Vec<_>>();
        let vk_tensor = unsafe {
            Self::new_uninitialized_with_stride(
                R::KnownDims::try_from_slice(source_shape.as_slice()).unwrap(),
                R::KnownDims::try_from_slice(source_stride.as_slice()).unwrap(),
                source_dtype,
                executor,
            )?
        };

        let raw_data = source_data.as_bytes();
        let bytes_to_transfer = raw_data.len();
        assert!(bytes_to_transfer <= vk_tensor.suballocation.size as usize);

        if bytes_to_transfer == 0 {
            // Nothing to copy for zero-sized tensors.
            return Ok(vk_tensor);
        }

        let transfer_buffer = &vk_tensor.buffer_transfer_kit.transfer_buffer;
        transfer_buffer.write().unwrap()[0..raw_data.len()].copy_from_slice(raw_data);

        let mut builder = AutoCommandBufferBuilder::primary(
            executor.context.command_buffer_allocator.clone(),
            executor.context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;
        let outer_offset = vk_tensor.suballocation.offset;
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                transfer_buffer.clone(),
                vk_tensor
                    .buffer
                    .clone()
                    .slice(outer_offset..outer_offset + bytes_to_transfer as u64),
            ))
            .unwrap();
        let command_buffer = builder.build()?;
        // Let's execute this command buffer now.
        let future = vulkano::sync::now(executor.context.device.clone())
            .then_execute(executor.context.queue.clone(), command_buffer)
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
        Ok(vk_tensor)
    }

    fn from_ndarray_inner_bool(
        source: ArcArray<bool, R::NDArrayDim>,
        executor: &mut VulkanImmediateExecutor,
    ) -> Result<Self, VulkanError> {
        let source = if source.as_slice_memory_order().is_none() {
            source.as_standard_layout()
        } else {
            CowArray::from(source.view())
        };
        let source_dtype = bool::DTYPE;
        let source_data = source.as_slice_memory_order().unwrap();
        let source_stride = source.strides();

        let source_shape = source.shape().iter().map(|x| *x as u64).collect::<Vec<_>>();
        let source_stride = source_stride.iter().map(|x| *x as u64).collect::<Vec<_>>();
        let vk_tensor = unsafe {
            Self::new_uninitialized_with_stride(
                R::KnownDims::try_from_slice(source_shape.as_slice()).unwrap(),
                R::KnownDims::try_from_slice(source_stride.as_slice()).unwrap(),
                source_dtype,
                executor,
            )?
        };

        let raw_data = source_data
            .iter()
            .map(|x| if *x { 1u8 } else { 0u8 })
            .collect::<Vec<_>>();
        let bytes_to_transfer = raw_data.len();
        assert!(bytes_to_transfer <= vk_tensor.suballocation.size as usize);

        if bytes_to_transfer == 0 {
            return Ok(vk_tensor);
        }

        let transfer_buffer = &vk_tensor.buffer_transfer_kit.transfer_buffer;
        transfer_buffer.write().unwrap()[0..raw_data.len()].copy_from_slice(&raw_data);

        let mut builder = AutoCommandBufferBuilder::primary(
            executor.context.command_buffer_allocator.clone(),
            executor.context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;
        let outer_offset = vk_tensor.suballocation.offset;
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                transfer_buffer.clone(),
                vk_tensor
                    .buffer
                    .clone()
                    .slice(outer_offset..outer_offset + bytes_to_transfer as u64),
            ))
            .unwrap();
        let command_buffer = builder.build()?;
        // Let's execute this command buffer now.
        let future = vulkano::sync::now(executor.context.device.clone())
            .then_execute(executor.context.queue.clone(), command_buffer)
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
        Ok(vk_tensor)
    }

    pub fn to_ndarray(&self) -> NDArrayNumericTensor<R> {
        {
            // Handle zero-sized dimensions: no bytes to read, construct empty NDArray directly.
            if self.shape.as_slice().contains(&0) {
                // Use zero strides to satisfy ndarray's bounds checks for empty arrays.
                let zero_stride =
                    R::KnownDims::try_from_slice(&vec![0u64; self.shape.len()]).unwrap();
                return NDArrayNumericTensor::from_bytes(
                    &[],
                    self.dtype,
                    &self.shape,
                    &zero_stride,
                )
                .unwrap();
            }

            let bytes_to_read = (self
                .shape
                .as_slice()
                .iter()
                .zip(self.stride.as_slice().iter())
                .map(|(a, b)| (a - 1) * b)
                .sum::<u64>()
                + 1) as usize
                * self.dtype.size().unwrap();

            if bytes_to_read == 0 {
                return NDArrayNumericTensor::from_bytes(
                    &[],
                    self.dtype,
                    &self.shape,
                    &self.stride,
                )
                .unwrap();
            }

            let start_offset = self.suballocation.offset + self.offset as u64;
            let transfer_buffer = &self.buffer_transfer_kit.transfer_buffer;
            let mut builder = AutoCommandBufferBuilder::primary(
                self.buffer_transfer_kit
                    .context
                    .command_buffer_allocator
                    .clone(),
                self.buffer_transfer_kit.context.queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    self.buffer
                        .clone()
                        .slice(start_offset..start_offset + bytes_to_read as u64),
                    transfer_buffer.clone(),
                ))
                .unwrap();
            let command_buffer = builder.build().unwrap();
            // Let's execute this command buffer now.
            let future = vulkano::sync::now(self.buffer_transfer_kit.context.device.clone())
                .then_execute(
                    self.buffer_transfer_kit.context.queue.clone(),
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

            let reader = transfer_buffer.read().unwrap();
            NDArrayNumericTensor::from_bytes(
                &reader[0..bytes_to_read],
                self.dtype,
                &self.shape,
                &self.stride,
            )
            .unwrap()
        }
    }

    pub fn shape(&self) -> &R::KnownDims {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn broadcast<R2: Rank>(&self, dim: &R2::KnownDims) -> Option<VulkanTensor<R2>> {
        let mut new_stride = dim.as_slice().to_vec();
        // begin at the back (the least significant dimension)
        // size of the axis has to either agree or `from` has to be 1
        if dim.len() < self.rank() {
            return None;
        }

        {
            let mut new_stride_iter = new_stride.iter_mut().rev();
            for ((er, es), dr) in self
                .shape()
                .as_slice()
                .iter()
                .rev()
                .zip(self.stride.as_slice().iter().rev())
                .zip(new_stride_iter.by_ref())
            {
                /* update strides */
                if *dr == *er {
                    /* keep stride */
                    *dr = *es;
                } else if *er == 1 {
                    /* dead dimension, zero stride */
                    *dr = 0
                } else {
                    return None;
                }
            }

            /* set remaining strides to zero */
            for dr in new_stride_iter {
                *dr = 0;
            }
        }

        let new_stride = R2::KnownDims::try_from_slice(&new_stride).unwrap();

        Some(VulkanTensor {
            shape: dim.clone(),
            stride: new_stride,
            suballocation: self.suballocation.clone(),
            offset: self.offset,
            dtype: self.dtype,
            buffer: self.buffer.clone(),
            buffer_transfer_kit: self.buffer_transfer_kit.clone(),
        })
    }

    pub fn is_contiguous(&self) -> bool {
        let mut v = 1;
        for i in (0..self.rank()).rev() {
            if self.stride[i] != v {
                return false;
            }
            v *= self.shape[i];
        }
        true
    }
}
