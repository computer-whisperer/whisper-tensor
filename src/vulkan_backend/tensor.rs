use std::sync::Arc;
use vulkano::buffer::Subbuffer;
use vulkano::memory::allocator::Suballocation;
use crate::dtype::DType;
use crate::NDArrayNumericTensor;
use crate::tensor_rank::{DimContainer, DimProduct, Rank};
use crate::vulkan_backend::{VulkanError, VulkanImmediateExecutor};



#[derive(Debug, Clone)]
pub struct VulkanTensor<R: Rank> {
    dtype: DType,
    shape: R::KnownDims,
    pub(crate) stride: R::KnownDims,
    pub(crate) suballocation: Arc<Suballocation>,
    pub(crate) offset: usize,
    pub(crate) buffer: Subbuffer<[u8]>
}

impl<R: Rank> VulkanTensor<R> {

    pub unsafe fn new_uninitialized(shape: R::KnownDims, dtype: DType, executor: &mut VulkanImmediateExecutor) -> Result<Self, VulkanError> {
        let needed_space = shape.as_slice().iter().product::<u64>() as usize*dtype.size();
        let (buffer, suballocation) = executor.alloc_space(needed_space);

        let mut stride = vec![];
        let mut v = 1;
        for &i in shape.as_slice() {
            stride.push(v);
            v = v * i;
        }
        let stride = R::KnownDims::try_from_slice(stride.as_slice()).unwrap();

        Ok(VulkanTensor {
            dtype,
            shape,
            suballocation: Arc::new(suballocation),
            offset: 0,
            buffer,
            stride
        })
    }
    
    pub fn from_ndarray(source: NDArrayNumericTensor<R>, executor: &mut VulkanImmediateExecutor) -> Result<Self, VulkanError> {

        let tensor = unsafe {Self::new_uninitialized(source.shape(), source.dtype(), executor)?};
        
        {
            let mut writer = tensor.buffer.write().unwrap();
            for i in 0..tensor.shape.dim_product() {
                let mut index = vec![];
                let mut v = i;
                for &j in tensor.shape.as_slice() {
                    index.push(v % j);
                    v = v / j;
                }
                let index = R::KnownDims::try_from_slice(index.as_slice()).unwrap();
                let value = source.get(&index).unwrap().to_bytes();
                // Calculate destination
                let outer_offset = tensor.suballocation.offset as usize;
                let inner_offset = index.as_slice().iter().zip(tensor.stride.as_slice().iter()).map(|(a, b)| a*b).sum::<u64>() as usize * tensor.dtype.size();
                writer[outer_offset+inner_offset..outer_offset+inner_offset+value.len()].copy_from_slice(&value);
            }
        }
        Ok(tensor)
    }

    pub fn to_ndarray(&self) -> NDArrayNumericTensor<R> {
        {
            let reader = self.buffer.read().unwrap();
            let bytes_to_read = self.shape.as_slice().iter().zip(self.stride.as_slice().iter()).map(|(a, b)| a*b).sum::<u64>() as usize*self.dtype.size();
            let start_offset = self.suballocation.offset as usize + self.offset;
            NDArrayNumericTensor::from_bytes(
                &reader[start_offset ..start_offset + bytes_to_read],
                self.dtype, &self.shape, &self.stride).unwrap()
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
}

impl<R: Rank> From<VulkanTensor<R>> for NDArrayNumericTensor<R> {
    fn from(value: VulkanTensor<R>) -> Self {
        value.to_ndarray()
    }
}