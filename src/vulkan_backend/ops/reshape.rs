use crate::tensor_rank::{DimContainer, DimProduct, Rank};
use crate::vulkan_backend::tensor::VulkanTensor;
use crate::vulkan_backend::{VulkanError, VulkanImmediateExecutor};

impl<R: Rank> VulkanTensor<R> {
    pub fn reshape(&self, new_shape: R::KnownDims, vulkan_immediate_executor: &mut VulkanImmediateExecutor) -> Result<Self, VulkanError> {
        if new_shape.dim_product() != self.shape().dim_product() {
            return Err(VulkanError::InvalidShape);
        }
        let VulkanTensor {
            dtype,
            suballocation,
            buffer,
            offset,
            ..
        } = self.to_contiguous(vulkan_immediate_executor)?;
        
        let new_stride = {
            let mut stride = vec![];
            let mut v = 1;
            for &i in new_shape.as_slice().iter().rev() {
                stride.push(v);
                v = v * i;
            }
            stride.reverse();
            R::KnownDims::try_from_slice(stride.as_slice()).unwrap()
        };
        
        Ok(VulkanTensor {
            shape: new_shape,
            stride: new_stride,
            dtype,
            suballocation,
            buffer,
            offset
        })
    }

    pub fn unsqueeze<R2: Rank>(&self, axis: usize) -> Result<VulkanTensor<R2>, VulkanError> {
        let new_shape = {
            let mut v = self.shape().as_slice().to_vec();
            v.insert(axis, 1);
            R2::KnownDims::try_from_slice(v.as_slice()).unwrap()
        };
        let new_strides = {
            let mut v = self.stride.as_slice().to_vec();
            v.insert(axis, 1);
            R2::KnownDims::try_from_slice(v.as_slice()).unwrap()
        };
        Ok(VulkanTensor::<R2>{
            shape: new_shape,
            stride: new_strides,
            dtype: self.dtype,
            suballocation: self.suballocation.clone(),
            buffer: self.buffer.clone(),
            offset: self.offset
        })
    }

    pub fn squeeze<R2: Rank>(&self, axis: usize) -> Result<VulkanTensor<R2>, VulkanError> {
        if self.shape[axis] == 1 {
            let new_shape = {
                let mut v = self.shape().as_slice().to_vec();
                v.remove(axis);
                R2::KnownDims::try_from_slice(v.as_slice()).unwrap()
            };
            let new_strides = {
                let mut v = self.stride.as_slice().to_vec();
                v.remove(axis);
                R2::KnownDims::try_from_slice(v.as_slice()).unwrap()
            };
            Ok(VulkanTensor::<R2>{
                shape: new_shape,
                stride: new_strides,
                dtype: self.dtype,
                suballocation: self.suballocation.clone(),
                buffer: self.buffer.clone(),
                offset: self.offset
            })
        }
        else {
            Err(VulkanError::InvalidShape)
        }
    }
}