use std::ops::Range;
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

    pub fn slice(&self, indices: &[Range<u64>]) -> Result<Self, VulkanError> {
        // Get indices for the lowest value in output tensor
        let start_index = indices.iter().map(|r| r.start).collect::<Vec<_>>();
        let new_shape = indices.iter().map(|r| r.end - r.start).collect::<Vec<_>>();
        let new_shape = R::KnownDims::try_from_slice(new_shape.as_slice()).unwrap();
        let new_strides = self.stride.clone();
        let new_offset = self.offset + (self.stride.as_slice().iter().zip(start_index.iter()).map(|(s, i)| s * i).sum::<u64>() as usize);
        Ok(VulkanTensor::<R>{
            shape: R::KnownDims::try_from_slice(new_shape.as_slice()).unwrap(),
            stride: R::KnownDims::try_from_slice(new_strides.as_slice()).unwrap(),
            dtype: self.dtype,
            suballocation: self.suballocation.clone(),
            buffer: self.buffer.clone(),
            offset: new_offset
        })
    }

    pub fn transpose(&self, axes: &[usize]) -> Result<Self, VulkanError> {
        let new_shape = axes.iter().map(|&i| self.shape[i]).collect::<Vec<_>>();
        let new_strides = axes.iter().map(|&i| self.stride[i]).collect::<Vec<_>>();
        Ok(VulkanTensor::<R>{
            shape: R::KnownDims::try_from_slice(new_shape.as_slice()).unwrap(),
            stride: R::KnownDims::try_from_slice(new_strides.as_slice()).unwrap(),
            dtype: self.dtype,
            suballocation: self.suballocation.clone(),
            buffer: self.buffer.clone(),
            offset: self.offset
        })
    }
}