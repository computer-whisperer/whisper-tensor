use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use memmap2::Mmap;
use safetensors::SafeTensors;
use safetensors::tensor::{Metadata, TensorInfo};
use crate::{onnx, Error};
use crate::onnx::{TensorProto};
use crate::tensor::{DType, Shape, Tensor, TensorData};

pub trait WeightExternalOutputManager<'a> {
    fn write_pth_tensor_data(&mut self, graph_tensor: &'a dyn Tensor, tensor_info: candle_core::pickle::TensorInfo, tensors: Arc<candle_core::pickle::PthTensors>) -> Result<(), Error> {
        let candle_tensor = tensors.get(&tensor_info.name).unwrap().unwrap();
        self.write_tensor_data(graph_tensor, TensorData::from_candle_tensor(candle_tensor)?)
    }
    fn write_safetensors_tensor_data(&mut self, graph_tensor: &'a SafetensorsTensor) -> Result<(), Error> {
        let file_index = graph_tensor.file_index;
        let name = &graph_tensor.name;
        let st = SafeTensors::deserialize(&graph_tensor.inner.safetensors_files[file_index]).map_err(|x| Error::SafeTensorError(x))?;
        let data = st.tensor(name).map_err(|x| Error::SafeTensorError(x))?;
        self.write_tensor_data(graph_tensor, TensorData::from_safetensors_view(data)?)
    }
    fn write_tensor_data(&mut self, graph_tensor: &'a dyn Tensor, data: TensorData) -> Result<(), Error>;
    fn get_initializer(&mut self, graph_tensor: &'a dyn Tensor, tensor_name: String) -> Result<Option<TensorProto>, Error>;
    fn finalize_tensor_data(&mut self) {}
}

pub struct NullOutputManager {}

impl NullOutputManager {
    pub(crate) fn new() -> Self {Self {}}
}

impl <'a> WeightExternalOutputManager<'a> for NullOutputManager {
    fn write_pth_tensor_data(&mut self, _graph_tensor: &'a dyn Tensor, _tensor_info: candle_core::pickle::TensorInfo, _tensors: Arc<candle_core::pickle::PthTensors>) -> Result<(), Error> {
        Ok(())
    }
    fn write_safetensors_tensor_data(&mut self, _graph_tensor: &'a SafetensorsTensor) -> Result<(), Error> {
        Ok(())
    }
    fn write_tensor_data(&mut self, _graph_tensor: &'a dyn Tensor, _data: TensorData) -> Result<(), Error> {
        Ok(())
    }

    fn get_initializer(&mut self, _graph_tensor: &'a dyn Tensor, _tensor_name: String) -> Result<Option<TensorProto>, Error> {
        Ok(None)
    }
}

pub struct EmbeddedOutputManager<'a> {
    tensor_data_map: HashMap<&'a dyn Tensor, TensorData>,
}

impl<'a> EmbeddedOutputManager<'a> {
    pub fn new() -> Self {
        Self {
            tensor_data_map: HashMap::new(),
        }
    }
}
impl <'a> WeightExternalOutputManager<'a> for EmbeddedOutputManager<'a> {
    fn write_tensor_data(&mut self, graph_tensor: &'a dyn Tensor, tensor_data: TensorData) -> Result<(), Error> {
        self.tensor_data_map.insert(graph_tensor, tensor_data);
        Ok(())
    }

    fn get_initializer(&mut self, graph_tensor: &'a dyn Tensor, tensor_name: String) -> Result<Option<TensorProto>, Error> {
        if let Some(tensor_data) = self.tensor_data_map.remove(&graph_tensor) {
            Ok(Some(tensor_data.to_tensor_data_proto(Some(tensor_name))?))
        }
        else {
            Ok(None)
        }
    }
}


pub struct BinOutputManager<'a> {
    output: File,
    output_path: PathBuf,
    tensor_data_map: HashMap<&'a dyn Tensor, (usize, usize)>,
    output_data: Option<Vec<onnx::StringStringEntryProto>>
}

impl<'a> BinOutputManager<'a> {
    pub fn new(output_location: &Path) -> Self {
        let output = File::create(output_location).unwrap();
        Self {
            output,
            output_path: output_location.to_path_buf(),
            tensor_data_map: HashMap::new(),
            output_data: None
        }
    }
}
impl<'a> WeightExternalOutputManager<'a> for BinOutputManager<'a> {

    fn write_tensor_data(&mut self, graph_tensor: &'a dyn Tensor, data: TensorData) -> Result<(), Error> {
        let offset = self.output.metadata().unwrap().len() as usize;
        let data = data.to_raw_encoding();
        self.output.write_all(&data).map_err(|e| Error::IoError(e))?;
        self.tensor_data_map.insert(graph_tensor, (offset, data.len()));
        Ok(())
    }

    fn get_initializer(&mut self, graph_tensor: &'a dyn Tensor, tensor_name: String) -> Result<Option<TensorProto>, Error> {

        if let Some((byte_offset, byte_len)) = self.tensor_data_map.remove(&graph_tensor) {
            let mut external_data = self.output_data.clone().unwrap();
            external_data.extend_from_slice(&[
                onnx::StringStringEntryProto {
                    key: "offset".to_string(),
                    value: format!("{byte_offset}"),
                },
                onnx::StringStringEntryProto {
                    key: "length".to_string(),
                    value: format!("{byte_len}"),
                }
            ]);
            Ok(Some(TensorProto {
                name: tensor_name,
                data_type: onnx::tensor_proto::DataType::from(graph_tensor.dtype()) as i32,
                dims: graph_tensor.shape().resolve()?.iter().map(|x| *x as i64).collect(),
                data_location: onnx::tensor_proto::DataLocation::External as i32,
                external_data,
                .. Default::default()
            }))
        }
        else {
            Ok(None)
        }
    }

    fn finalize_tensor_data(&mut self) {
        self.output.flush().unwrap();
        self.output_data = Some(vec![
            onnx::StringStringEntryProto {
                key: "location".to_string(),
                value: self.output_path.file_name().unwrap().to_str().unwrap().to_string(),
            }
        ]);
    }
}

/*
struct InplaceWeightOutputManager {

}
impl WeightExternalOutputManager for InplaceWeightOutputManager {}
*/

pub struct PthTensor {
    tensor_info: candle_core::pickle::TensorInfo,
    tensors: Arc<candle_core::pickle::PthTensors>,
    data_type: DType,
    shape: Shape
}

impl PthTensor {
    pub fn new(tensor_info: candle_core::pickle::TensorInfo, tensors: Arc<candle_core::pickle::PthTensors>) -> Result<Arc<Self>, Error> {
        let data_type = match tensor_info.dtype {
            candle_core::DType::F32 => DType::F32,
            candle_core::DType::F16 => DType::F16,
            candle_core::DType::BF16 => DType::BF16,
            _ => return Err(Error::UnsupportedDTypeError)
        };
        let shape = Shape::from(tensor_info.layout.shape());
        Ok(Arc::new(Self {
            tensors,
            shape,
            tensor_info,
            data_type
        }))
    }
}

impl Tensor for PthTensor {
    fn dtype(&self) -> DType {
        self.data_type
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn gather_weights<'a>(&'a self, manager: &mut dyn WeightExternalOutputManager<'a>) {
        manager.write_pth_tensor_data(self, self.tensor_info.clone(), self.tensors.clone()).unwrap()
    }

    fn get_initializer<'a>(&'a self, name: String, manager: &mut dyn WeightExternalOutputManager<'a>) -> Result<Option<TensorProto>, Error> {
        manager.get_initializer(self, name)
    }
    
    fn get_name(&self) -> Option<&str> {
        Some(&self.tensor_info.name)
    }
    
    fn is_input(&self) -> bool {
        true
    }
}

pub struct SafeTensor {

}

pub trait WeightManager {
    fn prefix(&self, name: &str) -> Self;
    fn get_tensor(&self, name: &str) -> Result<Arc<dyn Tensor>, Error>;
    fn get_prefix_tail(&self) -> Option<&str>;
    fn get_prefix(&self) -> Option<&str>;
    fn get_tensor_names(&self) -> Vec<String>;
    fn print_weight_list(&self) {
        for name in self.get_tensor_names() {
            println!("{}", name);
        }
    }
}

pub struct PthWeightManager {
    prefix_tail: Option<String>,
    prefix: Option<String>,
    pth_tensors: Arc<candle_core::pickle::PthTensors>
}

impl PthWeightManager {
    pub fn new(pth_tensors: Arc<candle_core::pickle::PthTensors>) -> Self {
        Self {
            prefix_tail: None,
            prefix: None,
            pth_tensors
        }
    }
}

impl WeightManager for PthWeightManager {
    fn prefix(&self, name: &str) -> Self {
        let prefix = Some(if let Some(prefix) = &self.prefix {
            format!("{}.{}", prefix, name)
        } else {
            name.to_string()
        });
        Self {
            prefix_tail: Some(name.to_string()),
            prefix,
            pth_tensors: self.pth_tensors.clone()
        }
    }
    fn get_tensor(&self, name: &str) -> Result<Arc<dyn Tensor>, Error> {
        let name = if let Some(prefix) = &self.prefix {
            format!("{}.{}", prefix, name)
        } else {
            name.to_string()
        };
        let tensor_infos = self.pth_tensors.tensor_infos();
        let tensor_info = tensor_infos.get(&name).ok_or(Error::NoSuchTensorError(name))?;
        Ok(PthTensor::new(tensor_info.clone(), self.pth_tensors.clone())?)
    }
    fn get_prefix_tail(&self) -> Option<&str> {
        self.prefix_tail.as_deref()
    }
    fn get_prefix(&self) -> Option<&str> {
        self.prefix.as_deref()
    }

    fn get_tensor_names(&self) -> Vec<String> {
        self.pth_tensors.tensor_infos().keys().map(|x| x.to_string()).collect()
    }
}

pub struct SafetensorsWeightManagerInner {
    safetensors_files: Vec<Arc<Mmap>>,
    safetensors_metadata: Vec<(usize, Metadata)>
}

impl SafetensorsWeightManagerInner {
    pub fn new(safetensors_files: Vec<Arc<Mmap>>) -> Result<Self, Error> {
        let safetensors_metadata = {
            let mut out = vec![];
            for safetensors_mmap in &safetensors_files {
                let metadata = SafeTensors::read_metadata(&safetensors_mmap).map_err(|x| Error::SafeTensorError(x))?;
                out.push(metadata);
            }
            out
        };
        Ok(Self {
            safetensors_files,
            safetensors_metadata
        })
    }

    pub fn get_tensor_info(&self, name: &str) -> Option<(usize, TensorInfo)> {
        for (i, metadata) in self.safetensors_metadata.iter().enumerate() {
            if let Some(tensor_info) = metadata.1.info(name) {
                return Some((i, tensor_info.clone()));
            }
        }
        None
    }

    fn get_tensor_names(&self) -> Vec<String> {
        let mut out = vec![];
        for metadata in &self.safetensors_metadata {
            let tensors = metadata.1.tensors();
            out.extend(tensors.keys().map(|x| x.clone()));
        }
        out
    }
}

pub struct SafetensorsWeightManager {
    prefix_tail: Option<String>,
    prefix: Option<String>,
    inner: Arc<SafetensorsWeightManagerInner>
}

impl SafetensorsWeightManager {
    pub fn new(safetensors_files: Vec<Arc<Mmap>>) -> Result<Self, Error> {
        Ok(Self {
            prefix_tail: None,
            prefix: None,
            inner: Arc::new(SafetensorsWeightManagerInner::new(safetensors_files)?)
        })
    }
}

impl WeightManager for SafetensorsWeightManager {
    fn prefix(&self, name: &str) -> Self {
        let prefix = Some(if let Some(prefix) = &self.prefix {
            format!("{}.{}", prefix, name)
        } else {
            name.to_string()
        });
        Self {
            prefix_tail: Some(name.to_string()),
            prefix,
            inner: self.inner.clone()
        }
    }
    fn get_tensor(&self, name: &str) -> Result<Arc<dyn Tensor>, Error> {
        let full_name = if let Some(prefix) = &self.prefix {
            format!("{}.{}", prefix, name)
        } else {
            name.to_string()
        };
        Ok(Arc::new(SafetensorsTensor::new(self.inner.clone(), full_name)?))
    }
    fn get_prefix_tail(&self) -> Option<&str> {
        self.prefix_tail.as_deref()
    }
    fn get_prefix(&self) -> Option<&str> {
        self.prefix.as_deref()
    }

    fn get_tensor_names(&self) -> Vec<String> {
        self.inner.get_tensor_names()
    }
}

pub struct SafetensorsTensor {
    name: String,
    inner: Arc<SafetensorsWeightManagerInner>,
    file_index: usize,
    _tensor_info: TensorInfo,
    data_type: DType,
    shape: Shape
}

impl SafetensorsTensor {
    pub fn new(inner: Arc<SafetensorsWeightManagerInner>, name: String) -> Result<Self, Error> {
        let (file_index, tensor_info) = inner.get_tensor_info(&name).ok_or(Error::NoSuchTensorError(name.to_string()))?;
        let data_type = DType::from_safetensors(tensor_info.dtype)?;
        let shape = Shape::from(tensor_info.shape.clone());
        Ok(Self {
            name,
            inner,
            file_index,
            _tensor_info: tensor_info,
            data_type,
            shape
        })
    }
}

impl Tensor for SafetensorsTensor {
    fn dtype(&self) -> DType {
        self.data_type
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }
    fn gather_weights<'a>(&'a self, manager: &mut dyn WeightExternalOutputManager<'a>) {
        manager.write_safetensors_tensor_data(self).unwrap()
    }

    fn get_initializer<'a>(&'a self, name: String, manager: &mut dyn WeightExternalOutputManager<'a>) -> Result<Option<TensorProto>, Error> {
        manager.get_initializer(self, name)
    }

    fn get_name(&self) -> Option<&str> {
        Some(&self.name)
    }

    fn is_input(&self) -> bool {
        true
    }
}