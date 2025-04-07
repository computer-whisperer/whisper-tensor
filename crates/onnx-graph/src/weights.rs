use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use crate::{onnx, Error};
use crate::onnx::{TensorProto};
use crate::tensor::{DType, Shape, Tensor};

pub trait WeightExternalOutputManager<'a> {
    fn write_pth_tensor_data(&mut self, graph_tensor: &'a dyn Tensor, tensor_info: candle_core::pickle::TensorInfo, tensors: Arc<candle_core::pickle::PthTensors>) -> Result<(), Error>;
    fn get_initializer(&mut self, graph_tensor: &'a dyn Tensor, tensor_name: String) -> Result<Option<TensorProto>, Error>;
}

pub struct EmbeddedOutputManager<'a> {
    tensor_data_map: HashMap<&'a dyn Tensor, (Vec<f32>, Vec<i32>)>,
}

impl<'a> EmbeddedOutputManager<'a> {
    pub fn new() -> Self {
        Self {
            tensor_data_map: HashMap::new(),
        }
    }
}
impl <'a> WeightExternalOutputManager<'a> for EmbeddedOutputManager<'a> {
    fn write_pth_tensor_data(&mut self, graph_tensor: &'a dyn Tensor, tensor_info: candle_core::pickle::TensorInfo, tensors: Arc<candle_core::pickle::PthTensors>) -> Result<(), Error> {
        let data = tensors.get(&tensor_info.name).unwrap().unwrap();

        let (float_data, int32_data) = match tensor_info.dtype {
            candle_core::DType::F32 => {
                (data.flatten_all().unwrap().to_vec1().unwrap(), vec![])
            }
            candle_core::DType::F16 => {
                let f_data = data.flatten_all().unwrap().to_vec1::<half::f16>().unwrap();
                let i_data = f_data.iter().map(|x| x.to_bits() as i32).collect();
                (vec![], i_data)
            }
            candle_core::DType::BF16 => {
                let f_data = data.flatten_all().unwrap().to_vec1::<half::bf16>().unwrap();
                let i_data = f_data.iter().map(|x| x.to_bits() as i32).collect();
                (vec![], i_data)
            }
            _ => {
                return Err(Error::UnsupportedDtypeError)
            }
        };
        self.tensor_data_map.insert(graph_tensor, (float_data, int32_data));
        Ok(())
    }

    fn get_initializer(&mut self, graph_tensor: &'a dyn Tensor, tensor_name: String) -> Result<Option<TensorProto>, Error> {

        if let Some((float_data, int32_data)) = self.tensor_data_map.remove(&graph_tensor) {
            Ok(Some(TensorProto {
                name: tensor_name,
                data_type: onnx::tensor_proto::DataType::from(graph_tensor.dtype()) as i32,
                dims: graph_tensor.shape().resolve()?.iter().map(|x| *x as i64).collect(),
                float_data,
                int32_data,
                .. Default::default()
            }))
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

    pub fn finalize_tensor_data(&mut self) {
        self.output.flush().unwrap();
        self.output_data = Some(vec![
            onnx::StringStringEntryProto {
                key: "location".to_string(),
                value: self.output_path.file_name().unwrap().to_str().unwrap().to_string(),
            }
        ]);

    }
}
impl<'a> WeightExternalOutputManager<'a> for BinOutputManager<'a> {
    fn write_pth_tensor_data(&mut self, graph_tensor: &'a dyn Tensor, tensor_info: candle_core::pickle::TensorInfo, tensors: Arc<candle_core::pickle::PthTensors>) -> Result<(), Error> {
        let pth_data = tensors.get(&tensor_info.name).unwrap().unwrap();

        self.tensor_data_map.insert(graph_tensor, match tensor_info.dtype {
            candle_core::DType::F32 => {
                let f_data: Vec<f32> = pth_data.flatten_all().unwrap().to_vec1().unwrap();
                let data: Vec<u8> = f_data.iter().map(|x| x.to_le_bytes()).flatten().collect();
                let offset = self.output.metadata().unwrap().len() as usize;
                self.output.write_all(&data).map_err(|e| Error::IoError(e))?;
                (offset, data.len())
            }
            candle_core::DType::F16 => {
                let f_data: Vec<half::f16> = pth_data.flatten_all().unwrap().to_vec1().unwrap();
                let data: Vec<u8> = f_data.iter().map(|x| x.to_le_bytes()).flatten().collect();
                let offset = self.output.metadata().unwrap().len() as usize;
                self.output.write_all(&data).map_err(|e| Error::IoError(e))?;
                (offset, data.len())
            }
            candle_core::DType::BF16 => {
                let f_data: Vec<half::bf16> = pth_data.flatten_all().unwrap().to_vec1().unwrap();
                let data: Vec<u8> = f_data.iter().map(|x| x.to_le_bytes()).flatten().collect();
                let offset = self.output.metadata().unwrap().len() as usize;
                self.output.write_all(&data).map_err(|e| Error::IoError(e))?;
                (offset, data.len())
            }
            _ => {
                return Err(Error::UnsupportedDtypeError)
            }
        });
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
            _ => return Err(Error::UnsupportedDtypeError)
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
}

pub trait WeightManager {
    fn prefix(&self, name: &str) -> Self;
    fn get_tensor(&self, name: &str) -> Result<Arc<dyn Tensor>, Error>;
    fn get_prefix_tail(&self) -> Option<&str>;
    fn get_prefix(&self) -> Option<&str>;
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
        let tensor_info = tensor_infos.get(&name).ok_or(Error::NoSuchTensorError)?;
        Ok(PthTensor::new(tensor_info.clone(), self.pth_tensors.clone())?)
    }
    fn get_prefix_tail(&self) -> Option<&str> {
        self.prefix_tail.as_deref()
    }
    fn get_prefix(&self) -> Option<&str> {
        self.prefix.as_deref()
    }
}