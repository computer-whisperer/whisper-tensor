use std::collections::HashMap;
use pyo3::{Py, PyAny, PyErr, Python};
use pyo3::prelude::PyAnyMethods;
use pyo3::types::{PyDict, PyNone};
use crate::dtype::DType;
use crate::ndarray_backend::NDArrayNumericTensor;
use crate::numeric_tensor::{NumericTensor, NumericTensorError};
use crate::onnx;

#[derive(Debug)]
pub struct ONNXReferenceTensor {
    value: Py<PyAny>,
}

impl Clone for ONNXReferenceTensor {
    fn clone(&self) -> Self {
        Python::with_gil(|py| {
            let value = self.value.bind(py).clone();
            ONNXReferenceTensor {
                value: value.unbind(),
            }
        })
    }
}

impl ONNXReferenceTensor {
    
    pub fn dtype(&self) -> DType {
        todo!()
    }
    
    pub fn rank(&self) -> usize {
        self.shape().len()
    }
    
    pub fn shape(&self) -> Vec<usize> {
        Python::with_gil(|py| {
            let value = self.value.bind(py);
            value.getattr("shape").unwrap().extract().unwrap()
        })
    }
}

impl TryFrom<&ONNXReferenceTensor> for NDArrayNumericTensor {
    type Error = NumericTensorError;

    fn try_from(value: &ONNXReferenceTensor) -> Result<Self, Self::Error> {
        let out = Python::with_gil(|py| {
            let value = value.value.bind(py);
            let v_flat = value.call_method0("flatten")?;
            let v_flat = v_flat.call_method0("tolist")?;
            let v_flat = v_flat.extract::<Vec<f32>>()?;
            let shape = value.getattr("shape")?;
            let shape = shape.extract::<Vec<usize>>()?;
            Ok(NDArrayNumericTensor::from_vec_shape(v_flat, &shape))
        }).map_err(|e| ONNXReferenceError::PyErr(e))?;
        Ok(out?)
    }
}

impl TryFrom<ONNXReferenceTensor> for NDArrayNumericTensor {
    type Error = NumericTensorError;
    
    fn try_from(value: ONNXReferenceTensor) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl TryFrom<&NDArrayNumericTensor> for ONNXReferenceTensor {
    type Error = ONNXReferenceError;
    fn try_from(input: &NDArrayNumericTensor) -> Result<Self, Self::Error> {
        Python::with_gil(|py| {
            let np = py.import("numpy")?;
            let np_array = match &input {
                NDArrayNumericTensor::F32(x) => {np.call_method("array", (x.flatten().to_vec(),), None)?},
                NDArrayNumericTensor::F64(x) => {np.call_method("array", (x.flatten().to_vec(),), None)?},
                NDArrayNumericTensor::U32(x) => {np.call_method("array", (x.flatten().to_vec(),), None)?}
                NDArrayNumericTensor::I32(x) => {np.call_method("array", (x.flatten().to_vec(),), None)?}
                NDArrayNumericTensor::U64(x) => {np.call_method("array", (x.flatten().to_vec(),), None)?}
                NDArrayNumericTensor::I64(x) => {np.call_method("array", (x.flatten().to_vec(),), None)?}
                _ => {Err(ONNXReferenceError::UnsupportedDType)?}
            };
            let np_array = np_array.call_method1("reshape", (input.shape(),))?;

            Ok(ONNXReferenceTensor{
                value: np_array.unbind()
            })
        }).map_err(|e| ONNXReferenceError::PyErr(e))
    }
}

impl TryFrom<NDArrayNumericTensor> for ONNXReferenceTensor {
    type Error = ONNXReferenceError;
    fn try_from(value: NDArrayNumericTensor) -> Result<Self, Self::Error> {
        Self::try_from(&value)
    }
}

impl TryFrom<NumericTensor> for ONNXReferenceTensor {
    type Error = NumericTensorError;
    fn try_from(value: NumericTensor) -> Result<Self, Self::Error> {
        Ok(match value {
            NumericTensor::NDArray(x) => ONNXReferenceTensor::try_from(x)?,
            _ => Self::try_from(NDArrayNumericTensor::try_from(value)?)?
        })
    }
}

impl TryFrom<&NumericTensor> for ONNXReferenceTensor {
    type Error = NumericTensorError;
    fn try_from(value: &NumericTensor) -> Result<Self, Self::Error> {
        Ok(match value {
            NumericTensor::NDArray(x) => ONNXReferenceTensor::try_from(x)?,
            _ => Self::try_from(NDArrayNumericTensor::try_from(value)?)?
        })
    }
}

impl From<ONNXReferenceTensor> for NumericTensor {
    fn from(value: ONNXReferenceTensor) -> Self {
        Self::ONNXReference(value)
    }
}


pub struct ONNXReferenceBackend {
    evaluator: Py<PyAny>,
}

#[derive(Debug, thiserror::Error)]
pub enum ONNXReferenceError {
    #[error(transparent)]
    PyErr(#[from] anyhow::Error),
    #[error("Unsupported dtype")]
    UnsupportedDType
}

impl ONNXReferenceBackend {
    pub fn new(onnx_bytes: &[u8]) -> Result<Self, ONNXReferenceError> {

        let evaluator = Python::with_gil(|py| {
            //let locals = [("os", py.import("os")?), ("sys", py.import("sys")?)].into_py_dict(py)?;
            //py.eval(c_str!("sys.path.append(\"libs/onnx\")"), None, Some(&locals))?;
            let onnx = py.import("onnx")?;
            let onnx_reference = py.import("onnx.reference")?;
            let onnx_checker = py.import("onnx.checker")?;
            let model = onnx.call_method1("load_model_from_string", (onnx_bytes, "protobuf"))?;
            let onnx_external_data_helper = py.import("onnx.external_data_helper")?;
            onnx_external_data_helper.call_method1("load_external_data_for_model", (model.clone(), "."))?;
            onnx_checker.call_method1("check_model", (model.clone(),))?;
            let evaluator = onnx_reference.call_method1("ReferenceEvaluator", (model,))?;
            Ok::<_, PyErr>(evaluator.unbind())
        }).map_err(|e| ONNXReferenceError::PyErr(e.into()))?;

        Ok(ONNXReferenceBackend {
            evaluator
        })
    }
    
    pub fn run(&self, inputs: HashMap<String, ONNXReferenceTensor>) -> Result<HashMap<String, ONNXReferenceTensor>, ONNXReferenceError> {
        Python::with_gil(|py| {
            let evaluator = self.evaluator.bind(py);
            let py_inputs = PyDict::new(py);
            for (key, value) in inputs {
                py_inputs.set_item(key, value.value.bind(py))?;
            }
            let res = evaluator.call_method1("run", (PyNone::get(py), py_inputs))?;
            let output_names: Vec<String> = evaluator.getattr("output_names")?.extract()?;
            let mut outputs = HashMap::new();
            for (i, name) in output_names.iter().enumerate() {
                let value = res.get_item(i)?;
                outputs.insert(name.clone(), ONNXReferenceTensor{value: value.unbind()});
            }
            Ok(outputs)
        }).map_err(|e| ONNXReferenceError::PyErr(e))
    }
    
    pub fn get_input_tensor_info(&self) -> Result<HashMap<String, (DType, Vec<Option<u64>>)>, ONNXReferenceError> {
        Python::with_gil(|py| {
            let mut output = HashMap::new();
            let evaluator = self.evaluator.bind(py);
            let input_names: Vec<String> = evaluator.getattr("input_names")?.extract()?;
            let input_types= evaluator.getattr("input_types")?;
            for (idx, name) in input_names.iter().enumerate() {
                let tensor_type = input_types.get_item(idx)?.getattr("tensor_type")?;
                let dtype: i32 = tensor_type.getattr("elem_type")?.extract()?;
                let dtype = DType::try_from(onnx::tensor_proto::DataType::try_from(dtype)?)?;
                let shape = tensor_type.getattr("shape")?.getattr("dim")?;
                let shape: Vec<Option<u64>> = shape.try_iter()?.map(|x| None).collect();
                output.insert(name.clone(), (dtype, shape));
            }
            Ok(output)
        }).map_err(|e| ONNXReferenceError::PyErr(e))
    }
}

