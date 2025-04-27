use std::collections::{HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use crate::{onnx, Error};
use crate::node::{Node, SingleOutputNode};
use crate::onnx::{TensorProto, ValueInfoProto};
use crate::weights::WeightExternalOutputManager;
use core::fmt;

#[derive(Clone, Debug)]
pub struct Dimension {
    pub value: Option<usize>,
    pub name: Option<String>,
    pub denotation: Option<String>
}

impl Dimension {
    pub fn new(value: Option<usize>, name: Option<String>, denotation: Option<String>) -> Arc<Self> {
        Arc::new(Dimension { value, name, denotation })
    }

    pub fn resolve(&self) -> Result<usize, Error> {
        self.value.ok_or_else(|| Error::UnresolvedDimensionError)
    }
}

impl From<&Dimension> for onnx::tensor_shape_proto::Dimension {
    fn from(value: &Dimension) -> Self {
        Self {
            value: match value.value {
                Some(value) => Some(onnx::tensor_shape_proto::dimension::Value::DimValue(value as i64)),
                None => match &value.name {
                    None => None,
                    Some(name) => Some(onnx::tensor_shape_proto::dimension::Value::DimParam(name.clone()))
                }
            },
            denotation: value.denotation.clone().unwrap_or_default()
        }
    }
}

impl From<usize> for Dimension {
    fn from(value: usize) -> Self {
        Self { value: Some(value), name: None, denotation: None }
    }
}

impl PartialEq for &Dimension {
    fn eq(&self, other: &Self) -> bool {
        core::ptr::eq(*self, *other) || if let (Some(a), Some(b)) = (self.value, other.value) {a == b} else {false}
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(value) = self.value {
            write!(f, "{}", value)
        }
        else if let Some(name) = &self.name {
            write!(f, "{}", name)
        }
        else {
            write!(f, "?")
        }
    }
}

#[derive(Clone, Debug)]
pub struct Shape {
    pub dims: Vec<Arc<Dimension>>
}

impl Shape {
    pub fn new(dims: Vec<Arc<Dimension>>) -> Self {
        Self { dims }
    }

    pub fn resolve(&self) -> Result<Vec<usize>, Error> {
        let mut res = vec![];
        for dim in &self.dims {
            res.push(dim.resolve()?);
        }
        Ok(res)
    }

    pub fn transpose(&self) -> Self {
        Self {
            dims: self.dims.iter().rev().cloned().collect()
        }
    }

    pub fn dim(&self, index: isize) -> &Arc<Dimension> {
        let rank = self.rank();
        let index = if index < 0 {
            rank - (-index) as usize
        }
        else {
            index as usize
        };
        &self.dims[index]
    }
    
    pub fn unsqueeze(&self, axis: isize) -> Self {
        let rank = self.rank();
        let axis = if axis < 0 {
            rank - (-axis) as usize
        }
        else {
            axis as usize
        };
        let mut new_dims = self.dims.clone();
        new_dims.insert(axis, Dimension::new(Some(1), None, None));
        Self::new(new_dims)
    }
    
    pub fn num_elements(&self) -> Result<usize, Error> {
        let mut v = 1;
        for dim in &self.dims {
            v *= dim.resolve()?;
        }
        Ok(v)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.dims.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("x"))
    }
}

impl PartialEq for Shape {
    fn eq(&self, other: &Self) -> bool {
        self.dims.len() == other.dims.len() && self.dims.iter().zip(other.dims.iter()).all(|(a, b)| a.as_ref() == b.as_ref())
    }
}

impl Shape {
    pub fn rank(&self) -> usize {
        self.dims.len()
    }
}

impl From<&Shape> for onnx::TensorShapeProto {
    fn from(value: &Shape) -> Self {
        Self {
            dim: value.dims.iter().map(|x| x.as_ref().into()).collect()
        }
    }
}

impl From<Shape> for onnx::TensorShapeProto {
    fn from(value: Shape) -> Self {
        Self {
            dim: value.dims.iter().map(|x| x.as_ref().into()).collect()
        }
    }
}

impl From<&candle_core::Shape> for Shape {
    fn from(value: &candle_core::Shape) -> Self {
        Shape { dims: value.clone().dims().into_iter().map(|x| Dimension::new(Some(*x), None, None)).collect() }
    }
}

impl From<candle_core::Shape> for Shape {
    fn from(value: candle_core::Shape) -> Self {
        Shape { dims: value.clone().dims().into_iter().map(|x| Dimension::new(Some(*x), None, None)).collect() }
    }
}

impl core::ops::Index<usize> for Shape {
    type Output = Arc<Dimension>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.dims[index]
    }
}

impl <T: Clone> From<&[T]> for Shape
where
    Dimension: From<T>
{
    fn from(value: &[T]) -> Self {
        Shape { dims: value.iter().map(|x| Arc::new(Dimension::from(x.clone()))).collect() }
    }
}

impl <T> From<Vec<T>> for Shape
where
    Dimension: From<T>
{
    fn from(value: Vec<T>) -> Self {
        Shape { dims: value.into_iter().map(|x| Arc::new(Dimension::from(x))).collect() }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum DType {
    F32,
    F16,
    BF16,
    U16,
    I32,
    I64
}

impl DType {
    pub fn from_safetensors(dtype: safetensors::Dtype) -> Result<Self, Error> {
        match dtype {
            safetensors::Dtype::F32 => Ok(DType::F32),
            safetensors::Dtype::F16 => Ok(DType::F16),
            safetensors::Dtype::BF16 => Ok(DType::BF16),
            safetensors::Dtype::U16 => Ok(DType::U16),
            safetensors::Dtype::I32 => Ok(DType::I32),
            _ => Err(Error::UnsupportedDTypeError)
        }
    }
    
    pub fn from_candle(dtype: candle_core::DType) -> Result<Self, Error> {
        match dtype {
            candle_core::DType::F32 => Ok(DType::F32),
            candle_core::DType::BF16 => Ok(DType::BF16),
            candle_core::DType::F16 => Ok(DType::F16),
            _ => Err(Error::UnsupportedDTypeError)
        }
    }
}

impl From<DType> for onnx::tensor_proto::DataType {
    fn from(value: DType) -> Self {
        match value {
            DType::F32 => onnx::tensor_proto::DataType::Float,
            DType::F16 => onnx::tensor_proto::DataType::Float16,
            DType::BF16 => onnx::tensor_proto::DataType::Bfloat16,
            DType::U16 => onnx::tensor_proto::DataType::Uint16,
            DType::I32 => onnx::tensor_proto::DataType::Int32,
            DType::I64 => onnx::tensor_proto::DataType::Int64,
        }
    }
}

impl core::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub trait Tensor  {
    fn dtype(&self) -> DType;
    fn shape(&self) -> &Shape;
    fn rank(&self) -> usize {
        self.shape().rank()
    }
    fn to_value_info_proto(&self, name: String) -> ValueInfoProto {
        ValueInfoProto{
            name,
            r#type: Some(
                onnx::TypeProto{
                    value: Some(
                        onnx::type_proto::Value::TensorType(onnx::type_proto::Tensor {
                            elem_type: onnx::tensor_proto::DataType::from(self.dtype()) as i32,
                            shape: Some(self.shape().into())
                        })
                    ),
                    denotation: "TENSOR".to_string()
                }
            ),
            .. Default::default()
        }
    }

    fn store_data_bin(&mut self, _output_buffer: &mut Vec<u8>) {}

    fn get_nodes<'a>(&'a self, _table: &mut HashSet<&'a dyn Node>) {}

    fn get_tensors<'a>(&'a self, table: &mut HashSet<&'a dyn Tensor>) where Self: Sized {
        let self_dyn = self as &dyn Tensor;
        if table.contains(&self_dyn) {
            table.insert(self as &dyn Tensor);
            self.get_sub_tensors(table);
        }
    }

    fn get_sub_tensors<'a>(&'a self, _table: &mut HashSet<&'a dyn Tensor>) {}

    fn gather_weights<'a>(&'a self, _manager: &mut dyn WeightExternalOutputManager<'a>) {}

    fn get_initializer<'a>(&'a self, _name: String, _manager: &mut dyn WeightExternalOutputManager<'a>) -> Result<Option<onnx::TensorProto>, Error> {
        Ok(None)
    }

    fn get_name(&self) -> Option<&str> {
        None
    }
    
    fn resolve_data(&self) -> Option<TensorData> {
        None
    }
    
    fn is_input(&self) -> bool;
}

impl<'a> PartialEq for &'a dyn Tensor{
    fn eq(&self, other:&Self) -> bool{
        std::ptr::addr_eq(*self, *other)
    }
}

impl<'a> Eq for &'a dyn Tensor{}

impl<'a> Hash for &'a dyn Tensor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let a: *const _ = *self;
        let address: *const u8 = a.cast();
        state.write_usize(address.addr());
    }
}


impl <T: SingleOutputNode> Tensor for T {
    fn dtype(&self) -> DType {
        self.get_output_dtype()
    }

    fn shape(&self) -> &Shape {
        self.get_output_shape()
    }
    
    fn resolve_data(&self) -> Option<TensorData> {
        self.resolve_output_data()
    }

    fn get_nodes<'a>(&'a self, table: &mut HashSet<&'a dyn Node>) {
        <Self as Node>::get_nodes(self, table)
    }

    fn get_sub_tensors<'a>(&'a self, table: &mut HashSet<&'a dyn Tensor>) {
        <Self as Node>::get_tensors(self, table);
    }
    
    fn is_input(&self) -> bool {
        false
    }
}

pub struct InputTensor {
    data_type: DType,
    name: String,
    shape: Shape
}

impl InputTensor {
    pub fn new(name: String, data_type: DType, shape: Shape) -> Arc<Self> {
        Arc::new(Self {name, data_type, shape })
    }
}

impl Tensor for InputTensor {
    fn dtype(&self) -> DType {
        self.data_type
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn get_name(&self) -> Option<&str> {
        Some(&self.name)
    }
    
    fn is_input(&self) -> bool {
        true
    }
}

pub struct InputTensorInitialized {
    name: String,
    initial_value: TensorData
}

impl InputTensorInitialized {
    pub fn new(name: String, initial_value: TensorData) -> Arc<Self> {
        Arc::new(Self {name, initial_value})
    }
}

impl Tensor for InputTensorInitialized {
    fn dtype(&self) -> DType {
        self.initial_value.dtype()
    }

    fn shape(&self) -> &Shape {
        &self.initial_value.shape()
    }

    fn get_name(&self) -> Option<&str> {
        Some(&self.name)
    }

    fn is_input(&self) -> bool {
        true
    }
    
    fn get_initializer<'a>(&'a self, name: String, _manager: &mut dyn WeightExternalOutputManager<'a>) -> Result<Option<TensorProto>, Error> {
        Ok(Some(self.initial_value.to_tensor_data_proto(Some(name))?))
    }
}

pub struct StubTensor {
    shape: Shape,
    dtype: DType
}

impl StubTensor {
    pub(crate) fn new(shape: Shape, dtype: DType) -> Arc<Self> {
        Arc::new(Self {shape, dtype})
    }
}

impl Tensor for StubTensor {
    fn dtype(&self) -> DType {
        self.dtype
    }
    fn shape(&self) -> &Shape {
        &self.shape
    }
    fn is_input(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone)]
pub enum TensorDataValue {
    F32(Vec<f32>),
    BF16(Vec<half::bf16>),
    F16(Vec<half::f16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
}

impl TensorDataValue {
    pub fn len(&self) -> usize {
        match self {
            TensorDataValue::F32(v) => v.len(),
            TensorDataValue::I32(v) => v.len(),
            TensorDataValue::BF16(v) => v.len(),
            TensorDataValue::F16(v) => v.len(),
            TensorDataValue::I64(v) => v.len(),
        }
    }
    
    pub fn dtype(&self) -> DType {
        match self {
            TensorDataValue::F32(_) => DType::F32,
            TensorDataValue::BF16(_) => DType::BF16,
            TensorDataValue::F16(_) => DType::F16,
            TensorDataValue::I32(_) => DType::I32,
            TensorDataValue::I64(_) => DType::I64,
        }
    }
    
    pub fn get_raw_encoding(&self) -> Vec<u8> {
        match self {
            TensorDataValue::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            TensorDataValue::BF16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            TensorDataValue::F16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            TensorDataValue::I32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            TensorDataValue::I64(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }
    
    pub fn from_raw_encoding(dtype: DType, data: &[u8]) -> Result<Self, Error> {
        match dtype {
            DType::F32 => {
                let mut v = Vec::new();
                for i in 0..data.len() / 4 {
                    v.push(f32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap()));
                }
                Ok(TensorDataValue::F32(v))
            }
            DType::F16 => {
                let mut v = Vec::new();
                for i in 0..data.len() / 2 {
                    v.push(half::f16::from_le_bytes(data[i * 2..i * 2 + 2].try_into().unwrap()));
                }
                Ok(TensorDataValue::F16(v))
            }
            DType::BF16 => {
                let mut v = Vec::new();
                for i in 0..data.len() / 2 {
                    v.push(half::bf16::from_le_bytes(data[i * 2..i * 2 + 2].try_into().unwrap()));
                }
                Ok(TensorDataValue::BF16(v))
            }
            _ => Err(Error::UnsupportedDTypeError)
        }
    }
}

impl From<Vec<f32>> for TensorDataValue {
    fn from(value: Vec<f32>) -> Self {
        TensorDataValue::F32(value)
    }
}

impl From<Vec<half::bf16>> for TensorDataValue {
    fn from(value: Vec<half::bf16>) -> Self {
        TensorDataValue::BF16(value)
    }
}

impl From<Vec<half::f16>> for TensorDataValue {
    fn from(value: Vec<half::f16>) -> Self {
        TensorDataValue::F16(value)
    }
}

impl From<Vec<i32>> for TensorDataValue {
    fn from(value: Vec<i32>) -> Self {
        TensorDataValue::I32(value)
    }
}

impl From<Vec<i64>> for TensorDataValue {
    fn from(value: Vec<i64>) -> Self {
        TensorDataValue::I64(value)
    }
}

#[derive(Debug, Clone)]
pub struct TensorData {
    value: TensorDataValue,
    shape: Shape
}

impl TensorData {
    pub fn new(value: TensorDataValue, shape: Shape) -> Result<Self, Error> {
        if shape.num_elements()? != value.len() {
            return Err(Error::InvalidInputError);
        }
        Ok(Self { value, shape })
    }
    
    pub fn fill<T>(shape: Shape, value: T) -> Result<Self, Error> 
    where
        T: Copy,
        TensorDataValue: From<Vec<T>>,
    {
        let num_elements = shape.num_elements()?;
        let data = vec![value; num_elements];
        Ok(Self::new(TensorDataValue::from(data), shape)?)
    }
    
    pub fn zeros(shape: Shape, dtype: DType) -> Result<Self, Error> {
        match dtype {
            DType::F32 => Self::fill(shape, 0.0f32),
            DType::F16 => Self::fill(shape, half::f16::from_f32(0.0f32)),
            DType::BF16 => Self::fill(shape, half::bf16::from_f32(0.0f32)),
            DType::I32 => Self::fill(shape, 0i32),
            DType::I64 => Self::fill(shape, 0i64),
            _ => Err(Error::UnsupportedDTypeError),
        }
    }
    
    pub fn dtype(&self) -> DType {
        self.value.dtype()
    }
    
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    
    pub fn to_int_vec(&self) -> Result<Vec<i64>, Error> {
        match &self.value {
            TensorDataValue::I32(x) => Ok(x.iter().map(|x| *x as i64).collect()),
            TensorDataValue::I64(x) => Ok(x.clone()),
            _ => Err(Error::UnsupportedDTypeError),
        }
    }
    
    pub fn to_tensor_data_proto(&self, name: Option<String>) -> Result<TensorProto, Error> {
        Ok(TensorProto{
            name: name.unwrap_or_default(),
            data_type: (onnx::tensor_proto::DataType::from(self.value.dtype()) as i32),
            dims: self.shape.resolve()?.iter().map(|x| *x as i64).collect(),
            raw_data: self.value.get_raw_encoding(),
            ..Default::default()
        })
    }
    
    pub fn from_candle_tensor(tensor: candle_core::Tensor) -> Result<Self, Error> {
        let shape = Shape::from(tensor.shape());
        let value = match tensor.dtype() {
            candle_core::DType::F32 => {
                let f_vec: Vec<f32> = tensor.flatten_all().unwrap().to_vec1().map_err(|x| Error::CandleCoreError(x))?;
                TensorDataValue::F32(f_vec)
            }
            candle_core::DType::BF16 => {
                let f_vec: Vec<half::bf16> = tensor.flatten_all().unwrap().to_vec1().map_err(|x| Error::CandleCoreError(x))?;
                TensorDataValue::BF16(f_vec)
            }
            candle_core::DType::F16 => {
                let f_vec: Vec<half::f16> = tensor.flatten_all().unwrap().to_vec1().map_err(|x| Error::CandleCoreError(x))?;
                TensorDataValue::F16(f_vec)
            }
            _ => return Err(Error::UnsupportedDTypeError),
        };
        Ok(Self{
            shape,
            value
        })
    }
    
    pub(crate) fn to_raw_encoding(&self) -> Vec<u8> {
        self.value.get_raw_encoding()
    }
    
    pub fn from_safetensors_view(tensor: safetensors::tensor::TensorView) -> Result<Self, Error> {
        let dtype = DType::from_safetensors(tensor.dtype())?;
        let shape = Shape::from(tensor.shape());
        let value = TensorDataValue::from_raw_encoding(dtype, tensor.data())?;
        if shape.num_elements()? != value.len() {
            Err(Error::OtherError)?;
        }
        Ok(Self{
            shape,
            value
        })
    }
}