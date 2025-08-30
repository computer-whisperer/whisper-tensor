use crate::backends::ModelLoadedTensorCache;
#[cfg(feature = "vulkan")]
use crate::backends::vulkan_backend::VulkanImmediateExecutor;
use crate::dtype::{DType, DTypeError};
use crate::numeric_tensor::NumericTensor;
use crate::symbolic_graph::observer::SymbolicGraphObserver;
use crate::symbolic_graph::ops::{EvalError, Operation};
use crate::symbolic_graph::tensor_store::TensorStore;
use crate::symbolic_graph::{
    GraphOperation, SymbolicGraph, SymbolicGraphNodePath, SymbolicGraphOperationId,
    SymbolicGraphTensorId, SymbolicGraphTensorPath, check_tensor_matches,
};
use crate::tensor_rank::{DynRank, Rank};
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::time::Instant;

#[derive(Debug)]
#[allow(unused_lifetimes)]
pub enum EvalBackend<'a> {
    #[cfg(feature = "candle")]
    Candle(candle_core::Device),
    NDArray,
    #[cfg(feature = "vulkan")]
    Vulkan(&'a mut VulkanImmediateExecutor),
    #[cfg(feature = "tch")]
    TCH,
    NotUsed(PhantomData<&'a ()>),
}

impl<'a> core::fmt::Display for EvalBackend<'a> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl<'a> EvalBackend<'a> {
    pub fn supports_dtype(&self, dtype: DType) -> bool {
        match self {
            #[cfg(feature = "candle")]
            EvalBackend::Candle(_) => matches!(
                dtype,
                DType::F32
                    | DType::F64
                    | DType::BF16
                    | DType::F16
                    | DType::U32
                    | DType::I64
                    | DType::U8
            ),
            EvalBackend::NDArray => true,
            #[cfg(feature = "vulkan")]
            EvalBackend::Vulkan(_) => !matches!(dtype, DType::STRING),
            #[cfg(feature = "tch")]
            EvalBackend::TCH => matches!(
                dtype,
                DType::F64
                    | DType::F32
                    | DType::BF16
                    | DType::F16
                    | DType::I64
                    | DType::I32
                    | DType::I16
                    | DType::I8
                    | DType::U8
                    | DType::BOOL
            ),
            _ => false,
        }
    }

    pub fn to_native_type(&mut self, tensor: &NumericTensor<DynRank>) -> NumericTensor<DynRank> {
        match self {
            #[cfg(feature = "candle")]
            EvalBackend::Candle(x) => tensor.to_candle(x).unwrap().into(),
            EvalBackend::NDArray => tensor.to_ndarray().unwrap().into(),
            #[cfg(feature = "vulkan")]
            EvalBackend::Vulkan(x) => NumericTensor::Vulkan(tensor.to_vulkan(x).unwrap()),
            #[cfg(feature = "tch")]
            EvalBackend::TCH => NumericTensor::TCH(tensor.to_tch()),
            _ => {
                unimplemented!()
            }
        }
    }

    pub fn is_on_backend<R: Rank>(&self, tensor: &NumericTensor<R>) -> bool {
        match (self, tensor) {
            (EvalBackend::NDArray, NumericTensor::NDArray(_)) => true,
            #[cfg(feature = "candle")]
            (EvalBackend::Candle(_), NumericTensor::Candle(_)) => true,
            #[cfg(feature = "vulkan")]
            (EvalBackend::Vulkan(vk_executor), NumericTensor::Vulkan(vk_tensor)) => {
                vk_executor.context.device == *vk_tensor.get_device()
            }
            #[cfg(feature = "tch")]
            (EvalBackend::TCH, NumericTensor::TCH(_)) => true,
            _ => false,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EvalRuntimeError {
    #[error(transparent)]
    DTypeError(#[from] DTypeError),
    #[error("Unexpected shape: expected {0:?}, got {1:?} in shape {2:?}")]
    UnexpectedDimension(u64, u64, Vec<u64>),
    #[error("Unexpected rank: expected {0}, got {1}")]
    UnexpectedRank(usize, usize),
    #[error("Unexpected dtype: expected {0}, got {1}")]
    UnexpectedDType(DType, DType),
    #[error("Missing input tensor: {0} {1:?} {2:?}")]
    MissingInputTensor(String, Option<DType>, Option<Vec<usize>>),
    #[error("Eval Error: {0:?} {1}")]
    EvalError(Option<String>, EvalError),
}

pub fn run<T: SymbolicGraphObserver>(
    model: &SymbolicGraph,
    tensor_store: &TensorStore,
    loaded_tensor_cache: Option<&mut ModelLoadedTensorCache>,
    eval_backend: &mut EvalBackend,
    observer: &mut T,
    inputs: HashMap<String, NumericTensor<DynRank>>,
) -> Result<HashMap<String, NumericTensor<DynRank>>, EvalRuntimeError> {
    let initialized_tensors = if let Some(tensor_cache) = loaded_tensor_cache {
        model.get_initialized_tensors_cached(tensor_store, tensor_cache, eval_backend)
    } else {
        model.get_initialized_tensors(tensor_store)
    };
    let mut tensor_uses_left: HashMap<SymbolicGraphTensorId, usize> = HashMap::new();
    let mut active_tensors: HashMap<SymbolicGraphTensorId, NumericTensor<DynRank>> = HashMap::new();
    for (tensor_id, tensor) in initialized_tensors {
        // Start with 1 extra use with initialized tensors, assuming they will stay on-device after
        *tensor_uses_left.entry(tensor_id).or_insert(0) += 1;
        active_tensors.insert(tensor_id, tensor);
    }
    let tensors_by_name = model.get_tensors_by_name();
    for (name, tensor) in inputs {
        if let Some(tensor_id) = tensors_by_name.get(&name) {
            observer.on_tensor_assigned(
                &SymbolicGraphTensorPath::Tensor(*tensor_id),
                &tensor,
                eval_backend,
            );
            active_tensors.insert(*tensor_id, tensor);
        }
    }

    for id in model.get_outputs() {
        let val = tensor_uses_left.entry(id).or_insert_with(|| 0);
        *val += 1;
    }
    let ops = model.get_operations();
    for op in ops.values() {
        for id in op.op.get_inputs() {
            let val = tensor_uses_left.entry(id).or_insert_with(|| 0);
            *val += 1;
        }
    }

    let mut tensors_just_created = vec![];

    let ops = model.get_operations();
    let mut remaining_ops_to_complete: HashSet<SymbolicGraphOperationId> =
        ops.keys().copied().collect();
    loop {
        // Pick the next op to use
        let mut best_op_id = None;
        let mut best_op_id_score = None;
        for op_id in &remaining_ops_to_complete {
            let GraphOperation { name: _, op } = ops.get(op_id).unwrap();
            let mut n_dropped_tensor = 0;
            let mut fast_reused_tensors = 0;
            let mut are_inputs_present = true;
            let inputs = op.get_inputs();
            for input in &inputs {
                if tensors_just_created.contains(input) {
                    fast_reused_tensors += 1;
                }
                if let Some(x) = tensor_uses_left.get(input)
                    && *x <= 1
                {
                    n_dropped_tensor += 1;
                }
                if !active_tensors.contains_key(input) {
                    are_inputs_present = false;
                }
            }
            if are_inputs_present {
                let score = n_dropped_tensor + inputs.len() + fast_reused_tensors * 4;
                if let Some(best_score) = best_op_id_score {
                    if score > best_score {
                        best_op_id_score = Some(score);
                        best_op_id = Some(*op_id);
                    }
                } else {
                    best_op_id_score = Some(score);
                    best_op_id = Some(*op_id);
                }
            }
        }

        if let Some(op_id) = best_op_id {
            let GraphOperation { name, op } = ops.get(&op_id).unwrap();
            let input_ids = op.get_inputs();
            let mut input_values = HashMap::new();
            for tensor_id in input_ids {
                if let Some(value) = active_tensors.get(&tensor_id) {
                    // Validate shape and dtype
                    let tensor_info = model.get_tensor_info(tensor_id).unwrap();
                    check_tensor_matches(value, tensor_info)
                        .map_err(|x| EvalRuntimeError::EvalError(name.clone(), x))?;
                    input_values.insert(tensor_id, value.clone());

                    let entry = tensor_uses_left.entry(tensor_id).or_insert_with(|| 0);
                    if *entry > 0 {
                        *entry -= 1;
                    }
                    if *entry == 0 {
                        active_tensors.remove(&tensor_id);
                    }
                } else {
                    // Should not happen!
                    panic!();
                }
            }

            let start_instant = Instant::now();
            let outputs = op
                .eval(eval_backend, &input_values)
                .map_err(|x| EvalRuntimeError::EvalError(name.clone(), x))?;
            let end_instant = Instant::now();
            observer.on_op_executed(
                &SymbolicGraphNodePath::Node(op_id),
                start_instant,
                end_instant,
                eval_backend,
            );
            let mut new_tensors = vec![];
            for (tensor_id, value) in outputs {
                //assert_eq!(value.has_nan().unwrap(), false);

                // Validate shape and dtype
                let tensor_info = model.get_tensor_info(tensor_id).unwrap();
                check_tensor_matches(&value, tensor_info)
                    .map_err(|x| EvalRuntimeError::EvalError(name.clone(), x))?;

                observer.on_tensor_assigned(
                    &SymbolicGraphTensorPath::Tensor(tensor_id),
                    &value,
                    eval_backend,
                );
                if let Some(x) = tensor_uses_left.get(&tensor_id)
                    && *x > 0
                {
                    active_tensors.insert(tensor_id, value);
                    new_tensors.push(tensor_id);
                }
            }
            tensors_just_created = new_tensors;
            remaining_ops_to_complete.remove(&op_id);
        } else {
            // No options, must exit
            break;
        }
    }

    let mut output_tensors = HashMap::new();
    let model_output_ids = model.get_outputs();
    for id in model_output_ids {
        if let Some(name) = model.get_tensor_name(id)
            && let Some(tensor) = active_tensors.get(&id)
        {
            output_tensors.insert(name.to_string(), tensor.clone());
        }
    }

    Ok(output_tensors)
}
