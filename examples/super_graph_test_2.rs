use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::model::{Model};
use whisper_tensor::dtype::DType;
use whisper_tensor::milli_graph::{MilliOpGraph};
use whisper_tensor::milli_graph::ops::{AnyMilliOp, MilliOpArgMax, MilliOpCast, MilliOpConstant, MilliOpShape, MilliOpSimpleBinary, MilliOpSlice, MilliOpSqueeze, MilliOpUnsqueeze};
use whisper_tensor::model::ModelExecutionRuntime::Eval;
use whisper_tensor::super_graph::links::{SuperGraphLink, SuperGraphLinkModel, SuperGraphLinkString, SuperGraphLinkTensor};
use whisper_tensor::super_graph::nodes::{SuperGraphNodeMilliOpGraph, SuperGraphNodeModelExecution, SuperGraphNodeTokenizerDecode, SuperGraphNodeTokenizerEncode, SuperGraphNodeTokenizerLoad};
use whisper_tensor::super_graph::SuperGraphBuilder;
use whisper_tensor::super_graph::data::SuperGraphData;
use whisper_tensor_import::{identify_and_load, ModelTypeHint};
use whisper_tensor_import::onnx_graph::{TokenizerInfo, WeightStorageStrategy};

fn main() {
    tracing_subscriber::fmt::init();

    let input_path = Path::new("gpt2-lm-head-10.onnx");
    let onnx_data = identify_and_load(input_path, WeightStorageStrategy::EmbeddedData, Some(ModelTypeHint::GPT2)).unwrap();

    let model = Arc::new(Model::new_from_onnx(&onnx_data).unwrap());

    let bla = model.text_inference_tokens_in_logits_out_interface.clone().unwrap();

    let prompt = "The fibonacci sequence is: 1, 1, 2, 3, 5, 8, 13,".to_string();
    let mut tokenizer_cache = HashMap::new();
    print!("{:}", prompt);
    let mut context = prompt.clone();
    for _ in 0..10 {
        let res = bla.run_string_in_string_out(model.clone(), context.clone(), &mut tokenizer_cache, &mut EvalBackend::NDArray).unwrap();
        print!("{:}", res);
        context.push_str(&res);
    }
}