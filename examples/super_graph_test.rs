use std::collections::HashMap;
use std::path::Path;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::dtype::DType;
use whisper_tensor::milli_graph::MilliOpGraph;
use whisper_tensor::milli_graph::ops::{
    AnyMilliOp, MilliOpArgMax, MilliOpCast, MilliOpConstant, MilliOpShape, MilliOpSimpleBinary,
    MilliOpSlice, MilliOpSqueeze, MilliOpUnsqueeze,
};
use whisper_tensor::model::Model;
use whisper_tensor::super_graph::SuperGraphBuilder;
use whisper_tensor::super_graph::data::SuperGraphData;
use whisper_tensor::super_graph::links::{
    SuperGraphLink, SuperGraphLinkModel, SuperGraphLinkString, SuperGraphLinkTensor,
};
use whisper_tensor::super_graph::nodes::{
    SuperGraphNodeMilliOpGraph, SuperGraphNodeModelExecution, SuperGraphNodeTokenizerDecode,
    SuperGraphNodeTokenizerEncode, SuperGraphNodeTokenizerLoad,
};
use whisper_tensor_import::onnx_graph::{TokenizerInfo, WeightStorageStrategy};
use whisper_tensor_import::{ModelTypeHint, identify_and_load};

fn main() {
    tracing_subscriber::fmt::init();

    let input_path = Path::new("gpt2-lm-head-10.onnx");
    let onnx_data = identify_and_load(
        input_path,
        WeightStorageStrategy::EmbeddedData,
        Some(ModelTypeHint::GPT2),
    )
    .unwrap();

    let model = Model::new_from_onnx(&onnx_data).unwrap();

    let mut builder = SuperGraphBuilder::new();

    let model_link = SuperGraphLinkModel::new(builder.get_next_link_id());
    let text_input_link = SuperGraphLinkString::new(builder.get_next_link_id());

    let tokenizer_link = SuperGraphNodeTokenizerLoad::new_and_add(
        &mut builder,
        TokenizerInfo::HFTokenizer("gpt2".to_string()),
    );

    let tokens = SuperGraphNodeTokenizerEncode::new_and_add(
        &mut builder,
        tokenizer_link.clone(),
        text_input_link.clone(),
    );

    // Model invocation
    let logit_output = {
        let inputs = {
            let mut inputs = HashMap::new();
            inputs.insert(tokens.clone(), "input1".to_string());
            inputs
        };
        let (outputs, logit_output) = {
            let mut outputs = HashMap::new();
            let tensor = SuperGraphLinkTensor::new(builder.get_next_link_id());
            outputs.insert("output1".to_string(), tensor.clone());
            (outputs, tensor)
        };
        let node = SuperGraphNodeModelExecution::new(model_link.clone(), inputs, outputs);
        builder.add_node(node.into());
        logit_output
    };

    // Sampler
    let chosen_token = {
        let (mut milli_graph, inputs_map) = MilliOpGraph::new(&[logit_output.clone()]);
        let logits_input = inputs_map[&logit_output];

        // Slice to last token

        let logits_in = {
            let input_shape =
                milli_graph.push_op(AnyMilliOp::Shape(MilliOpShape::new(logits_input)));
            let const_a = milli_graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new(
                NDArrayNumericTensor::from_vec(vec![0i64, 0, 1, 0]).to_dyn(),
            )));
            let value = milli_graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::mul(
                input_shape,
                const_a,
            )));
            let value = milli_graph.push_op(AnyMilliOp::SimpleBinary(MilliOpSimpleBinary::sub(
                value, const_a,
            )));
            milli_graph.push_op(AnyMilliOp::Slice(MilliOpSlice::new(
                logits_input,
                value,
                input_shape,
                None,
                None,
            )))
        };

        // Cull unnecessary dims
        let const_0 = milli_graph.push_op(AnyMilliOp::Constant(MilliOpConstant::new(
            NDArrayNumericTensor::from_vec(vec![0i64]).to_dyn(),
        )));
        let logits_in =
            milli_graph.push_op(AnyMilliOp::Squeeze(MilliOpSqueeze::new(logits_in, const_0)));
        let logits_in =
            milli_graph.push_op(AnyMilliOp::Squeeze(MilliOpSqueeze::new(logits_in, const_0)));
        let logits_in =
            milli_graph.push_op(AnyMilliOp::Squeeze(MilliOpSqueeze::new(logits_in, const_0)));

        let output = milli_graph.push_op(AnyMilliOp::ArgMax(MilliOpArgMax::new(
            logits_in, 0, false, false,
        )));
        let output = milli_graph.push_op(AnyMilliOp::Cast(MilliOpCast::new(output, DType::U32)));
        let output = milli_graph.push_op(AnyMilliOp::Unsqueeze(MilliOpUnsqueeze::new(
            output, const_0,
        )));
        let mut output_map = HashMap::new();

        let output_tensor = SuperGraphLinkTensor::new(builder.get_next_link_id());
        output_map.insert(output, output_tensor.clone());
        milli_graph.set_output_map(output_map);

        let node = SuperGraphNodeMilliOpGraph::new(milli_graph);
        builder.add_node(node.into());
        output_tensor
    };

    let text_output =
        SuperGraphNodeTokenizerDecode::new_and_add(&mut builder, tokenizer_link, chosen_token);

    let inputs = vec![model_link.to_any(), text_input_link.to_any()];
    let outputs = vec![text_output.to_any()];

    let super_graph = builder.build(inputs.as_slice(), outputs.as_slice());

    let mut super_graph_data = SuperGraphData::new();
    super_graph_data.models.insert(model_link, &model);
    super_graph_data.strings.insert(
        text_input_link,
        "The fibonacci sequence is: 1, 1, 2, 3, 5, 8, 13,".to_string(),
    );
    let res = super_graph
        .run(super_graph_data, None, &mut EvalBackend::NDArray)
        .unwrap();
    let res = res.strings.get(&text_output).unwrap();
    println!("{res:?}");
}
