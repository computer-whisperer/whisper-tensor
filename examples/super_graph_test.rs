use std::collections::HashMap;
use std::path::Path;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::backends::ndarray_backend::NDArrayNumericTensor;
use whisper_tensor::dtype::DType;
use whisper_tensor::milli_graph::MilliOpGraph;
use whisper_tensor::milli_graph::ops::{
    ArgMax, Cast, Constant, Shape, SimpleBinary, Slice, Squeeze, Unsqueeze,
};
use whisper_tensor::model::Model;
use whisper_tensor::super_graph::cache::SuperGraphTensorCache;
use whisper_tensor::super_graph::data::SuperGraphData;
use whisper_tensor::super_graph::links::{
    SuperGraphLink, SuperGraphLinkString, SuperGraphLinkTensor, SuperGraphLinkTensorMap,
};
use whisper_tensor::super_graph::nodes::{SuperGraphNode, SuperGraphNodeMilliOpGraph, SuperGraphNodeModelExecution, SuperGraphNodeTokenizerDecode, SuperGraphNodeTokenizerEncode, SuperGraphNodeTokenizerLoad};
use whisper_tensor::super_graph::{SuperGraphBuilder, SuperGraphContext};
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

    let mut rng = rand::rng();
    let model = Model::new_from_onnx(&onnx_data, &mut rng).unwrap();

    let mut builder = SuperGraphBuilder::new();

    let model_link = SuperGraphLinkTensorMap::new(&mut rng);
    let text_input_link = SuperGraphLinkString::new(&mut rng);

    let tokenizer_link = SuperGraphNodeTokenizerLoad::new_and_add(
        &mut builder,
        TokenizerInfo::HFTokenizer("gpt2".to_string()),
        &mut rng,
    );

    let tokens =
        SuperGraphNodeTokenizerEncode::new_and_add(&mut builder, tokenizer_link, text_input_link, &mut rng);

    // Model invocation
    let logit_output = {
        let inputs = vec![(tokens, "input1".to_string())];
        let (outputs, logit_output) = {
            let tensor = SuperGraphLinkTensor::new(&mut rng);
            let outputs = vec![("output1".to_string(), tensor)];
            (outputs, tensor)
        };
        let node = SuperGraphNodeModelExecution::new(&mut rng, model_link, 0, inputs, outputs);
        builder.add_node(node.to_any());
        logit_output
    };

    // Sampler
    let chosen_token = {
        let (mut milli_graph, inputs_map) = MilliOpGraph::new([logit_output.global_id()], &mut rng);
        let logits_input = inputs_map[&logit_output.global_id()];

        // Slice to last token

        let logits_in = {
            let input_shape = Shape::push_new(&mut milli_graph, logits_input, &mut rng);
            let const_a = Constant::push_new(
                &mut milli_graph,
                NDArrayNumericTensor::from_vec(vec![0i64, 0, 1, 0]).to_dyn(),
                &mut rng,
            );
            let value = SimpleBinary::mul(&mut milli_graph, input_shape, const_a, &mut rng);
            let value = SimpleBinary::sub(&mut milli_graph, value, const_a, &mut rng);
            Slice::push_new(
                &mut milli_graph,
                logits_input,
                value,
                input_shape,
                None,
                None,
                &mut rng,
            )
        };

        // Cull unnecessary dims
        let const_0 = Constant::push_new(
            &mut milli_graph,
            NDArrayNumericTensor::from_vec(vec![0i64]).to_dyn(),
            &mut rng,
        );
        let logits_in = Squeeze::push_new(&mut milli_graph, logits_in, const_0, &mut rng);
        let logits_in = Squeeze::push_new(&mut milli_graph, logits_in, const_0, &mut rng);
        let logits_in = Squeeze::push_new(&mut milli_graph, logits_in, const_0, &mut rng);

        let output = ArgMax::push_new(&mut milli_graph, logits_in, 0, false, false, &mut rng);
        let output = Cast::push_new(&mut milli_graph, output, DType::U32, &mut rng);
        let output = Unsqueeze::push_new(&mut milli_graph, output, const_0, &mut rng);
        let mut output_map = HashMap::new();

        let output_tensor = SuperGraphLinkTensor::new(&mut rng);
        output_map.insert(output, output_tensor.global_id());
        milli_graph.set_output_map(output_map);

        let node = SuperGraphNodeMilliOpGraph::new(milli_graph, &mut rng);
        builder.add_node(node.to_any());
        output_tensor
    };

    let text_output =
        SuperGraphNodeTokenizerDecode::new_and_add(&mut builder, tokenizer_link, chosen_token, &mut rng);

    let inputs = vec![model_link.to_any(), text_input_link.to_any()];
    let outputs = vec![text_output.to_any()];

    let super_graph = builder.build(&mut rng, inputs.as_slice(), outputs.as_slice());

    let mut super_graph_data = SuperGraphData::new();
    super_graph_data
        .tensor_maps
        .insert(model_link, model.get_tensor_store());
    super_graph_data.strings.insert(
        text_input_link,
        "The fibonacci sequence is: 1, 1, 2, 3, 5, 8, 13,".to_string(),
    );
    let res = super_graph
        .run(
            super_graph_data,
            &mut SuperGraphContext {
                observer: &mut (),
                eval_backend: &mut EvalBackend::NDArray,
                caches: None,
                super_graph_tensor_cache: &mut SuperGraphTensorCache::new(),
                symbolic_graphs: vec![model.get_symbolic_graph()],
                use_compiled_models: false,
                compiled_models: None,
            },
        )
        .unwrap();
    let res = res.strings.get(&text_output).unwrap();
    println!("{res:?}");
}
