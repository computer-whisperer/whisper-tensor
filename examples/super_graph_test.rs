use std::collections::HashMap;
use std::path::Path;
use llm_samplers::prelude::{SampleFreqPresence, SampleGreedy, SampleTemperature, SamplerChain, SimpleSamplerResources};
use rand::prelude::StdRng;
use rand::SeedableRng;
use typenum::P1;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::language_model::LanguageModelManager;
use whisper_tensor::model::{Model};
use whisper_tensor::numeric_tensor::{NumericTensor};
use whisper_tensor::sampler::{LLMSamplersBundle};
use whisper_tensor::tokenizer::Tokenizer;
use whisper_tensor::backends::vulkan_backend::{VulkanContext, VulkanImmediateExecutor};
use whisper_tensor::milli_graph::MilliOpGraph;
use whisper_tensor::super_graph::links::SuperGraphLinkTensor;
use whisper_tensor::super_graph::nodes::{SuperGraphNodeModelExecution, SuperGraphNodeModelLoad, SuperGraphNodeStringInput, SuperGraphNodeStringOutput, SuperGraphNodeTokenizerDecode, SuperGraphNodeTokenizerEncode, SuperGraphNodeTokenizerLoad};
use whisper_tensor::super_graph::SuperGraphBuilder;
use whisper_tensor_import::{identify_and_load, ModelTypeHint};
use whisper_tensor_import::onnx_graph::{TokenizerInfo, WeightStorageStrategy};

fn main() {
    tracing_subscriber::fmt::init();

    let input_path = Path::new("gpt2-lm-head-10.onnx");
    let onnx_data = identify_and_load(input_path, WeightStorageStrategy::EmbeddedData, Some(ModelTypeHint::GPT2)).unwrap();

    let model = Model::new_from_onnx(&onnx_data).unwrap();

    let mut builder = SuperGraphBuilder::new();

    let model_load = SuperGraphNodeModelLoad::new_and_add(&mut builder, "test".to_string());

    let text_input = SuperGraphNodeStringInput::new_and_add(&mut builder);

    let tokenizer_link = SuperGraphNodeTokenizerLoad::new_and_add(&mut builder, TokenizerInfo::RWKVWorld);

    let tokens = SuperGraphNodeTokenizerEncode::new_and_add(&mut builder, tokenizer_link.clone(), text_input);

    // Model invocation
    let logit_output = {
        let inputs = {
            let mut inputs = HashMap::new();
            inputs.insert(tokens.clone(), "tokens".to_string());
            inputs
        };
        let (outputs, logit_output) = {
            let mut outputs = HashMap::new();
            let tensor = SuperGraphLinkTensor::new(builder.get_next_link_id());
            outputs.insert("logits".to_string(), tensor.clone());
            (outputs, tensor)
        };
        let node = SuperGraphNodeModelExecution::new(model_load, inputs, outputs);
        builder.add_node(node.into());
        logit_output
    };

    // Sampler
    let chosen_token = {
        let (mut milli_graph_builder, inputs_map) = MilliOpGraph::new(&[logit_output.clone()]);
        let logits_input = inputs_map[&logit_output].clone();

        //milli_graph_builder.push_op(AnyMilliOp::)
    };

    let text_output = SuperGraphNodeTokenizerDecode::new_and_add(&mut builder, tokenizer_link, logit_output);

    SuperGraphNodeStringOutput::new_and_add(&mut builder, text_output);

    let super_graph = builder.build();
}