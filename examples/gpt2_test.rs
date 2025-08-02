use llm_samplers::prelude::{
    SampleFreqPresence, SampleGreedy, SampleTemperature, SamplerChain, SimpleSamplerResources,
};
use rand::SeedableRng;
use rand::prelude::StdRng;
use std::path::Path;
use typenum::P1;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::language_model::LanguageModelManager;
use whisper_tensor::model::Model;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::sampler::LLMSamplersBundle;
use whisper_tensor::tokenizer::Tokenizer;
//use whisper_tensor::backends::vulkan_backend::{VulkanContext, VulkanImmediateExecutor};
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;
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

    let runtime = Model::new_from_onnx(&onnx_data).unwrap();
    let mut llm = LanguageModelManager::new(runtime).unwrap();

    let tokenizer = llm.get_tokenizer().unwrap();

    let prompt = "The fibonacci sequence is: 1, 1, 2, 3, 5, 8, 13,";
    let input = tokenizer
        .encode(prompt)
        .iter()
        .map(|x| *x as i64)
        .collect::<Vec<_>>();
    let input_tensor = NumericTensor::from_vec(input)
        .to_dyn_rank()
        .unsqueeze(0)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    let mut sampler = {
        let mut sc = SamplerChain::new();
        sc += SampleFreqPresence::new(0.1, 0.1, 128);
        sc += SampleTemperature::new(1.0);
        sc.push_sampler(SampleGreedy::new());

        let res =
            SimpleSamplerResources::new(Some(Box::new(StdRng::seed_from_u64(12345))), Some(vec![]));

        LLMSamplersBundle { chain: sc, res }
    };

    //let vulkan_context = VulkanContext::new().unwrap();
    //let mut vulkan_runtime = VulkanImmediateExecutor::new(vulkan_context).unwrap();

    let (output, _) = llm
        .run(
            input_tensor.clone(),
            None,
            &mut sampler,
            &mut EvalBackend::NDArray,
        )
        .unwrap();
    let output_tensor = output
        .squeeze(0)
        .unwrap()
        .squeeze(0)
        .unwrap()
        .try_to_type::<u32>()
        .unwrap()
        .try_to_rank::<P1>()
        .unwrap();
    let output_values: Vec<u32> = output_tensor.to_vec();
    let output = tokenizer.decode(&output_values).unwrap();
    println!("{}", output);
}
