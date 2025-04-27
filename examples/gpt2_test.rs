use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use llm_samplers::prelude::{SampleFreqPresence, SampleGreedy, SampleTemperature, SamplerChain, SimpleSamplerResources};
use rand::prelude::StdRng;
use rand::SeedableRng;
use tokenizers::Tokenizer;
use onnx_graph::WeightStorageStrategy;
use onnx_import::identify_and_load;
use whisper_tensor::{RuntimeModel, Backend};
use whisper_tensor::language_model::LanguageModelManager;
use whisper_tensor::numeric_tensor::{NumericTensor};
use whisper_tensor::sampler::{GreedySampler, LLMSamplersBundle, Sampler};

fn main() {
    tracing_subscriber::fmt::init();

    let input_path = Path::new("gpt2-lm-head-10.onnx");
    let mut onnx_data = Vec::new();
    File::open(input_path).unwrap().read_to_end(&mut onnx_data).unwrap();

    let runtime = RuntimeModel::load_onnx(&onnx_data, Backend::ONNXReference).unwrap();
    let mut llm = LanguageModelManager::new(runtime, "input1", "output1");

    let tokenizer = Tokenizer::from_pretrained("gpt2", None).unwrap();

    let prompt = "The fibbonacci sequence is: 1, 1, 2, 3, 5, 8, 13,";
    let input = whisper_tensor::tokenizer::Tokenizer::encode(&tokenizer, prompt);
    let input_tensor = NumericTensor::from_vec1(input).unsqueeze(0).unwrap().unsqueeze(0).unwrap();

    let mut sampler = {
        let mut sc = SamplerChain::new();
        sc += SampleFreqPresence::new(0.1, 0.1, 128);
        sc += SampleTemperature::new(1.0);
        sc.push_sampler(SampleGreedy::new());

        let res = SimpleSamplerResources::new(
            Some(Box::new(StdRng::seed_from_u64(12345))),
            Some(vec![])
        );

        LLMSamplersBundle{
            chain: sc,
            res
        }
    };

    let output = llm.run(input_tensor.clone(), &mut sampler).unwrap();
    let output_values: Vec<u32> = output.squeeze(0).unwrap().squeeze(0).unwrap().try_into().unwrap();
    let output = whisper_tensor::tokenizer::Tokenizer::decode(&tokenizer, &output_values);
    println!("{}", output);
}