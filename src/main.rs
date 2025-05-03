use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use llm_samplers::prelude::{SampleFreqPresence, SampleGreedy, SampleTemperature, SamplerChain, SimpleSamplerResources};
use tokenizers::Tokenizer;
use onnx_graph::WeightStorageStrategy;
use onnx_import::{identify_and_load, ModelTypeHint};
use rand::prelude::StdRng;
use rand::SeedableRng;
use rwkv_tokenizer::WorldTokenizer;
use whisper_tensor::{RuntimeModel, RuntimeBackend, RuntimeEnvironment};
use whisper_tensor::eval_backend::EvalBackend;
use whisper_tensor::language_model::LanguageModelManager;
use whisper_tensor::numeric_tensor::{NumericTensor};
use whisper_tensor::sampler::{GreedySampler, LLMSamplersBundle};

fn main() {
    tracing_subscriber::fmt::init();

    //let input_path = Path::new("/mnt/secondary/rwkv7-g1/rwkv7-g1-0.1b-20250307-ctx4096.pth");
    let input_path = Path::new("/ceph-fuse/public/neural_models/llms/rwkv7-g1/rwkv7-g1-0.1b-20250307-ctx4096.pth");
    let onnx_out = Path::new("out.onnx");
    let bin_out = Path::new("out.bin");
    let onnx_data = identify_and_load(input_path, WeightStorageStrategy::BinFile(bin_out.to_path_buf()), Some(ModelTypeHint::RWKV7)).unwrap();
    File::create(onnx_out).unwrap().write_all(&onnx_data).unwrap();

    let runtime = RuntimeModel::load_onnx(&onnx_data, RuntimeBackend::ORT, Default::default()).unwrap();
    let mut llm = LanguageModelManager::new(runtime).unwrap();

    let tokenizer = llm.get_tokenizer().unwrap();

    let prompt = "The fibbonacci sequence is: 1, 1, 2, 3, 5, 8, 13,";
    let input = tokenizer.encode(prompt).iter().map(|x| *x as i64).collect::<Vec<_>>();
    let input_tensor = NumericTensor::from_vec1(input).unsqueeze(0, &EvalBackend::NDArray).unwrap().unsqueeze(0, &EvalBackend::NDArray).unwrap();

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

    let (output, _) = llm.run(input_tensor.clone(), None, &mut sampler).unwrap();
    let output_values: Vec<u32> = output.squeeze(0, &EvalBackend::NDArray).unwrap().squeeze(0, &EvalBackend::NDArray).unwrap().try_into().unwrap();
    let output = tokenizer.decode(&output_values).unwrap();
    println!("{}", output);
}