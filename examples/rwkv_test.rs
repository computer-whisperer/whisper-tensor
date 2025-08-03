use llm_samplers::prelude::{SampleGreedy, SamplerChain, SimpleSamplerResources};
use rand::SeedableRng;
use rand::prelude::StdRng;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use typenum::P1;
use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::language_model::LanguageModelManager;
use whisper_tensor::model::Model;
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::sampler::LLMSamplersBundle;
use whisper_tensor::tokenizer::Tokenizer;
use whisper_tensor_import::onnx_graph::WeightStorageStrategy;
use whisper_tensor_import::{ModelTypeHint, identify_and_load};

fn main() {
    tracing_subscriber::fmt::init();

    let input_path =
        Path::new("/mnt/secondary/rwkv-7-world/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth");
    //let input_path = Path::new("/ceph-fuse/public/neural_models/llms/rwkv7-g1/rwkv7-g1-0.1b-20250307-ctx4096.pth");
    let onnx_out = Path::new("out.onnx");
    let _bin_out = Path::new("out.bin");
    let onnx_data = identify_and_load(
        input_path,
        WeightStorageStrategy::EmbeddedData,
        Some(ModelTypeHint::RWKV7),
    )
    .unwrap();
    File::create(onnx_out)
        .unwrap()
        .write_all(&onnx_data)
        .unwrap();

    let runtime = Model::new_from_onnx(&onnx_data).unwrap();
    let mut llm = LanguageModelManager::new(runtime).unwrap();

    let tokenizer = llm.get_tokenizer().unwrap();

    let prompt = "Mary had a little lamb";
    let input = tokenizer
        .encode(prompt)
        .iter()
        .map(|x| *x as i64)
        .collect::<Vec<_>>();
    let input_tensor = NumericTensor::from_vec(input)
        .to_dyn_rank()
        .unsqueeze(0)
        .unwrap();

    let mut sampler = {
        let mut sc = SamplerChain::new();
        //sc += SampleFreqPresence::new(0.1, 0.1, 128);
        //sc += SampleTemperature::new(1.0);
        sc.push_sampler(SampleGreedy::new());

        let res =
            SimpleSamplerResources::new(Some(Box::new(StdRng::seed_from_u64(12345))), Some(vec![]));

        LLMSamplersBundle { chain: sc, res }
    };

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
        .try_to_type::<u32>()
        .unwrap()
        .try_to_rank::<P1>()
        .unwrap();
    let output_values: Vec<u32> = output_tensor.to_vec();
    println!("{output_values:?}");
    let output = tokenizer.decode(&output_values).unwrap();
    println!("{output}");
}
