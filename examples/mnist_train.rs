/// MNIST training example using whisper-tensor's training infrastructure.
///
/// Architecture: 2-layer MLP (784 → 128 → 10) with ReLU activation.
/// Loss: Cross-entropy
/// Optimizer: SGD
///
/// This example exercises the full training pipeline:
///   SymbolicGraph → generate_milli_graph_with_options (forward + backward + optimizer)
///   → training loop via MilliOpGraph::eval
///
/// Place MNIST IDX files in data/mnist/:
///   train-images-idx3-ubyte, train-labels-idx1-ubyte,
///   t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte
///
/// If files are not found, generates synthetic data for testing the pipeline.
use std::collections::HashMap;

use whisper_tensor::backends::eval_backend::EvalBackend;
use whisper_tensor::dtype::DType;
use whisper_tensor::graph::GlobalId;
use whisper_tensor::milli_graph::{
    BackwardGenOptions, LossInputSource, LossWiring, MilliGraphGenOptions, MilliOpGraph,
    OptimizerGenOptions, OptimizerKind,
};
use whisper_tensor::numeric_tensor::NumericTensor;
use whisper_tensor::scalar_info::ScalarInfoTyped;
use whisper_tensor::symbolic_graph::{SymbolicGraphMutator, TensorType};
use whisper_tensor::tensor_rank::DynRank;

// --- Data ---

struct Dataset {
    images: Vec<f32>, // [N * 784], normalized [0,1]
    labels: Vec<u8>,  // [N]
    num_samples: usize,
}

struct MnistData {
    train: Dataset,
    test: Dataset,
}

fn load_mnist() -> MnistData {
    let dir = std::path::Path::new("data/mnist");
    let train_img_path = dir.join("train-images-idx3-ubyte");
    let train_lbl_path = dir.join("train-labels-idx1-ubyte");
    let test_img_path = dir.join("t10k-images-idx3-ubyte");
    let test_lbl_path = dir.join("t10k-labels-idx1-ubyte");

    if train_img_path.exists()
        && train_lbl_path.exists()
        && test_img_path.exists()
        && test_lbl_path.exists()
    {
        eprintln!("Loading MNIST from data/mnist/...");
        let parse_images = |path: &std::path::Path| -> Dataset {
            let raw = std::fs::read(path).unwrap();
            let n = u32::from_be_bytes(raw[4..8].try_into().unwrap()) as usize;
            let images: Vec<f32> = raw[16..].iter().map(|&b| b as f32 / 255.0).collect();
            assert_eq!(images.len(), n * 784);
            Dataset {
                images,
                labels: vec![],
                num_samples: n,
            }
        };
        let parse_labels = |path: &std::path::Path| -> Vec<u8> {
            let raw = std::fs::read(path).unwrap();
            raw[8..].to_vec()
        };

        let mut train = parse_images(&train_img_path);
        train.labels = parse_labels(&train_lbl_path);
        let mut test = parse_images(&test_img_path);
        test.labels = parse_labels(&test_lbl_path);

        MnistData { train, test }
    } else {
        eprintln!("MNIST files not found in data/mnist/, generating synthetic data...");
        eprintln!("(Place IDX files there for real training)");
        generate_synthetic_data()
    }
}

/// Generate synthetic linearly-separable data for pipeline testing.
fn generate_synthetic_data() -> MnistData {
    let mut rng = rand::rng();

    fn make_dataset(n: usize, rng: &mut impl rand::Rng) -> Dataset {
        let mut images = vec![0.0f32; n * 784];
        let mut labels = vec![0u8; n];
        for i in 0..n {
            let label = (i % 10) as u8;
            labels[i] = label;
            let stripe_start = (label as usize) * 78;
            for j in stripe_start..stripe_start + 78 {
                images[i * 784 + j] = 0.8 + rng.random::<f32>() * 0.2;
            }
            for j in 0..784 {
                images[i * 784 + j] += rng.random::<f32>() * 0.1;
                images[i * 784 + j] = images[i * 784 + j].clamp(0.0, 1.0);
            }
        }
        Dataset {
            images,
            labels,
            num_samples: n,
        }
    }

    MnistData {
        train: make_dataset(2000, &mut rng),
        test: make_dataset(500, &mut rng),
    }
}

fn get_batch(
    dataset: &Dataset,
    start: usize,
    batch_size: usize,
) -> (NumericTensor<DynRank>, NumericTensor<DynRank>) {
    let actual_batch = batch_size.min(dataset.num_samples - start);
    let img_slice = &dataset.images[start * 784..(start + actual_batch) * 784];
    let img_tensor =
        NumericTensor::<DynRank>::from_vec_shape(img_slice.to_vec(), vec![actual_batch, 784])
            .unwrap();

    let mut one_hot = vec![0.0f32; actual_batch * 10];
    for i in 0..actual_batch {
        one_hot[i * 10 + dataset.labels[start + i] as usize] = 1.0;
    }
    let label_tensor =
        NumericTensor::<DynRank>::from_vec_shape(one_hot, vec![actual_batch, 10]).unwrap();

    (img_tensor, label_tensor)
}

// --- Model construction ---

/// Build a 2-layer MLP as a SymbolicGraph.
///
/// Returns (mutator, param_ids=[W1,b1,W2,b2], image_input_id, logits_id)
fn build_mlp_graph(
    rng: &mut impl rand::Rng,
) -> (SymbolicGraphMutator, Vec<GlobalId>, GlobalId, GlobalId) {
    let mut m = SymbolicGraphMutator::new(rng);
    let s = |v: u64| ScalarInfoTyped::Numeric(v);

    // Inputs: images and weight/bias parameters
    let images = m.push_typed_tensor(
        "images",
        TensorType::Input(None),
        Some(DType::F32),
        Some(vec![s(0), s(784)]),
        rng,
    );
    let w1 = m.push_typed_tensor(
        "W1",
        TensorType::Input(None),
        Some(DType::F32),
        Some(vec![s(784), s(128)]),
        rng,
    );
    let b1 = m.push_typed_tensor(
        "b1",
        TensorType::Input(None),
        Some(DType::F32),
        Some(vec![s(128)]),
        rng,
    );
    let w2 = m.push_typed_tensor(
        "W2",
        TensorType::Input(None),
        Some(DType::F32),
        Some(vec![s(128), s(10)]),
        rng,
    );
    let b2 = m.push_typed_tensor(
        "b2",
        TensorType::Input(None),
        Some(DType::F32),
        Some(vec![s(10)]),
        rng,
    );

    m.push_input(images);
    m.push_input(w1);
    m.push_input(b1);
    m.push_input(w2);
    m.push_input(b2);

    // Layer 1: relu(images @ W1 + b1)
    let h1 = m.push_matmul("matmul1", images, w1, rng);
    let h1 = m.push_add("bias1", h1, b1, rng);
    let h1 = m.push_relu("relu1", h1, rng);

    // Layer 2: h1 @ W2 + b2
    let logits = m.push_matmul("matmul2", h1, w2, rng);
    let logits = m.push_add("bias2", logits, b2, rng);

    m.push_output(logits);
    (m, vec![w1, b1, w2, b2], images, logits)
}

fn rand_normal(rng: &mut impl rand::Rng) -> f32 {
    let u1: f32 = rng.random::<f32>().max(1e-7);
    let u2: f32 = rng.random::<f32>();
    (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * std::f32::consts::PI * u2).cos()
}

fn init_weights(
    rng: &mut impl rand::Rng,
    param_ids: &[GlobalId],
) -> HashMap<GlobalId, NumericTensor<DynRank>> {
    let mut params = HashMap::new();

    // W1 [784, 128] — He init
    let std1 = (2.0f32 / 784.0).sqrt();
    let w1: Vec<f32> = (0..784 * 128).map(|_| rand_normal(rng) * std1).collect();
    params.insert(
        param_ids[0],
        NumericTensor::from_vec_shape(w1, vec![784, 128]).unwrap(),
    );

    // b1 [128] — zeros
    params.insert(
        param_ids[1],
        NumericTensor::from_vec_shape(vec![0.0f32; 128], vec![128]).unwrap(),
    );

    // W2 [128, 10] — He init
    let std2 = (2.0f32 / 128.0).sqrt();
    let w2: Vec<f32> = (0..128 * 10).map(|_| rand_normal(rng) * std2).collect();
    params.insert(
        param_ids[2],
        NumericTensor::from_vec_shape(w2, vec![128, 10]).unwrap(),
    );

    // b2 [10] — zeros
    params.insert(
        param_ids[3],
        NumericTensor::from_vec_shape(vec![0.0f32; 10], vec![10]).unwrap(),
    );

    params
}

fn main() {
    let mut rng = rand::rng();

    // 1. Load data
    let data = load_mnist();
    eprintln!(
        "Data: {} train, {} test samples",
        data.train.num_samples, data.test.num_samples
    );

    // 2. Build model graph
    let (mutator, param_ids, image_id, logits_id) = build_mlp_graph(&mut rng);
    let (graph, _tensor_store) = mutator.get_inner();

    // 3. Build training graph (forward + backward + optimizer)
    let (loss_graph, loss_info) = MilliOpGraph::cross_entropy_loss(&mut rng);

    let options = MilliGraphGenOptions {
        backward: Some(BackwardGenOptions {
            loss_graph,
            loss_wiring: vec![
                LossWiring {
                    loss_input: loss_info.predictions_input,
                    source: LossInputSource::ForwardOutput(logits_id),
                },
                LossWiring {
                    loss_input: loss_info.targets_input,
                    source: LossInputSource::ExternalInput {
                        name: "targets".into(),
                    },
                },
            ],
            loss_output: loss_info.loss_output,
            trainable_params: param_ids.clone(),
            stop_gradients: std::collections::HashSet::new(),
        }),
        optimizer: Some(OptimizerGenOptions {
            kind: OptimizerKind::SGD { lr: 0.1 },
        }),
    };

    eprintln!("Generating training graph...");
    let training_graph = graph
        .generate_milli_graph_with_options(&options, &mut rng)
        .unwrap();
    let meta = training_graph.training_metadata.as_ref().unwrap();
    eprintln!(
        "  {} trainable params, {} external inputs (labels)",
        meta.param_to_grad.len(),
        meta.external_inputs.len(),
    );

    // 4. Forward-only graph for evaluation
    let fwd_graph = graph.generate_milli_graph(&mut rng);

    // 5. Initialize weights
    let mut params = init_weights(&mut rng, &param_ids);

    // 6. Training loop
    let mut backend = EvalBackend::NDArray;
    let batch_size = 64;
    let num_epochs = 5;

    let acc = eval_accuracy(
        &fwd_graph,
        &data.test,
        &params,
        image_id,
        logits_id,
        &mut backend,
    );
    eprintln!("Initial test accuracy: {:.2}%", acc * 100.0);

    for epoch in 0..num_epochs {
        let mut epoch_loss = 0.0f64;
        let mut num_batches = 0u32;
        let mut batch_start = 0;

        while batch_start < data.train.num_samples {
            let (img_batch, label_batch) = get_batch(&data.train, batch_start, batch_size);

            let mut inputs: HashMap<GlobalId, NumericTensor<DynRank>> = params.clone();
            inputs.insert(image_id, img_batch);
            inputs.insert(meta.external_inputs[0], label_batch);

            let results: HashMap<_, _> = training_graph
                .eval(&inputs, &mut (), &mut backend)
                .unwrap()
                .collect();

            let loss_val: Vec<f32> = results[&meta.loss.unwrap()]
                .flatten()
                .unwrap()
                .try_into()
                .unwrap();
            epoch_loss += loss_val[0] as f64;
            num_batches += 1;

            // Feed updated parameters back
            for (&ext_param, &new_param_output) in &meta.param_updates {
                params.insert(ext_param, results[&new_param_output].clone());
            }

            batch_start += batch_size;
        }

        let avg_loss = epoch_loss / num_batches as f64;
        let acc = eval_accuracy(
            &fwd_graph,
            &data.test,
            &params,
            image_id,
            logits_id,
            &mut backend,
        );
        eprintln!(
            "Epoch {}/{}: loss = {:.4}, test accuracy = {:.2}%",
            epoch + 1,
            num_epochs,
            avg_loss,
            acc * 100.0
        );
    }

    eprintln!("Done!");
}

fn eval_accuracy(
    fwd_graph: &MilliOpGraph,
    test: &Dataset,
    params: &HashMap<GlobalId, NumericTensor<DynRank>>,
    image_id: GlobalId,
    logits_id: GlobalId,
    backend: &mut EvalBackend,
) -> f32 {
    let batch_size = 256;
    let mut correct = 0usize;
    let mut total = 0usize;
    let mut i = 0;

    while i < test.num_samples {
        let (img_batch, _) = get_batch(test, i, batch_size);
        let actual = batch_size.min(test.num_samples - i);

        let mut inputs = params.clone();
        inputs.insert(image_id, img_batch);

        let results: HashMap<_, _> = fwd_graph.eval(&inputs, &mut (), backend).unwrap().collect();
        let logits: Vec<f32> = results[&logits_id].flatten().unwrap().try_into().unwrap();

        for b in 0..actual {
            let row = &logits[b * 10..(b + 1) * 10];
            let pred = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            if pred == test.labels[i + b] as usize {
                correct += 1;
            }
            total += 1;
        }
        i += batch_size;
    }

    correct as f32 / total as f32
}
