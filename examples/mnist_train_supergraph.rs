/// MNIST training example using a SuperGraph Scan node for the training loop.
///
/// Same model/training graph as mnist_train.rs, but the batch iteration is driven
/// by a SuperGraph Scan node instead of a manual Rust loop. Epochs are still a
/// Rust loop (so we can eval accuracy between them).
///
/// This exercises the SuperGraph → Scan → MilliOpGraph nesting and highlights
/// the ergonomic gaps in wiring a training loop this way.
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
use whisper_tensor::super_graph::cache::SuperGraphTensorCache;
use whisper_tensor::super_graph::data::SuperGraphData;
use whisper_tensor::super_graph::links::{SuperGraphLinkDouble, SuperGraphLinkTriple};
use whisper_tensor::super_graph::nodes::{
    SuperGraphNode, SuperGraphNodeMilliOpGraph, SuperGraphNodeScan,
};
use whisper_tensor::super_graph::{
    SuperGraphAnyLink, SuperGraphBuilder, SuperGraphContext, SuperGraphLinkTensor,
};
use whisper_tensor::symbolic_graph::{SymbolicGraphMutator, TensorType};
use whisper_tensor::tensor_rank::DynRank;

// --- Data (shared with mnist_train.rs) ---

struct Dataset {
    images: Vec<f32>,
    labels: Vec<u8>,
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
        generate_synthetic_data()
    }
}

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

// --- Model construction (shared) ---

fn build_mlp_graph(
    rng: &mut impl rand::Rng,
) -> (SymbolicGraphMutator, Vec<GlobalId>, GlobalId, GlobalId) {
    let mut m = SymbolicGraphMutator::new(rng);
    let s = |v: u64| ScalarInfoTyped::Numeric(v);

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

    let h1 = m.push_matmul("matmul1", images, w1, rng);
    let h1 = m.push_add("bias1", h1, b1, rng);
    let h1 = m.push_relu("relu1", h1, rng);
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
    let std1 = (2.0f32 / 784.0).sqrt();
    let w1: Vec<f32> = (0..784 * 128).map(|_| rand_normal(rng) * std1).collect();
    params.insert(
        param_ids[0],
        NumericTensor::from_vec_shape(w1, vec![784, 128]).unwrap(),
    );
    params.insert(
        param_ids[1],
        NumericTensor::from_vec_shape(vec![0.0f32; 128], vec![128]).unwrap(),
    );
    let std2 = (2.0f32 / 128.0).sqrt();
    let w2: Vec<f32> = (0..128 * 10).map(|_| rand_normal(rng) * std2).collect();
    params.insert(
        param_ids[2],
        NumericTensor::from_vec_shape(w2, vec![128, 10]).unwrap(),
    );
    params.insert(
        param_ids[3],
        NumericTensor::from_vec_shape(vec![0.0f32; 10], vec![10]).unwrap(),
    );
    params
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

// --- SuperGraph training loop ---

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
    let meta = training_graph.training_metadata.clone().unwrap();

    // Forward-only graph for eval
    let fwd_graph = graph.generate_milli_graph(&mut rng);

    // 4. Pre-batch training data into [num_batches, batch_size, ...]
    //    Scan will slice along axis 0, one batch per iteration.
    let batch_size = 64usize;
    let num_batches = data.train.num_samples / batch_size; // drop incomplete tail
    let used_samples = num_batches * batch_size;

    let batched_images = NumericTensor::<DynRank>::from_vec_shape(
        data.train.images[..used_samples * 784].to_vec(),
        vec![num_batches, batch_size, 784],
    )
    .unwrap();

    let mut one_hot_all = vec![0.0f32; used_samples * 10];
    for i in 0..used_samples {
        one_hot_all[i * 10 + data.train.labels[i] as usize] = 1.0;
    }
    let batched_labels =
        NumericTensor::<DynRank>::from_vec_shape(one_hot_all, vec![num_batches, batch_size, 10])
            .unwrap();

    eprintln!(
        "Pre-batched: {} batches of {} (dropped {} samples)",
        num_batches,
        batch_size,
        data.train.num_samples - used_samples
    );

    // 5. Build inner SuperGraph (one training step = one MilliOpGraph eval)
    //
    //    The MilliOpGraph node reads/writes tensors keyed by its own external IDs.
    //    The inner SuperGraph just wraps it, declaring which of those IDs are
    //    inputs vs outputs.
    let labels_ext_id = meta.external_inputs[0];
    let loss_id = meta.loss.unwrap();

    let inner_input_links: Vec<_> = training_graph
        .get_inputs()
        .iter()
        .map(|&id| SuperGraphAnyLink::tensor(id))
        .collect();

    let mut inner_output_links: Vec<_> = meta
        .param_updates
        .values()
        .map(|&new_param| SuperGraphAnyLink::tensor(new_param))
        .collect();
    inner_output_links.push(SuperGraphAnyLink::tensor(loss_id));

    let mut inner_builder = SuperGraphBuilder::new();
    inner_builder.add_node(SuperGraphNodeMilliOpGraph::new(training_graph, &mut rng).to_any());
    let inner_graph = inner_builder.build(&mut rng, &inner_input_links, &inner_output_links);

    // 6. Build outer SuperGraph with Scan
    //
    //    Outer links are fresh IDs that live in the outer namespace.
    //    The Scan maps: outer links → inner links (MilliOpGraph IDs) → outputs.

    // -- Outer links for inputs/outputs --
    let iter_count_link = SuperGraphLinkTensor::new(&mut rng);
    let outer_images_link = SuperGraphLinkTensor::new(&mut rng);
    let outer_labels_link = SuperGraphLinkTensor::new(&mut rng);
    let collected_losses_link = SuperGraphLinkTensor::new(&mut rng);

    // One (outer_initial, outer_final) pair per trainable parameter
    let outer_param_links: Vec<(GlobalId, SuperGraphLinkTensor, SuperGraphLinkTensor)> = meta
        .param_updates
        .keys()
        .map(|&ext_param| {
            (
                ext_param,
                SuperGraphLinkTensor::new(&mut rng), // initial
                SuperGraphLinkTensor::new(&mut rng), // final
            )
        })
        .collect();

    // -- State links: thread params across iterations --
    //    (outer_initial, inner_input=ext_param_id, inner_output=new_param_id)
    let state_links: Vec<SuperGraphLinkTriple> = outer_param_links
        .iter()
        .map(|&(ext_param, outer_initial, _)| {
            let new_param = meta.param_updates[&ext_param];
            SuperGraphLinkTriple::Tensor(
                outer_initial,
                SuperGraphLinkTensor(ext_param),
                SuperGraphLinkTensor(new_param),
            )
        })
        .collect();

    // -- Scan inputs: slice training data along axis 0 --
    let scan_inputs = vec![
        (outer_images_link, SuperGraphLinkTensor(image_id), 0u32),
        (outer_labels_link, SuperGraphLinkTensor(labels_ext_id), 0),
    ];

    // -- Scan outputs: collect per-batch loss along axis 0 --
    let scan_outputs = vec![(
        SuperGraphLinkTensor(loss_id), // inner (from iter_outputs)
        collected_losses_link,         // outer (accumulated tensor)
        0u32,                          // concat axis
    )];

    // -- Simple outputs: final param values from last iteration --
    let simple_outputs: Vec<SuperGraphLinkDouble> = outer_param_links
        .iter()
        .map(|&(ext_param, _, outer_final)| {
            let new_param = meta.param_updates[&ext_param];
            SuperGraphLinkDouble::Tensor(SuperGraphLinkTensor(new_param), outer_final)
        })
        .collect();

    let scan_node = SuperGraphNodeScan::new(
        inner_graph,
        iter_count_link,
        vec![], // simple_inputs: none needed
        state_links,
        scan_inputs,
        scan_outputs,
        simple_outputs,
        &mut rng,
    );

    let mut outer_builder = SuperGraphBuilder::new();
    outer_builder.add_node(scan_node.to_any());

    let mut outer_input_links = vec![
        SuperGraphAnyLink::Tensor(iter_count_link),
        SuperGraphAnyLink::Tensor(outer_images_link),
        SuperGraphAnyLink::Tensor(outer_labels_link),
    ];
    for &(_, outer_initial, _) in &outer_param_links {
        outer_input_links.push(SuperGraphAnyLink::Tensor(outer_initial));
    }

    let mut outer_output_links = vec![SuperGraphAnyLink::Tensor(collected_losses_link)];
    for &(_, _, outer_final) in &outer_param_links {
        outer_output_links.push(SuperGraphAnyLink::Tensor(outer_final));
    }

    let epoch_graph = outer_builder.build(&mut rng, &outer_input_links, &outer_output_links);

    // 7. Initialize weights
    let mut params = init_weights(&mut rng, &param_ids);

    // 8. Training loop (one Scan = one epoch over all batches)
    let num_epochs = 5;
    let mut backend = EvalBackend::NDArray;

    let acc = eval_accuracy(
        &fwd_graph,
        &data.test,
        &params,
        image_id,
        logits_id,
        &mut backend,
    );
    eprintln!("Initial test accuracy: {:.2}%", acc * 100.0);

    let iter_count_tensor =
        NumericTensor::<DynRank>::from_vec_shape(vec![num_batches as i64], vec![1]).unwrap();

    for epoch in 0..num_epochs {
        // Populate outer SuperGraph data
        let mut sg_data = SuperGraphData::new();
        sg_data
            .tensors
            .insert(iter_count_link, iter_count_tensor.clone());
        sg_data
            .tensors
            .insert(outer_images_link, batched_images.clone());
        sg_data
            .tensors
            .insert(outer_labels_link, batched_labels.clone());
        for &(ext_param, outer_initial, _) in &outer_param_links {
            sg_data
                .tensors
                .insert(outer_initial, params[&ext_param].clone());
        }

        // Run one epoch
        let mut tensor_cache = SuperGraphTensorCache::new();
        let mut observer = ();
        let mut context = SuperGraphContext::new(&mut backend, &mut observer, &mut tensor_cache);

        let results = epoch_graph.run(sg_data, &mut context).unwrap();

        // Extract collected losses and average
        let losses: Vec<f32> = results.tensors[&collected_losses_link]
            .flatten()
            .unwrap()
            .try_into()
            .unwrap();
        let avg_loss = losses.iter().map(|&v| v as f64).sum::<f64>() / losses.len() as f64;

        // Extract updated params
        for &(ext_param, _, outer_final) in &outer_param_links {
            params.insert(ext_param, results.tensors[&outer_final].clone());
        }

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
