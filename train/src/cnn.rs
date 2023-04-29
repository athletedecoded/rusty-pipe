use tch::{CModule, Device, Tensor};
use anyhow::{bail, Result};
use tch::nn::{self, OptimizerConfig, ModuleT};
use tch::vision::imagenet;

#[derive(Debug)]
struct Net {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Net {
    fn new(vs: &nn::Path) -> Net {
        let conv1 = nn::conv2d(vs, 3, 32, 3, Default::default());
        let conv2 = nn::conv2d(vs, 32, 64, 3, Default::default());
        let fc1 = nn::linear(vs, 64 * 54 * 54, 512, Default::default());
        let fc2 = nn::linear(vs, 512, 2, Default::default());
        Net { conv1, conv2, fc1, fc2 }
    }
}

impl nn::ModuleT for Net {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.view([-1, 3, 224, 224])
            .apply(&self.conv1)
            .relu()
            .max_pool2d_default(2)
            .apply(&self.conv2)
            .relu()
            .max_pool2d_default(2)
            .view([-1, 64 * 54 * 54])
            .apply(&self.fc1)
            .relu()
            .dropout(0.3, train)
            .apply(&self.fc2)
            .log_softmax(-1, tch::Kind::Float)
    }
}


pub fn run(dataset_dir: String) -> Result<()> {
    // Run Prechecks
    println!("Running Prechecks...");
    println!("...Cuda available: {}", tch::Cuda::is_available());
    println!("...Cudnn available: {}", tch::Cuda::cudnn_is_available());

    // Load the dataset and resize it to the usual imagenet dimension of 224x224.
    let dataset = imagenet::load_from_dir(dataset_dir)?;
    println!("{:?}", dataset.train_labels);

    let mut vs = nn::VarStore::new(Device::cuda_if_available());
    let net = Net::new(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, 1e-4)?;
    for epoch in 1..5 {
        for (bimages, blabels) in dataset.train_iter(64).shuffle().to_device(vs.device()) {
            let bimages = tch::vision::dataset::augmentation(&bimages, true, 4, 8);
            let loss = net.forward_t(&bimages, true).cross_entropy_for_logits(&blabels);
            opt.backward_step(&loss);
        }
        let test_accuracy =
            net.batch_accuracy_for_logits(&dataset.test_images, &dataset.test_labels, vs.device(), 64);
        println!("epoch: {:4} test acc: {:5.2}%", epoch, 100. * test_accuracy,);
    }

    vs.freeze();
    let mut closure = |input: &[Tensor]| vec![net.forward_t(&input[0], false)];
    let model = CModule::create_by_tracing(
        "RustyPipe",
        "forward",
        // The input tensor shape is [1, 3, 224, 224]
        &[Tensor::zeros(&[3*224*224], (tch::Kind::Float, tch::Device::Cpu))],
        &mut closure,
    )?;
    // Save the model to a file
    println!("Saving model to model.pt...");
    model.save("model.pt")?;

    Ok(())
}