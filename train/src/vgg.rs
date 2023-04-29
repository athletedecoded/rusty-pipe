use tch::{CModule, Device, Tensor};
use anyhow::{bail, Result};
use tch::nn::{self, OptimizerConfig, ModuleT};
use tch::vision::imagenet;

#[derive(Debug)]
struct Net {
    conv1_1: nn::Conv2D,
    conv1_2: nn::Conv2D,
    conv2_1: nn::Conv2D,
    conv2_2: nn::Conv2D,
    conv3_1: nn::Conv2D,
    conv3_2: nn::Conv2D,
    conv3_3: nn::Conv2D,
    conv4_1: nn::Conv2D,
    conv4_2: nn::Conv2D,
    conv4_3: nn::Conv2D,
    conv5_1: nn::Conv2D,
    conv5_2: nn::Conv2D,
    conv5_3: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
    fc3: nn::Linear,
}


impl Net {
    fn new(vs: &nn::Path, num_classes: i64) -> Net {
        let conv1_1 = nn::conv2d(vs, 3, 32, 3, Default::default());
        let conv1_2 = nn::conv2d(vs, 32, 64, 3, Default::default());
        let conv2_1 = nn::conv2d(vs, 64, 128, 3, Default::default());
        let conv2_2 = nn::conv2d(vs, 128, 128, 3, Default::default());
        let conv3_1 = nn::conv2d(vs, 128, 256, 3, Default::default());
        let conv3_2 = nn::conv2d(vs, 256, 256, 3, Default::default());
        let conv3_3 = nn::conv2d(vs, 256, 256, 3, Default::default());
        let conv4_1 = nn::conv2d(vs, 256, 512, 3, Default::default());
        let conv4_2 = nn::conv2d(vs, 512, 512, 3, Default::default());
        let conv4_3 = nn::conv2d(vs, 512, 512, 3, Default::default());
        let conv5_1 = nn::conv2d(vs, 512, 512, 3, Default::default());
        let conv5_2 = nn::conv2d(vs, 512, 512, 3, Default::default());
        let conv5_3 = nn::conv2d(vs, 512, 512, 3, Default::default());
        let fc1 = nn::linear(vs, 512 * 7 * 7, 4096, Default::default());
        let fc2 = nn::linear(vs, 4096, 4096, Default::default());
        let fc3 = nn::linear(vs, 4096, num_classes, Default::default());
        Net { 
            conv1_1,
            conv1_2,
            conv2_1,
            conv2_2,
            conv3_1,
            conv3_2,
            conv3_3,
            conv4_1,
            conv4_2,
            conv4_3,
            conv5_1,
            conv5_2,
            conv5_3,
            fc1,
            fc2,
            fc3,
        }
    }
}

impl nn::ModuleT for Net {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.view([-1, 3, 224, 224])
            .apply(&self.conv1_1)
            .relu()
            .apply(&self.conv1_2)
            .relu()
            .max_pool2d_default(2)
            .apply(&self.conv2_1)
            .relu()
            .apply(&self.conv2_2)
            .relu()
            .max_pool2d_default(2)
            .apply(&self.conv3_1)
            .relu()
            .apply(&self.conv3_2)
            .relu()
            .apply(&self.conv3_3)
            .relu()
            .max_pool2d_default(2)
            .apply(&self.conv4_1)
            .relu()
            .apply(&self.conv4_2)
            .relu()
            .apply(&self.conv4_3)
            .relu()
            .max_pool2d_default(2)
            .apply(&self.conv5_1)
            .relu()
            .apply(&self.conv5_2)
            .relu()
            .apply(&self.conv5_3)
            .relu()
            .max_pool2d_default(2)
            .view([-1, 512*7*7])
            .apply(&self.fc1)
            .relu()
            .dropout(0.5, train)
            .apply(&self.fc2)
            .relu()
            .dropout(0.5, train)
            .apply(&self.fc3)
            .log_softmax(-1, tch::Kind::Float)
    }
}


fn main() -> Result<()> {
    // Run Prechecks
    println!("Running Prechecks...");
    println!("...Cuda available: {}", tch::Cuda::is_available());
    println!("...Cudnn available: {}", tch::Cuda::cudnn_is_available());

    let args: Vec<_> = std::env::args().collect();    
    let dataset_dir = match args.as_slice() {
        [_, d] => d.to_owned(),
        _ => bail!("usage: cargo run dataset-path"),
    };
    // Load the dataset and resize it to the usual imagenet dimension of 224x224.
    let dataset = imagenet::load_from_dir(dataset_dir)?;
    println!("{dataset:?}");

    let mut vs = nn::VarStore::new(Device::cuda_if_available());
    let net = Net::new(&vs.root(),2);
    let mut opt = nn::Adam::default().build(&vs, 1e-4)?;
    for epoch in 1..10 {
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
    model.save("model.pt")?;

    Ok(())
}