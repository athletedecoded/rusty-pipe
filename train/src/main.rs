use anyhow::{bail, Result};
mod cnn;
mod vgg;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();    
    let dataset_dir = match args.as_slice() {
        [_, d] => d.to_owned(),
        _ => bail!("usage: cargo run dataset-path"),
    };
    cnn::run(dataset_dir)
}