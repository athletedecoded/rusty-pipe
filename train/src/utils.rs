use rand::{seq::SliceRandom, thread_rng};
use tch::vision::imagenet;

pub fn load_from_dir<T: AsRef<Path>>(dir: T) -> Result<Dataset, TchError> {
    let mut train_images: Vec<Tensor> = vec![];
    let mut train_labels: Vec<Tensor> = vec![];
    let mut test_images: Vec<Tensor> = vec![];
    let mut test_labels: Vec<Tensor> = vec![];
    
    let classes = std::fs::read_dir(&dir)?
        .filter_map(|d| d.ok().map(|d| d.path()))
        .filter(|d| d.is_dir())
        .filter_map(|d| d.file_name().map(|d| d.to_os_string()))
        .collect::<Vec<_>>();
    println!("classes: {classes:?}");

    for (label_index, label_dir) in classes.iter().enumerate() {
        let label_index = label_index as i64;
        let images = imagenet::load_images_from_dir(&dir.join(label_dir))?;
        // Random train_val permutation
        let mut rng = thread_rng();
        images.shuffle(&mut rng);
        let split_index = (permutation.len() as f32 * 0.8) as usize;
        let (train_set, val_set) = images.split_at(split_index);
        // Count
        let n_train = train_set.size()[0];
        let n_val = val_set.size()[0];
        // Training data
        train_images.push(train_set);
        train_labels.push(Tensor::ones(&[n_train], kind::INT64_CPU) * label_index);
        // Val data
        test_images.push(val_set);
        test_labels.push(Tensor::ones(&[n_val], kind::INT64_CPU) * label_index);
    }
    Ok(Dataset {
        train_images: Tensor::f_cat(&train_images, 0)?,
        train_labels: Tensor::f_cat(&train_labels, 0)?,
        test_images: Tensor::f_cat(&test_images, 0)?,
        test_labels: Tensor::f_cat(&test_labels, 0)?,
        labels: classes.len() as i64,
    })
}