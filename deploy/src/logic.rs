//use actix_multipart::Multipart;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use tch::nn::ModuleT;
use tch::vision::imagenet;
use tch::Kind;
use tch::{Device, Tensor};

#[derive(Serialize, Deserialize, Debug)]
pub struct Prediction {
    pub probabilities: Vec<f64>,
    pub classes: Vec<String>,
}

pub mod files {
    use std::io::Write;

    use actix_multipart::Multipart;
    use actix_web::{web, Error};
    use futures::{StreamExt, TryStreamExt};
    use log::error;

    pub async fn save_file(mut payload: Multipart, file_path: String) -> Result<String, Error> {
        if let Ok(Some(mut field)) = payload.try_next().await {
            let filepath = format!(".{}", file_path);
            log::info!("func: save_file: filepath: {:?}", filepath);

            let mut f = web::block({
                let filepath_clone = filepath.clone();
                move || std::fs::File::create(&filepath_clone)
            })
            .await
            .map_err(|e| {
                error!("Error creating file: {:?}", e);
                actix_web::error::ErrorInternalServerError("Error creating file")
            })?;

            while let Some(chunk) = field.next().await {
                let data = chunk.unwrap();
                f = web::block({
                    let _filepath_clone = filepath.clone();
                    move || {
                        let mut file = f?;
                        file.write_all(&data)?;
                        Ok(file)
                    }
                })
                .await
                .map_err(|e| {
                    error!("Error writing to file: {:?}", e);
                    actix_web::error::ErrorInternalServerError("Error writing to file")
                })?;
            }
            let metadata = std::fs::metadata(&filepath).unwrap();
            let file_size = metadata.len();
            log::info!("func: save_file: file_size: {:?}", file_size);
            return Ok(filepath);
        }

        Err(actix_web::error::ErrorBadRequest("Error processing file"))
    }
}

pub async fn verify_image(image_path: String) -> Result<bool, Box<dyn std::error::Error>> {
    if fs::metadata(&image_path).is_err() {
        log::error!(
            "func: predict_image: image file not found: {:?}",
            image_path
        );
        return Err(format!("File not found: {}", image_path).into());
    }

    let metadata = fs::metadata(&image_path)?;
    let file_size = metadata.len();
    log::info!("Image path exists, size: {:?} bytes", file_size);

    Ok(true)
}

pub const CLASS_COUNT: i64 = 2;

pub const CLASSES: [&str; 2] = [
    "ant",
    "bee",
];

/// Returns the top k classes as well as the associated scores.
pub fn top_preds(tensor: &Tensor, k: i64) -> Vec<(f64, String)> {
    let tensor = match tensor.size().as_slice() {
        [CLASS_COUNT] => tensor.shallow_clone(),
        [1, CLASS_COUNT] => tensor.view((CLASS_COUNT,)),
        [1, 1, CLASS_COUNT] => tensor.view((CLASS_COUNT,)),
        _ => panic!("unexpected tensor shape {tensor:?}"),
    };
    let (values, indexes) = tensor.topk(k, 0, true, true);
    let values = Vec::<f64>::from(values);
    let indexes = Vec::<i64>::from(indexes);
    values
        .iter()
        .zip(indexes.iter())
        .map(|(&value, &index)| (value, CLASSES[index as usize].to_owned()))
        .collect()
}

pub async fn predict_image(image_path: String) -> Result<Prediction, Box<dyn std::error::Error>> {
    log::info!("route: /predict function: predict_image()");
    log::info!("func: predict_image: loading image: {:?}", image_path);
    //lets add logging to ensure that the image is loaded and the path is correct with error handling
    let verify_image = match verify_image(image_path.clone()).await {
        Ok(verify_image) => verify_image,
        Err(error) => return Err(error.into()),
    };
    if !verify_image {
        log::error!(
            "func: predict_image: image file not found: {:?}",
            image_path
        );
        return Err(actix_web::error::ErrorBadRequest("Image file not found").into());
    }

    let image = match imagenet::load_image_and_resize224(&image_path) {
        Ok(image) => image,
        Err(error) => return Err(error.into()),
    };

    log::info!("func: predict_image: starting");
    let mut vs = tch::nn::VarStore::new(tch::Device::Cpu);

    log::info!("func: predict_image: loading model: model.ot");
    let weight_pth = "model.ot";
    let model = tch::CModule::load(weight_pth)?;
    let output = model.forward_ts(&[image.unsqueeze(0)])?.softmax(-1, Kind::Float);
    // Dump output to console
    println!("output: {:?}", output);
    for (probability, class) in top_preds(&output, 2).iter() {
        println!("{:50} {:5.2}%", class, 100.0 * probability)
    }
    
    // let resnet34 = tch::vision::resnet::resnet18(&vs.root(), imagenet::CLASS_COUNT);
    // let weight_pth = "resnet34.ot";
    // vs.load(&weight_pth).unwrap();
    // log::info!("func: predict_image:  applying forward pass of the model to get the logits and convert them to probabilities via a softmax");
    // let output = resnet34
    //     .forward_t(&image.unsqueeze(0), /*train=*/ false)
    //     .softmax(-1, Kind::Float);
    // // Dump output to console
    // println!("output: {:?}", output);

    // for (probability, class) in imagenet::top(&output, 5).iter() {
    //     println!("{:50} {:5.2}%", class, 100.0 * probability)
    // }


    log::info!(
        "func: predict_image: : prediction results: {:?}",
        imagenet::top(&output, 5)
    );

    let top_result = top_preds(&output, 1);
    log::info!("Top result: {:?}", top_result);
    let (probability, class) = top_result.first().unwrap(); // Swapped variables
    log::info!("Class: {:?}", class);
    log::info!("Confidence: {:?}", probability);
    let confidence_f64 = *probability; // Directly use the probability value

    log::info!("func: predict_image: : prediction result: {:?}", class);
    log::info!(
        "func: predict_image: : prediction result: {:?}",
        confidence_f64
    );

    let prediction = Prediction {
        probabilities: vec![confidence_f64],
        classes: vec![class.to_string()], // Updated variable
    };

    log::info!("func: predict_image: : prediction result: {:?}", prediction);
    Ok(prediction)
}