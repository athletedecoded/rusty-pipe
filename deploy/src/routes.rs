use actix_multipart::Multipart;
use actix_web::post;
use actix_web::{get, Error, HttpResponse, Result};
use serde_json::json;
use std::path::Path;

use crate::logic::files;
use crate::logic::predict_image;

#[get("/")]
pub async fn index() -> HttpResponse {
    log::info!("route: / function: index()");
    let message = "Send an image payload using curl with the following command:\ncurl -X POST -H \"Content-Type: multipart/form-data\" -F \"image=@/path/to/your/image.jpg\" http://127.0.0.1:8080/predict";
    HttpResponse::Ok().content_type("text/plain").body(message)
}

#[post("/predict")]
pub async fn predict(payload: Multipart) -> Result<HttpResponse, Error> {
    //log starting upload and include route and function name
    log::info!("route: /predict function: predict()");
    // create the path if it doesn't exist
    let temp_dir = Path::new("./tmp/");
    if !temp_dir.exists() {
        log::info!("Creating temp directory: {:?}", temp_dir);
        std::fs::create_dir_all(temp_dir)?;
    }
    // save the file to the temp directory
    let file_path = match files::save_file(payload, "/tmp/image.jpg".to_string()).await {
        Ok(path) => path,
        Err(e) => {
            let error_message = format!("File upload failed with error: {:?}", e);
            log::error!(
                "Route: /predict, Function: predict_image, Error: {}",
                error_message
            );
            return Ok(
                HttpResponse::Ok().json(json!({ "status": "error", "message": error_message }))
            );
        }
    };
    let cloned_file_path = file_path.clone();
    let prediction = match predict_image(cloned_file_path).await {
        Ok(p) => p,
        Err(e) => {
            let error_message = format!("Prediction failed with error: {:?}", e);
            log::error!(
                "Route: /predict, Function: predict_image, Error: {}",
                error_message
            );
            return Ok(HttpResponse::InternalServerError()
                .json(json!({ "status": "error", "message": error_message })));
        }
    };
    //delete file after prediction
    std::fs::remove_file(file_path)?;
    log::info!(
        "Route: /predict, Function: predict_image, Result: {:?}",
        prediction
    );

    Ok(HttpResponse::Ok().json(json!({ "status": "success", "result": prediction })))
}