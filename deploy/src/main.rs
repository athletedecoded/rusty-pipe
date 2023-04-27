use actix_web::middleware::Logger;
use actix_web::{App, HttpServer};
use log::LevelFilter;

mod routes;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::Builder::from_default_env()
        .filter_level(LevelFilter::Debug)
        .init();
    println!("Starting pytorch model server...");
    HttpServer::new(|| {
        App::new()
            .wrap(Logger::default())
            .service(routes::index)
            // .service(routes::predict)
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}