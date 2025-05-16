use std::time;
use tokio::time::sleep;
use tower_http::{
    services::{ServeDir, ServeFile},
    trace::TraceLayer,
};


#[tokio::main]
async fn main() {
    // initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::TRACE)
        .init();

    tokio::spawn(async move {
        use axum::{routing::get, Router};
        
        let app = Router::new()
            .route("/health", get(|| async { "ok" }))
            .nest_service("/pkg", ServeDir::new("./crates/whisper-tensor-webui/pkg/"))
            .nest_service("/assets", ServeDir::new("./crates/whisper-tensor-webui/assets/"))
            .route_service("/index.html", ServeFile::new("./crates/whisper-tensor-webui/assets/index.html"))
            .route_service("/", ServeFile::new("./crates/whisper-tensor-webui/assets/index.html"))
            .layer(TraceLayer::new_for_http());

        let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
        axum::serve(listener, app).await.unwrap();
    });

    loop {
        sleep(time::Duration::from_millis(1000)).await;
    }
}
