mod gemini;

use std::sync::Arc;

use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::Html,
    routing::{get, post},
};
use gemini::{Gemini, Report};

// `cargo run -- photo.jpg` analyzes one image; `cargo run` starts the web server.
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let gemini = Gemini::from_env()?;

    if let Some(path) = std::env::args().nth(1) {
        let report = gemini.analyze(&std::fs::read(&path)?).await?;
        println!("{report:#?}");
        return Ok(());
    }

    let app = Router::new()
        .route("/", get(|| async { Html(include_str!("../static/index.html")) }))
        .route("/analyze", post(analyze))
        .layer(DefaultBodyLimit::max(10 * 1024 * 1024)) // phone photos can exceed axum's 2 MB default
        .with_state(Arc::new(gemini));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    println!("open http://localhost:3000");
    axum::serve(listener, app).await?;
    Ok(())
}

// The browser POSTs raw JPEG bytes; we reply with the hazard report as JSON.
async fn analyze(
    State(gemini): State<Arc<Gemini>>,
    jpeg: Bytes,
) -> Result<Json<Report>, (StatusCode, String)> {
    gemini
        .analyze(&jpeg)
        .await
        .map(Json)
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))
}
