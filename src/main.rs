mod gemini;
mod rules;

use std::sync::{Arc, Mutex};
use std::time::Instant;

use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::Html,
    routing::{get, post},
};
use gemini::{Gemini, Hazard};
use rules::{Active, Event, Tracker};
use serde::Serialize;

struct AppState {
    gemini: Gemini,
    session: Mutex<Session>, // one camera for now, so one shared session
}

#[derive(Default)]
struct Session {
    tracker: Tracker,
    last_raw: Vec<Hazard>, // Gemini's latest answer, reused while the scene is unchanged
}

#[derive(Serialize)]
struct Analysis {
    raw: Vec<Hazard>,    // what Gemini saw in this frame (unconfirmed)
    active: Vec<Active>, // confirmed hazards, most urgent first
    events: Vec<Event>,  // what changed this frame: detected / critical / resolved
}

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
        .with_state(Arc::new(AppState { gemini, session: Mutex::default() }));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    println!("open http://localhost:3000");
    axum::serve(listener, app).await?;
    Ok(())
}

// The browser POSTs raw JPEG bytes; Gemini judges the frame, the tracker decides what to alert on.
// An empty body means "scene unchanged": skip Gemini, just let time pass in the tracker.
async fn analyze(
    State(state): State<Arc<AppState>>,
    jpeg: Bytes,
) -> Result<Json<Analysis>, (StatusCode, String)> {
    let report = if jpeg.is_empty() {
        None
    } else {
        Some(
            state
                .gemini
                .analyze(&jpeg)
                .await
                .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?,
        )
    };

    let now = Instant::now();
    let mut session = state.session.lock().unwrap();
    let events = match report {
        Some(report) => {
            let events = session.tracker.update(&report, now);
            session.last_raw = report.hazards;
            events
        }
        None => session.tracker.tick(now),
    };
    Ok(Json(Analysis {
        active: session.tracker.active(now),
        events,
        raw: session.last_raw.clone(),
    }))
}
