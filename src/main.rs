mod config;
mod gemini;
mod rules;

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Query, State},
    http::{HeaderMap, StatusCode, header},
    response::{Html, IntoResponse},
    routing::{get, post},
};
use gemini::{Gemini, Hazard};
use rules::{Active, Event, Tracker};
use serde::{Deserialize, Serialize};

const LOG_LEN: usize = 50; // events kept for the dashboard
const OFFLINE_AFTER: Duration = Duration::from_secs(10);

struct AppState {
    gemini: Gemini,
    cameras: Mutex<HashMap<String, Camera>>,
    log: Mutex<VecDeque<Logged>>,
    // All cameras share one Gemini quota, so calls are spaced globally (15/min on Flash Lite).
    next_gemini_call: Mutex<Instant>,
    gemini_gap: Duration,
    critical_after: HashMap<String, Duration>, // copied into each new camera's tracker
}

struct Camera {
    tracker: Tracker,
    last_raw: Vec<Hazard>,  // Gemini's latest answer, reused while the scene is unchanged
    last_frame: Bytes,      // the latest frame Gemini judged, shown on the dashboard
    frame_version: u64,     // bumps with each new frame so the dashboard knows to reload it
    last_seen: Option<Instant>,
}

#[derive(Serialize, Clone)]
struct Logged {
    camera: String,
    at_ms: u64, // unix time, so the browser can format it
    #[serde(flatten)]
    event: Event,
}

#[derive(Deserialize)]
struct CameraQuery {
    camera: Option<String>,
}

impl CameraQuery {
    fn name(&self) -> String {
        let name = self.camera.as_deref().unwrap_or("").trim();
        if name.is_empty() { "Camera".into() } else { name.chars().take(40).collect() }
    }
}

// What a camera page gets back after each frame/tick.
#[derive(Serialize)]
struct Analysis {
    checked: bool,       // false = Gemini was skipped (unchanged scene or quota spacing)
    raw: Vec<Hazard>,    // what Gemini saw in the latest frame (unconfirmed)
    active: Vec<Active>, // confirmed hazards, most urgent first
    events: Vec<Event>,  // what changed this tick: detected / critical / resolved
}

// Everything the dashboard needs, in one poll.
#[derive(Serialize)]
struct Overview {
    cameras: Vec<CameraView>,
    log: Vec<Logged>, // newest first
}

#[derive(Serialize)]
struct CameraView {
    name: String,
    online: bool,
    frame_version: u64, // 0 = no frame yet
    raw: Vec<Hazard>,
    active: Vec<Active>,
}

// `hackpsu`            start the server on port 3000
// `hackpsu --tunnel`   also open a public https:// link (phones need https for camera access)
// `hackpsu photo.jpg`  analyze one image and print the result
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = config::Config::load()?;
    let gemini = Gemini::new(&config)?;
    let arg = std::env::args().nth(1);
    let tunnel = arg.as_deref() == Some("--tunnel");

    if let Some(path) = arg.filter(|_| !tunnel) {
        let report = gemini.analyze(&std::fs::read(&path)?, None).await?;
        println!("{report:#?}");
        return Ok(());
    }

    let rpm: u64 = std::env::var("GEMINI_RPM").ok().and_then(|v| v.parse().ok()).unwrap_or(15);
    let state_has_key = gemini.has_server_key();
    let state = AppState {
        gemini,
        cameras: Mutex::default(),
        log: Mutex::default(),
        next_gemini_call: Mutex::new(Instant::now()),
        gemini_gap: Duration::from_millis(60_000 / rpm),
        critical_after: config.critical_after(),
    };

    let app = Router::new()
        .route("/", get(|| async { Html(include_str!("../static/dashboard.html")) }))
        .route("/camera", get(|| async { Html(include_str!("../static/camera.html")) }))
        .route("/analyze", post(analyze))
        .route("/state", get(overview))
        .route("/frame", get(frame))
        .layer(DefaultBodyLimit::max(10 * 1024 * 1024)) // phone photos can exceed axum's 2 MB default
        .with_state(Arc::new(state));

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    println!("dashboard: http://localhost:3000   camera: http://localhost:3000/camera");
    if !state_has_key {
        println!("no server Gemini key: each camera must enter its own key on the camera page");
    }
    // Held until the server stops; dropping it kills cloudflared.
    let _tunnel = if tunnel { Some(start_tunnel()?) } else { None };
    axum::serve(listener, app).await?;
    Ok(())
}

// Runs a Cloudflare quick tunnel (no account needed) and prints its link + a QR code
// for phones once cloudflared reports it.
fn start_tunnel() -> anyhow::Result<tokio::process::Child> {
    use anyhow::Context;
    use tokio::io::{AsyncBufReadExt, BufReader};

    let mut child = tokio::process::Command::new("cloudflared")
        .args(["tunnel", "--url", "http://localhost:3000"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .context("cloudflared not found; install it: winget install --id Cloudflare.cloudflared")?;

    // cloudflared logs to stderr; keep draining it so its pipe never fills up.
    let stderr = child.stderr.take().expect("stderr is piped");
    tokio::spawn(async move {
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            let url = line
                .split_whitespace()
                .find(|w| w.starts_with("https://") && w.ends_with(".trycloudflare.com"));
            if let Some(url) = url {
                println!("\npublic dashboard: {url}\npublic camera:    {url}/camera\n");
                print_qr(&format!("{url}/camera"));
            }
        }
    });
    Ok(child)
}

fn print_qr(url: &str) {
    use qrcode::{QrCode, render::unicode::Dense1x2};
    if let Ok(code) = QrCode::new(url) {
        // Inverted colors so it scans on a dark terminal background.
        let qr = code.render::<Dense1x2>().dark_color(Dense1x2::Light).light_color(Dense1x2::Dark).build();
        println!("scan to open the camera page:\n{qr}");
    }
}

// A camera POSTs JPEG bytes (or an empty body for "scene unchanged").
// Gemini judges the frame, that camera's tracker decides what to alert on.
async fn analyze(
    State(state): State<Arc<AppState>>,
    Query(q): Query<CameraQuery>,
    headers: HeaderMap,
    jpeg: Bytes,
) -> Result<Json<Analysis>, (StatusCode, String)> {
    let name = q.name();
    // BYOK: a camera may send its own Gemini key. Used for this request only, never stored or logged.
    let user_key = headers
        .get("x-gemini-key")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|k| !k.is_empty());

    // Take a Gemini slot if one is free; otherwise treat this frame as unchanged and
    // let the camera try again next second. Cameras end up taking turns.
    // Cameras with their own key have their own quota, so they skip the shared slot.
    let use_gemini = !jpeg.is_empty() && (user_key.is_some() || {
        let mut next = state.next_gemini_call.lock().unwrap();
        let now = Instant::now();
        if now >= *next {
            *next = now + state.gemini_gap;
            true
        } else {
            false
        }
    });

    // The mutexes are never held across this await.
    let report = if use_gemini {
        Some(state.gemini.analyze(&jpeg, user_key).await.map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?)
    } else {
        None
    };

    let now = Instant::now();
    let mut cameras = state.cameras.lock().unwrap();
    let cam = cameras.entry(name.clone()).or_insert_with(|| Camera {
        tracker: Tracker::new(state.critical_after.clone()),
        last_raw: Vec::new(),
        last_frame: Bytes::new(),
        frame_version: 0,
        last_seen: None,
    });
    cam.last_seen = Some(now);
    let events = match report {
        Some(report) => {
            cam.last_frame = jpeg;
            cam.frame_version += 1;
            let events = cam.tracker.update(&report, now);
            cam.last_raw = report.hazards;
            events
        }
        None => cam.tracker.tick(now),
    };

    let at_ms = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64;
    let mut log = state.log.lock().unwrap();
    for event in &events {
        log.push_front(Logged { camera: name.clone(), at_ms, event: event.clone() });
    }
    log.truncate(LOG_LEN);

    Ok(Json(Analysis {
        checked: use_gemini,
        active: cam.tracker.active(now),
        raw: cam.last_raw.clone(),
        events,
    }))
}

async fn overview(State(state): State<Arc<AppState>>) -> Json<Overview> {
    let now = Instant::now();
    let cameras = state.cameras.lock().unwrap();
    let mut views: Vec<CameraView> = cameras
        .iter()
        .map(|(name, cam)| CameraView {
            name: name.clone(),
            online: cam.last_seen.is_some_and(|t| now - t < OFFLINE_AFTER),
            frame_version: cam.frame_version,
            raw: cam.last_raw.clone(),
            active: cam.tracker.active(now),
        })
        .collect();
    views.sort_by(|a, b| a.name.cmp(&b.name));
    let log = state.log.lock().unwrap().iter().cloned().collect();
    Json(Overview { cameras: views, log })
}

async fn frame(State(state): State<Arc<AppState>>, Query(q): Query<CameraQuery>) -> impl IntoResponse {
    let cameras = state.cameras.lock().unwrap();
    match cameras.get(&q.name()) {
        Some(cam) if !cam.last_frame.is_empty() => {
            ([(header::CONTENT_TYPE, "image/jpeg")], cam.last_frame.clone()).into_response()
        }
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}
