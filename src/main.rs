mod board;
mod config;
mod gemini;
mod openai;
mod vision;

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Path, Query, State},
    http::{HeaderMap, StatusCode, header},
    response::{Html, IntoResponse},
    routing::{get, post},
};
use board::{Board, CAMERA_WEIGHT, Event, Pos, Sighting, ThreatView, WORKER_WEIGHT};
use gemini::{Hazard, Severity};
use serde::{Deserialize, Serialize};
use vision::Vision;

const LOG_LEN: usize = 50; // events kept for the dashboard
const OFFLINE_AFTER: Duration = Duration::from_secs(10);
const WORKER_GONE_AFTER: Duration = Duration::from_secs(30); // worker page re-sends every 10 s
const POSITIONS_FILE: &str = "positions.json"; // where cameras sit on the map, kept across restarts

struct AppState {
    vision: Vision,
    cameras: Mutex<HashMap<String, Camera>>,
    positions: Mutex<HashMap<String, Pos>>, // camera name -> spot on the floor-plan map
    workers: Mutex<HashMap<String, (Pos, Instant)>>, // worker name -> where they said they are, when
    board: Mutex<Board>,
    log: Mutex<VecDeque<Logged>>,
    // All cameras share one Gemini quota, so calls are spaced globally (15/min on Flash Lite).
    next_gemini_call: Mutex<Instant>,
    gemini_gap: Duration,
    kinds: Vec<String>, // hazard kinds from hazards.toml, offered on the worker report page
    gmaps_key: Option<String>, // server-side Google Maps key for floor-plan snapshots
}

#[derive(Default)]
struct Camera {
    last_raw: Vec<Hazard>, // Gemini's latest answer, reused while the scene is unchanged
    last_frame: Bytes,     // the latest frame Gemini judged, shown on the dashboard
    frame_version: u64,    // bumps with each new frame so the dashboard knows to reload it
    last_seen: Option<Instant>,
}

#[derive(Serialize, Clone)]
struct Logged {
    source: String, // camera name, "worker: <name>", "dashboard", or "board" (fading/escalation)
    at_ms: u64,     // unix time, so the browser can format it
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

#[derive(Deserialize)]
struct IdQuery {
    id: u64,
}

// The worker page saying "I'm here".
#[derive(Deserialize)]
struct CheckIn {
    name: String,
    pos: Pos,
}

#[derive(Serialize)]
struct WorkerView {
    name: String,
    pos: Pos,
}

#[derive(Deserialize)]
struct PlaceQuery {
    camera: String,
    x: f32,
    y: f32,
}

// What the worker report page sends.
#[derive(Deserialize)]
struct Report {
    name: String,
    kind: String,
    severity: Severity,
    location: String,
    note: String,
    pos: Option<Pos>, // where they tapped on the map, if they did
}

// What a camera page gets back after each frame/tick.
#[derive(Serialize)]
struct Analysis {
    checked: bool,           // false = Gemini was skipped (unchanged scene or quota spacing)
    raw: Vec<Hazard>,        // what Gemini saw in the latest frame (unconfirmed)
    active: Vec<ThreatView>, // this camera's confirmed threats, most urgent first
}

// Everything the dashboard needs, in one poll.
#[derive(Serialize)]
struct Overview {
    cameras: Vec<CameraView>,
    threats: Vec<ThreatView>, // the whole board, most urgent first
    workers: Vec<WorkerView>, // workers who checked in recently
    log: Vec<Logged>,         // newest first
}

#[derive(Serialize)]
struct CameraView {
    name: String,
    online: bool,
    frame_version: u64, // 0 = no frame yet
    raw: Vec<Hazard>,
    pos: Option<Pos>,   // None = not placed on the map yet
}

// `hackpsu`            start the server on port 3000
// `hackpsu --tunnel`   also open a public https:// link (phones need https for camera access)
// `hackpsu photo.jpg`  analyze one image and print the result
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = config::Config::load()?;
    let vision = Vision::from_env(&config)?;
    let arg = std::env::args().nth(1);
    let tunnel = arg.as_deref() == Some("--tunnel");

    if let Some(path) = arg.filter(|_| !tunnel) {
        let report = vision.analyze(&std::fs::read(&path)?, None).await?;
        println!("{report:#?}");
        return Ok(());
    }

    let rpm: u64 = std::env::var("GEMINI_RPM").ok().and_then(|v| v.parse().ok()).unwrap_or(15);
    // Google Maps key: env var or a gitignored key file holding only the raw key.
    let gmaps_key = std::env::var("GMAPS_KEY")
        .or_else(|_| std::fs::read_to_string(".env/gmaps_api.env"))
        .ok()
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty());
    if gmaps_key.is_none() {
        println!("no Google Maps key: floor-plan alignment disabled (set GMAPS_KEY or .env/gmaps_api.env)");
    }
    let has_server_key = vision.has_server_key();
    let state = Arc::new(AppState {
        vision,
        cameras: Mutex::default(),
        positions: Mutex::new(load_positions()),
        workers: Mutex::default(),
        board: Mutex::new(Board::new(config.critical_after())),
        log: Mutex::default(),
        next_gemini_call: Mutex::new(Instant::now()),
        gemini_gap: Duration::from_millis(60_000 / rpm),
        kinds: config.kinds(),
        gmaps_key,
    });

    // The board decays and escalates every second, even when no camera is sending.
    let ticker = state.clone();
    tokio::spawn(async move {
        let mut every = tokio::time::interval(Duration::from_secs(1));
        loop {
            every.tick().await;
            let events = ticker.board.lock().unwrap().tick(Instant::now());
            ticker.log_events("board", events);
        }
    });

    let app = Router::new()
        .route("/", get(landing))
        .route("/camera", get(|| async { Html(include_str!("../static/camera.html")) }))
        .route("/worker", get(|| async { Html(include_str!("../static/worker.html")) }))
        .route("/checkin", post(checkin))
        .route("/report", get(|| async { Html(include_str!("../static/report.html")) }).post(report))
        .route("/analyze", post(analyze))
        .route("/resolve", post(resolve))
        .route("/place", post(place))
        .route("/floorplan", get(floorplan).post(upload_floorplan))
        .route("/sfx/{file}", get(sfx))
        .route("/align", post(align))
        .route("/bounds", get(bounds))
        .route("/state", get(overview))
        .route("/kinds", get(|State(s): State<Arc<AppState>>| async move { Json(s.kinds.clone()) }))
        .route("/frame", get(frame))
        .layer(DefaultBodyLimit::max(10 * 1024 * 1024)) // phone photos can exceed axum's 2 MB default
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    println!("dashboard: http://localhost:3000   camera: /camera   worker: /worker");
    if !has_server_key {
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
                println!("\npublic dashboard: {url}\npublic camera:    {url}/camera\nworker app:       {url}/worker\n");
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

fn load_positions() -> HashMap<String, Pos> {
    std::fs::read_to_string(POSITIONS_FILE)
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_default()
}

fn clamp(p: Pos) -> Pos {
    Pos { x: p.x.clamp(0.0, 1.0), y: p.y.clamp(0.0, 1.0) }
}

impl AppState {
    fn log_events(&self, source: &str, events: Vec<Event>) {
        if events.is_empty() {
            return;
        }
        let at_ms = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64;
        let mut log = self.log.lock().unwrap();
        for event in events {
            log.push_front(Logged { source: source.to_string(), at_ms, event });
        }
        log.truncate(LOG_LEN);
    }
}

// A camera POSTs JPEG bytes (or an empty body for "scene unchanged").
// Gemini judges the frame; each hazard it sees becomes a sighting on the board.
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
        Some(state.vision.analyze(&jpeg, user_key).await.map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?)
    } else {
        None
    };

    let now = Instant::now();
    let raw = {
        let mut cameras = state.cameras.lock().unwrap();
        let cam = cameras.entry(name.clone()).or_default();
        cam.last_seen = Some(now);
        if let Some(report) = &report {
            cam.last_frame = jpeg;
            cam.frame_version += 1;
            cam.last_raw = report.hazards.clone();
        }
        cam.last_raw.clone()
    };

    // Hazards a camera sees are placed at the camera's spot on the map.
    let pos = state.positions.lock().unwrap().get(&name).copied();
    let mut events = Vec::new();
    let active = {
        let mut board = state.board.lock().unwrap();
        for hazard in report.map(|r| r.hazards).unwrap_or_default() {
            let sighting = Sighting { hazard, source: name.clone(), pos, weight: CAMERA_WEIGHT };
            events.extend(board.sight(sighting, now));
        }
        board.confirmed(now, Some(&name))
    };
    state.log_events(&name, events);

    Ok(Json(Analysis { checked: use_gemini, raw, active }))
}

// A worker reports a hazard by hand. It goes straight onto the board, already confirmed.
async fn report(State(state): State<Arc<AppState>>, Json(r): Json<Report>) -> Result<StatusCode, (StatusCode, String)> {
    if !state.kinds.contains(&r.kind) {
        return Err((StatusCode::BAD_REQUEST, format!("unknown hazard kind: {}", r.kind)));
    }
    let name: String = r.name.trim().chars().take(40).collect();
    let source = format!("worker: {}", if name.is_empty() { "anonymous" } else { &name });
    let hazard = Hazard {
        kind: r.kind,
        severity: r.severity,
        location: r.location.chars().take(200).collect(),
        reason: r.note.chars().take(500).collect(),
    };
    let sighting = Sighting { hazard, source: source.clone(), pos: r.pos.map(clamp), weight: WORKER_WEIGHT };
    let events = state.board.lock().unwrap().sight(sighting, Instant::now());
    state.log_events(&source, events);
    Ok(StatusCode::NO_CONTENT)
}

// A person marks a threat as dealt with.
async fn resolve(State(state): State<Arc<AppState>>, Query(q): Query<IdQuery>) -> StatusCode {
    let event = state.board.lock().unwrap().resolve(q.id);
    match event {
        Some(event) => {
            state.log_events("dashboard", vec![event]);
            StatusCode::NO_CONTENT
        }
        None => StatusCode::NOT_FOUND,
    }
}

// A worker taps "I'm here" on the map (and the page repeats it while open).
async fn checkin(State(state): State<Arc<AppState>>, Json(c): Json<CheckIn>) -> StatusCode {
    let name: String = c.name.trim().chars().take(40).collect();
    if name.is_empty() {
        return StatusCode::BAD_REQUEST;
    }
    state.workers.lock().unwrap().insert(name, (clamp(c.pos), Instant::now()));
    StatusCode::NO_CONTENT
}

// The dashboard drags a camera onto the map.
async fn place(State(state): State<Arc<AppState>>, Query(q): Query<PlaceQuery>) -> StatusCode {
    let mut positions = state.positions.lock().unwrap();
    positions.insert(q.camera, clamp(Pos { x: q.x, y: q.y }));
    match serde_json::to_string_pretty(&*positions).map(|json| std::fs::write(POSITIONS_FILE, json)) {
        Ok(Ok(())) => StatusCode::NO_CONTENT,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

// The dashboard carries a __GMAPS_KEY__ placeholder the server fills from its own key,
// so the key never lands in the repository.
async fn landing(State(state): State<Arc<AppState>>) -> Html<String> {
    Html(include_str!("../static/dashboard.html").replace("__GMAPS_KEY__", state.gmaps_key.as_deref().unwrap_or("")))
}

// Where the manager last framed the site, so the aligner reopens there.
async fn bounds() -> impl IntoResponse {
    let text = tokio::fs::read_to_string("bounds.json").await.unwrap_or_else(|_| "{}".into());
    ([(header::CONTENT_TYPE, "application/json"), (header::CACHE_CONTROL, "no-cache")], text).into_response()
}

// The manager framed the site on Google Maps; fetch exactly that frame as a satellite
// snapshot (same Web-Mercator projection, so zoom + center + the frame's pixel size
// cover the identical area) and make it the floor plan.
async fn align(
    State(state): State<Arc<AppState>>,
    Query(q): Query<AlignQuery>,
) -> Result<StatusCode, (StatusCode, String)> {
    let key = state.gmaps_key.as_deref().ok_or_else(|| {
        (StatusCode::BAD_REQUEST, "no Google Maps key on the server; set GMAPS_KEY or .env/gmaps_api.env".to_string())
    })?;
    let sane = q.lat.is_finite() && q.lat.abs() <= 90.0
        && q.lng.is_finite() && q.lng.abs() <= 180.0
        && (64..=1280).contains(&q.w) && (64..=1280).contains(&q.h)
        && q.zoom <= 21;
    if !sane {
        return Err((StatusCode::BAD_REQUEST, "bad align parameters".into()));
    }
    let url = format!(
        "https://maps.googleapis.com/maps/api/staticmap?center={},{}&zoom={}&size={}x{}&scale=2&maptype=satellite&key={}",
        q.lat, q.lng, q.zoom, q.w, q.h, key
    );
    let resp = reqwest::get(&url).await.map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;
    let status = resp.status();
    let content_type = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    let bytes = resp.bytes().await.map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;
    if !status.is_success() || !content_type.starts_with("image/") {
        // Google answers with JSON/text on bad keys or quota errors.
        let detail = String::from_utf8_lossy(&bytes[..bytes.len().min(300)]).to_string();
        return Err((StatusCode::BAD_GATEWAY, format!("staticmap returned {status} ({content_type}): {detail}")));
    }
    tokio::fs::write("floorplan.png", &bytes)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let frame = serde_json::json!({ "lat": q.lat, "lng": q.lng, "zoom": q.zoom, "w": q.w, "h": q.h });
    let _ = tokio::fs::write("bounds.json", frame.to_string()).await;
    Ok(StatusCode::NO_CONTENT)
}

// Optional floor-plan image next to the server; the dashboard draws a plain grid without one.
async fn floorplan() -> impl IntoResponse {
    for (file, mime) in [("floorplan.png", "image/png"), ("floorplan.jpg", "image/jpeg")] {
        if let Ok(bytes) = tokio::fs::read(file).await {
            return (
                [(header::CONTENT_TYPE, mime), (header::CACHE_CONTROL, "no-cache")],
                bytes,
            )
                .into_response();
        }
    }
    StatusCode::NOT_FOUND.into_response()
}

// A custom floor plan uploaded from the dashboard (PNG or JPEG, sniffed by magic bytes).
// Whichever of the two files isn't written is removed so the freshest upload wins.
async fn upload_floorplan(body: Bytes) -> StatusCode {
    let (name, other) = if body.starts_with(&[0x89, b'P', b'N', b'G']) {
        ("floorplan.png", "floorplan.jpg")
    } else if body.starts_with(&[0xFF, 0xD8]) {
        ("floorplan.jpg", "floorplan.png")
    } else {
        return StatusCode::UNSUPPORTED_MEDIA_TYPE;
    };
    match tokio::fs::write(name, &body).await {
        Ok(()) => {
            let _ = tokio::fs::remove_file(other).await;
            StatusCode::NO_CONTENT
        }
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

#[derive(Deserialize)]
struct AlignQuery {
    lat: f64,
    lng: f64,
    zoom: u8,
    w: u16,
    h: u16,
}

// Sound effects shipped in the sfx/ folder (warning, submitted); swap files without rebuilding.
async fn sfx(Path(file): Path<String>) -> impl IntoResponse {
    let safe = !file.starts_with('.')
        && file.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    let mime = match file.rsplit_once('.').map(|(_, ext)| ext) {
        Some("mp3") => "audio/mpeg",
        Some("wav") => "audio/wav",
        Some("ogg") => "audio/ogg",
        _ => "application/octet-stream",
    };
    match safe {
        true if let Ok(bytes) = tokio::fs::read(format!("sfx/{file}")).await => {
            ([(header::CONTENT_TYPE, mime)], bytes).into_response()
        }
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn overview(State(state): State<Arc<AppState>>) -> Json<Overview> {
    let now = Instant::now();
    let positions = state.positions.lock().unwrap().clone();
    let mut cameras: Vec<CameraView> = state
        .cameras
        .lock()
        .unwrap()
        .iter()
        .map(|(name, cam)| CameraView {
            name: name.clone(),
            online: cam.last_seen.is_some_and(|t| now - t < OFFLINE_AFTER),
            frame_version: cam.frame_version,
            raw: cam.last_raw.clone(),
            pos: positions.get(name).copied(),
        })
        .collect();
    cameras.sort_by(|a, b| a.name.cmp(&b.name));
    let threats = state.board.lock().unwrap().confirmed(now, None);
    let workers = {
        let mut workers = state.workers.lock().unwrap();
        workers.retain(|_, (_, seen)| now - *seen < WORKER_GONE_AFTER);
        let mut list: Vec<WorkerView> =
            workers.iter().map(|(name, (pos, _))| WorkerView { name: name.clone(), pos: *pos }).collect();
        list.sort_by(|a, b| a.name.cmp(&b.name));
        list
    };
    let log = state.log.lock().unwrap().iter().cloned().collect();
    Json(Overview { cameras, threats, workers, log })
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
