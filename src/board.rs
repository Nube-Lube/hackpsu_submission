use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::gemini::{Hazard, Severity};

// Every sighting adds weight to a threat's score; the score halves every HALF_LIFE.
// Two camera sightings within about one half-life confirm a threat; one alone fades away.
const HALF_LIFE: Duration = Duration::from_secs(30);
const CONFIRM_SCORE: f64 = 1.5;
const FADE_SCORE: f64 = 0.25; // below this, unconfirmed and low-severity threats are dropped
const STALE_AFTER: Duration = Duration::from_secs(60); // "not seen recently" on the dashboard
// Sightings of the same kind this close (fraction of the map) are treated as the same threat,
// if that threat is confirmed or was seen within MATCH_WINDOW. This lets a moving threat
// build up score across several cameras.
const MATCH_RADIUS: f32 = 0.15;
const MATCH_WINDOW: Duration = Duration::from_secs(60);
const TRAIL_LEN: usize = 10;
pub const CAMERA_WEIGHT: f64 = 1.0;
pub const WORKER_WEIGHT: f64 = 2.0; // a person reporting it confirms on the spot

/// A point on the floor-plan map; both axes run 0..1.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Pos {
    pub x: f32,
    pub y: f32,
}

impl Pos {
    fn distance(self, other: Pos) -> f32 {
        (self.x - other.x).hypot(self.y - other.y)
    }
}

/// One report of a hazard, from a camera frame or a worker.
pub struct Sighting {
    pub hazard: Hazard,
    pub source: String,   // camera name, or "worker: <name>"
    pub pos: Option<Pos>, // None if the camera isn't placed on the map / worker didn't tap one
    pub weight: f64,
}

/// Site-wide list of threats.
/// Unconfirmed sightings and low-severity threats decay away on their own;
/// confirmed medium/high threats stay until a person resolves them.
pub struct Board {
    threats: Vec<Threat>,
    next_id: u64,
    critical_after: HashMap<String, Duration>, // from hazards.toml `critical_after_secs`
}

struct Threat {
    id: u64,
    hazard: Hazard, // latest description; severity is the highest seen
    sources: Vec<String>, // every camera/worker that has reported it, first one first
    trail: Vec<Pos>,      // where it has been seen, oldest first; last = current position
    score: f64,
    scored_at: Instant, // when `score` was last decayed
    last_seen: Instant,
    confirmed_at: Option<Instant>,
    critical: bool,
}

impl Threat {
    fn decay(&mut self, now: Instant) {
        let halvings = (now - self.scored_at).as_secs_f64() / HALF_LIFE.as_secs_f64();
        self.score *= 0.5f64.powf(halvings);
        self.scored_at = now;
    }

    fn needs_person(&self) -> bool {
        self.hazard.severity >= Severity::Medium
    }

    fn matches(&self, s: &Sighting, now: Instant) -> bool {
        if self.hazard.kind != s.hazard.kind {
            return false;
        }
        match (self.trail.last(), s.pos) {
            (Some(&here), Some(there)) => {
                here.distance(there) <= MATCH_RADIUS
                    && (self.confirmed_at.is_some() || now - self.last_seen <= MATCH_WINDOW)
            }
            // Without positions, fall back to "same camera / same worker".
            _ => self.sources.contains(&s.source),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ThreatView {
    pub id: u64,
    #[serde(flatten)]
    pub hazard: Hazard,
    pub sources: Vec<String>,
    pub trail: Vec<Pos>,
    pub critical: bool,
    pub stale: bool,          // confirmed but not seen for a while (still needs resolving)
    pub seconds: u64,         // since confirmed
    pub needs_person: bool,   // false = will fade on its own once it stops being seen
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    Detected { id: u64, kind: String, reason: String },
    Critical { id: u64, kind: String, reason: String },
    Resolved { id: u64, kind: String, by: Resolver },
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Resolver {
    Person, // someone pressed Resolve
    Faded,  // low severity and no longer seen
}

impl Board {
    pub fn new(critical_after: HashMap<String, Duration>) -> Self {
        Self { threats: Vec::new(), next_id: 1, critical_after }
    }

    /// Add a sighting: reinforces a matching threat or starts a new one.
    pub fn sight(&mut self, s: Sighting, now: Instant) -> Vec<Event> {
        let mut events = Vec::new();
        let existing = self.threats.iter().position(|t| t.matches(&s, now));
        let t = match existing {
            Some(i) => {
                let t = &mut self.threats[i];
                t.decay(now);
                let severity = t.hazard.severity.max(s.hazard.severity);
                t.hazard = Hazard { severity, ..s.hazard };
                if !t.sources.contains(&s.source) {
                    t.sources.push(s.source);
                }
                if let Some(pos) = s.pos
                    && t.trail.last() != Some(&pos)
                {
                    t.trail.push(pos);
                    if t.trail.len() > TRAIL_LEN {
                        t.trail.remove(0);
                    }
                }
                t
            }
            None => {
                self.threats.push(Threat {
                    id: self.next_id,
                    hazard: s.hazard,
                    sources: vec![s.source],
                    trail: s.pos.into_iter().collect(),
                    score: 0.0,
                    scored_at: now,
                    last_seen: now,
                    confirmed_at: None,
                    critical: false,
                });
                self.next_id += 1;
                self.threats.last_mut().unwrap()
            }
        };
        t.score += s.weight;
        t.last_seen = now;
        if t.confirmed_at.is_none() && t.score >= CONFIRM_SCORE {
            t.confirmed_at = Some(now);
            events.push(Event::Detected { id: t.id, kind: t.hazard.kind.clone(), reason: t.hazard.reason.clone() });
        }
        events.extend(self.tick(now));
        events
    }

    /// Time passes: scores decay, small threats fade, long-running ones escalate.
    pub fn tick(&mut self, now: Instant) -> Vec<Event> {
        let mut events = Vec::new();
        self.threats.retain_mut(|t| {
            t.decay(now);
            if t.score >= FADE_SCORE {
                return true;
            }
            match t.confirmed_at {
                None => false, // a one-off sighting that was never confirmed: drop silently
                Some(_) if !t.needs_person() => {
                    events.push(Event::Resolved { id: t.id, kind: t.hazard.kind.clone(), by: Resolver::Faded });
                    false
                }
                Some(_) => true, // confirmed medium/high: only a person can clear it
            }
        });
        for t in &mut self.threats {
            if let Some(since) = t.confirmed_at
                && let Some(&limit) = self.critical_after.get(&t.hazard.kind)
                && !t.critical
                && now - since >= limit
            {
                t.critical = true;
                events.push(Event::Critical { id: t.id, kind: t.hazard.kind.clone(), reason: t.hazard.reason.clone() });
            }
        }
        events
    }

    /// A person marks a threat as dealt with.
    pub fn resolve(&mut self, id: u64) -> Option<Event> {
        let i = self.threats.iter().position(|t| t.id == id)?;
        let t = self.threats.remove(i);
        Some(Event::Resolved { id, kind: t.hazard.kind, by: Resolver::Person })
    }

    /// Confirmed threats, most urgent first. `source` limits it to ones that camera has reported.
    pub fn confirmed(&self, now: Instant, source: Option<&str>) -> Vec<ThreatView> {
        let mut out: Vec<ThreatView> = self
            .threats
            .iter()
            .filter(|t| source.is_none_or(|s| t.sources.iter().any(|x| x == s)))
            .filter_map(|t| {
                Some(ThreatView {
                    id: t.id,
                    hazard: t.hazard.clone(),
                    sources: t.sources.clone(),
                    trail: t.trail.clone(),
                    critical: t.critical,
                    stale: now - t.last_seen >= STALE_AFTER,
                    seconds: (now - t.confirmed_at?).as_secs(),
                    needs_person: t.needs_person(),
                })
            })
            .collect();
        out.sort_by(|a, b| {
            (b.critical, b.hazard.severity, a.stale).cmp(&(a.critical, a.hazard.severity, b.stale))
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOWN: &str = "person_down";
    const CAM: &str = "Dock";

    fn board() -> Board {
        Board::new(HashMap::from([(DOWN.to_string(), Duration::from_secs(15))]))
    }

    fn seen(kind: &str, severity: Severity, weight: f64) -> Sighting {
        Sighting {
            hazard: Hazard { kind: kind.into(), severity, location: "floor".into(), reason: "test".into() },
            source: CAM.into(),
            pos: None,
            weight,
        }
    }

    fn seen_at(kind: &str, source: &str, x: f32) -> Sighting {
        Sighting { source: source.into(), pos: Some(Pos { x, y: 0.5 }), ..seen(kind, Severity::High, CAMERA_WEIGHT) }
    }

    fn secs(n: u64) -> Duration {
        Duration::from_secs(n)
    }

    #[test]
    fn one_sighting_fades_silently() {
        let mut b = board();
        let t0 = Instant::now();
        assert!(b.sight(seen("spill", Severity::High, CAMERA_WEIGHT), t0).is_empty());
        assert!(b.tick(t0 + secs(120)).is_empty());
        assert!(b.threats.is_empty());
    }

    #[test]
    fn two_quick_sightings_confirm_once() {
        let mut b = board();
        let t0 = Instant::now();
        b.sight(seen("spill", Severity::High, CAMERA_WEIGHT), t0);
        let events = b.sight(seen("spill", Severity::High, CAMERA_WEIGHT), t0 + secs(5));
        assert!(matches!(&events[..], [Event::Detected { .. }]));
        assert!(b.sight(seen("spill", Severity::High, CAMERA_WEIGHT), t0 + secs(10)).is_empty());
    }

    #[test]
    fn sightings_far_apart_never_confirm() {
        let mut b = board();
        let t0 = Instant::now();
        b.sight(seen("spill", Severity::High, CAMERA_WEIGHT), t0);
        // Three half-lives later the first sighting is mostly forgotten.
        assert!(b.sight(seen("spill", Severity::High, CAMERA_WEIGHT), t0 + secs(90)).is_empty());
    }

    #[test]
    fn worker_report_confirms_immediately() {
        let mut b = board();
        let events = b.sight(seen("spill", Severity::Medium, WORKER_WEIGHT), Instant::now());
        assert!(matches!(&events[..], [Event::Detected { .. }]));
    }

    #[test]
    fn low_severity_fades_after_confirmation() {
        let mut b = board();
        let t0 = Instant::now();
        b.sight(seen("trip_hazard", Severity::Low, WORKER_WEIGHT), t0);
        let events = b.tick(t0 + secs(300));
        assert!(matches!(&events[..], [Event::Resolved { by: Resolver::Faded, .. }]));
        assert!(b.confirmed(t0 + secs(300), None).is_empty());
    }

    #[test]
    fn medium_severity_waits_for_a_person() {
        let mut b = board();
        let t0 = Instant::now();
        b.sight(seen("trip_hazard", Severity::Medium, WORKER_WEIGHT), t0);
        assert!(b.tick(t0 + secs(3600)).is_empty());
        let view = b.confirmed(t0 + secs(3600), None);
        assert!(view[0].stale);
        let id = view[0].id;
        assert!(matches!(b.resolve(id), Some(Event::Resolved { by: Resolver::Person, .. })));
        assert!(b.confirmed(t0 + secs(3600), None).is_empty());
    }

    #[test]
    fn person_down_escalates_after_15s() {
        let mut b = board();
        let t0 = Instant::now();
        b.sight(seen(DOWN, Severity::High, CAMERA_WEIGHT), t0);
        b.sight(seen(DOWN, Severity::High, CAMERA_WEIGHT), t0); // confirmed at t0
        assert!(b.tick(t0 + secs(10)).is_empty());
        let events = b.tick(t0 + secs(16));
        assert!(matches!(&events[..], [Event::Critical { .. }]));
        assert!(b.confirmed(t0 + secs(16), None)[0].critical);
    }

    #[test]
    fn moving_threat_confirms_across_nearby_cameras() {
        let mut b = board();
        let t0 = Instant::now();
        // Seen once by each of two cameras 10% of the map apart: one threat, now confirmed.
        assert!(b.sight(seen_at("intruder", "Dock", 0.40), t0).is_empty());
        let events = b.sight(seen_at("intruder", "Lobby", 0.50), t0 + secs(5));
        assert!(matches!(&events[..], [Event::Detected { .. }]));
        let view = b.confirmed(t0 + secs(5), None);
        assert_eq!(view.len(), 1);
        assert_eq!(view[0].sources, ["Dock", "Lobby"]);
        assert_eq!(view[0].trail.len(), 2);
    }

    #[test]
    fn far_apart_sightings_stay_separate() {
        let mut b = board();
        let t0 = Instant::now();
        b.sight(seen_at("spill", "Dock", 0.1), t0);
        b.sight(seen_at("spill", "Gate", 0.9), t0);
        assert_eq!(b.threats.len(), 2);
        assert!(b.confirmed(t0, None).is_empty());
    }

    #[test]
    fn severity_only_ratchets_up() {
        let mut b = board();
        let t0 = Instant::now();
        b.sight(seen("spill", Severity::High, CAMERA_WEIGHT), t0);
        b.sight(seen("spill", Severity::Low, CAMERA_WEIGHT), t0);
        assert_eq!(b.confirmed(t0, None)[0].hazard.severity, Severity::High);
    }
}
