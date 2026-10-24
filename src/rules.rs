use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::gemini::{Hazard, Report};

const WINDOW: usize = 3; // frames remembered per hazard
const CONFIRM_HITS: usize = 2; // must be seen in this many of the last WINDOW frames
const CLEAR_MISSES: usize = 2; // a confirmed hazard clears after this many misses in a row

// Hazards are matched across frames by kind only; Gemini's location text varies too much
// between frames to match on. One tracker per camera, one hazard per kind.
pub struct Tracker {
    tracks: HashMap<String, Track>,
    critical_after: HashMap<String, Duration>, // from hazards.toml `critical_after_secs`
}

struct Track {
    seen: VecDeque<bool>, // newest at the back
    latest: Hazard,
    confirmed_at: Option<Instant>,
    critical: bool,
}

#[derive(Debug, Serialize)]
pub struct Active {
    #[serde(flatten)]
    pub hazard: Hazard,
    pub critical: bool,
    pub seconds: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    Detected { kind: String, reason: String },
    Critical { kind: String, reason: String },
    Resolved { kind: String },
}

impl Tracker {
    pub fn new(critical_after: HashMap<String, Duration>) -> Self {
        Self { tracks: HashMap::new(), critical_after }
    }

    /// Feed one frame's report; returns the events this frame caused.
    pub fn update(&mut self, report: &Report, now: Instant) -> Vec<Event> {
        let mut events = Vec::new();

        for h in &report.hazards {
            self.tracks.entry(h.kind.clone()).or_insert_with(|| Track {
                seen: VecDeque::new(),
                latest: h.clone(),
                confirmed_at: None,
                critical: false,
            });
        }

        self.tracks.retain(|kind, t| {
            let hit = report.hazards.iter().find(|h| h.kind == *kind);
            t.seen.push_back(hit.is_some());
            if t.seen.len() > WINDOW {
                t.seen.pop_front();
            }
            if let Some(h) = hit {
                t.latest = h.clone();
            }

            match t.confirmed_at {
                None if t.seen.iter().filter(|&&s| s).count() >= CONFIRM_HITS => {
                    t.confirmed_at = Some(now);
                    events.push(Event::Detected { kind: kind.clone(), reason: t.latest.reason.clone() });
                }
                Some(_) if t.seen.iter().rev().take(CLEAR_MISSES).all(|&s| !s)
                    && t.seen.len() >= CLEAR_MISSES =>
                {
                    events.push(Event::Resolved { kind: kind.clone() });
                    return false;
                }
                _ => {}
            }

            // Drop unconfirmed tracks once they fall out of the window entirely.
            t.confirmed_at.is_some() || t.seen.iter().any(|&s| s)
        });

        events.extend(self.tick(now));
        events
    }

    /// Time passed without a new frame (scene unchanged): only time-based escalation can happen.
    pub fn tick(&mut self, now: Instant) -> Vec<Event> {
        let mut events = Vec::new();
        for (kind, t) in &mut self.tracks {
            if let Some(since) = t.confirmed_at
                && let Some(&limit) = self.critical_after.get(kind)
                && !t.critical
                && now - since >= limit
            {
                t.critical = true;
                events.push(Event::Critical { kind: kind.clone(), reason: t.latest.reason.clone() });
            }
        }
        events
    }

    /// Confirmed hazards right now, most urgent first.
    pub fn active(&self, now: Instant) -> Vec<Active> {
        let mut out: Vec<Active> = self
            .tracks
            .values()
            .filter_map(|t| {
                Some(Active {
                    hazard: t.latest.clone(),
                    critical: t.critical,
                    seconds: (now - t.confirmed_at?).as_secs(),
                })
            })
            .collect();
        out.sort_by(|a, b| (b.critical, b.hazard.severity).cmp(&(a.critical, a.hazard.severity)));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemini::Severity;

    fn frame(kinds: &[&str]) -> Report {
        Report {
            hazards: kinds
                .iter()
                .map(|&kind| Hazard {
                    kind: kind.into(),
                    severity: Severity::High,
                    location: "floor".into(),
                    reason: "test".into(),
                })
                .collect(),
        }
    }

    const DOWN: &str = "person_down";
    const SPILL: &str = "spill";

    fn tracker() -> Tracker {
        Tracker::new(HashMap::from([(DOWN.to_string(), Duration::from_secs(15))]))
    }

    #[test]
    fn single_frame_is_not_enough() {
        let mut t = tracker();
        let now = Instant::now();
        assert!(t.update(&frame(&[SPILL]), now).is_empty());
        assert!(t.update(&frame(&[]), now).is_empty());
        assert!(t.update(&frame(&[]), now).is_empty());
        assert!(t.active(now).is_empty());
    }

    #[test]
    fn two_of_three_confirms_once() {
        let mut t = tracker();
        let now = Instant::now();
        t.update(&frame(&[SPILL]), now);
        let events = t.update(&frame(&[SPILL]), now);
        assert_eq!(events, vec![Event::Detected { kind: SPILL.into(), reason: "test".into() }]);
        // Still there: no duplicate alert.
        assert!(t.update(&frame(&[SPILL]), now).is_empty());
    }

    #[test]
    fn resolves_after_two_misses() {
        let mut t = tracker();
        let now = Instant::now();
        t.update(&frame(&[SPILL]), now);
        t.update(&frame(&[SPILL]), now);
        assert!(t.update(&frame(&[]), now).is_empty()); // one miss tolerated
        assert_eq!(t.update(&frame(&[]), now), vec![Event::Resolved { kind: SPILL.into() }]);
        assert!(t.active(now).is_empty());
    }

    #[test]
    fn person_down_escalates_after_15s() {
        let mut t = tracker();
        let start = Instant::now();
        t.update(&frame(&[DOWN]), start);
        t.update(&frame(&[DOWN]), start); // confirmed at `start`
        assert!(t.update(&frame(&[DOWN]), start + Duration::from_secs(10)).is_empty());
        let events = t.update(&frame(&[DOWN]), start + Duration::from_secs(16));
        assert_eq!(events, vec![Event::Critical { kind: DOWN.into(), reason: "test".into() }]);
        assert!(t.active(start + Duration::from_secs(16))[0].critical);
    }

    #[test]
    fn tick_escalates_without_new_frames() {
        let mut t = tracker();
        let start = Instant::now();
        t.update(&frame(&[DOWN]), start);
        t.update(&frame(&[DOWN]), start);
        // Person lies still, so the browser only sends "unchanged" ticks.
        let events = t.tick(start + Duration::from_secs(16));
        assert_eq!(events, vec![Event::Critical { kind: DOWN.into(), reason: "test".into() }]);
    }
}
