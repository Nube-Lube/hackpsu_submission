use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Deserialize;

const DEFAULT: &str = include_str!("../hazards.toml");

/// Optional KEY=VALUE overrides from the gitignored `.env/openai.env`, so the
/// local-inference setup survives restarts without re-exporting shell variables.
/// Real environment variables win over the file.
pub fn env_file(path: &str) -> HashMap<String, String> {
    std::fs::read_to_string(path)
        .map(|text| {
            text.lines()
                .filter_map(|line| {
                    let line = line.trim();
                    line.split_once('=')
                        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
                        .filter(|(k, _)| !k.is_empty())
                })
                .collect()
        })
        .unwrap_or_default()
}

#[derive(Deserialize)]
pub struct Config {
    prompt: String,
    #[serde(rename = "hazard")]
    hazards: Vec<HazardDef>,
}

#[derive(Deserialize)]
struct HazardDef {
    kind: String,
    description: String,
    critical_after_secs: Option<u64>,
}

impl Config {
    /// `hazards.toml` in the working directory (or HAZARDS_CONFIG), else the built-in default.
    pub fn load() -> Result<Self> {
        let path = std::env::var("HAZARDS_CONFIG").unwrap_or_else(|_| "hazards.toml".into());
        let (text, source) = match std::fs::read_to_string(&path) {
            Ok(text) => (text, path.as_str()),
            Err(_) => (DEFAULT.to_string(), "built-in defaults"),
        };
        let config: Config = toml::from_str(&text).with_context(|| format!("invalid {source}"))?;
        anyhow::ensure!(!config.hazards.is_empty(), "{source} defines no [[hazard]] entries");
        println!("hazards from {source}: {}", config.kinds().join(", "));
        Ok(config)
    }

    /// The prompt sent to Gemini: the configured text plus the list of hazard kinds.
    pub fn prompt(&self) -> String {
        let mut prompt = self.prompt.trim().to_string();
        prompt.push_str("\n\nHazard kinds:\n");
        for h in &self.hazards {
            prompt.push_str(&format!("- {}: {}\n", h.kind, h.description));
        }
        prompt
    }

    pub fn kinds(&self) -> Vec<String> {
        self.hazards.iter().map(|h| h.kind.clone()).collect()
    }

    /// Kinds that escalate to critical, and after how long.
    pub fn critical_after(&self) -> HashMap<String, Duration> {
        self.hazards
            .iter()
            .filter_map(|h| Some((h.kind.clone(), Duration::from_secs(h.critical_after_secs?))))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_parses() {
        let config: Config = toml::from_str(DEFAULT).unwrap();
        assert!(config.kinds().contains(&"person_down".to_string()));
        assert_eq!(config.critical_after()["person_down"], Duration::from_secs(15));
        assert!(config.prompt().contains("- spill: "));
    }
}
