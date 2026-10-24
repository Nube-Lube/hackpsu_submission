use anyhow::{Context, Result};
use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::config::Config;

// What we want back. Gemini is forced to match this via `responseSchema`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub hazards: Vec<Hazard>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hazard {
    pub kind: String, // one of the kinds in hazards.toml
    pub severity: Severity,
    pub location: String,
    pub reason: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Low,
    Medium,
    High,
}

pub struct Gemini {
    http: reqwest::Client,
    api_key: String,
    model: String,
    prompt: String,
    kinds: Vec<String>,
}

impl Gemini {
    pub fn new(config: &Config) -> Result<Self> {
        Ok(Self {
            http: reqwest::Client::new(),
            // GEMINI_API_KEY (handy on a server), else the gitignored key file holding only the raw key.
            api_key: match std::env::var("GEMINI_API_KEY") {
                Ok(key) => key,
                Err(_) => std::fs::read_to_string(".env/gemini_api.env")
                    .context("set GEMINI_API_KEY or create .env/gemini_api.env")?,
            }
            .trim()
            .to_string(),
            model: std::env::var("GEMINI_MODEL").unwrap_or_else(|_| "gemini-3.5-flash-lite".into()),
            prompt: config.prompt(),
            kinds: config.kinds(),
        })
    }

    pub async fn analyze(&self, jpeg: &[u8]) -> Result<Report> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
            self.model
        );
        let body = json!({
            "contents": [{
                "parts": [
                    { "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64::engine::general_purpose::STANDARD.encode(jpeg),
                    }},
                    { "text": self.prompt },
                ]
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": schema(&self.kinds),
            }
        });

        // Gemini returns 503 (overloaded) / 429 (rate limited) fairly often; retry with backoff.
        let mut attempt = 0;
        let resp: serde_json::Value = loop {
            let r = self
                .http
                .post(&url)
                .header("x-goog-api-key", &self.api_key)
                .json(&body)
                .send()
                .await?;
            let status = r.status();
            if (status.is_server_error() || status.as_u16() == 429) && attempt < 3 {
                attempt += 1;
                tokio::time::sleep(std::time::Duration::from_millis(500 * 2u64.pow(attempt))).await;
                continue;
            }
            break r.error_for_status()?.json().await?;
        };

        // The JSON we asked for arrives as a string inside the first candidate.
        let text = resp["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .with_context(|| format!("unexpected Gemini response: {resp}"))?;
        Ok(serde_json::from_str(text)?)
    }
}

fn schema(kinds: &[String]) -> serde_json::Value {
    json!({
        "type": "OBJECT",
        "properties": {
            "hazards": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "kind": { "type": "STRING", "enum": kinds },
                        "severity": { "type": "STRING", "enum": ["low", "medium", "high"] },
                        "location": { "type": "STRING" },
                        "reason": { "type": "STRING" },
                    },
                    "required": ["kind", "severity", "location", "reason"],
                }
            }
        },
        "required": ["hazards"],
    })
}
