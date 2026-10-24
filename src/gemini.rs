use anyhow::{Context, Result};
use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};

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
    api_key: Option<String>, // the server's own key; cameras may bring theirs instead (BYOK)
    models: Vec<String>, // tried in order; moves to the next when one runs out of daily quota
    current: AtomicUsize,
    prompt: String,
    kinds: Vec<String>,
}

impl Gemini {
    pub fn new(config: &Config) -> Result<Self> {
        Ok(Self {
            http: reqwest::Client::new(),
            // GEMINI_API_KEY (handy on a server), else the gitignored key file holding only the raw key.
            // Neither is required: without one, every camera must bring its own key.
            api_key: std::env::var("GEMINI_API_KEY")
                .or_else(|_| std::fs::read_to_string(".env/gemini_api.env"))
                .ok()
                .map(|k| k.trim().to_string())
                .filter(|k| !k.is_empty()),
            // Comma-separated; each model has its own free-tier quota.
            models: std::env::var("GEMINI_MODEL")
                .unwrap_or_else(|_| "gemini-3.5-flash-lite,gemini-3.1-flash-lite".into())
                .split(',')
                .map(|m| m.trim().to_string())
                .filter(|m| !m.is_empty())
                .collect(),
            current: AtomicUsize::new(0),
            prompt: config.prompt(),
            kinds: config.kinds(),
        })
    }

    pub fn has_server_key(&self) -> bool {
        self.api_key.is_some()
    }

    /// `user_key` is a camera's own key (BYOK); otherwise the server's key is used.
    pub async fn analyze(&self, jpeg: &[u8], user_key: Option<&str>) -> Result<Report> {
        let key = user_key
            .or(self.api_key.as_deref())
            .context("no Gemini API key: add one on the camera page, or give the server one")?;
        // Model fallback progress is only shared for the server's key; a user's key has its own quota.
        let shared = user_key.is_none();

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

        let mut i = if shared { self.current.load(Ordering::Relaxed) } else { 0 };
        let resp = loop {
            let model = self.models.get(i).with_context(|| {
                format!("every Gemini model is out of daily quota: {}", self.models.join(", "))
            })?;
            match self.call(model, key, &body).await? {
                Some(resp) => break resp,
                None => {
                    // compare_exchange: if two requests hit the limit at once, only one moves us on.
                    if shared
                        && self.current.compare_exchange(i, i + 1, Ordering::Relaxed, Ordering::Relaxed).is_ok()
                    {
                        eprintln!("{model} is out of daily quota, moving to the next model");
                    }
                    i += 1;
                }
            }
        };

        // The JSON we asked for arrives as a string inside the first candidate.
        let text = resp["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .with_context(|| format!("unexpected Gemini response: {resp}"))?;
        Ok(serde_json::from_str(text)?)
    }

    /// One request to one model. `Ok(None)` means that model's daily quota is used up.
    async fn call(&self, model: &str, key: &str, body: &serde_json::Value) -> Result<Option<serde_json::Value>> {
        let url = format!("https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent");

        // Gemini returns 503 (overloaded) / 429 (rate limited) fairly often; retry with backoff.
        let mut attempt = 0;
        loop {
            let r = self.http.post(&url).header("x-goog-api-key", key).json(body).send().await?;
            let status = r.status();
            if status.is_success() {
                return Ok(Some(r.json().await?));
            }
            let text = r.text().await.unwrap_or_default();
            // A per-day limit won't recover by waiting a few seconds; give up on this model.
            if status.as_u16() == 429 && text.contains("PerDay") {
                return Ok(None);
            }
            if (status.is_server_error() || status.as_u16() == 429) && attempt < 3 {
                attempt += 1;
                tokio::time::sleep(std::time::Duration::from_millis(500 * 2u64.pow(attempt))).await;
                continue;
            }
            anyhow::bail!("{model} returned {status}: {text}");
        }
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
