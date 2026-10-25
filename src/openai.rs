use anyhow::{Context, Result};
use base64::Engine;
use serde_json::json;

use crate::config::Config;
use crate::gemini::Report;

/// Any OpenAI-compatible chat completions server with a vision model:
/// Ollama, LM Studio, llama.cpp, vLLM, ...
pub struct OpenAi {
    http: reqwest::Client,
    url: String,
    model: String,
    api_key: Option<String>, // local servers usually don't need one
    prompt: String,
    kinds: Vec<String>,
}

impl OpenAi {
    pub fn from_env(config: &Config) -> Result<Self> {
        let vars = crate::config::env_file(".env/openai.env");
        let get = |name: &str| {
            std::env::var(name).ok().filter(|v| !v.is_empty()).or_else(|| vars.get(name).cloned())
        };
        let base = get("OPENAI_BASE_URL").unwrap_or_else(|| "http://localhost:11434/v1".into());
        Ok(Self {
            http: reqwest::Client::new(),
            url: format!("{}/chat/completions", base.trim_end_matches('/')),
            model: get("OPENAI_MODEL").context("set OPENAI_MODEL to a vision model, e.g. qwen2.5vl:7b")?,
            api_key: get("OPENAI_API_KEY"),
            // Not every local model honors response_format, so also ask for JSON in words.
            prompt: format!("{}\n\nRespond with JSON only, matching the given schema.", config.prompt()),
            kinds: config.kinds(),
        })
    }

    pub fn describe(&self) -> String {
        format!("{} at {}", self.model, self.url)
    }

    pub async fn analyze(&self, jpeg: &[u8]) -> Result<Report> {
        let image = base64::engine::general_purpose::STANDARD.encode(jpeg);
        let body = json!({
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text", "text": self.prompt },
                    { "type": "image_url", "image_url": { "url": format!("data:image/jpeg;base64,{image}") } },
                ],
            }],
            "response_format": {
                "type": "json_schema",
                "json_schema": { "name": "hazard_report", "strict": true, "schema": schema(&self.kinds) },
            },
        });

        let mut req = self.http.post(&self.url).json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        let r = req.send().await.with_context(|| format!("can't reach {}", self.url))?;
        let status = r.status();
        if !status.is_success() {
            anyhow::bail!("{} returned {status}: {}", self.model, r.text().await.unwrap_or_default());
        }
        let resp: serde_json::Value = r.json().await?;
        let text = resp["choices"][0]["message"]["content"]
            .as_str()
            .with_context(|| format!("unexpected response: {resp}"))?;
        parse_report(text)
    }
}

// Local models sometimes wrap the JSON in a ```json code fence anyway.
fn parse_report(text: &str) -> Result<Report> {
    let json = text
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    serde_json::from_str(json).with_context(|| format!("model didn't return the expected JSON: {text}"))
}

// Standard JSON Schema (lowercase types), unlike Gemini's native API.
fn schema(kinds: &[String]) -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "hazards": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": { "type": "string", "enum": kinds },
                        "severity": { "type": "string", "enum": ["low", "medium", "high"] },
                        "location": { "type": "string" },
                        "reason": { "type": "string" },
                    },
                    "required": ["kind", "severity", "location", "reason"],
                    "additionalProperties": false,
                }
            }
        },
        "required": ["hazards"],
        "additionalProperties": false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fenced_json() {
        let text = "```json\n{\"hazards\":[{\"kind\":\"spill\",\"severity\":\"low\",\"location\":\"floor\",\"reason\":\"water\"}]}\n```";
        assert_eq!(parse_report(text).unwrap().hazards[0].kind, "spill");
    }
}
