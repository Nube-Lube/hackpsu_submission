use anyhow::Result;

use crate::config::Config;
use crate::gemini::{Gemini, Report};
use crate::openai::OpenAi;

/// Which model judges the frames. LLM_PROVIDER=openai switches to an OpenAI-compatible
/// server (e.g. a local model); otherwise Gemini.
pub enum Vision {
    Gemini(Gemini),
    OpenAi(OpenAi),
}

impl Vision {
    pub fn from_env(config: &Config) -> Result<Self> {
        let vision = match std::env::var("LLM_PROVIDER").as_deref() {
            Ok("openai") => Vision::OpenAi(OpenAi::from_env(config)?),
            _ => Vision::Gemini(Gemini::new(config)?),
        };
        match &vision {
            Vision::Gemini(_) => println!("model: Gemini"),
            Vision::OpenAi(o) => println!("model: {}", o.describe()),
        }
        Ok(vision)
    }

    /// False only for Gemini without a server key, where every camera must bring its own.
    pub fn has_server_key(&self) -> bool {
        match self {
            Vision::Gemini(g) => g.has_server_key(),
            Vision::OpenAi(_) => true,
        }
    }

    /// `user_key` is a camera's own Gemini key (BYOK); ignored by OpenAI-compatible servers.
    pub async fn analyze(&self, jpeg: &[u8], user_key: Option<&str>) -> Result<Report> {
        match self {
            Vision::Gemini(g) => g.analyze(jpeg, user_key).await,
            Vision::OpenAi(o) => o.analyze(jpeg).await,
        }
    }
}
