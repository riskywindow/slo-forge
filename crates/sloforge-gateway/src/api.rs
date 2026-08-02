use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Prompt {
    Text(String),
    Texts(Vec<String>),
    Tokens(Vec<u32>),
    TokenBatches(Vec<Vec<u32>>),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: Prompt,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sloforge: Option<RequestSlo>,
    #[serde(flatten)]
    pub extensions: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sloforge: Option<RequestSlo>,
    #[serde(flatten)]
    pub extensions: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Value,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RequestSlo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deadline_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_class: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct InferenceRequest {
    pub endpoint: &'static str,
    pub payload: Value,
    pub stream: bool,
    pub model: String,
    pub max_tokens: u32,
    pub slo: Option<RequestSlo>,
}

impl TryFrom<CompletionRequest> for InferenceRequest {
    type Error = serde_json::Error;

    fn try_from(request: CompletionRequest) -> Result<Self, Self::Error> {
        let stream = request.stream;
        let model = request.model.clone();
        let max_tokens = request.max_tokens;
        let slo = request.sloforge.clone();
        let mut payload = serde_json::to_value(request)?;
        strip_internal_metadata(&mut payload);
        Ok(Self {
            endpoint: "/v1/completions",
            payload,
            stream,
            model,
            max_tokens,
            slo,
        })
    }
}

impl TryFrom<ChatRequest> for InferenceRequest {
    type Error = serde_json::Error;

    fn try_from(request: ChatRequest) -> Result<Self, Self::Error> {
        let stream = request.stream;
        let model = request.model.clone();
        let max_tokens = request.max_tokens;
        let slo = request.sloforge.clone();
        let mut payload = serde_json::to_value(request)?;
        strip_internal_metadata(&mut payload);
        Ok(Self {
            endpoint: "/v1/chat/completions",
            payload,
            stream,
            model,
            max_tokens,
            slo,
        })
    }
}

fn strip_internal_metadata(payload: &mut Value) {
    if let Some(object) = payload.as_object_mut() {
        object.remove("sloforge");
    }
}

const fn default_max_tokens() -> u32 {
    16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal_slo_metadata_is_not_forwarded_to_engines() -> Result<(), serde_json::Error> {
        let request: CompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "prompt": "prompt",
            "sloforge": {"deadline_ms": 100}
        }))?;
        let inference = InferenceRequest::try_from(request)?;
        assert!(inference.slo.is_some());
        assert!(inference.payload.get("sloforge").is_none());
        Ok(())
    }
}
