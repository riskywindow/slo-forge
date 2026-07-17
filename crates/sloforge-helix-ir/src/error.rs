//! Errors returned by strict Helix parsing and semantic validation.

use thiserror::Error;

#[derive(Clone, Debug, Eq, Error, PartialEq)]
#[error("{path}: {message}")]
pub struct ValidationError {
    pub path: String,
    pub message: String,
}

impl ValidationError {
    #[must_use]
    pub fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("invalid Helix JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid Helix document: {0}")]
    Validation(#[from] ValidationError),
}
