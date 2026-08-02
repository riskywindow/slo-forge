//! Error types for strict Genesis IR parsing and migration.

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
    #[error("invalid Genesis JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid Genesis document: {0}")]
    Validation(#[from] ValidationError),
    #[error("unsupported Genesis migration: {0}")]
    Migration(String),
}
