//! Errors returned by strict Continuum wire parsing and validation.

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
    #[error("invalid Continuum JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid Continuum document: {0}")]
    Validation(#[from] ValidationError),
    #[error("unsupported Continuum migration: {0}")]
    Migration(String),
}
