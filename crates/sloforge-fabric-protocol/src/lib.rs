//! Strict versioned wire contracts for topology-aware physical inference plans.

mod canonical;
mod error;
mod migration;
mod model;

pub use canonical::{canonical_hash, canonical_json};
pub use error::{ProtocolError, ValidationError};
pub use migration::migrate_document;
pub use model::*;

use serde::de::DeserializeOwned;

/// Parse JSON and enforce semantic invariants.
///
/// # Errors
///
/// Returns a JSON decoding error or a semantic validation error.
pub fn from_json<T: DeserializeOwned + Validate>(bytes: &[u8]) -> Result<T, ProtocolError> {
    let document: T = serde_json::from_slice(bytes)?;
    document.validate()?;
    Ok(document)
}

/// Cross-field validation for top-level Fabric documents.
pub trait Validate {
    /// Validate version, identity, and cross-field invariants.
    ///
    /// # Errors
    ///
    /// Returns the first semantic validation failure with its field path.
    fn validate(&self) -> Result<(), ValidationError>;
}
