//! Strict shared wire types for portable active-execution state.

mod canonical;
mod error;
mod migration;
mod model;

pub use canonical::{canonical_hash, canonical_json};
pub use error::{ProtocolError, ValidationError};
pub use migration::migrate_document;
pub use model::*;

use serde::de::DeserializeOwned;

/// Decode a strict wire document and check cross-field invariants.
///
/// # Errors
///
/// Returns a decoding or semantic validation error. Partial documents are not
/// returned.
pub fn from_json<T: DeserializeOwned + Validate>(bytes: &[u8]) -> Result<T, ProtocolError> {
    let document: T = serde_json::from_slice(bytes)?;
    document.validate()?;
    Ok(document)
}

/// Semantic validation applied after strict JSON decoding.
pub trait Validate {
    /// Check version, identity, integrity, ownership, and cross-field invariants.
    ///
    /// # Errors
    ///
    /// Returns the first violated invariant with a stable field path.
    fn validate(&self) -> Result<(), ValidationError>;
}
