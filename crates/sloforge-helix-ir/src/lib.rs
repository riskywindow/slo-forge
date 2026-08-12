//! Canonical trusted wire models for the `SLOForge` Helix learning loop.

#![allow(clippy::too_many_lines)]

mod canonical;
mod error;
mod model;
mod trace;

pub use canonical::{canonical_hash, canonical_json};
pub use error::{ProtocolError, ValidationError};
pub use model::*;
pub use trace::*;

use serde::de::DeserializeOwned;

/// Decode strict JSON and apply semantic invariants.
///
/// # Errors
///
/// Returns a decoding or semantic validation error.
pub fn from_json<T: DeserializeOwned + Validate>(bytes: &[u8]) -> Result<T, ProtocolError> {
    let document: T = serde_json::from_slice(bytes)?;
    document.validate()?;
    Ok(document)
}

/// Post-deserialization semantic validation for Helix roots.
pub trait Validate {
    /// Check version, provenance, policy, and lineage invariants.
    ///
    /// # Errors
    ///
    /// Returns the first violated invariant with a stable path.
    fn validate(&self) -> Result<(), ValidationError>;
}
