//! Versioned wire contracts shared by `SLOForge`'s Rust data plane and Python compiler.
//!
//! The canonical interchange is JSON over bounded subprocess stdin/stdout.  Core
//! fields reject unknown members; controlled extension maps require namespaced keys.

mod canonical;
mod error;
mod migration;
mod model;

pub use canonical::{canonical_hash, canonical_json};
pub use error::{ProtocolError, ValidationError};
pub use migration::migrate_document;
pub use model::*;

use serde::de::DeserializeOwned;

/// Parse and semantically validate a protocol document.
///
/// # Errors
///
/// Returns [`ProtocolError`] when JSON decoding or semantic validation fails.
pub fn from_json<T: DeserializeOwned + Validate>(bytes: &[u8]) -> Result<T, ProtocolError> {
    let value: T = serde_json::from_slice(bytes)?;
    value.validate()?;
    Ok(value)
}

/// Semantic validation implemented by top-level protocol documents.
pub trait Validate {
    /// Validate cross-field and domain invariants.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationError`] identifying the first invalid field.
    fn validate(&self) -> Result<(), ValidationError>;
}
