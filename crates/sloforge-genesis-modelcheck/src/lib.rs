//! A deterministic, bounded explicit-state checker for Genesis serving protocols.
//!
//! Results from this crate are evidence only for the bounds and assumptions in
//! the [`ModelCheckRequest`]. They are not universal proofs.

mod checker;
mod model;

pub use checker::{check, replay_counterexample};
pub use model::*;

use schemars::Schema;

/// Return the JSON Schema for model-check requests.
#[must_use]
pub fn request_schema() -> Schema {
    schemars::schema_for!(ModelCheckRequest)
}

/// Return the JSON Schema for model-check results.
#[must_use]
pub fn result_schema() -> Schema {
    schemars::schema_for!(ModelCheckResult)
}
