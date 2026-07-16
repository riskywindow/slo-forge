//! Deterministic bounded explicit-state checking for Continuum.
//!
//! Results from this crate are evidence for the configured finite bounds and
//! declared assumptions only. They are not universal protocol proofs.

mod checker;
mod model;

pub use checker::{check, replay_counterexample, validate_result};
pub use model::*;

use schemars::Schema;

/// JSON Schema for checker requests.
#[must_use]
pub fn request_schema() -> Schema {
    schemars::schema_for!(ModelCheckRequest)
}

/// JSON Schema for checker evidence.
#[must_use]
pub fn result_schema() -> Schema {
    schemars::schema_for!(ModelCheckResult)
}
