//! Deterministic, communication-aware physical execution simulator.
//!
//! The crate consumes and emits bounded, versioned JSON. It is deliberately
//! independent of CUDA and networking libraries: calibrated hardware curves are
//! explicit inputs, so CPU CI can exercise the same scheduler without pretending
//! that synthetic data was measured.

mod curve;
mod engine;
mod model;
mod validate;

pub use engine::simulate;
pub use model::*;
pub use validate::{SimError, validate};

/// Current subprocess schema. Additive changes retain major version one.
pub const FABRIC_SIM_SCHEMA_VERSION: &str = "1.0";
