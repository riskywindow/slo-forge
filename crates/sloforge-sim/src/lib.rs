//! Deterministic, calibrated discrete-event inference simulator.
//!
//! The public request/response structures are the stable JSON subprocess API.

mod engine;
mod model;

pub use engine::{SimError, simulate};
pub use model::*;

/// Current subprocess protocol version. Versions with the same major number are additive.
pub const SIM_SCHEMA_VERSION: &str = "1.0";
