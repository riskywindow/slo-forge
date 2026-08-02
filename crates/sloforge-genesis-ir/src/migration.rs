//! Lossless migrations from the pre-stable Genesis alpha fixture.

use serde_json::{Map, Value};

use crate::{API_VERSION, ProtocolError, SCHEMA_VERSION};

fn rename(values: &mut Map<String, Value>, old: &str, new: &str) -> Result<(), ProtocolError> {
    let Some(value) = values.remove(old) else {
        return Ok(());
    };
    if values.contains_key(new) {
        return Err(ProtocolError::Migration(format!(
            "legacy document contains both {old:?} and {new:?}"
        )));
    }
    values.insert(new.to_owned(), value);
    Ok(())
}

/// Migrate a stable v1 or known v1alpha1 object without guessing semantics.
///
/// # Errors
///
/// Rejects unknown versions, kinds, and ambiguous field aliases.
pub fn migrate_document(document: &Value) -> Result<Value, ProtocolError> {
    let mut result = document.clone();
    let values = result
        .as_object_mut()
        .ok_or_else(|| ProtocolError::Migration("document must be an object".into()))?;
    let version = values
        .get("schema_version")
        .or_else(|| values.get("version"))
        .and_then(Value::as_str);
    if version == Some(SCHEMA_VERSION) {
        return Ok(result);
    }
    if !matches!(version, Some("v1alpha1" | "0.1.0")) {
        return Err(ProtocolError::Migration(format!(
            "unsupported schema version {version:?}"
        )));
    }
    values.remove("version");
    values.insert(
        "schema_version".into(),
        Value::String(SCHEMA_VERSION.into()),
    );
    values.insert("api_version".into(), Value::String(API_VERSION.into()));
    let kind = values
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let stable_kind = match kind {
        "InferenceGenome" | "inference_genome" => "InferenceGenome",
        "Transformation" | "transformation" => "Transformation",
        "Candidate" | "candidate" => "Candidate",
        "Counterexample" | "counterexample" => "Counterexample",
        other => {
            return Err(ProtocolError::Migration(format!(
                "unsupported document kind {other:?}"
            )));
        }
    };
    values.insert("kind".into(), Value::String(stable_kind.into()));
    match stable_kind {
        "InferenceGenome" => {
            for region in [
                "workflow",
                "request",
                "serving",
                "state",
                "distributed",
                "tensor",
                "kernel",
                "recovery",
            ] {
                rename(values, &format!("{region}_genome"), region)?;
            }
        }
        "Transformation" => {
            rename(values, "id", "transformation_id")?;
            rename(values, "verification", "verification_obligations")?;
        }
        "Candidate" => {
            rename(values, "id", "candidate_id")?;
            rename(values, "events", "lifecycle")?;
        }
        "Counterexample" => {
            rename(values, "id", "counterexample_id")?;
            rename(values, "command", "reproduction")?;
        }
        _ => unreachable!("stable kind was exhaustively matched"),
    }
    Ok(result)
}
