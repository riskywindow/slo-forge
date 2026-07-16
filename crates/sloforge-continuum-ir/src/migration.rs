//! Lossless migrations from known pre-stable Continuum documents.

use serde_json::{Map, Value};

use crate::{API_VERSION, ProtocolError, SCHEMA_VERSION};

const KINDS: &[&str] = &[
    "LogicalStateSchema",
    "PhysicalStateLayout",
    "ExecutionStateCapsule",
    "CompatibilityReport",
    "StateTransformationIR",
    "MigrationPlan",
    "StateTransaction",
    "MigrationVerificationEvidence",
];

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

/// Migrate a known alpha object without guessing state semantics.
///
/// # Errors
///
/// Rejects unknown versions, kinds, and ambiguous legacy aliases.
pub fn migrate_document(document: &Value) -> Result<Value, ProtocolError> {
    let mut result = document.clone();
    let values = result
        .as_object_mut()
        .ok_or_else(|| ProtocolError::Migration("document must be an object".into()))?;
    let version = values
        .get("schema_version")
        .or_else(|| values.get("version"))
        .and_then(Value::as_str);
    let kind = values
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    if version == Some(SCHEMA_VERSION) {
        if !KINDS.contains(&kind.as_str()) {
            return Err(ProtocolError::Migration(format!(
                "unsupported document kind {kind:?}"
            )));
        }
        return Ok(result);
    }
    if !matches!(version, Some("v1alpha1" | "0.1.0")) {
        return Err(ProtocolError::Migration(format!(
            "unsupported schema version {version:?}"
        )));
    }
    let stable_kind = match kind.as_str() {
        "logical_state" => "LogicalStateSchema".to_owned(),
        "physical_state" => "PhysicalStateLayout".to_owned(),
        "execution_state_capsule" => "ExecutionStateCapsule".to_owned(),
        "compatibility_report" => "CompatibilityReport".to_owned(),
        "state_transformation" => "StateTransformationIR".to_owned(),
        "migration_plan" => "MigrationPlan".to_owned(),
        "state_transaction" => "StateTransaction".to_owned(),
        "verification_evidence" => "MigrationVerificationEvidence".to_owned(),
        other if KINDS.contains(&other) => other.to_owned(),
        other => {
            return Err(ProtocolError::Migration(format!(
                "unsupported document kind {other:?}"
            )));
        }
    };
    values.remove("version");
    values.insert(
        "schema_version".into(),
        Value::String(SCHEMA_VERSION.into()),
    );
    values.insert("api_version".into(), Value::String(API_VERSION.into()));
    values.insert("kind".into(), Value::String(stable_kind.clone()));
    match stable_kind.as_str() {
        "ExecutionStateCapsule" => {
            rename(values, "logical", "logical_state")?;
            rename(values, "physical", "physical_state")?;
        }
        "PhysicalStateLayout" => rename(values, "runtime_identity", "runtime")?,
        "StateTransformationIR" => rename(values, "id", "transformation_id")?,
        "MigrationPlan" => rename(values, "id", "plan_id")?,
        "StateTransaction" => rename(values, "phase", "current_phase")?,
        _ => {}
    }
    Ok(result)
}
