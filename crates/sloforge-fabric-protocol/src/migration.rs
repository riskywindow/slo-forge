use serde_json::Value;

use crate::{ProtocolError, ValidationError};

fn rename(
    object: &mut serde_json::Map<String, Value>,
    old: &str,
    new: &str,
) -> Result<(), ProtocolError> {
    if let Some(value) = object.remove(old) {
        if object.contains_key(new) {
            return Err(ValidationError::new(
                old,
                format!("document contains both {old:?} and {new:?}"),
            )
            .into());
        }
        object.insert(new.to_owned(), value);
    }
    Ok(())
}

/// Migrate a v1alpha1 document into the stable v1 wire representation.
///
/// # Errors
///
/// Returns an error for an unknown version, kind, or ambiguous renamed field.
pub fn migrate_document(document: &Value) -> Result<Value, ProtocolError> {
    let mut result = document.clone();
    let object = result
        .as_object_mut()
        .ok_or_else(|| ValidationError::new("$", "Fabric document must be an object"))?;
    let version = object
        .get("schema_version")
        .or_else(|| object.get("version"))
        .and_then(Value::as_str);
    if version == Some("1.0.0") {
        return Ok(result);
    }
    if !matches!(version, Some("v1alpha1" | "0.1.0")) {
        return Err(ValidationError::new(
            "schema_version",
            format!("unsupported Fabric IR schema version: {version:?}"),
        )
        .into());
    }
    object.remove("version");
    object.insert(
        "schema_version".to_owned(),
        Value::String("1.0.0".to_owned()),
    );
    object.insert(
        "api_version".to_owned(),
        Value::String("sloforge.io/fabric/v1".to_owned()),
    );
    match object.get("kind").and_then(Value::as_str) {
        Some("TopologyGraph") => {
            rename(object, "id", "topology_id")?;
            rename(object, "links", "edges")?;
        }
        Some("ModelGraph") => {
            rename(object, "revision", "model_revision")?;
            rename(object, "digest", "model_digest")?;
        }
        Some("FabricProfile") => {
            rename(object, "id", "profile_id")?;
            rename(object, "series", "measurements")?;
        }
        Some("PhysicalExecutionPlan") => {
            rename(object, "deployment_plan", "logical_deployment_plan")?;
            rename(object, "placement", "rank_placement")?;
            rename(object, "overlap", "communication_overlap")?;
            rename(object, "predictions", "predicted_metrics")?;
            rename(object, "rejected_candidates", "rejected_alternatives")?;
        }
        Some("RecoveryPlan") => {
            rename(object, "id", "recovery_id")?;
            rename(object, "proposal_actions", "actions")?;
        }
        kind => {
            return Err(ValidationError::new(
                "kind",
                format!("unsupported Fabric IR document kind: {kind:?}"),
            )
            .into());
        }
    }
    Ok(result)
}
