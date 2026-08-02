use serde_json::{Map, Value};

use crate::{API_VERSION, SCHEMA_VERSION, ValidationError};

fn rename(object: &mut Map<String, Value>, old: &str, new: &str) -> Result<(), ValidationError> {
    if let Some(value) = object.remove(old) {
        if object.contains_key(new) {
            return Err(ValidationError::new(
                new,
                format!("legacy document contains both {old:?} and {new:?}"),
            ));
        }
        object.insert(new.to_owned(), value);
    }
    Ok(())
}

/// Upgrade v1alpha1/0.1.0 JSON values to the stable v1 wire shape.
///
/// # Errors
///
/// Returns [`ValidationError`] for a non-object, unknown version or kind, or
/// conflicting legacy and stable field names.
pub fn migrate_document(source: &Value) -> Result<Value, ValidationError> {
    let mut result = source.clone();
    let object = result
        .as_object_mut()
        .ok_or_else(|| ValidationError::new("$", "IR document must be a JSON object"))?;
    if object.get("schema_version").and_then(Value::as_str) == Some(SCHEMA_VERSION) {
        return Ok(result);
    }
    let version = object
        .get("version")
        .or_else(|| object.get("schema_version"))
        .and_then(Value::as_str);
    if !matches!(version, Some("v1alpha1" | "0.1.0")) {
        return Err(ValidationError::new(
            "schema_version",
            format!("unsupported IR schema version: {version:?}"),
        ));
    }
    let kind = object
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or("DeploymentPlan")
        .to_owned();
    object.remove("version");
    object.insert(
        "schema_version".to_owned(),
        Value::String(SCHEMA_VERSION.to_owned()),
    );
    object.insert(
        "api_version".to_owned(),
        Value::String(API_VERSION.to_owned()),
    );

    if matches!(kind.as_str(), "DeploymentPlan" | "deployment_plan") {
        object.insert(
            "kind".to_owned(),
            Value::String("DeploymentPlan".to_owned()),
        );
        for (old, new) in [
            ("replicas", "replica_topology"),
            ("routing_policy", "routing"),
            ("admission_policy", "admission"),
            ("batching_policy", "batching"),
            ("autoscaling_policy", "autoscaling"),
            ("cold_start_strategy", "cold_start"),
            ("canary_policy", "canary"),
            ("rollback_policy", "rollback"),
        ] {
            rename(object, old, new)?;
        }
        if let Some(model) = object.get_mut("model").and_then(Value::as_object_mut) {
            rename(model, "id", "model_id")?;
        }
        if let Some(engine) = object.get_mut("engine").and_then(Value::as_object_mut) {
            rename(engine, "runtime_name", "runtime")?;
        }
    } else if matches!(kind.as_str(), "EvidenceBundle" | "evidence_bundle") {
        object.insert(
            "kind".to_owned(),
            Value::String("EvidenceBundle".to_owned()),
        );
        rename(object, "optimizer_decisions", "optimizer_history")?;
    } else {
        return Err(ValidationError::new(
            "kind",
            format!("unsupported IR document kind: {kind:?}"),
        ));
    }
    Ok(result)
}
