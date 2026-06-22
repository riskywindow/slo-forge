//! Canonical JSON serialization compatible with the Python IR boundary.

use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::ProtocolError;

fn canonicalize(value: Value) -> Value {
    match value {
        Value::Object(values) => {
            let ordered: BTreeMap<_, _> = values
                .into_iter()
                .map(|(key, value)| (key, canonicalize(value)))
                .collect();
            Value::Object(ordered.into_iter().collect())
        }
        Value::Array(values) => Value::Array(values.into_iter().map(canonicalize).collect()),
        other => other,
    }
}

fn python_float_repr(value: f64) -> String {
    let raw = format!("{value:?}");
    let Some((mantissa, exponent)) = raw.split_once('e') else {
        return raw;
    };
    let (sign, digits) = exponent
        .strip_prefix('-')
        .map_or(('+', exponent), |digits| ('-', digits));
    format!("{mantissa}e{sign}{digits:0>2}")
}

fn encode_value(value: &Value, output: &mut Vec<u8>) -> Result<(), serde_json::Error> {
    match value {
        Value::Null => output.extend_from_slice(b"null"),
        Value::Bool(true) => output.extend_from_slice(b"true"),
        Value::Bool(false) => output.extend_from_slice(b"false"),
        Value::Number(number) => {
            let rendered = if let Some(value) = number.as_i64() {
                value.to_string()
            } else if let Some(value) = number.as_u64() {
                value.to_string()
            } else if let Some(value) = number.as_f64() {
                python_float_repr(value)
            } else {
                number.to_string()
            };
            output.extend_from_slice(rendered.as_bytes());
        }
        Value::String(text) => output.extend_from_slice(serde_json::to_string(text)?.as_bytes()),
        Value::Array(items) => {
            output.push(b'[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                encode_value(item, output)?;
            }
            output.push(b']');
        }
        Value::Object(items) => {
            output.push(b'{');
            for (index, (key, item)) in items.iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                output.extend_from_slice(serde_json::to_string(key)?.as_bytes());
                output.push(b':');
                encode_value(item, output)?;
            }
            output.push(b'}');
        }
    }
    Ok(())
}

/// Serialize sorted UTF-8 JSON without insignificant whitespace.
///
/// # Errors
///
/// Returns an error for values that JSON cannot represent.
pub fn canonical_json<T: Serialize>(document: &T) -> Result<Vec<u8>, ProtocolError> {
    let value = canonicalize(serde_json::to_value(document)?);
    let mut output = Vec::new();
    encode_value(&value, &mut output)?;
    Ok(output)
}

/// Compute a lowercase SHA-256 content identifier over canonical JSON.
///
/// # Errors
///
/// Returns an error when the document cannot be represented as canonical JSON.
pub fn canonical_hash<T: Serialize>(document: &T) -> Result<String, ProtocolError> {
    let payload = canonical_json(document)?;
    Ok(format!("{:x}", Sha256::digest(payload)))
}
