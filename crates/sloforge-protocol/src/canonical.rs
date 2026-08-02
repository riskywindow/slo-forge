use std::collections::BTreeMap;

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{ArtifactDigest, ProtocolError};

fn sort_value(value: Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.into_iter().map(sort_value).collect()),
        Value::Object(values) => {
            let sorted: BTreeMap<_, _> = values
                .into_iter()
                .map(|(key, value)| (key, sort_value(value)))
                .collect();
            Value::Object(sorted.into_iter().collect())
        }
        scalar => scalar,
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

/// Serialize using the cross-language canonical JSON profile.
///
/// # Errors
///
/// Returns [`ProtocolError`] when the document cannot be represented as JSON.
pub fn canonical_json<T: Serialize>(document: &T) -> Result<Vec<u8>, ProtocolError> {
    let sorted = sort_value(serde_json::to_value(document)?);
    let mut output = Vec::new();
    encode_value(&sorted, &mut output)?;
    Ok(output)
}

/// Compute the content-addressed SHA-256 digest of canonical JSON.
///
/// # Errors
///
/// Returns [`ProtocolError`] when the document cannot be represented as JSON.
pub fn canonical_hash<T: Serialize>(document: &T) -> Result<ArtifactDigest, ProtocolError> {
    let payload = canonical_json(document)?;
    let value = format!("{:x}", Sha256::digest(payload));
    Ok(ArtifactDigest {
        algorithm: DigestAlgorithm::Sha256,
        value,
    })
}

use crate::DigestAlgorithm;
