use std::io::{self, Read};

use sloforge_genesis_modelcheck::{ModelCheckRequest, check, request_schema, result_schema};

const MAX_INPUT_BYTES: u64 = 1_048_576;

fn main() {
    if let Err(message) = run() {
        eprintln!("{message}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    if arguments.as_slice() == ["--schema", "request"] {
        return write_json(&request_schema());
    }
    if arguments.as_slice() == ["--schema", "result"] {
        return write_json(&result_schema());
    }
    if !arguments.is_empty() {
        return Err("usage: sloforge-genesis-modelcheck [--schema request|result]".to_owned());
    }
    let mut bytes = Vec::new();
    io::stdin()
        .take(MAX_INPUT_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("failed to read request: {error}"))?;
    if bytes.len() as u64 > MAX_INPUT_BYTES {
        return Err(format!("request exceeds {MAX_INPUT_BYTES} byte limit"));
    }
    let request: ModelCheckRequest =
        serde_json::from_slice(&bytes).map_err(|error| format!("invalid request JSON: {error}"))?;
    let result = check(&request).map_err(|diagnostics| {
        serde_json::to_string(&diagnostics)
            .unwrap_or_else(|_| "request validation failed".to_owned())
    })?;
    write_json(&result)
}

fn write_json(value: &impl serde::Serialize) -> Result<(), String> {
    serde_json::to_writer_pretty(io::stdout().lock(), value)
        .map_err(|error| format!("failed to write JSON: {error}"))?;
    println!();
    Ok(())
}
