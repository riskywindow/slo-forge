use std::io::{self, Read};

use sloforge_state_modelcheck::{
    ModelCheckRequest, check, request_schema, result_schema, validate_result,
};

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
    let request = if arguments.first().map(String::as_str) == Some("--safe") {
        if arguments.len() != 2 {
            return Err("usage: sloforge-state-modelcheck --safe <seed>".to_owned());
        }
        let seed = arguments[1]
            .parse::<u64>()
            .map_err(|error| format!("invalid seed: {error}"))?;
        ModelCheckRequest::safe(seed)
    } else {
        if !arguments.is_empty() {
            return Err(
                "usage: sloforge-state-modelcheck [--schema request|result|--safe <seed>]"
                    .to_owned(),
            );
        }
        let mut bytes = Vec::new();
        io::stdin()
            .take(MAX_INPUT_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| format!("failed to read request: {error}"))?;
        if bytes.len() as u64 > MAX_INPUT_BYTES {
            return Err(format!("request exceeds {MAX_INPUT_BYTES} byte limit"));
        }
        serde_json::from_slice(&bytes).map_err(|error| format!("invalid request JSON: {error}"))?
    };
    let result = check(&request).map_err(|diagnostics| {
        serde_json::to_string(&diagnostics)
            .unwrap_or_else(|_| "request validation failed".to_owned())
    })?;
    validate_result(&request, &result).map_err(|errors| format!("invalid result: {errors:?}"))?;
    write_json(&result)
}

fn write_json(value: &impl serde::Serialize) -> Result<(), String> {
    serde_json::to_writer_pretty(io::stdout().lock(), value)
        .map_err(|error| format!("failed to write JSON: {error}"))?;
    println!();
    Ok(())
}
