use clap::Parser;
use serde::Deserialize;
use sloforge_sim::{RequestSpec, SimulationRequest, simulate};
use std::io::{Read, Write};
use std::path::PathBuf;

const MAX_INPUT_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TraceLine {
    Canonical(CanonicalTraceRequest),
    Scenario(RequestSpec),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CanonicalTraceRequest {
    request_id: String,
    arrival_ms: f64,
    prompt_tokens: u32,
    output_tokens: u32,
    #[serde(default = "default_trace_priority")]
    priority: u8,
    #[serde(default = "default_request_class")]
    request_class: String,
    #[serde(default)]
    deadline_ms: Option<f64>,
    #[serde(default)]
    adapter_id: Option<String>,
    #[serde(default)]
    prefix_group: Option<String>,
    #[serde(default)]
    cancelled_at_ms: Option<f64>,
}

const fn default_trace_priority() -> u8 {
    1
}

fn default_request_class() -> String {
    "interactive".into()
}

#[derive(Debug, Parser)]
#[command(about = "Run the deterministic SLOForge discrete-event simulator")]
struct Args {
    /// Optional compatibility subcommand.
    #[arg(value_parser = ["simulate"])]
    command: Option<String>,
    /// JSON input path. Reads stdin when omitted.
    #[arg(long, visible_alias = "config")]
    input: Option<PathBuf>,
    /// Optional JSONL request trace. Overrides the `requests` array in the input.
    #[arg(long)]
    trace: Option<PathBuf>,
    /// JSON output path. Writes stdout when omitted.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Optional Chrome/Perfetto trace-event output path.
    #[arg(long)]
    chrome_trace: Option<PathBuf>,
    /// Emit compact JSON rather than human-readable JSON.
    #[arg(long)]
    compact: bool,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("sloforge-sim: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let bytes = if let Some(path) = args.input {
        let metadata = std::fs::metadata(&path)?;
        if metadata.len() > MAX_INPUT_BYTES {
            return Err("input exceeds 64 MiB safety limit".into());
        }
        std::fs::read(path)?
    } else {
        let mut bytes = Vec::new();
        std::io::stdin()
            .take(MAX_INPUT_BYTES + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() as u64 > MAX_INPUT_BYTES {
            return Err("stdin exceeds 64 MiB safety limit".into());
        }
        bytes
    };
    let mut input: SimulationRequest = serde_json::from_slice(&bytes)?;
    if let Some(trace) = args.trace {
        let metadata = std::fs::metadata(&trace)?;
        if metadata.len() > MAX_INPUT_BYTES {
            return Err("trace exceeds 64 MiB safety limit".into());
        }
        let contents = std::fs::read_to_string(trace)?;
        input.requests = parse_trace(&contents)?;
    }
    let result = simulate(&input)?;
    if let Some(path) = args.chrome_trace {
        let chrome = serde_json::json!({
            "traceEvents": &result.trace_events,
            "displayTimeUnit": "ms",
            "sloforgeProvenance": &result.provenance,
        });
        std::fs::write(path, serde_json::to_vec_pretty(&chrome)?)?;
    }
    let output = if args.compact {
        serde_json::to_vec(&result)?
    } else {
        serde_json::to_vec_pretty(&result)?
    };
    if let Some(path) = args.output {
        std::fs::write(path, output)?;
    } else {
        let mut stdout = std::io::stdout().lock();
        stdout.write_all(&output)?;
        stdout.write_all(b"\n")?;
    }
    Ok(())
}

fn parse_trace(contents: &str) -> Result<Vec<RequestSpec>, Box<dyn std::error::Error>> {
    let mut requests = Vec::new();
    for (line_idx, line) in contents.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let record: TraceLine = serde_json::from_str(line)
            .map_err(|error| format!("trace line {}: {error}", line_idx + 1))?;
        requests.push(match record {
            TraceLine::Scenario(request) => request,
            TraceLine::Canonical(request) => canonical_request(request)?,
        });
    }
    Ok(requests)
}

fn canonical_request(
    request: CanonicalTraceRequest,
) -> Result<RequestSpec, Box<dyn std::error::Error>> {
    let arrival_ms = rounded_millis("arrival_ms", request.arrival_ms)?;
    let deadline_ms = request
        .deadline_ms
        .map(|value| rounded_millis("deadline_ms", value))
        .transpose()?;
    let cancel_after_ms = request
        .cancelled_at_ms
        .map(|cancelled| {
            if cancelled < request.arrival_ms {
                Err("cancelled_at_ms precedes arrival_ms".into())
            } else {
                rounded_millis("cancel_after_ms", cancelled - request.arrival_ms)
            }
        })
        .transpose()?;
    Ok(RequestSpec {
        id: request.request_id,
        arrival_ms,
        prompt_tokens: request.prompt_tokens,
        output_tokens: request.output_tokens,
        priority: 10_u8.saturating_sub(request.priority),
        request_class: request.request_class,
        deadline_ms,
        cancel_after_ms,
        canary_eligible: true,
        adapter_id: request.adapter_id,
        prefix_group: request.prefix_group,
    })
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn rounded_millis(name: &str, value: f64) -> Result<u64, Box<dyn std::error::Error>> {
    if !value.is_finite() || value < 0.0 || value > u64::MAX as f64 {
        return Err(format!("{name} must be a finite non-negative millisecond value").into());
    }
    Ok(value.round() as u64)
}
