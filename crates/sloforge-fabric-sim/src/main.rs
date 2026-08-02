use clap::Parser;
use sloforge_fabric_sim::{FabricSimulationRequest, simulate};
use std::io::{Read, Write};
use std::path::PathBuf;

const MAX_INPUT_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Debug, Parser)]
#[command(about = "Run the deterministic SLOForge Fabric physical simulator")]
struct Args {
    /// Optional compatibility subcommand.
    #[arg(value_parser = ["simulate"])]
    command: Option<String>,
    /// JSON request path. Reads stdin when omitted.
    #[arg(long)]
    input: Option<PathBuf>,
    /// JSON response path. Writes stdout when omitted.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Optional Chrome/Perfetto trace file.
    #[arg(long)]
    chrome_trace: Option<PathBuf>,
    /// Emit compact output.
    #[arg(long)]
    compact: bool,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("sloforge-fabric-sim: {error}");
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
    let request: FabricSimulationRequest = serde_json::from_slice(&bytes)?;
    let output = simulate(&request)?;
    if let Some(path) = args.chrome_trace {
        let chrome = serde_json::json!({
            "traceEvents": &output.trace_events,
            "displayTimeUnit": "us",
            "sloforgeProvenance": &output.provenance,
        });
        std::fs::write(path, serde_json::to_vec_pretty(&chrome)?)?;
    }
    let encoded = if args.compact {
        serde_json::to_vec(&output)?
    } else {
        serde_json::to_vec_pretty(&output)?
    };
    if let Some(path) = args.output {
        std::fs::write(path, encoded)?;
    } else {
        let mut stdout = std::io::stdout().lock();
        stdout.write_all(&encoded)?;
        stdout.write_all(b"\n")?;
    }
    Ok(())
}
