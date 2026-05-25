use clap::{Parser, Subcommand};
use sloforge_loadgen::{
    ReplayConfig, WorkloadConfig, generate, read_trace, replay, summarize_trace, write_trace,
};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[command(about = "Generate, validate, and replay SLOForge JSONL workload traces")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Generate {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Validate {
        trace: PathBuf,
    },
    Replay {
        #[arg(long)]
        trace: PathBuf,
        #[arg(long)]
        target: String,
        #[arg(long, default_value = "sloforge-mock")]
        model: String,
        #[arg(long, default_value_t = 1.0)]
        time_scale: f64,
        #[arg(long, default_value_t = 64)]
        max_concurrency: usize,
        #[arg(long, default_value_t = 30_000)]
        request_timeout_ms: u64,
        #[arg(long)]
        output: PathBuf,
    },
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("sloforge-loadgen: {error}");
        std::process::exit(2);
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    match Args::parse().command {
        Command::Generate { config, output } => {
            let config: WorkloadConfig = serde_json::from_slice(&std::fs::read(config)?)?;
            let records = generate(&config)?;
            write_trace(BufWriter::new(std::fs::File::create(&output)?), &records)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&summarize_trace(&records))?
            );
        }
        Command::Validate { trace } => {
            let records = read_file(&trace)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&summarize_trace(&records))?
            );
        }
        Command::Replay {
            trace,
            target,
            model,
            time_scale,
            max_concurrency,
            request_timeout_ms,
            output,
        } => {
            let records = read_file(&trace)?;
            let measurements = replay(
                &records,
                &ReplayConfig {
                    target,
                    model,
                    time_scale,
                    max_concurrency,
                    request_timeout_ms,
                },
            )
            .await?;
            write_jsonl(&output, &measurements)?;
            println!(
                "replayed {} requests to {}",
                measurements.len(),
                output.display()
            );
        }
    }
    Ok(())
}

fn read_file(path: &Path) -> Result<Vec<sloforge_sim::RequestSpec>, Box<dyn std::error::Error>> {
    Ok(read_trace(BufReader::new(std::fs::File::open(path)?))?)
}

fn write_jsonl<T: serde::Serialize>(
    path: &Path,
    records: &[T],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = BufWriter::new(std::fs::File::create(path)?);
    for record in records {
        serde_json::to_writer(&mut writer, record)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}
