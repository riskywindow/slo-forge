use clap::{Parser, Subcommand};
use sloforge_gateway::Gateway;
use sloforge_gateway::config::GatewayConfig;
use sloforge_gateway::mock::{MockBackend, MockBackendConfig};
use std::net::SocketAddr;
use std::path::PathBuf;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(
    name = "sloforge-gateway",
    version,
    about = "SLO-aware streaming inference data plane"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run the gateway using an offline JSON configuration generated from a `DeploymentPlan`.
    Serve {
        #[arg(long)]
        config: PathBuf,
    },
    /// Run an explicit deterministic mock inference backend for CPU evaluation.
    MockBackend {
        #[arg(long)]
        config: PathBuf,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
    let cli = Cli::parse();
    match cli.command {
        Command::Serve { config } => {
            let config = GatewayConfig::from_path(config)?;
            let bind: SocketAddr = config.bind.parse()?;
            let gateway = Gateway::new(config)?;
            let health_checks = gateway.spawn_health_checks();
            let listener = tokio::net::TcpListener::bind(bind).await?;
            info!(address = %listener.local_addr()?, "gateway listening");
            axum::serve(listener, gateway.router())
                .with_graceful_shutdown(shutdown_signal())
                .await?;
            drop(health_checks);
            gateway.write_trace_artifact()?;
        }
        Command::MockBackend { config } => {
            let config = MockBackendConfig::from_path(config)?;
            let bind: SocketAddr = config.bind.parse()?;
            let backend = MockBackend::new(config)?;
            let listener = tokio::net::TcpListener::bind(bind).await?;
            info!(address = %listener.local_addr()?, "mock backend listening");
            axum::serve(listener, backend.router())
                .with_graceful_shutdown(shutdown_signal())
                .await?;
        }
    }
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        if tokio::signal::ctrl_c().await.is_err() {
            std::future::pending::<()>().await;
        }
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }
}
