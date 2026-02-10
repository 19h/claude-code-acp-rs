//! Claude Code ACP Agent binary
//!
//! Run with: cargo run
//!
//! For help: cargo run -- --help

use clap::Parser;
use claude_code_acp::{cli::Cli, run_acp_with_cli, shutdown_otel};
use std::io::IsTerminal;
use tokio::signal;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Run the ACP agent with graceful shutdown on SIGTERM/SIGINT
    let result = tokio::select! {
        result = run_acp_with_cli(&cli) => result,
        _ = signal::ctrl_c() => {
            eprintln!("Received SIGINT, shutting down...");
            Ok(())
        }
        _ = async {
            #[cfg(unix)]
            {
                let mut sigterm = signal::unix::signal(signal::unix::SignalKind::terminate())
                    .expect("Failed to register SIGTERM handler");
                sigterm.recv().await
            }
            #[cfg(not(unix))]
            {
                std::future::pending::<()>().await
            }
        } => {
            eprintln!("Received SIGTERM, shutting down...");
            Ok(())
        }
    };

    // Shutdown OpenTelemetry to flush all pending spans
    shutdown_otel();

    if let Err(e) = result {
        // Output error to stderr (ACP protocol uses stdout for messages)
        eprintln!("Error: {}", e);

        // If running in interactive mode, show more details
        if std::io::stdin().is_terminal() {
            eprintln!("\nFor debugging, run with --diagnostic to log to a file.");
            eprintln!("Or use -v/-vv/-vvv for more verbose logging.");
        }

        std::process::exit(1);
    }

    Ok(())
}
