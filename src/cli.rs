//! Command-line interface definitions
//!
//! Provides CLI argument parsing using clap for the Claude Code ACP Agent.

use std::path::PathBuf;

use clap::Parser;

/// Claude Code ACP Agent (Rust) - Use Claude Code from any ACP client
#[derive(Parser, Debug, Clone)]
#[command(name = "claude-code-acp-rs")]
#[command(version, about, long_about = None)]
pub struct Cli {
    /// Run in ACP server mode (stdio communication with editor)
    #[arg(long)]
    pub acp: bool,

    /// Execute prompt directly in headless mode
    #[arg(short = 'p', long, value_name = "PROMPT")]
    pub prompt: Option<String>,

    /// Enable diagnostic mode (auto-log to temp file)
    #[arg(short, long)]
    pub diagnostic: bool,

    /// Log directory (implies diagnostic mode)
    #[arg(short = 'l', long, value_name = "DIR")]
    pub log_dir: Option<PathBuf>,

    /// Log file name (implies diagnostic mode)
    #[arg(short = 'f', long, value_name = "FILE")]
    pub log_file: Option<String>,

    /// Increase logging verbosity (-v, -vv, -vvv)
    /// Note: RUST_LOG env var takes priority over this flag
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Quiet mode (only errors)
    /// Note: RUST_LOG env var takes priority over this flag
    #[arg(short, long)]
    pub quiet: bool,

    /// Resume an existing session by id (mirrors `claude -r <id>`).
    ///
    /// Accepted for compatibility with the wrapper invocation. The actual
    /// session restoration is driven by the ACP `session/load` request that
    /// follows the initialize handshake, so this flag is recorded but not
    /// required by the agent runtime.
    #[arg(short = 'r', long, value_name = "SESSION_ID")]
    pub resume: Option<String>,

    /// OpenTelemetry OTLP endpoint (e.g., http://localhost:4317)
    /// When otel feature is enabled, this configures the OTLP exporter.
    /// When otel feature is disabled, this argument is accepted but ignored.
    #[arg(long, value_name = "URL", env = "OTEL_EXPORTER_OTLP_ENDPOINT")]
    pub otel_endpoint: Option<String>,

    /// OpenTelemetry service name
    #[arg(long, value_name = "NAME", default_value = "claude-code-acp-rs")]
    pub otel_service_name: String,
}

#[allow(clippy::derivable_impls)]
impl Default for Cli {
    fn default() -> Self {
        Self {
            acp: false,
            prompt: None,
            diagnostic: false,
            log_dir: None,
            log_file: None,
            verbose: 0,
            quiet: false,
            resume: None,
            otel_endpoint: None,
            otel_service_name: "claude-code-acp-rs".to_string(),
        }
    }
}

impl Cli {
    /// Check if diagnostic mode is enabled (output to file)
    ///
    /// Returns true if `--diagnostic` is set, or if `--log-dir` or `--log-file` is specified.
    pub fn is_diagnostic(&self) -> bool {
        self.diagnostic || self.log_dir.is_some() || self.log_file.is_some()
    }

    /// Check if OpenTelemetry tracing is enabled
    ///
    /// Returns true if `--otel-endpoint` is specified and the otel feature is enabled.
    #[cfg(feature = "otel")]
    pub fn is_otel_enabled(&self) -> bool {
        self.otel_endpoint.is_some()
    }

    /// Check if OpenTelemetry tracing is enabled (always false without otel feature)
    /// Note: --otel-endpoint argument is still accepted but ignored when feature is disabled
    #[cfg(not(feature = "otel"))]
    pub fn is_otel_enabled(&self) -> bool {
        if self.otel_endpoint.is_some() {
            tracing::warn!("--otel-endpoint specified but otel feature is not enabled, ignoring");
        }
        false
    }

    /// Get the log level based on CLI arguments
    ///
    /// - `--quiet`: ERROR
    /// - default: INFO
    /// - `-v`: DEBUG
    /// - `-vv` or more: TRACE
    pub fn log_level(&self) -> tracing::Level {
        if self.quiet {
            tracing::Level::ERROR
        } else {
            match self.verbose {
                0 => tracing::Level::INFO,
                1 => tracing::Level::DEBUG,
                _ => tracing::Level::TRACE,
            }
        }
    }

    /// Get the log file path for diagnostic mode
    ///
    /// Uses the specified log directory and file name, or defaults to:
    /// - Directory: system temp directory
    /// - File: `claude-code-acp-rs-YYYYMMDD-HHMMSS.log`
    pub fn log_path(&self) -> PathBuf {
        let dir = self.log_dir.clone().unwrap_or_else(std::env::temp_dir);

        let filename = self.log_file.clone().unwrap_or_else(|| {
            let timestamp = chrono::Local::now().format("%Y%m%d-%H%M%S");
            format!("claude-code-acp-rs-{timestamp}.log")
        });

        dir.join(filename)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_cli() {
        let cli = Cli::default();
        assert!(!cli.is_diagnostic());
        assert_eq!(cli.log_level(), tracing::Level::INFO);
    }

    #[test]
    fn test_diagnostic_mode() {
        let cli = Cli {
            diagnostic: true,
            ..Default::default()
        };
        assert!(cli.is_diagnostic());
    }

    #[test]
    fn test_log_dir_implies_diagnostic() {
        let cli = Cli {
            log_dir: Some(PathBuf::from("/tmp")),
            ..Default::default()
        };
        assert!(cli.is_diagnostic());
    }

    #[test]
    fn test_log_file_implies_diagnostic() {
        let cli = Cli {
            log_file: Some("test.log".to_string()),
            ..Default::default()
        };
        assert!(cli.is_diagnostic());
    }

    #[test]
    fn test_log_levels() {
        // Quiet mode
        let cli = Cli {
            quiet: true,
            ..Default::default()
        };
        assert_eq!(cli.log_level(), tracing::Level::ERROR);

        // Default
        let cli = Cli::default();
        assert_eq!(cli.log_level(), tracing::Level::INFO);

        // Verbose
        let cli = Cli {
            verbose: 1,
            ..Default::default()
        };
        assert_eq!(cli.log_level(), tracing::Level::DEBUG);

        // Very verbose
        let cli = Cli {
            verbose: 2,
            ..Default::default()
        };
        assert_eq!(cli.log_level(), tracing::Level::TRACE);
    }

    #[test]
    fn test_log_path_custom_dir() {
        let cli = Cli {
            log_dir: Some(PathBuf::from("/var/log")),
            log_file: Some("test.log".to_string()),
            ..Default::default()
        };
        assert_eq!(cli.log_path(), PathBuf::from("/var/log/test.log"));
    }

    #[test]
    fn test_log_path_default_generates_timestamp() {
        let cli = Cli::default();
        let path = cli.log_path();

        // Should be in temp directory
        assert!(path.starts_with(std::env::temp_dir()));

        // Should have correct prefix
        let filename = path.file_name().unwrap().to_str().unwrap();
        assert!(filename.starts_with("claude-code-acp-rs-"));
        assert!(
            std::path::Path::new(filename)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("log"))
        );
    }

    #[test]
    fn test_cli_acp_mode() {
        let cli = Cli::parse_from(["claude-code-acp-rs", "--acp"]);
        assert!(cli.acp);
        assert!(cli.prompt.is_none());
    }

    #[test]
    fn test_cli_headless_mode() {
        let cli = Cli::parse_from(["claude-code-acp-rs", "-p", "test prompt"]);
        assert!(!cli.acp);
        assert_eq!(cli.prompt, Some("test prompt".to_string()));
    }

    #[test]
    fn test_cli_headless_mode_long() {
        let cli = Cli::parse_from(["claude-code-acp-rs", "--prompt", "test prompt"]);
        assert!(!cli.acp);
        assert_eq!(cli.prompt, Some("test prompt".to_string()));
    }

    #[test]
    fn test_cli_no_mode() {
        let cli = Cli::parse_from(["claude-code-acp-rs"]);
        assert!(!cli.acp);
        assert!(cli.prompt.is_none());
    }
}
