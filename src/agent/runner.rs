//! ACP Agent runner
//!
//! Entry point for running the Claude ACP Agent.

use std::io::IsTerminal;
use std::sync::Arc;

use sacp::link::AgentToClient;
use sacp::schema::{
    CancelNotification, InitializeRequest, LoadSessionRequest, NewSessionRequest, PromptRequest,
    SetSessionModeRequest,
};
use sacp::{ByteStreams, JrConnectionCx, MessageCx};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};
use tokio_util::sync::CancellationToken;
use tracing::Instrument;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use super::core::ClaudeAcpAgent;
use super::handlers;
use crate::cli::Cli;
use crate::types::AgentError;

// OpenTelemetry imports (only when feature is enabled)
#[cfg(feature = "otel")]
use opentelemetry::global;
#[cfg(feature = "otel")]
use opentelemetry::trace::TracerProvider;
#[cfg(feature = "otel")]
use opentelemetry_otlp::WithExportConfig;
#[cfg(feature = "otel")]
use opentelemetry_sdk::trace::SdkTracerProvider;

// Global storage for OpenTelemetry provider (for proper shutdown)
#[cfg(feature = "otel")]
static OTEL_PROVIDER: std::sync::OnceLock<SdkTracerProvider> = std::sync::OnceLock::new();

/// Shutdown OpenTelemetry provider (flush all pending spans)
///
/// This should be called before the application exits to ensure all
/// telemetry data is properly flushed to the backend.
#[cfg(feature = "otel")]
pub fn shutdown_otel() {
    if let Some(provider) = OTEL_PROVIDER.get() {
        tracing::info!("Shutting down OpenTelemetry provider...");
        if let Err(e) = provider.shutdown() {
            eprintln!("Failed to shutdown OpenTelemetry provider: {:?}", e);
        } else {
            tracing::info!("OpenTelemetry provider shutdown complete");
        }
    }
}

/// Shutdown OpenTelemetry provider (no-op when feature is disabled)
#[cfg(not(feature = "otel"))]
pub fn shutdown_otel() {}

/// Initialize OpenTelemetry tracer provider
///
/// Following OpenTelemetry Rust best practices:
/// 1. Create provider with batch exporter for production use
/// 2. Set global tracer provider for easy access
/// 3. Store provider for proper shutdown
#[cfg(feature = "otel")]
fn init_otel(endpoint: &str, service_name: &str) -> anyhow::Result<SdkTracerProvider> {
    use opentelemetry_sdk::Resource;

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()?;

    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(
            Resource::builder()
                .with_service_name(service_name.to_owned())
                .build(),
        )
        .build();

    // Set global tracer provider (best practice)
    global::set_tracer_provider(provider.clone());

    Ok(provider)
}

/// Build an EnvFilter based on CLI args and RUST_LOG environment variable
///
/// Priority: RUST_LOG environment variable > CLI arguments (-v, -vv, -q)
fn build_env_filter(cli: &Cli) -> tracing_subscriber::EnvFilter {
    // Check if RUST_LOG is set and non-empty
    if let Ok(rust_log) = std::env::var("RUST_LOG") {
        if !rust_log.is_empty() {
            // RUST_LOG takes priority - use it directly
            return tracing_subscriber::EnvFilter::new(rust_log);
        }
    }

    // No RUST_LOG set, use CLI arguments to determine level
    let level = cli.log_level();
    tracing_subscriber::EnvFilter::from_default_env().add_directive(level.into())
}

/// Initialize logging with file output (diagnostic mode)
fn init_logging_to_file(cli: &Cli) -> anyhow::Result<()> {
    let filter = build_env_filter(cli);

    let log_path = cli.log_path();

    // Ensure directory exists
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let file = std::fs::File::create(&log_path)?;

    // Output log file location to stderr (user needs to know)
    eprintln!("Diagnostic mode: logging to {}", log_path.display());

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::sync::Mutex::new(file))
        .with_ansi(false);

    #[cfg(feature = "otel")]
    {
        if cli.is_otel_enabled() {
            let endpoint = cli.otel_endpoint.as_ref().unwrap();
            let service_name = &cli.otel_service_name;

            eprintln!(
                "OpenTelemetry enabled: endpoint={}, service={}",
                endpoint, service_name
            );

            let provider = init_otel(endpoint, service_name)?;
            let tracer = provider.tracer("claude-code-acp-rs");
            let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

            // Store provider globally for proper shutdown
            drop(OTEL_PROVIDER.set(provider));

            tracing_subscriber::registry()
                .with(filter)
                .with(fmt_layer)
                .with(otel_layer)
                .init();
        } else {
            tracing_subscriber::registry()
                .with(filter)
                .with(fmt_layer)
                .init();
        }
    }

    #[cfg(not(feature = "otel"))]
    {
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt_layer)
            .init();
    }

    Ok(())
}

/// Initialize logging with stderr output (normal mode)
fn init_logging_to_stderr(cli: &Cli) {
    let filter = build_env_filter(cli);

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_ansi(false);

    #[cfg(feature = "otel")]
    {
        if cli.is_otel_enabled() {
            let endpoint = cli.otel_endpoint.as_ref().unwrap();
            let service_name = &cli.otel_service_name;

            eprintln!(
                "OpenTelemetry enabled: endpoint={}, service={}",
                endpoint, service_name
            );

            let provider = init_otel(endpoint, service_name).expect("Failed to init OpenTelemetry");
            let tracer = provider.tracer("claude-code-acp-rs");
            let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

            // Store provider globally for proper shutdown
            drop(OTEL_PROVIDER.set(provider));

            tracing_subscriber::registry()
                .with(filter)
                .with(fmt_layer)
                .with(otel_layer)
                .init();
        } else {
            tracing_subscriber::registry()
                .with(filter)
                .with(fmt_layer)
                .init();
        }
    }

    #[cfg(not(feature = "otel"))]
    {
        tracing_subscriber::registry()
            .with(filter)
            .with(fmt_layer)
            .init();
    }
}

/// Initialize logging based on CLI arguments
fn init_logging(cli: &Cli) -> anyhow::Result<()> {
    if cli.is_diagnostic() {
        init_logging_to_file(cli)
    } else {
        init_logging_to_stderr(cli);
        Ok(())
    }
}

/// Run the ACP agent with CLI arguments
///
/// This is the main entry point when using CLI argument parsing.
/// It initializes logging based on CLI args and starts the ACP handler chain.
pub async fn run_acp_with_cli(cli: &Cli) -> anyhow::Result<()> {
    let startup_time = std::time::Instant::now();

    // Initialize logging first (must happen before any tracing)
    init_logging(cli)?;

    // Record startup as a SHORT-LIVED span that closes immediately
    // This ensures it appears in Jaeger right away, not just when agent shuts down
    {
        let startup_span = tracing::info_span!(
            "agent_startup",
            version = %env!("CARGO_PKG_VERSION"),
            pid = %std::process::id(),
            diagnostic = %cli.is_diagnostic(),
            otel_enabled = %cli.otel_endpoint.is_some(),
        );
        let _enter = startup_span.enter();

        tracing::info!("========== Claude Code ACP Agent Starting ==========");
        tracing::info!(
            version = %env!("CARGO_PKG_VERSION"),
            pid = %std::process::id(),
            "Agent process info"
        );

        // Log CLI configuration
        if cli.is_diagnostic() {
            tracing::info!(
                log_path = %cli.log_path().display(),
                "Diagnostic mode enabled"
            );
        }

        if let Some(otel_endpoint) = &cli.otel_endpoint {
            tracing::info!(
                otel_endpoint = %otel_endpoint,
                "OpenTelemetry tracing enabled"
            );
        }

        // Log startup timing
        let init_elapsed = startup_time.elapsed();
        tracing::info!(
            init_elapsed_ms = init_elapsed.as_millis(),
            "Logging initialized"
        );
    } // <-- startup_span closes here and gets exported to Jaeger immediately!

    // Emit a separate "agent ready" trace that will show in Jaeger
    emit_agent_ready_trace(startup_time.elapsed()).await;

    // Run the server (this is a long-running span, won't appear until shutdown)
    // Use Box::pin to reduce stack size of large future (~23KB)
    let result = Box::pin(run_acp_server()).await.map_err(Into::into);

    // Emit shutdown trace
    emit_agent_shutdown_trace(startup_time.elapsed()).await;

    result
}

/// Emit a short-lived trace to indicate agent is ready
/// This will appear in Jaeger immediately
#[tracing::instrument(name = "agent_ready", skip_all, fields(
    startup_ms = %startup_duration.as_millis(),
    version = %env!("CARGO_PKG_VERSION"),
    pid = %std::process::id(),
))]
async fn emit_agent_ready_trace(startup_duration: std::time::Duration) {
    tracing::info!(
        startup_ms = startup_duration.as_millis(),
        "Agent ready and waiting for ACP messages"
    );
    // Small delay to ensure span is exported before continuing
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
}

/// Emit a short-lived trace to indicate agent is shutting down
#[tracing::instrument(name = "agent_shutdown", skip_all, fields(
    uptime_secs = %total_uptime.as_secs(),
    uptime_ms = %total_uptime.as_millis(),
))]
async fn emit_agent_shutdown_trace(total_uptime: std::time::Duration) {
    tracing::info!(
        uptime_secs = total_uptime.as_secs(),
        uptime_ms = total_uptime.as_millis(),
        "========== Agent Shutdown Complete =========="
    );
}

/// Run the ACP agent
///
/// This is the main entry point for the Claude Code ACP Agent.
/// It sets up the JSON-RPC handler chain and serves requests over stdio.
///
/// For CLI usage with argument parsing, use `run_acp_with_cli()` instead.
pub async fn run_acp() -> Result<(), sacp::Error> {
    // Initialize tracing with default settings
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .with_writer(std::io::stderr)
        .init();

    // Use Box::pin to reduce stack size of large future (~23KB)
    Box::pin(run_acp_server()).await
}

/// Internal server implementation
///
/// This contains the actual ACP server logic, shared by both `run_acp()` and `run_acp_with_cli()`.
///
/// # Large Future Note
///
/// This function generates a large future (~22KB) due to the complex handler chain with
/// nested closures for tracing spans. This is acceptable because:
/// 1. The function is always called via `Box::pin` at the call sites, which moves the
///    future to the heap and reduces stack memory usage
/// 2. The builder pattern with nested closures is the idiomatic way to use the sacp SDK
/// 3. Extracting handlers into separate functions doesn't help due to the builder's
///    ownership requirements
#[tracing::instrument(name = "acp_server_main")]
#[allow(clippy::large_futures)]
async fn run_acp_server() -> Result<(), sacp::Error> {
    let server_start_time = std::time::Instant::now();

    // Check if running in interactive terminal (for debugging)
    let is_tty = std::io::stdin().is_terminal();

    // Print startup banner for easy log identification
    let agent_session_id = uuid::Uuid::new_v4();
    let start_time = chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f");

    tracing::info!(
        "================================================================================"
    );
    tracing::info!("  Claude Code ACP Agent - Session Start");
    tracing::info!(
        "--------------------------------------------------------------------------------"
    );
    tracing::info!("  Version:    {}", env!("CARGO_PKG_VERSION"));
    tracing::info!("  Start Time: {}", start_time);
    tracing::info!("  Session ID: {}", agent_session_id);
    tracing::info!("  PID:        {}", std::process::id());
    tracing::info!(
        "  TTY Mode:   {}",
        if is_tty { "interactive" } else { "subprocess" }
    );
    tracing::info!(
        "================================================================================"
    );

    // Log environment info
    tracing::debug!(
        rust_log = ?std::env::var("RUST_LOG").ok(),
        cwd = ?std::env::current_dir().ok(),
        "Environment configuration"
    );

    if is_tty {
        // Running in interactive terminal - provide helpful message
        eprintln!("Claude Code ACP Agent is running in interactive mode.");
        eprintln!("This agent communicates via ACP protocol over stdin/stdout.");
        eprintln!("To use with an editor, configure it to run this binary.");
        eprintln!("Waiting for ACP protocol messages on stdin...");
        eprintln!("(Press Ctrl+C to exit)");
    } else {
        // Running as subprocess (e.g., from editor) - minimal logging
        tracing::info!("Waiting for ACP protocol messages on stdin...");
    }

    // Create the agent
    let agent_create_start = std::time::Instant::now();
    let agent = ClaudeAcpAgent::new();
    let config = Arc::new(agent.config().clone());
    let sessions = agent.sessions().clone();
    let prompt_manager = agent.prompt_manager().clone();
    let agent_create_elapsed = agent_create_start.elapsed();

    tracing::info!(
        agent_name = %agent.name(),
        elapsed_ms = agent_create_elapsed.as_millis(),
        has_base_url = config.base_url.is_some(),
        has_api_key = config.api_key.is_some(),
        has_model = config.model.is_some(),
        "Agent created"
    );

    // Build the handler chain
    tracing::debug!("Building ACP handler chain");
    AgentToClient::builder()
        .name(agent.name())
        // Handle initialize request
        .on_receive_request(
            {
                let config = config.clone();
                async move |request: InitializeRequest, request_cx, _connection_cx| {
                    let protocol_version = format!("{:?}", request.protocol_version);
                    let span = tracing::info_span!(
                        "handle_initialize",
                        protocol_version = %protocol_version,
                    );

                    async {
                        tracing::info!(
                            "Received initialize request (protocol version: {})",
                            protocol_version
                        );
                        let response = handlers::handle_initialize(request, &config);
                        tracing::debug!("Sending initialize response");
                        request_cx.respond(response)
                    }
                    .instrument(span)
                    .await
                }
            },
            sacp::on_receive_request!(),
        )
        // Handle session/new request
        .on_receive_request(
            {
                let config = config.clone();
                let sessions = sessions.clone();
                async move |request: NewSessionRequest, request_cx, connection_cx| {
                    let cwd = request.cwd.display().to_string();
                    let span = tracing::info_span!(
                        "handle_session_new",
                        cwd = %cwd,
                        mcp_server_count = request.mcp_servers.len(),
                    );

                    async {
                        tracing::debug!("Received session/new request");
                        match handlers::handle_new_session(request, &config, &sessions, connection_cx).await {
                            Ok(response) => request_cx.respond(response),
                            Err(e) => request_cx
                                .respond_with_error(sacp::util::internal_error(e.to_string())),
                        }
                    }
                    .instrument(span)
                    .await
                }
            },
            sacp::on_receive_request!(),
        )
        // Handle session/load request
        .on_receive_request(
            {
                let config = config.clone();
                let sessions = sessions.clone();
                async move |request: LoadSessionRequest, request_cx, _connection_cx| {
                    let session_id = request.session_id.0.clone();
                    let span = tracing::info_span!(
                        "handle_session_load",
                        session_id = %session_id,
                    );

                    async {
                        tracing::debug!("Received session/load request for session {}", session_id);
                        match handlers::handle_load_session(request, &config, &sessions) {
                            Ok(response) => request_cx.respond(response),
                            Err(e) => request_cx
                                .respond_with_error(sacp::util::internal_error(e.to_string())),
                        }
                    }
                    .instrument(span)
                    .await
                }
            },
            sacp::on_receive_request!(),
        )
        // Handle session/prompt request
        //
        // IMPORTANT: This handler uses `spawn` to run handle_prompt in a background task.
        // This is necessary because handle_prompt may call `can_use_tool` callback,
        // which sends `session/request_permission` request and blocks waiting for response.
        // If we block in the handler directly, the incoming_protocol_actor cannot process
        // the response, causing a deadlock.
        //
        // NEW PROMPT CANCELLATION: When a new prompt arrives for a session that already
        // has a prompt running, we first cancel the previous prompt to avoid multiple
        // concurrent prompts for the same session.
        .on_receive_request(
            {
                let config = config.clone();
                let sessions = sessions.clone();
                let prompt_manager = prompt_manager.clone();
                async move |request: PromptRequest, request_cx, connection_cx| {
                    let session_id = request.session_id.0.clone();
                    let prompt_len = request.prompt.len();

                    // Create a span for the entire request handling
                    let span = tracing::info_span!(
                        "handle_session_prompt",
                        session_id = %session_id,
                        prompt_blocks = prompt_len,
                    );

                    tracing::debug!(
                        "Received session/prompt request for session {}, spawning handler",
                        session_id
                    );

                    // IMPORTANT: Cancel any previous prompt for this session first
                    // This prevents issues like cargo build blocking new prompts
                    let session_id_str = session_id.to_string();
                    prompt_manager.cancel_session_prompt(&session_id_str).await;

                    // Create a cancellation token for this prompt
                    let cancel_token = CancellationToken::new();

                    // Spawn the handler in a background task to avoid blocking the event loop.
                    //
                    // DESIGN NOTE: We use a double-spawn pattern here:
                    //
                    // 1. Outer spawn (connection_cx.spawn):
                    //    - Required by ACP protocol to avoid blocking the incoming_protocol_actor
                    //    - Allows the protocol to continue processing messages (like permission responses)
                    //    - Runs the prompt handling asynchronously
                    //
                    // 2. Inner spawn (tokio::spawn):
                    //    - Required by PromptManager to track and cancel the prompt task
                    //    - The PromptManager needs a JoinHandle to wait for task completion
                    //    - This handle is registered and can be used for timeout-based cancellation
                    //
                    // 3. Oneshot channel:
                    //    - Bridges the inner task (tracked by PromptManager) with the outer task
                    //    - Sends the result back to be used for the ACP response
                    //
                    // Alternative considered: Use AbortHandle instead of JoinHandle
                    // - Pros: Simpler, single spawn
                    // - Cons: Can't wait for graceful shutdown, may leave resources leaked
                    // - Decision: Current design is safer for resource cleanup
                    let config = config.clone();
                    let sessions = sessions.clone();
                    let prompt_manager = prompt_manager.clone();
                    connection_cx.spawn({
                        let connection_cx = connection_cx.clone();
                        async move {
                            async {
                                tracing::debug!(
                                    "Starting prompt handling for session {}",
                                    session_id_str
                                );

                                // Create a channel to send the result back
                                let (result_tx, result_rx) = tokio::sync::oneshot::channel();

                                // Spawn the actual prompt handling task
                                let handle = tokio::spawn({
                                    let cancel_token = cancel_token.clone();
                                    let session_id_for_log = session_id_str.clone();
                                    // Clone connection_cx for use in spawned task
                                    // Note: We clone here because connection_cx will be moved into the spawned async block
                                    let connection_cx_inner = connection_cx.clone();
                                    async move {
                                        tracing::debug!(
                                            session_id = %session_id_for_log,
                                            "Prompt task started"
                                        );

                                        let result = handlers::handle_prompt(
                                            request,
                                            &config,
                                            &sessions,
                                            connection_cx_inner,
                                            cancel_token,
                                        )
                                        .await;

                                        // Send result through channel
                                        // If receiver is dropped, the task panicked or was cancelled
                                        if result_tx.send(result).is_err() {
                                            tracing::warn!("Failed to send prompt result - receiver was dropped (task may have panicked)");
                                        }

                                        tracing::debug!(
                                            session_id = %session_id_for_log,
                                            "Prompt task completed"
                                        );
                                    }
                                });

                                // Register the prompt task
                                let prompt_id = prompt_manager.register_prompt(
                                    session_id_str.clone(),
                                    handle,
                                    cancel_token,
                                );

                                // Wait for the result
                                let result = result_rx.await;
                                let result = match result {
                                    Ok(Ok(response)) => Ok(response),
                                    Ok(Err(e)) => Err(e),
                                    Err(_) => {
                                        // Sender was dropped (task panicked)
                                        Err(AgentError::Internal(
                                            "Prompt task failed unexpectedly".to_string(),
                                        ))
                                    }
                                };

                                // Complete the prompt
                                prompt_manager.complete_prompt(&session_id_str, &prompt_id);

                                // Respond to the request
                                match result {
                                    Ok(response) => request_cx.respond(response),
                                    Err(e) => {
                                        tracing::error!("Prompt error: {}", e);
                                        request_cx.respond_with_error(sacp::util::internal_error(
                                            e.to_string(),
                                        ))
                                    }
                                }
                            }
                            .instrument(span)
                            .await
                        }
                    })
                }
            },
            sacp::on_receive_request!(),
        )
        // Handle session/setMode request
        .on_receive_request(
            {
                let sessions = sessions.clone();
                async move |request: SetSessionModeRequest, request_cx, connection_cx| {
                    let session_id = request.session_id.0.clone();
                    let mode_id = request.mode_id.0.clone();
                    let span = tracing::info_span!(
                        "handle_session_setMode",
                        session_id = %session_id,
                        mode_id = %mode_id,
                    );

                    async {
                        tracing::debug!("Received session/setMode request");
                        match handlers::handle_set_mode(request, &sessions, connection_cx).await {
                            Ok(response) => request_cx.respond(response),
                            Err(e) => request_cx
                                .respond_with_error(sacp::util::internal_error(e.to_string())),
                        }
                    }
                    .instrument(span)
                    .await
                }
            },
            sacp::on_receive_request!(),
        )
        // Note: SetSessionModel is not yet supported by sacp SDK (JrRequest not implemented)
        // The model selection is returned in NewSessionResponse, but changing it mid-session
        // is not yet available. When sacp adds support, uncomment the following handler.
        // Handle session/cancel notification
        .on_receive_notification(
            {
                let sessions = sessions.clone();
                async move |notification: CancelNotification, _connection_cx| {
                    let session_id = notification.session_id.0.clone();
                    let span = tracing::info_span!(
                        "handle_session_cancel",
                        session_id = %session_id,
                    );

                    async {
                        tracing::debug!(
                            "Received session/cancel notification for session {}",
                            session_id
                        );
                        if let Err(e) = handlers::handle_cancel(&session_id, &sessions).await {
                            tracing::error!("Cancel error: {}", e);
                        }
                        Ok(())
                    }
                    .instrument(span)
                    .await
                }
            },
            sacp::on_receive_notification!(),
        )
        // Handle unknown messages and unstable protocol methods
        .on_receive_message(
            {
                let config = agent.config().clone();
                let sessions = agent.sessions().clone();
                async move |message: MessageCx, connection_cx: JrConnectionCx<AgentToClient>| {
                    let method = message.method().to_string();
                    let span = tracing::info_span!(
                        "handle_message",
                        method = ?method,
                    );

                    async {
                        // Route unstable protocol methods that sacp doesn't know about
                        match method.as_str() {
                            "session/set_model" => {
                                dispatch_unstable_request(message, |params| async {
                                    let request: agent_client_protocol_schema::SetSessionModelRequest =
                                        serde_json::from_value(params)?;
                                    let response = handlers::handle_set_session_model(request, &sessions).await
                                        .map_err(|e| anyhow::anyhow!("{}", e))?;
                                    Ok(serde_json::to_value(response)?)
                                }).await
                            }
                            "session/fork" => {
                                let cx = connection_cx.clone();
                                dispatch_unstable_request(message, |params| async {
                                    let request: agent_client_protocol_schema::ForkSessionRequest =
                                        serde_json::from_value(params)?;
                                    let response = handlers::handle_fork_session(request, &config, &sessions, cx)
                                        .map_err(|e| anyhow::anyhow!("{}", e))?;
                                    Ok(serde_json::to_value(response)?)
                                }).await
                            }
                            "session/resume" => {
                                let cx = connection_cx.clone();
                                dispatch_unstable_request(message, |params| async {
                                    let request: agent_client_protocol_schema::ResumeSessionRequest =
                                        serde_json::from_value(params)?;
                                    let response = handlers::handle_resume_session(request, &config, &sessions, cx)
                                        .map_err(|e| anyhow::anyhow!("{}", e))?;
                                    Ok(serde_json::to_value(response)?)
                                }).await
                            }
                            "session/list" => {
                                dispatch_unstable_request(message, |params| async move {
                                    let request: agent_client_protocol_schema::ListSessionsRequest =
                                        serde_json::from_value(params)?;
                                    let response = handlers::handle_list_sessions(request)
                                        .map_err(|e| anyhow::anyhow!("{}", e))?;
                                    Ok(serde_json::to_value(response)?)
                                }).await
                            }
                            _ => {
                                tracing::warn!("Received unknown message: {:?}", method);
                                message.respond_with_error(
                                    sacp::util::internal_error("Unknown method"),
                                    connection_cx,
                                )
                            }
                        }
                    }
                    .instrument(span)
                    .await
                }
            },
            sacp::on_receive_message!(),
        )
        // Serve over stdio
        // Note: stdout is used for ACP protocol messages, stderr is for logging
        .serve(ByteStreams::new(
            tokio::io::stdout().compat_write(),
            tokio::io::stdin().compat(),
        ))
        .await
        .map_err(|e| {
            let uptime = server_start_time.elapsed();
            tracing::error!(
                error = %e,
                uptime_ms = uptime.as_millis(),
                "ACP server error"
            );
            e
        })
        .inspect(|_result| {
            let uptime = server_start_time.elapsed();
            tracing::info!(
                uptime_secs = uptime.as_secs(),
                uptime_ms = uptime.as_millis(),
                "ACP server shutting down gracefully"
            );
        })
}

/// Dispatch an unstable protocol request.
///
/// Extracts the params from the untyped MessageCx, passes them to the handler,
/// and sends the response back. The handler is responsible for deserializing the
/// params into the expected type and returning a serialized response.
async fn dispatch_unstable_request<Fut>(
    message: MessageCx,
    handler: impl FnOnce(serde_json::Value) -> Fut,
) -> Result<(), sacp::Error>
where
    Fut: std::future::Future<Output = Result<serde_json::Value, anyhow::Error>>,
{
    match message {
        MessageCx::Request(untyped, request_cx) => match handler(untyped.params.clone()).await {
            Ok(response_value) => request_cx.respond(response_value),
            Err(e) => {
                tracing::error!(error = %e, "Unstable handler error");
                request_cx.respond_with_error(sacp::util::internal_error(e.to_string()))
            }
        },
        MessageCx::Notification(_) => {
            tracing::warn!("Received notification for request-only method");
            Ok(())
        }
    }
}
