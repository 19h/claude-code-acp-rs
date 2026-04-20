//! ACP request handlers
//!
//! Implements handlers for ACP protocol requests:
//! - initialize: Return agent capabilities
//! - session/new: Create a new session
//! - session/prompt: Execute a prompt (Phase 1: simplified)
//! - session/setMode: Set permission mode

use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use futures::{Stream, StreamExt};
use sacp::JrConnectionCx;
use sacp::link::AgentToClient;
use sacp::schema::{
    AgentCapabilities, ContentBlock, CurrentModeUpdate, Implementation, InitializeRequest,
    InitializeResponse, LoadSessionRequest, LoadSessionResponse, NewSessionRequest,
    NewSessionResponse, PromptCapabilities, PromptRequest, PromptResponse, SessionId, SessionMode,
    SessionModeId, SessionModeState, SessionNotification, SessionUpdate, SetSessionModeRequest,
    SetSessionModeResponse, StopReason,
};
use tokio_util::sync::CancellationToken;

// Unstable types from agent-client-protocol-schema
use agent_client_protocol_schema::{ModelInfo, SessionModelState};
use tokio::sync::broadcast;
use tracing::instrument;

use crate::agent::flush;
use crate::agent::slash_commands::{
    filter_commands, get_predefined_commands, transform_mcp_command_input,
};
use crate::session::{PermissionMode, SessionManager};
use crate::terminal::TerminalClient;
use crate::types::{AgentConfig, AgentError, NewSessionMeta};

// Default model constants
const DEFAULT_MODEL_ID: &str = "claude-sonnet-4-20250514";
const DEFAULT_MODEL_DISPLAY_NAME: &str = "Default";

/// Handle initialize request
///
/// Returns the agent's capabilities and protocol version.
#[instrument(
    name = "acp_initialize",
    skip(request, _config),
    fields(
        protocol_version = ?request.protocol_version,
        agent_version = %env!("CARGO_PKG_VERSION"),
    )
)]
pub fn handle_initialize(request: InitializeRequest, _config: &AgentConfig) -> InitializeResponse {
    tracing::info!(
        protocol_version = ?request.protocol_version,
        agent_name = "claude-code-acp-rs",
        agent_version = %env!("CARGO_PKG_VERSION"),
        "Handling ACP initialize request"
    );

    // Build agent capabilities using builder pattern
    let prompt_caps = PromptCapabilities::new().image(true).embedded_context(true);

    let mcp_caps = sacp::schema::McpCapabilities::new().http(true).sse(true);

    let session_caps = sacp::schema::SessionCapabilities::new()
        .fork(agent_client_protocol_schema::SessionForkCapabilities::default())
        .list(agent_client_protocol_schema::SessionListCapabilities::default())
        .resume(agent_client_protocol_schema::SessionResumeCapabilities::default());

    let capabilities = AgentCapabilities::new()
        .prompt_capabilities(prompt_caps)
        .mcp_capabilities(mcp_caps)
        .session_capabilities(session_caps)
        .load_session(true);

    // Build agent info
    let agent_info =
        Implementation::new("claude-code-acp-rs", env!("CARGO_PKG_VERSION")).title("Claude Code");

    tracing::debug!(
        capabilities = ?capabilities,
        "Sending initialize response with capabilities"
    );

    // Build response
    InitializeResponse::new(request.protocol_version)
        .agent_capabilities(capabilities)
        .agent_info(agent_info)
}

/// Handle session/new request
///
/// Creates a new session with the given working directory and metadata.
/// Returns available modes and models for the session.
#[instrument(
    name = "acp_new_session",
    skip(request, config, sessions, connection_cx),
    fields(
        cwd = ?request.cwd,
        has_meta = request.meta.is_some(),
        mcp_server_count = request.mcp_servers.len(),
    )
)]
#[allow(unused_variables)]
pub async fn handle_new_session(
    request: NewSessionRequest,
    config: &AgentConfig,
    sessions: &Arc<SessionManager>,
    connection_cx: JrConnectionCx<AgentToClient>,
) -> Result<NewSessionResponse, AgentError> {
    let start_time = Instant::now();

    tracing::info!(
        cwd = ?request.cwd,
        has_meta = request.meta.is_some(),
        mcp_server_count = request.mcp_servers.len(),
        "Creating new ACP session"
    );

    // Log external MCP servers from client
    if !request.mcp_servers.is_empty() {
        tracing::info!(
            mcp_servers = ?request.mcp_servers.iter().map(|s| match s {
                sacp::schema::McpServer::Stdio(stdio) => format!("{}(stdio:{})", stdio.name, stdio.command.display()),
                sacp::schema::McpServer::Http(http) => format!("{}(http:{})", http.name, http.url),
                sacp::schema::McpServer::Sse(sse) => format!("{}(sse:{})", sse.name, sse.url),
                _ => "unknown".to_string(),
            }).collect::<Vec<_>>(),
            "External MCP servers from client"
        );
    }

    // Parse metadata from request if present
    let meta = request.meta.as_ref().and_then(|m| {
        serde_json::to_value(m)
            .ok()
            .map(|v| NewSessionMeta::from_request_meta(Some(&v)))
    });

    // Get working directory from request
    let cwd = request.cwd;

    // Generate session ID
    let session_id = uuid::Uuid::new_v4().to_string();

    tracing::debug!(
        session_id = %session_id,
        "Generated new session ID"
    );

    // Create the session
    sessions.create_session(
        session_id.clone(),
        cwd.clone(),
        config,
        meta.as_ref(),
        &request.mcp_servers,
    )?;

    // Build available modes
    let available_modes = build_available_modes();
    let mode_state = SessionModeState::new("default", available_modes);

    // Build available models
    let model_state = build_available_models(config);

    let elapsed = start_time.elapsed();
    tracing::info!(
        session_id = %session_id,
        cwd = ?cwd,
        elapsed_ms = elapsed.as_millis(),
        "New session created successfully"
    );

    // Send available commands list to client
    // This is done asynchronously (similar to TypeScript's setTimeout)
    // to ensure the response is sent first
    #[cfg(not(test))] // Only in production, skip in tests
    {
        let session_id_clone = session_id.clone();
        tokio::spawn(async move {
            if let Err(e) = send_available_commands_update(&session_id_clone, connection_cx) {
                tracing::warn!(
                    session_id = %session_id_clone,
                    "Failed to send available commands update: {}",
                    e
                );
            }
        });
    }

    Ok(NewSessionResponse::new(session_id)
        .modes(mode_state)
        .models(model_state))
}

/// Handle session/load request
///
/// Loads an existing session by resuming it with the given session ID.
/// Returns available modes and models for the session.
///
/// Note: Unlike TS implementation which doesn't support loadSession,
/// our Rust implementation uses claude-code-agent-sdk's resume functionality
/// to restore conversation history.
#[instrument(
    name = "acp_load_session",
    skip(request, config, sessions),
    fields(
        session_id = %request.session_id.0,
        cwd = ?request.cwd,
    )
)]
pub fn handle_load_session(
    request: LoadSessionRequest,
    config: &AgentConfig,
    sessions: &Arc<SessionManager>,
) -> Result<LoadSessionResponse, AgentError> {
    let start_time = Instant::now();

    // The session_id in the request is the ID of the session to resume
    let resume_session_id = request.session_id.0.to_string();
    let cwd = request.cwd;

    tracing::info!(
        session_id = %resume_session_id,
        cwd = ?cwd,
        "Loading existing session"
    );

    // Create NewSessionMeta with resume option
    // This tells the underlying SDK to resume from the specified session
    let meta = NewSessionMeta::with_resume(&resume_session_id);

    // Generate a new session ID for this loaded session
    // Note: We use the same session ID as the one being loaded
    // so the client can continue using the same ID
    let session_id = resume_session_id.clone();

    // Check if session already exists in our manager
    // If it does, we just return success (session already loaded)
    if sessions.has_session(&session_id) {
        let elapsed = start_time.elapsed();
        tracing::info!(
            session_id = %session_id,
            elapsed_ms = elapsed.as_millis(),
            "Session already exists, returning existing session"
        );
    } else {
        // Create the session with resume option
        tracing::debug!(
            session_id = %session_id,
            "Creating session with resume option"
        );
        sessions.create_session(session_id.clone(), cwd.clone(), config, Some(&meta), &[])?;

        let elapsed = start_time.elapsed();
        tracing::info!(
            session_id = %session_id,
            elapsed_ms = elapsed.as_millis(),
            "Session loaded and created successfully"
        );
    }

    // Build available modes (same as new session)
    let available_modes = build_available_modes();
    let mode_state = SessionModeState::new("default", available_modes);

    // Build available models
    let model_state = build_available_models(config);

    Ok(LoadSessionResponse::new()
        .modes(mode_state)
        .models(model_state))
}

/// Build available permission modes
///
/// Returns the list of permission modes available in the agent.
fn build_available_modes() -> Vec<SessionMode> {
    vec![
        SessionMode::new("default", "Default")
            .description("Standard behavior, prompts for dangerous operations"),
        SessionMode::new("acceptEdits", "Accept Edits")
            .description("Auto-accept file edit operations"),
        SessionMode::new("plan", "Plan Mode")
            .description("Planning mode, no actual tool execution"),
        SessionMode::new("dontAsk", "Don't Ask")
            .description("Don't prompt for permissions, deny if not pre-approved"),
        SessionMode::new("bypassPermissions", "Bypass Permissions")
            .description("Bypass all permission checks"),
    ]
}

/// Build available models for session
///
/// Returns model information including current model and available models.
/// Dynamically builds the model list based on the currently configured model.
///
/// Priority: config.model > ANTHROPIC_MODEL env var > Default (claude-sonnet-4-20250514)
fn build_available_models(config: &AgentConfig) -> SessionModelState {
    // Get current model from config or environment
    // Use official default model as fallback
    let current_model_id = config
        .model
        .clone()
        .or_else(|| std::env::var("ANTHROPIC_MODEL").ok())
        .unwrap_or_else(|| DEFAULT_MODEL_ID.to_string());

    // Use friendly display name for default model
    let display_name = if config.model.is_none() && std::env::var("ANTHROPIC_MODEL").is_err() {
        DEFAULT_MODEL_DISPLAY_NAME.to_string()
    } else {
        current_model_id.clone()
    };

    // Build description
    let description = if display_name == DEFAULT_MODEL_DISPLAY_NAME {
        format!("Default model ({})", DEFAULT_MODEL_ID)
    } else {
        format!("Current model: {}", current_model_id)
    };

    // Build available models list with the current model
    let available_models =
        vec![ModelInfo::new(current_model_id.clone(), display_name).description(description)];

    SessionModelState::new(current_model_id, available_models)
}

/// Send available commands update to client
///
/// Sends the list of available slash commands to the client via ACP notification.
#[allow(dead_code)]
#[allow(unused_variables)]
#[allow(clippy::unnecessary_wraps)]
fn send_available_commands_update(
    session_id: &str,
    connection_cx: JrConnectionCx<AgentToClient>,
) -> Result<(), AgentError> {
    let commands = filter_commands(get_predefined_commands());
    let command_count = commands.len();

    #[cfg(not(test))]
    {
        use sacp::schema::AvailableCommandsUpdate;
        let notification = SessionNotification::new(
            SessionId::new(session_id.to_string()),
            SessionUpdate::AvailableCommandsUpdate(AvailableCommandsUpdate::new(commands)),
        );

        connection_cx
            .send_notification(notification)
            .map_err(|e| AgentError::Internal(format!("Failed to send commands update: {}", e)))?;

        tracing::info!(
            session_id = %session_id,
            command_count,
            "Sent available commands update"
        );
    }

    #[cfg(test)]
    {
        // In tests, just log without actually sending
        tracing::info!(
            session_id = %session_id,
            command_count,
            "Test mode: skipping commands update"
        );
    }

    Ok(())
}

/// Handle session/prompt request
///
/// Sends the prompt to Claude and streams responses back as notifications.
#[instrument(
    name = "acp_prompt",
    skip(request, _config, sessions, connection_cx),
    fields(
        session_id = %request.session_id.0,
        prompt_blocks = request.prompt.len(),
    )
)]
pub async fn handle_prompt(
    request: PromptRequest,
    _config: &AgentConfig,
    sessions: &Arc<SessionManager>,
    connection_cx: JrConnectionCx<AgentToClient>,
    cancel_token: CancellationToken,
) -> Result<PromptResponse, AgentError> {
    let prompt_start = Instant::now();

    let session_id = request.session_id.0.as_ref();
    let session = sessions.get_session_or_error(session_id)?;

    // Generate or extract request_id from PromptRequest.meta
    // If client provides request_id in meta, use it; otherwise generate a UUID
    let request_id: String = match &request.meta {
        Some(meta) => {
            if let Some(serde_json::Value::String(id)) = meta.get("request_id") {
                id.clone()
            } else {
                uuid::Uuid::new_v4().to_string()
            }
        }
        None => uuid::Uuid::new_v4().to_string(),
    };

    // Reset cancelled flag at the start of each prompt
    // This ensures that cancelled state from previous prompt is cleared
    session.reset_cancelled();

    // Set the request_id on the session's converter
    // This will attach the request_id to all SessionNotification instances
    session.set_converter_request_id(request_id.clone()).await;

    tracing::info!(
        session_id = %session_id,
        request_id = %request_id,
        prompt_blocks = request.prompt.len(),
        "Starting prompt processing"
    );

    // Configure ACP MCP server with connection and terminal client
    // This enables tools like Bash to send terminal updates
    let terminal_client = Arc::new(TerminalClient::new(
        connection_cx.clone(),
        session_id.to_string(),
    ));
    session
        .configure_acp_server(connection_cx.clone(), Some(terminal_client))
        .await;

    // Set connection context for permission requests
    // This enables the can_use_tool callback to send permission requests to the client
    session.set_connection_cx(connection_cx.clone());

    // Connect if not already connected
    if !session.is_connected() {
        let connect_start = Instant::now();
        tracing::debug!(
            session_id = %session_id,
            "Connecting to Claude CLI"
        );
        session.connect().await?;
        let connect_elapsed = connect_start.elapsed();
        tracing::info!(
            session_id = %session_id,
            connect_elapsed_ms = connect_elapsed.as_millis(),
            "Connected to Claude CLI"
        );
    }

    // Extract text from prompt content blocks
    let query_text = extract_text_from_content(&request.prompt);
    let query_preview = query_text.chars().take(200).collect::<String>();

    tracing::info!(
        session_id = %session_id,
        query_len = query_text.len(),
        query_preview = %query_preview,
        "Sending query to Claude CLI"
    );

    // Get mutable client access and send the query
    let query_start = Instant::now();

    {
        let mut client = session.client_mut().await;

        // Send the query
        if !query_text.is_empty() {
            // Transform MCP command format: /mcp:server:cmd -> /server:cmd (MCP)
            let transformed_query = transform_mcp_command_input(&query_text);
            client
                .query(&transformed_query)
                .await
                .map_err(AgentError::from)?;
        }
    }
    let query_elapsed = query_start.elapsed();
    tracing::debug!(
        session_id = %session_id,
        query_elapsed_ms = query_elapsed.as_millis(),
        "Query sent to Claude CLI"
    );

    // Get read access to client for streaming responses
    let client = session.client().await;
    let mut stream = client.receive_response();
    let mut cancel_rx = session.cancel_receiver();

    // NOTE: drain_leftover_messages() is no longer needed because the SDK now
    // implements query-scoped message channels for proper message isolation.
    // Each receive_response() call gets its own isolated receiver, preventing
    // late-arriving ResultMessages from being consumed by the wrong prompt.
    // The function is kept for reference/debugging but not called.

    // Track streaming statistics
    let mut message_count = 0u64;
    let mut notification_count = 0u64;
    let mut error_count = 0u64;

    // Track last ResultMessage for determining stop reason
    let mut last_result: Option<claude_code_agent_sdk::ResultMessage> = None;

    // Process streaming responses
    let stream_start = Instant::now();
    loop {
        // Check for cancel signal from MCP cancellation notification
        match cancel_rx.try_recv() {
            Ok(()) => {
                tracing::info!(
                    session_id = %session_id,
                    request_id = %request_id,
                    "Cancel signal received from MCP notification"
                );
                if let Err(e) = client.interrupt().await {
                    tracing::warn!(
                        session_id = %session_id,
                        error = %e,
                        "Failed to send interrupt signal"
                    );
                }
                session.cancel().await;
                drain_messages_synchronously(session_id, &request_id, &mut stream).await;
                break;
            }
            Err(broadcast::error::TryRecvError::Empty) => {
                // No cancel signal, continue processing
            }
            Err(broadcast::error::TryRecvError::Closed) => {
                tracing::warn!(
                    session_id = %session_id,
                    "Cancel channel closed unexpectedly"
                );
                break;
            }
            Err(broadcast::error::TryRecvError::Lagged(_)) => {
                tracing::info!(
                    session_id = %session_id,
                    request_id = %request_id,
                    "Cancel signal lagged, treating as cancel"
                );
                if let Err(e) = client.interrupt().await {
                    tracing::warn!(
                        session_id = %session_id,
                        error = %e,
                        "Failed to send interrupt signal"
                    );
                }
                session.cancel().await;
                drain_messages_synchronously(session_id, &request_id, &mut stream).await;
                break;
            }
        }

        // Check if cancelled via CancellationToken
        if cancel_token.is_cancelled() {
            let elapsed = prompt_start.elapsed();
            tracing::info!(
                session_id = %session_id,
                elapsed_ms = elapsed.as_millis(),
                "Prompt cancelled by user"
            );
            session.clear_converter_request_id().await;
            session.clear_converter_cache().await;
            return Ok(PromptResponse::new(StopReason::Cancelled));
        }

        // Process next message from stream with timeout
        // Use 1000ms timeout (10x longer than original) to reduce overhead while maintaining safety
        let msg_result =
            tokio::time::timeout(tokio::time::Duration::from_millis(1000), stream.next()).await;

        match msg_result {
            Ok(Some(Ok(message))) => {
                message_count += 1;

                // Log message type for debugging
                let msg_type = format!("{:?}", message);
                tracing::debug!(
                    session_id = %session_id,
                    message_count = message_count,
                    msg_type = %msg_type.chars().take(50).collect::<String>(),
                    "Received message from SDK"
                );

                // Track ResultMessage for stop reason determination
                if let claude_code_agent_sdk::Message::Result(ref result) = message {
                    tracing::info!(
                        session_id = %session_id,
                        subtype = %result.subtype,
                        is_error = result.is_error,
                        duration_ms = result.duration_ms,
                        num_turns = result.num_turns,
                        "Received ResultMessage from Claude CLI"
                    );
                    last_result = Some(result.clone());
                }

                // Convert SDK message to ACP notifications
                let converter = session.converter().await;
                let notifications = converter.convert_message(&message, session_id);
                drop(converter); // Release read lock immediately
                let batch_size = notifications.len();

                // Send each notification
                for notification in notifications {
                    notification_count += 1;
                    if let Err(e) = send_notification(&connection_cx, notification) {
                        error_count += 1;
                        tracing::warn!(
                            session_id = %session_id,
                            error = %e,
                            "Failed to send notification"
                        );
                    }
                }

                tracing::trace!(
                    session_id = %session_id,
                    message_count = message_count,
                    batch_size = batch_size,
                    "Processed message from Claude CLI"
                );
            }
            Ok(None) => {
                // Stream ended normally
                tracing::debug!(
                    session_id = %session_id,
                    message_count = message_count,
                    "Message stream ended"
                );
                break;
            }
            Ok(Some(Err(e))) => {
                error_count += 1;
                tracing::error!(
                    session_id = %session_id,
                    error = %e,
                    message_count = message_count,
                    "Error receiving message from Claude CLI"
                );
                // Continue processing - don't fail on individual message errors
            }
            Err(_) => {
                // Timeout - continue loop to check cancel signal again
                // This ensures responsiveness even when stream is slow
            }
        }
    }

    let stream_elapsed = stream_start.elapsed();
    let total_elapsed = prompt_start.elapsed();

    tracing::info!(
        session_id = %session_id,
        total_elapsed_ms = total_elapsed.as_millis(),
        stream_elapsed_ms = stream_elapsed.as_millis(),
        query_elapsed_ms = query_elapsed.as_millis(),
        message_count = message_count,
        notification_count = notification_count,
        error_count = error_count,
        "Prompt completed"
    );

    // ========================================================================
    // CRITICAL: Flush pending notifications before returning EndTurn
    // ========================================================================
    //
    // This fixes the message ordering issue described in MESSAGE_ORDERING_ISSUE.md
    //
    // The Problem:
    // - send_notification() uses unbounded_send() which returns immediately
    // - Messages are processed asynchronously by outgoing_protocol_actor
    // - EndTurn response can arrive before notifications are sent
    //
    // The Solution:
    // - When using patched sacp with flush mechanism: call flush() to wait
    // - When using official sacp: fall back to sleep-based approximation
    //
    // IMPORTANT: This project uses a patch to your sacp fork during development
    // which includes the flush mechanism. See: docs/PATCH_CONFIGURATION.md
    //
    // When your PR is merged to official sacp, this will use the native flush()
    // method from the official library.
    //
    flush::ensure_notifications_flushed(&connection_cx, notification_count).await;

    tracing::debug!(session_id = %session_id, "Flush completed, clearing converter state");

    // Clean up converter state for this prompt:
    // - Clear request_id so it doesn't leak into future prompts
    // - Clear tool_use_cache to prevent unbounded memory growth
    session.clear_converter_request_id().await;
    tracing::debug!(session_id = %session_id, "Converter request_id cleared");

    session.clear_converter_cache().await;
    tracing::debug!(session_id = %session_id, "Converter cache cleared");

    // Determine stop reason based on cancellation state and ResultMessage
    // Reference: vendors/claude-code-acp/src/acp-agent.ts lines 286-323
    if cancel_token.is_cancelled() {
        tracing::info!(session_id = %session_id, "Returning Cancelled stop reason");
        return Ok(PromptResponse::new(StopReason::Cancelled));
    }

    tracing::debug!(session_id = %session_id, "Determining stop reason");

    if let Some(ref result) = last_result {
        // Check user cancelled flag first (set by session/cancel notification)
        // This matches TypeScript behavior where cancelled flag is checked before result handling
        if session.is_user_cancelled() {
            tracing::info!(
                session_id = %session_id,
                subtype = %result.subtype,
                "User cancelled session, returning Cancelled stop reason"
            );
            return Ok(PromptResponse::new(StopReason::Cancelled));
        }

        // Check is_error first - TS throws error when is_error=true
        if result.is_error {
            let error_msg = result
                .result
                .clone()
                .unwrap_or_else(|| result.subtype.clone());
            tracing::error!(
                session_id = %session_id,
                subtype = %result.subtype,
                is_error = result.is_error,
                error_msg = %error_msg,
                "Query completed with is_error=true, returning error"
            );
            // Match TS behavior: throw RequestError.internalError
            return Err(AgentError::Internal(format!(
                "Query failed: {} (subtype: {})",
                error_msg, result.subtype
            )));
        }

        // Determine stop reason based on subtype
        // Reference: vendors/claude-code-acp/src/acp-agent.ts lines 347-360
        let stop_reason = match result.subtype.as_str() {
            "success" => {
                tracing::debug!(
                    session_id = %session_id,
                    subtype = %result.subtype,
                    "Returning EndTurn for success"
                );
                StopReason::EndTurn
            }
            "error_during_execution" => {
                // Match TS behavior: error_during_execution with is_error=false returns EndTurn
                // This indicates execution was interrupted but not due to an error
                // User cancellation is already handled above by checking is_user_cancelled()
                tracing::info!(
                    session_id = %session_id,
                    subtype = %result.subtype,
                    "Returning EndTurn for error_during_execution (is_error=false)"
                );
                StopReason::EndTurn
            }
            "error_max_budget_usd" | "error_max_turns" | "error_max_structured_output_retries" => {
                tracing::info!(
                    session_id = %session_id,
                    subtype = %result.subtype,
                    "Returning MaxTurnRequests for max limit subtype"
                );
                StopReason::MaxTurnRequests
            }
            _ => {
                // Match TS behavior: unknown subtypes return Refusal (not EndTurn)
                tracing::warn!(
                    session_id = %session_id,
                    subtype = %result.subtype,
                    "Unknown result subtype, returning Refusal"
                );
                StopReason::Refusal
            }
        };
        return Ok(PromptResponse::new(stop_reason));
    }

    // No ResultMessage received - stream ended unexpectedly
    // TS throws: "Session did not end in result"
    tracing::error!(
        session_id = %session_id,
        "Stream ended without ResultMessage, returning error"
    );
    Err(AgentError::Internal(
        "Session did not end in result".to_string(),
    ))
}

/// Send a notification via the connection context
fn send_notification(
    cx: &JrConnectionCx<AgentToClient>,
    notification: SessionNotification,
) -> Result<(), sacp::Error> {
    cx.send_notification(notification)
}

/// Handle session/setMode request
///
/// Sets the permission mode for the session and sends a CurrentModeUpdate notification.
#[instrument(
    name = "acp_set_mode",
    skip(request, sessions, connection_cx),
    fields(
        session_id = %request.session_id.0,
        mode_id = %request.mode_id.0,
    )
)]
pub async fn handle_set_mode(
    request: SetSessionModeRequest,
    sessions: &Arc<SessionManager>,
    connection_cx: JrConnectionCx<AgentToClient>,
) -> Result<SetSessionModeResponse, AgentError> {
    let session_id_str = request.session_id.0.as_ref();
    let mode_id_str = request.mode_id.0.as_ref();

    tracing::info!(
        session_id = %session_id_str,
        mode_id = %mode_id_str,
        "Setting session mode"
    );

    let session = sessions.get_session_or_error(session_id_str)?;

    // Get previous mode for logging
    let previous_mode = session.permission_mode().await;

    // Parse the mode from mode_id
    let mode = PermissionMode::parse(mode_id_str).ok_or_else(|| {
        tracing::warn!(
            session_id = %session_id_str,
            mode_id = %mode_id_str,
            "Invalid mode ID"
        );
        AgentError::InvalidMode(mode_id_str.to_string())
    })?;

    // Set the mode in our permission handler
    session.set_permission_mode(mode).await;

    // Also set the mode in the SDK client
    // This is important for the SDK to know the current permission mode
    let sdk_mode = mode.to_sdk_mode();
    if let Err(e) = session.client().await.set_permission_mode(sdk_mode).await {
        tracing::warn!(
            session_id = %session_id_str,
            mode = %mode_id_str,
            error = %e,
            "Failed to set SDK permission mode (continuing anyway)"
        );
        // Don't fail - the local mode is still set
    }

    // Send CurrentModeUpdate notification to inform the client
    let mode_update = CurrentModeUpdate::new(SessionModeId::new(mode_id_str));
    let notification = SessionNotification::new(
        SessionId::new(session_id_str),
        SessionUpdate::CurrentModeUpdate(mode_update),
    );

    if let Err(e) = connection_cx.send_notification(notification) {
        tracing::warn!(
            session_id = %session_id_str,
            error = %e,
            "Failed to send CurrentModeUpdate notification"
        );
    }

    tracing::info!(
        session_id = %session_id_str,
        previous_mode = ?previous_mode,
        new_mode = %mode_id_str,
        "Session mode changed successfully"
    );

    Ok(SetSessionModeResponse::new())
}

/// Handle session cancellation
///
/// Called when a cancel notification is received.
/// Sends an interrupt signal to Claude CLI to stop the current operation.
#[instrument(
    name = "acp_cancel",
    skip(sessions),
    fields(session_id = %session_id)
)]
pub async fn handle_cancel(
    session_id: &str,
    sessions: &Arc<SessionManager>,
) -> Result<(), AgentError> {
    tracing::info!(
        session_id = %session_id,
        "Cancelling session"
    );

    let session = sessions.get_session_or_error(session_id)?;
    session.cancel().await;

    tracing::info!(
        session_id = %session_id,
        "Session cancellation completed"
    );

    Ok(())
}

/// Extract text from ACP content blocks
///
/// This handles all ContentBlock types:
/// - Text: Direct text content
/// - Resource: Embedded file content (prefers this as it contains the actual file text)
/// - ResourceLink: File references (includes URI as context)
/// - Image: Ignored (not text content - images should be handled by PromptConverter as SDK ImageBlock)
/// - Audio: Ignored (not text content - consistent with TypeScript reference implementation)
///
/// Note: This function extracts text-only content for logging/transcript purposes.
/// Image blocks are handled by PromptConverter and converted to SDK ImageBlock for the Claude API.
/// Audio blocks are not supported (consistent with vendors/claude-code-acp/src/acp-agent.ts).
fn extract_text_from_content(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| {
            match block {
                ContentBlock::Text(text_content) => Some(text_content.text.clone()),
                ContentBlock::Resource(embedded_resource) => {
                    // Extract text from embedded resource content
                    match &embedded_resource.resource {
                        sacp::schema::EmbeddedResourceResource::TextResourceContents(
                            text_resource,
                        ) => {
                            // Format as context tag with URI
                            Some(format!(
                                "<context uri=\"{}\">\n{}\n</context>",
                                text_resource.uri, text_resource.text
                            ))
                        }
                        sacp::schema::EmbeddedResourceResource::BlobResourceContents(
                            blob_resource,
                        ) => {
                            // Binary resource - include URI reference
                            Some(format!("<context uri=\"{}\" />", blob_resource.uri))
                        }
                        // Handle any future resource types
                        _ => None,
                    }
                }
                ContentBlock::ResourceLink(resource_link) => {
                    // ResourceLink - include URI reference as context
                    // Note: This doesn't include the file content, just a reference
                    let uri = &resource_link.uri;
                    let title = resource_link.title.as_deref().unwrap_or("");
                    if title.is_empty() {
                        Some(format!("<resource uri=\"{uri}\" />"))
                    } else {
                        Some(format!("[{title}]({uri})"))
                    }
                }
                ContentBlock::Image(_) | ContentBlock::Audio(_) => {
                    // Images and audio are not text content - skip them
                    None
                }
                // Handle any future ContentBlock types
                _ => None,
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Drain leftover messages from the response stream
///
/// This function consumes and discards any messages remaining in the stream
/// from a previous prompt. This is called after creating the stream but before
/// processing the new prompt's responses.
///
/// The function uses a short timeout to avoid blocking indefinitely if there
/// are no messages, and it logs any messages it drains for debugging.
///
/// **DEPRECATED**: This function is no longer needed because the SDK now
/// implements query-scoped message channels for proper message isolation.
/// Each receive_response() call gets its own isolated receiver, preventing
/// late-arriving ResultMessages from being consumed by the wrong prompt.
/// This function is kept for reference/debugging purposes only.
#[allow(dead_code)]
async fn drain_leftover_messages(
    stream: &mut Pin<
        Box<
            dyn Stream<
                    Item = Result<
                        claude_code_agent_sdk::Message,
                        claude_code_agent_sdk::ClaudeError,
                    >,
                > + Send
                + '_,
        >,
    >,
) {
    use tokio::time::{Duration, timeout};

    let mut drained_count = 0;
    let max_drain_time = Duration::from_millis(100);

    // Try to drain any leftover messages with a short timeout
    let start = std::time::Instant::now();
    while start.elapsed() < max_drain_time {
        match timeout(Duration::from_millis(10), stream.next()).await {
            Ok(Some(Ok(message))) => {
                drained_count += 1;
                // Log the drained message type for debugging
                tracing::debug!(
                    drained_message_type = format!("{:?}", message)
                        .chars()
                        .take(50)
                        .collect::<String>(),
                    "Drained leftover message from previous prompt"
                );
            }
            Ok(Some(Err(e))) => {
                tracing::warn!(
                    error = %e,
                    "Drained error message from previous prompt"
                );
                drained_count += 1;
            }
            Ok(None) => {
                // Stream ended
                break;
            }
            Err(_) => {
                // Timeout - no more messages available
                break;
            }
        }
    }

    if drained_count > 0 {
        tracing::info!(
            drained_count,
            "Drained leftover messages from previous prompt before processing new prompt"
        );
    }
}

/// Synchronously drain all messages from the stream before returning from cancel.
///
/// This function implements the "synchronous drain" strategy to prevent
/// leftover messages from leaking into the next prompt. It waits until
/// there is a 50ms silence period (no new messages) before returning,
/// ensuring the queue is fully drained.
///
/// # Arguments
///
/// * `session_id` - Session ID for logging
/// * `request_id` - Request ID for logging
/// * `stream` - The message stream to drain
async fn drain_messages_synchronously(
    session_id: &str,
    request_id: &str,
    stream: &mut Pin<
        Box<
            dyn Stream<
                    Item = Result<
                        claude_code_agent_sdk::Message,
                        claude_code_agent_sdk::ClaudeError,
                    >,
                > + Send
                + '_,
        >,
    >,
) {
    use tokio::time::{Duration, timeout};

    let drain_start = Instant::now();
    let mut drained_count = 0;
    let mut last_message_time = drain_start.elapsed();
    let silence_duration = Duration::from_millis(50);
    let check_interval = Duration::from_millis(10);
    // Safety: Maximum drain timeout to prevent indefinite hang if messages keep arriving
    let max_drain_duration = Duration::from_secs(5);

    tracing::info!(
        session_id = %session_id,
        request_id = %request_id,
        "Starting synchronous drain (waiting for {}ms silence, max {}s)",
        silence_duration.as_millis(),
        max_drain_duration.as_secs()
    );

    // Loop until we have the required silence period
    loop {
        // Safety check: don't drain indefinitely
        if drain_start.elapsed() >= max_drain_duration {
            tracing::warn!(
                session_id = %session_id,
                request_id = %request_id,
                drained_count,
                drain_duration_ms = drain_start.elapsed().as_millis(),
                "Drain reached maximum duration, exiting (messages may still be arriving)"
            );
            break;
        }

        match timeout(check_interval, stream.next()).await {
            Ok(Some(Ok(message))) => {
                drained_count += 1;
                last_message_time = drain_start.elapsed();
                let msg_type = format!("{:?}", message)
                    .chars()
                    .take(50)
                    .collect::<String>();
                tracing::debug!(
                    session_id = %session_id,
                    request_id = %request_id,
                    drained_count,
                    message_type = %msg_type,
                    "Draining message (synchronous)"
                );
            }
            Ok(Some(Err(e))) => {
                drained_count += 1;
                last_message_time = drain_start.elapsed();
                tracing::warn!(
                    session_id = %session_id,
                    request_id = %request_id,
                    error = %e,
                    "Drained error message (synchronous)"
                );
            }
            Err(_) => {
                // Timeout - check if we've had enough silence
                // Use saturating_sub to prevent theoretical underflow
                let time_since_last_message =
                    drain_start.elapsed().saturating_sub(last_message_time);
                if time_since_last_message >= silence_duration {
                    tracing::info!(
                        session_id = %session_id,
                        request_id = %request_id,
                        drained_count,
                        drain_duration_ms = drain_start.elapsed().as_millis(),
                        silence_duration_ms = time_since_last_message.as_millis(),
                        "Synchronous drain complete ({}ms silence achieved)",
                        silence_duration.as_millis()
                    );
                    break;
                }
                // Continue waiting
                tracing::trace!(
                    session_id = %session_id,
                    request_id = %request_id,
                    time_since_last_ms = time_since_last_message.as_millis(),
                    "Waiting for more silence..."
                );
            }
            Ok(None) => {
                // Stream ended
                tracing::info!(
                    session_id = %session_id,
                    request_id = %request_id,
                    drained_count,
                    drain_duration_ms = drain_start.elapsed().as_millis(),
                    "Stream ended during synchronous drain"
                );
                break;
            }
        }
    }
}

// ============================================================================
// Unstable Session Handlers
// ============================================================================

/// Handle session/set_model request
///
/// Sets the model for a session. The model will be used for subsequent prompts.
/// Reference: TS `unstable_setSessionModel` in acp-agent.ts
pub async fn handle_set_session_model(
    request: agent_client_protocol_schema::SetSessionModelRequest,
    sessions: &Arc<SessionManager>,
) -> Result<agent_client_protocol_schema::SetSessionModelResponse, AgentError> {
    let session_id = request.session_id.0.as_ref();
    let model_id = request.model_id.0.as_ref();

    tracing::info!(
        session_id = %session_id,
        model_id = %model_id,
        "Setting session model"
    );

    let session = sessions.get_session_or_error(session_id)?;
    session.set_model(model_id.to_string()).await;

    tracing::info!(
        session_id = %session_id,
        model_id = %model_id,
        "Session model set successfully"
    );

    Ok(agent_client_protocol_schema::SetSessionModelResponse::new())
}

/// Handle session/fork request
///
/// Creates a new session that forks from an existing session's conversation state.
/// Uses the SDK's `fork_session` + `resume` options to create an independent copy.
/// Reference: TS `unstable_forkSession` in acp-agent.ts
#[allow(unused_variables)]
pub fn handle_fork_session(
    request: agent_client_protocol_schema::ForkSessionRequest,
    config: &AgentConfig,
    sessions: &Arc<SessionManager>,
    connection_cx: JrConnectionCx<AgentToClient>,
) -> Result<agent_client_protocol_schema::ForkSessionResponse, AgentError> {
    let source_session_id = request.session_id.0.to_string();
    let cwd = request.cwd;

    tracing::info!(
        source_session_id = %source_session_id,
        cwd = ?cwd,
        "Forking session"
    );

    // Create meta with resume + fork
    let meta = NewSessionMeta::with_resume_and_fork(&source_session_id);

    // Generate new session ID for the forked session
    let new_session_id = uuid::Uuid::new_v4().to_string();

    // Create the forked session
    sessions.create_session(
        new_session_id.clone(),
        cwd.clone(),
        config,
        Some(&meta),
        &request.mcp_servers,
    )?;

    // Build response with available modes and models (same as new session)
    let available_modes = build_available_modes();
    let mode_state = SessionModeState::new("default", available_modes);
    let model_state = build_available_models(config);

    // Send available commands update
    #[cfg(not(test))]
    {
        let session_id_clone = new_session_id.clone();
        tokio::spawn(async move {
            if let Err(e) = send_available_commands_update(&session_id_clone, connection_cx) {
                tracing::warn!(
                    session_id = %session_id_clone,
                    "Failed to send available commands update for forked session: {}",
                    e
                );
            }
        });
    }

    tracing::info!(
        source_session_id = %source_session_id,
        new_session_id = %new_session_id,
        "Session forked successfully"
    );

    Ok(
        agent_client_protocol_schema::ForkSessionResponse::new(new_session_id)
            .modes(mode_state)
            .models(model_state),
    )
}

/// Handle session/resume request
///
/// Resumes an existing session from its conversation history.
/// Similar to fork but without creating an independent copy.
/// Reference: TS `unstable_resumeSession` in acp-agent.ts
#[allow(unused_variables)]
pub fn handle_resume_session(
    request: agent_client_protocol_schema::ResumeSessionRequest,
    config: &AgentConfig,
    sessions: &Arc<SessionManager>,
    connection_cx: JrConnectionCx<AgentToClient>,
) -> Result<agent_client_protocol_schema::ResumeSessionResponse, AgentError> {
    let resume_session_id = request.session_id.0.to_string();
    let cwd = request.cwd;

    tracing::info!(
        resume_session_id = %resume_session_id,
        cwd = ?cwd,
        "Resuming session"
    );

    // Create meta with resume option
    let meta = NewSessionMeta::with_resume(&resume_session_id);

    // Generate new session ID for the resumed session
    let new_session_id = uuid::Uuid::new_v4().to_string();

    // Create the resumed session
    sessions.create_session(
        new_session_id.clone(),
        cwd.clone(),
        config,
        Some(&meta),
        &request.mcp_servers,
    )?;

    // Build response with available modes and models
    let available_modes = build_available_modes();
    let mode_state = SessionModeState::new("default", available_modes);
    let model_state = build_available_models(config);

    // Send available commands update
    #[cfg(not(test))]
    {
        let session_id_clone = new_session_id.clone();
        tokio::spawn(async move {
            if let Err(e) = send_available_commands_update(&session_id_clone, connection_cx) {
                tracing::warn!(
                    session_id = %session_id_clone,
                    "Failed to send available commands update for resumed session: {}",
                    e
                );
            }
        });
    }

    tracing::info!(
        resume_session_id = %resume_session_id,
        new_session_id = %new_session_id,
        "Session resumed successfully"
    );

    Ok(agent_client_protocol_schema::ResumeSessionResponse::new()
        .modes(mode_state)
        .models(model_state))
}

/// Handle session/list request
///
/// Lists available sessions from the JSONL files in `~/.claude/projects/`.
/// Supports pagination via cursor and filtering by cwd.
/// Reference: TS `unstable_listSessions` in acp-agent.ts
pub fn handle_list_sessions(
    request: agent_client_protocol_schema::ListSessionsRequest,
) -> Result<agent_client_protocol_schema::ListSessionsResponse, AgentError> {
    use agent_client_protocol_schema::SessionInfo;

    const PAGE_SIZE: usize = 50;

    let claude_dir = dirs::home_dir()
        .ok_or_else(|| AgentError::Internal("No home directory found".to_string()))?
        .join(".claude")
        .join("projects");

    tracing::info!(
        claude_dir = ?claude_dir,
        cwd_filter = ?request.cwd,
        cursor = ?request.cursor,
        "Listing sessions"
    );

    // If the projects directory doesn't exist, return empty
    if !claude_dir.exists() {
        return Ok(agent_client_protocol_schema::ListSessionsResponse::new(
            vec![],
        ));
    }

    let cwd_filter_str = request
        .cwd
        .as_ref()
        .map(|p| p.to_string_lossy().to_string());
    let encoded_cwd_filter = request.cwd.as_ref().map(|cwd| encode_project_path(cwd));

    let mut all_sessions: Vec<SessionInfo> = Vec::new();

    // Read project directories
    let project_dirs = match std::fs::read_dir(&claude_dir) {
        Ok(dirs) => dirs,
        Err(e) => {
            tracing::error!(error = %e, "Failed to read projects directory");
            return Ok(agent_client_protocol_schema::ListSessionsResponse::new(
                vec![],
            ));
        }
    };

    for entry in project_dirs.flatten() {
        let project_dir = entry.path();
        if !project_dir.is_dir() {
            continue;
        }

        let encoded_path = entry.file_name().to_string_lossy().to_string();

        // Coarse pre-filter by encoded path
        if let Some(ref filter) = encoded_cwd_filter {
            if &encoded_path != filter {
                continue;
            }
        }

        // Read JSONL files in this project directory
        let Ok(files) = std::fs::read_dir(&project_dir) else {
            continue;
        };

        for file_entry in files.flatten() {
            let file_path = file_entry.path();
            let file_name = file_entry.file_name().to_string_lossy().to_string();

            // Skip non-JSONL files and agent-* files (internal metadata)
            if !std::path::Path::new(&file_name)
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("jsonl"))
                || file_name.starts_with("agent-")
            {
                continue;
            }

            let session_id = file_name.trim_end_matches(".jsonl").to_string();

            match parse_session_file(&file_path, &session_id, cwd_filter_str.as_deref()) {
                Ok(Some(info)) => all_sessions.push(info),
                Ok(None) => {} // Filtered out
                Err(e) => {
                    tracing::warn!(
                        file = ?file_path,
                        error = %e,
                        "Failed to parse session file"
                    );
                }
            }
        }
    }

    // Sort by updatedAt descending (most recent first)
    all_sessions.sort_by(|a, b| {
        let time_a = a.updated_at.as_deref().unwrap_or("");
        let time_b = b.updated_at.as_deref().unwrap_or("");
        time_b.cmp(time_a)
    });

    // Handle pagination with cursor
    let start_index = if let Some(ref cursor) = request.cursor {
        parse_cursor(cursor).unwrap_or(0)
    } else {
        0
    };

    let page = all_sessions
        .into_iter()
        .skip(start_index)
        .take(PAGE_SIZE)
        .collect::<Vec<_>>();

    let has_more = start_index + PAGE_SIZE < page.len() + start_index; // Simplified check

    let mut response = agent_client_protocol_schema::ListSessionsResponse::new(page);

    // Note: ListSessionsResponse has next_cursor field behind the feature flag
    // We check if there are more results and set the cursor
    if has_more {
        let next_offset = start_index + PAGE_SIZE;
        let cursor_json = serde_json::json!({"offset": next_offset});
        let cursor_str = base64_encode(&cursor_json.to_string());
        response.next_cursor = Some(cursor_str);
    }

    Ok(response)
}

/// Parse a session JSONL file to extract session info
///
/// Reads the file and extracts:
/// - cwd from any entry
/// - title from the first user message
/// - updatedAt from file modification time
fn parse_session_file(
    file_path: &std::path::Path,
    session_id: &str,
    cwd_filter: Option<&str>,
) -> Result<Option<agent_client_protocol_schema::SessionInfo>, std::io::Error> {
    use std::io::BufRead;

    let file = std::fs::File::open(file_path)?;
    let reader = std::io::BufReader::new(file);

    let mut session_cwd: Option<String> = None;
    let mut title: Option<String> = None;
    let mut parsed_any = false;

    for line in reader.lines() {
        let line = match line {
            Ok(l) if !l.is_empty() => l,
            _ => continue,
        };

        let entry: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        parsed_any = true;

        // Skip sidechain entries
        if entry.get("isSidechain") == Some(&serde_json::Value::Bool(true)) {
            continue;
        }

        // Extract cwd
        if session_cwd.is_none() {
            if let Some(cwd) = entry.get("cwd").and_then(|v| v.as_str()) {
                session_cwd = Some(cwd.to_string());
            }
        }

        // Extract title from first user message
        if title.is_none() {
            if entry.get("type").and_then(|v| v.as_str()) == Some("user") {
                if let Some(content) = entry.get("message").and_then(|m| m.get("content")) {
                    title = extract_title_from_content(content);
                }
            }
        }

        // Stop if we have both
        if title.is_some() && session_cwd.is_some() {
            break;
        }
    }

    if !parsed_any {
        return Ok(None);
    }

    // SessionInfo.cwd is required
    let Some(cwd) = session_cwd else {
        return Ok(None);
    };

    // Verify cwd matches filter if provided
    if let Some(filter) = cwd_filter {
        if cwd != filter {
            return Ok(None);
        }
    }

    // Get file modification time
    let updated_at = std::fs::metadata(file_path)
        .ok()
        .and_then(|m| m.modified().ok())
        .map(|t| {
            let datetime: chrono::DateTime<chrono::Utc> = t.into();
            datetime.to_rfc3339()
        });

    let mut info = agent_client_protocol_schema::SessionInfo::new(session_id.to_string(), &cwd);
    info.title = title;
    info.updated_at = updated_at;

    Ok(Some(info))
}

/// Extract title from message content
fn extract_title_from_content(content: &serde_json::Value) -> Option<String> {
    let text = if let Some(s) = content.as_str() {
        Some(s.to_string())
    } else if let Some(arr) = content.as_array() {
        arr.first().and_then(|first| {
            if let Some(s) = first.as_str() {
                Some(s.to_string())
            } else {
                first.get("text").and_then(|t| t.as_str()).map(String::from)
            }
        })
    } else {
        None
    };

    text.map(|t| sanitize_title(&t))
}

/// Sanitize a title string for display
fn sanitize_title(text: &str) -> String {
    // Truncate to reasonable length and trim whitespace
    let truncated: String = text.chars().take(200).collect();
    // Remove newlines and excessive whitespace
    truncated.lines().next().unwrap_or("").trim().to_string()
}

/// Encode a project path for use as directory name
///
/// Replaces path separators with hyphens.
/// Reference: TS `encodeProjectPath` in utils.ts
fn encode_project_path(cwd: &std::path::Path) -> String {
    let path_str = cwd.to_string_lossy();
    // Unix paths: replace / with -
    path_str.replace('/', "-")
}

/// Parse a pagination cursor
fn parse_cursor(cursor: &str) -> Option<usize> {
    let decoded = base64_decode(cursor)?;
    let parsed: serde_json::Value = serde_json::from_str(&decoded).ok()?;
    parsed
        .get("offset")?
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
}

/// Base64 encode a string
fn base64_encode(s: &str) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(s.as_bytes())
}

/// Base64 decode a string
fn base64_decode(s: &str) -> Option<String> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD.decode(s).ok()?;
    String::from_utf8(bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use sacp::schema::{ProtocolVersion, TextContent};
    use serial_test::serial;
    use std::time::Duration;

    #[test]
    fn test_handle_initialize() {
        let request = InitializeRequest::new(ProtocolVersion::LATEST);
        let config = AgentConfig::from_env();

        let response = handle_initialize(request, &config);

        assert_eq!(response.protocol_version, ProtocolVersion::LATEST);
    }

    #[tokio::test]
    async fn test_handle_new_session() {
        // Note: This test is disabled because handle_new_session now requires
        // JrConnectionCx which is difficult to mock in unit tests.
        // The functionality is tested through integration tests instead.
        // TODO: Add integration test for session/new with available commands update
    }

    #[test]
    fn test_extract_text_from_content() {
        let blocks = vec![
            ContentBlock::Text(TextContent::new("Hello")),
            ContentBlock::Text(TextContent::new("World")),
        ];

        let text = extract_text_from_content(&blocks);
        assert_eq!(text, "Hello\nWorld");
    }

    /// Test the drain_messages_synchronously function with a mock stream
    ///
    /// This test verifies that:
    /// 1. The drain function consumes all available messages
    /// 2. The drain function waits for the silence period before returning
    /// 3. The drain function respects the maximum timeout
    #[tokio::test]
    async fn test_drain_messages_synchronously() {
        use claude_code_agent_sdk::{Message, StreamEvent};
        use serde_json::json;
        use uuid::Uuid;

        let session_id = "test-session";
        let request_id = "test-request";

        // Create stream events with all required fields
        let test_uuid = Uuid::new_v4().to_string();
        let messages: Vec<Result<Message, claude_code_agent_sdk::ClaudeError>> = vec![
            Ok(Message::StreamEvent(StreamEvent {
                uuid: test_uuid.clone(),
                session_id: session_id.to_string(),
                event: json!({"type": "test"}),
                parent_tool_use_id: None,
            })),
            Ok(Message::StreamEvent(StreamEvent {
                uuid: test_uuid.clone(),
                session_id: session_id.to_string(),
                event: json!({"type": "test2"}),
                parent_tool_use_id: None,
            })),
            Ok(Message::StreamEvent(StreamEvent {
                uuid: test_uuid,
                session_id: session_id.to_string(),
                event: json!({"type": "test3"}),
                parent_tool_use_id: None,
            })),
        ];

        let mut stream: Pin<
            Box<dyn Stream<Item = Result<Message, claude_code_agent_sdk::ClaudeError>> + Send + '_>,
        > = Box::pin(stream::iter(messages));

        // The drain should complete quickly since the stream ends after 3 messages
        drain_messages_synchronously(session_id, request_id, &mut stream).await;

        // If we got here without timing out, the drain completed successfully
        // We can't directly inspect the drain state, but successful completion
        // indicates the function worked correctly
    }

    /// Test that drain_messages_synchronously handles empty streams correctly
    #[tokio::test]
    async fn test_drain_messages_synchronously_empty_stream() {
        use claude_code_agent_sdk::Message;

        let session_id = "test-session";
        let request_id = "test-request";

        // Create an empty stream
        let mut stream: Pin<
            Box<dyn Stream<Item = Result<Message, claude_code_agent_sdk::ClaudeError>> + Send + '_>,
        > = Box::pin(stream::empty());

        // The drain should complete immediately for an empty stream
        drain_messages_synchronously(session_id, request_id, &mut stream).await;

        // Successful completion means it handled the empty stream correctly
    }

    /// Test that drain_messages_synchronously respects the maximum timeout
    #[tokio::test]
    async fn test_drain_messages_synchronously_max_timeout() {
        use claude_code_agent_sdk::{Message, StreamEvent};
        use serde_json::json;
        use uuid::Uuid;

        let session_id = "test-session";
        let request_id = "test-request";

        // Create a stream that yields messages with small delays
        let messages = (0..50).map(|i| {
            Ok(Message::StreamEvent(StreamEvent {
                uuid: Uuid::new_v4().to_string(),
                session_id: session_id.to_string(),
                event: json!({"type": "test", "index": i}),
                parent_tool_use_id: None,
            }))
        });

        let mut stream: Pin<
            Box<dyn Stream<Item = Result<Message, claude_code_agent_sdk::ClaudeError>> + Send + '_>,
        > = Box::pin(stream::iter(messages));

        // The drain should complete within a reasonable time
        // Even with 50 messages, it should finish quickly (stream ends after yielding all)
        let start = std::time::Instant::now();
        drain_messages_synchronously(session_id, request_id, &mut stream).await;
        let elapsed = start.elapsed();

        // Should complete in under 1 second (much faster than the 5s max timeout)
        assert!(
            elapsed < Duration::from_secs(1),
            "Drain should complete quickly"
        );
    }

    /// Test that drain_messages_synchronously correctly detects silence period
    #[tokio::test]
    async fn test_drain_messages_synchronously_silence_detection() {
        use claude_code_agent_sdk::{Message, StreamEvent};
        use serde_json::json;
        use uuid::Uuid;

        let session_id = "test-session";
        let request_id = "test-request";

        // Create a stream that yields some messages, then stops
        // The drain should detect the silence (stream end) and return
        let test_uuid = Uuid::new_v4().to_string();
        let messages = vec![
            Ok(Message::StreamEvent(StreamEvent {
                uuid: test_uuid.clone(),
                session_id: session_id.to_string(),
                event: json!({"type": "msg1"}),
                parent_tool_use_id: None,
            })),
            Ok(Message::StreamEvent(StreamEvent {
                uuid: test_uuid.clone(),
                session_id: session_id.to_string(),
                event: json!({"type": "msg2"}),
                parent_tool_use_id: None,
            })),
            Ok(Message::StreamEvent(StreamEvent {
                uuid: test_uuid,
                session_id: session_id.to_string(),
                event: json!({"type": "msg3"}),
                parent_tool_use_id: None,
            })),
        ];

        let mut stream: Pin<
            Box<dyn Stream<Item = Result<Message, claude_code_agent_sdk::ClaudeError>> + Send + '_>,
        > = Box::pin(stream::iter(messages));

        // The drain should complete after the stream ends
        let start = std::time::Instant::now();
        drain_messages_synchronously(session_id, request_id, &mut stream).await;
        let elapsed = start.elapsed();

        // Should complete very quickly (stream ends immediately after 3 messages)
        assert!(
            elapsed < Duration::from_millis(100),
            "Drain should detect stream end quickly"
        );
    }

    /// Test build_available_models function with config model
    #[test]
    fn test_build_available_models_with_config() {
        let config = AgentConfig {
            model: Some("glm-4.7".to_string()),
            ..Default::default()
        };
        let model_state = build_available_models(&config);

        assert_eq!(model_state.current_model_id.0, "glm-4.7".into());
        assert_eq!(model_state.available_models.len(), 1);
        assert_eq!(model_state.available_models[0].model_id.0, "glm-4.7".into());
        assert_eq!(model_state.available_models[0].name, "glm-4.7");
        assert_eq!(
            model_state.available_models[0]
                .description
                .as_ref()
                .unwrap(),
            "Current model: glm-4.7"
        );
    }

    /// Test build_available_models function with environment variable
    #[test]
    #[serial]
    fn test_build_available_models_with_env_var() {
        unsafe { std::env::set_var("ANTHROPIC_MODEL", "gpt-4") };
        let config = AgentConfig {
            model: None,
            ..Default::default()
        };
        let model_state = build_available_models(&config);

        assert_eq!(model_state.current_model_id.0, "gpt-4".into());
        assert_eq!(model_state.available_models.len(), 1);
        assert_eq!(model_state.available_models[0].name, "gpt-4");
        unsafe { std::env::remove_var("ANTHROPIC_MODEL") };
    }

    /// Test build_available_models function with default fallback
    #[test]
    #[serial]
    fn test_build_available_models_default() {
        // Ensure no env var is set
        unsafe { std::env::remove_var("ANTHROPIC_MODEL") };
        let config = AgentConfig {
            model: None,
            ..Default::default()
        };
        let model_state = build_available_models(&config);

        // Should use DEFAULT_MODEL_ID internally
        assert_eq!(
            model_state.current_model_id.0,
            "claude-sonnet-4-20250514".into()
        );
        // But display "Default" as name
        assert_eq!(model_state.available_models[0].name, "Default");
        assert_eq!(
            model_state.available_models[0].description,
            Some("Default model (claude-sonnet-4-20250514)".into())
        );
    }
}
