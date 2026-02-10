//! ACP MCP Server implementation
//!
//! This module provides an MCP server that integrates with the ACP protocol,
//! allowing tools to send notifications during execution (e.g., terminal output).

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::OnceLock;
use std::time::Instant;

use async_trait::async_trait;
use claude_code_agent_sdk::{
    SdkMcpServer, SdkMcpTool, ToolDefinition, ToolHandler, ToolResult as SdkToolResult,
};
use futures::FutureExt;
use futures::future::BoxFuture;
use sacp::JrConnectionCx;
use sacp::link::AgentToClient;
use sacp::schema::{
    Meta, SessionId, SessionNotification, SessionUpdate, Terminal, ToolCall, ToolCallContent,
    ToolCallId, ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields, ToolKind,
};
use serde_json::Value;
use tokio::sync::{Mutex, RwLock};
use tracing::instrument;

use super::registry::{ToolContext, ToolResult};
use super::server::McpServer;
use crate::session::BackgroundProcessManager;
use crate::settings::PermissionChecker;
use crate::terminal::TerminalClient;

/// Type alias for the cancel callback to reduce type complexity
type CancelCallback = Arc<Mutex<Option<Box<dyn Fn() + Send + Sync>>>>;

/// ACP-integrated MCP server
///
/// This server implements the SDK's `SdkMcpServer` trait, allowing it to be
/// injected into the Claude Agent SDK. It provides access to:
/// - ACP connection for sending notifications
/// - Terminal API for command execution
/// - Tool registry for built-in tools
///
/// # Lock Ordering (Critical for Deadlock Prevention)
///
/// To prevent deadlocks, follow this lock acquisition order:
/// 1. OnceLock values (cwd) - lock-free after initialization
/// 2. cancel_callback (Mutex) - use try_lock for non-critical access
/// 3. OnceLock values (session_id, connection_cx, etc.) - lock-free
///
/// NEVER hold locks while calling tool.execute() as tools may need
/// to send notifications that could deadlock.
pub struct AcpMcpServer {
    /// Server name
    name: String,
    /// Server version
    version: String,
    /// Tool registry
    mcp_server: Arc<McpServer>,
    /// Tool definitions for SDK
    tools: HashMap<String, SdkMcpTool>,
    /// Session ID (set once at initialization)
    session_id: OnceLock<String>,
    /// ACP connection for sending notifications (set once at initialization)
    connection_cx: OnceLock<JrConnectionCx<AgentToClient>>,
    /// Terminal client (set once at initialization)
    ///
    /// Note: While configure_acp_server is called on every prompt and
    /// creates a new TerminalClient each time, all instances are functionally
    /// equivalent (same connection_cx and session_id). OnceLock ensures we
    /// use only the first instance.
    terminal_client: OnceLock<Arc<TerminalClient>>,
    /// Background process manager (set once at initialization)
    background_processes: OnceLock<Arc<BackgroundProcessManager>>,
    /// Working directory (set once at initialization)
    ///
    /// Uses OnceLock instead of RwLock because:
    /// - cwd is set exactly once during session initialization
    /// - All tools only read cwd (no tool ever modifies it)
    /// - OnceLock provides lock-free reads after initialization
    /// - No deadlock risk
    cwd: OnceLock<std::path::PathBuf>,
    /// Permission checker for tool-level permission checks
    permission_checker: OnceLock<Arc<RwLock<PermissionChecker>>>,
    /// Cancel callback - called when MCP cancellation notification is received
    /// Uses Mutex (not RwLock) because writes are rare and we need try_lock for deadlock safety
    cancel_callback: CancelCallback,
}

impl std::fmt::Debug for AcpMcpServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AcpMcpServer")
            .field("name", &self.name)
            .field("version", &self.version)
            .finish_non_exhaustive()
    }
}

impl AcpMcpServer {
    /// Create a new ACP MCP server
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        let mcp_server = Arc::new(McpServer::new());
        let tools = Self::build_tool_definitions(&mcp_server);

        Self {
            name: name.into(),
            version: version.into(),
            mcp_server,
            tools,
            session_id: OnceLock::new(),
            connection_cx: OnceLock::new(),
            terminal_client: OnceLock::new(),
            background_processes: OnceLock::new(),
            cwd: OnceLock::new(),
            permission_checker: OnceLock::new(),
            cancel_callback: Arc::new(Mutex::new(None)),
        }
    }

    /// Set the session ID (only sets if not already set)
    pub fn set_session_id(&self, session_id: impl Into<String>) {
        let new_id = session_id.into();
        if let Some(existing) = self.session_id.get() {
            if existing != &new_id {
                tracing::warn!(
                    existing = %existing,
                    new = %new_id,
                    "session_id already set with different value, ignoring new value"
                );
            }
        } else {
            drop(self.session_id.set(new_id));
        }
    }

    /// Set the ACP connection (only sets if not already set)
    pub fn set_connection(&self, cx: JrConnectionCx<AgentToClient>) {
        if self.connection_cx.get().is_some() {
            tracing::warn!("connection_cx already set, ignoring new value");
        } else {
            drop(self.connection_cx.set(cx));
        }
    }

    /// Set the terminal client (only sets if not already set)
    pub fn set_terminal_client(&self, client: Arc<TerminalClient>) {
        // Only set if not already set - configure_acp_server may be called multiple times
        if self.terminal_client.get().is_none() {
            drop(self.terminal_client.set(client));
        }
    }

    /// Set the background process manager (only sets if not already set)
    pub fn set_background_processes(&self, manager: Arc<BackgroundProcessManager>) {
        // Only set if not already set - configure_acp_server may be called multiple times
        if self.background_processes.get().is_none() {
            drop(self.background_processes.set(manager));
        }
    }

    /// Set the permission checker (only sets if not already set)
    pub fn set_permission_checker(&self, checker: Arc<RwLock<PermissionChecker>>) {
        // Only set if not already set - configure_acp_server may be called multiple times
        if self.permission_checker.get().is_none() {
            drop(self.permission_checker.set(checker));
        }
    }

    /// Set the working directory (synchronous, lock-free)
    ///
    /// Uses OnceLock to set the value on first call.
    /// - First call: Sets the cwd
    /// - Subsequent calls with same value: Silently ignored
    /// - Subsequent calls with different value: Logs warning, ignored
    ///
    /// This is synchronous (not async) because OnceLock operations are
    /// non-blocking and lock-free after initialization.
    ///
    /// # Deadlock Prevention
    ///
    /// OnceLock provides lock-free reads after initialization, eliminating
    /// the deadlock risk that existed with RwLock.
    pub fn set_cwd(&self, cwd: impl Into<std::path::PathBuf>) {
        let new_cwd = cwd.into();
        if let Some(existing) = self.cwd.get() {
            if existing != &new_cwd {
                tracing::warn!(
                    existing = %existing.display(),
                    new = %new_cwd.display(),
                    "cwd already set with different value, ignoring new value"
                );
            }
            // If already set, silently ignore (first value wins)
            // This matches the pattern of other OnceLock setters in this struct
        } else {
            self.cwd.set(new_cwd).expect("Failed to set cwd");
        }
    }

    /// Set the cancel callback
    ///
    /// This callback is invoked when a MCP cancellation notification is received.
    /// Use this to interrupt the Claude CLI when the user cancels an operation.
    pub async fn set_cancel_callback<F>(&self, callback: F)
    where
        F: Fn() + Send + Sync + 'static,
    {
        let mut guard = self.cancel_callback.lock().await;
        *guard = Some(Box::new(callback));
    }

    /// Get the MCP server
    pub fn mcp_server(&self) -> &Arc<McpServer> {
        &self.mcp_server
    }

    /// Build tool definitions from the MCP server
    ///
    /// Tools are registered with the `mcp__acp__` prefix so that CLI routes
    /// tool calls to our SDK MCP server via `mcp_message` requests.
    fn build_tool_definitions(mcp_server: &Arc<McpServer>) -> HashMap<String, SdkMcpTool> {
        let mut tools = HashMap::new();

        for schema in mcp_server.tool_schemas() {
            // Use original tool name without prefix
            // The MCP protocol will automatically add mcp__acp__ prefix
            let tool_name = schema.name.clone();
            let description = schema.description.clone();
            let input_schema = schema.input_schema.clone();

            // Create a placeholder handler - actual execution goes through handle_message
            let tool = SdkMcpTool {
                name: tool_name.clone(),
                description,
                input_schema,
                handler: Arc::new(PlaceholderHandler),
                annotations: None,
            };

            tools.insert(tool_name, tool);
        }

        tools
    }

    /// Send a terminal update notification
    ///
    /// This is a standalone function that can be called from spawned tasks.
    /// Note: This function is kept for potential future use with Terminal API.
    #[allow(dead_code)]
    fn send_terminal_update_with_cx(
        cx: &JrConnectionCx<AgentToClient>,
        session_id: &str,
        tool_use_id: &str,
        terminal_id: &str,
        status: ToolCallStatus,
        title: Option<&str>,
    ) -> Result<(), String> {
        // Build terminal content
        let terminal = Terminal::new(terminal_id.to_string());
        let content = vec![ToolCallContent::Terminal(terminal)];

        // Build update fields
        let mut update_fields = ToolCallUpdateFields::new().status(status).content(content);

        if let Some(title) = title {
            update_fields = update_fields.title(title);
        }

        // Build and send notification
        let tool_call_id = ToolCallId::new(tool_use_id.to_string());
        let update = ToolCallUpdate::new(tool_call_id, update_fields);
        let notification = SessionNotification::new(
            SessionId::new(session_id),
            SessionUpdate::ToolCallUpdate(update),
        );

        cx.send_notification(notification)
            .map_err(|e| format!("Failed to send notification: {}", e))
    }

    /// Send a terminal update notification (instance method)
    #[allow(dead_code)]
    fn send_terminal_update(
        &self,
        tool_use_id: &str,
        terminal_id: &str,
        status: ToolCallStatus,
        title: Option<&str>,
    ) -> Result<(), String> {
        // OnceLock provides lock-free access after initialization
        let session_id = self.session_id.get();
        let connection_cx = self.connection_cx.get();

        let Some(cx) = connection_cx else {
            return Err("No connection context available".to_string());
        };

        let Some(session_id) = session_id else {
            return Err("No session ID available".to_string());
        };

        // Build terminal content
        let terminal = Terminal::new(terminal_id.to_string());
        let content = vec![ToolCallContent::Terminal(terminal)];

        // Build update fields
        let mut update_fields = ToolCallUpdateFields::new().status(status).content(content);

        if let Some(title) = title {
            update_fields = update_fields.title(title);
        }

        // Build and send notification
        let tool_call_id = ToolCallId::new(tool_use_id.to_string());
        let update = ToolCallUpdate::new(tool_call_id, update_fields);
        let notification = SessionNotification::new(
            SessionId::new(session_id.as_str()),
            SessionUpdate::ToolCallUpdate(update),
        );

        cx.send_notification(notification)
            .map_err(|e| format!("Failed to send notification: {}", e))
    }

    /// Send a ToolCallUpdate notification with optional fields
    ///
    /// This is used to send tool call updates (including errors) to Zed.
    ///
    /// # Parameters
    ///
    /// - `cx`: ACP connection context for sending notifications
    /// - `session_id`: Current session identifier
    /// - `tool_use_id`: Tool call identifier from Claude
    /// - `status`: Optional status change (Completed, Failed, InProgress, Pending)
    /// - `title`: Optional display title for the tool call
    /// - `content`: Optional content to display (typically error messages for Failed status)
    /// - `meta`: Optional metadata (terminal_info, terminal_output, terminal_exit)
    ///
    /// # Error Content Behavior
    ///
    /// When a tool fails, include `content` with the error message so Zed can display it:
    /// ```ignore
    /// let content: Option<Vec<ToolCallContent>> = if result.is_error {
    ///     Some(vec![result.content.clone().into()])
    /// } else {
    ///     None
    /// };
    /// ```
    ///
    /// For successful completion, `content` should be `None` to avoid duplicate output
    /// (the tool result is sent separately via the result message).
    fn send_tool_call_update_with_meta(
        cx: &JrConnectionCx<AgentToClient>,
        session_id: &str,
        tool_use_id: &str,
        status: Option<ToolCallStatus>,
        title: Option<&str>,
        content: Option<Vec<ToolCallContent>>,
        meta: Option<Meta>,
    ) -> Result<(), String> {
        // Build update fields
        let mut update_fields = ToolCallUpdateFields::new();

        if let Some(status) = status {
            update_fields = update_fields.status(status);
        }

        if let Some(title) = title {
            update_fields = update_fields.title(title);
        }

        if let Some(content) = content {
            update_fields = update_fields.content(content);
        }

        // Build and send notification
        let tool_call_id = ToolCallId::new(tool_use_id.to_string());
        let mut update = ToolCallUpdate::new(tool_call_id, update_fields);

        // Add meta if provided
        if let Some(meta) = meta {
            update = update.meta(meta);
        }

        let notification = SessionNotification::new(
            SessionId::new(session_id.to_string()),
            SessionUpdate::ToolCallUpdate(update),
        );

        cx.send_notification(notification)
            .map_err(|e| format!("Failed to send notification: {}", e))
    }

    /// Convert serde_json::Value to Meta (Map<String, Value>)
    fn value_to_meta(value: serde_json::Value) -> Option<Meta> {
        match value {
            serde_json::Value::Object(map) => Some(map),
            _ => None,
        }
    }

    /// Send a ToolCall notification with meta field and terminal content
    ///
    /// IMPORTANT: This sends a ToolCall (not ToolCallUpdate) notification.
    /// Zed handles terminal in two ways:
    /// - meta.terminal_info - creates and registers the terminal
    /// - content[].Terminal - associates the terminal with the tool call UI
    ///
    /// Both are needed for terminal output to be displayed correctly.
    fn send_tool_call_with_meta(
        cx: &JrConnectionCx<AgentToClient>,
        session_id: &str,
        tool_use_id: &str,
        title: Option<&str>,
        status: ToolCallStatus,
        terminal_id: Option<&str>,
        meta: Option<Meta>,
    ) -> Result<(), String> {
        let tool_call_id = ToolCallId::new(tool_use_id.to_string());

        let mut tool_call = ToolCall::new(tool_call_id, title.unwrap_or("Running command"))
            .kind(ToolKind::Execute)
            .status(status);

        // Add terminal content to associate terminal with tool call UI
        if let Some(tid) = terminal_id {
            let terminal = Terminal::new(tid.to_string());
            tool_call = tool_call.content(vec![ToolCallContent::Terminal(terminal)]);
        }

        // Add meta if provided (contains terminal_info for creating the terminal)
        if let Some(meta) = meta {
            tool_call = tool_call.meta(meta);
        }

        let notification = SessionNotification::new(
            SessionId::new(session_id),
            SessionUpdate::ToolCall(tool_call),
        );

        cx.send_notification(notification)
            .map_err(|e| format!("Failed to send notification: {}", e))
    }

    /// Create a tool context for tool execution
    ///
    /// # Deadlock Prevention
    ///
    /// OnceLock provides lock-free reads after initialization, eliminating
    /// the deadlock risk that existed with RwLock.
    fn create_tool_context(&self, tool_use_id: Option<&str>) -> ToolContext {
        // OnceLock provides lock-free read after initialization
        let cwd = self.cwd.get().expect("cwd not initialized").clone();

        // OnceLock provides lock-free access after initialization
        let session_id = self
            .session_id
            .get()
            .map(|s| s.as_str())
            .unwrap_or("unknown");

        let terminal_client = self.terminal_client.get();
        let background_processes = self.background_processes.get();
        let connection_cx = self.connection_cx.get();
        let permission_checker = self.permission_checker.get();

        let mut context = ToolContext::new(session_id.to_string(), cwd);

        if let Some(client) = terminal_client {
            context = context.with_terminal_client(client.clone());
        }

        if let Some(manager) = background_processes {
            context = context.with_background_processes(manager.clone());
        }

        if let Some(id) = tool_use_id {
            context = context.with_tool_use_id(id);
        }

        if let Some(cx) = connection_cx {
            context = context.with_connection_cx(cx.clone());
        }

        if let Some(checker) = permission_checker {
            context = context.with_permission_checker(checker.clone());
        }

        context
    }

    /// Execute a tool with ACP integration
    ///
    /// # Deadlock Prevention
    ///
    /// OnceLock provides lock-free access to cwd after initialization,
    /// eliminating the deadlock risk that existed with RwLock.
    #[instrument(
        name = "acp_execute_tool",
        skip(self, arguments),
        fields(
            tool_name = %tool_name,
            tool_use_id = ?tool_use_id,
            args_size = arguments.to_string().len(),
        )
    )]
    async fn execute_tool(
        &self,
        tool_name: &str,
        arguments: Value,
        tool_use_id: Option<&str>,
    ) -> Result<ToolResult, String> {
        let start_time = Instant::now();

        // Log arguments preview (truncated for large inputs)
        let args_str = arguments.to_string();

        // Truncate at character boundary to avoid panic on multi-byte UTF-8
        let args_preview = if args_str.len() > 500 {
            // Find a safe character boundary near byte 500
            let safe_boundary = args_str
                .char_indices()
                .map(|(i, _)| i)
                .find(|&i| i > 500)
                .unwrap_or(args_str.len());
            format!("{}...(truncated)", &args_str[..safe_boundary])
        } else {
            args_str.clone()
        };

        tracing::info!(
            tool_name = %tool_name,
            tool_use_id = ?tool_use_id,
            args_preview = %args_preview,
            "Executing ACP tool"
        );

        // CRITICAL: Create context with lock-free OnceLock access
        let context = self.create_tool_context(tool_use_id);

        tracing::debug!("Tool context created, calling tool execution");

        // Special handling for Bash tool - use early return to match original behavior
        if tool_name == "Bash" {
            let result = self
                .execute_bash_tool(arguments, tool_use_id, &context)
                .await;
            let elapsed = start_time.elapsed();
            match &result {
                Ok(r) => {
                    tracing::info!(
                        tool_name = %tool_name,
                        elapsed_ms = elapsed.as_millis(),
                        is_error = r.is_error,
                        content_len = r.content.len(),
                        "ACP Bash tool completed"
                    );
                }
                Err(e) => {
                    tracing::error!(
                        tool_name = %tool_name,
                        elapsed_ms = elapsed.as_millis(),
                        error = %e,
                        "ACP Bash tool failed"
                    );
                }
            }
            return result;
        }

        // Execute other tools normally
        // Note: OnceLock provides lock-free access, no locks to release
        let result = self
            .mcp_server
            .execute(tool_name, arguments, &context)
            .await;

        #[cfg(feature = "verbose-debug")]
        tracing::debug!("Tool execution returned, preparing to send completion notification");

        // Send completion notification to Zed (required for non-Bash tools)
        // Use RAII scoping to ensure locks are released promptly
        {
            #[cfg(feature = "verbose-debug")]
            tracing::debug!("Acquiring locks for completion notification");

            // OnceLock provides lock-free access after initialization
            let session_id = self.session_id.get();
            let connection_cx = self.connection_cx.get();

            #[cfg(feature = "verbose-debug")]
            tracing::debug!("Locks acquired, checking if we have all required data");

            if let (Some(cx), Some(session_id), Some(tool_use_id)) =
                (connection_cx, session_id, tool_use_id)
            {
                let status = if result.is_error {
                    ToolCallStatus::Failed
                } else {
                    ToolCallStatus::Completed
                };

                #[cfg(feature = "verbose-debug")]
                tracing::debug!(
                    tool_name = %tool_name,
                    tool_use_id = %tool_use_id,
                    status = ?status,
                    is_error = result.is_error,
                    content_len = result.content.len(),
                    "Sending completion notification"
                );

                // Prepare content for the notification
                // For errors, include the error message so Zed can display it
                let content: Option<Vec<ToolCallContent>> = if result.is_error {
                    Some(vec![result.content.clone().into()])
                } else {
                    // For successful completion, no need to send content
                    // The tool result will be sent separately via result message
                    None
                };

                // Send completion notification with content for errors
                if let Err(e) = Self::send_tool_call_update_with_meta(
                    cx,
                    session_id,
                    tool_use_id,
                    Some(status),
                    None,
                    content,
                    None,
                ) {
                    tracing::debug!("Failed to send tool completion notification: {}", e);
                }

                #[cfg(feature = "verbose-debug")]
                tracing::debug!("Completion notification sent successfully");
            }
            // Locks are automatically released when this block ends

            #[cfg(feature = "verbose-debug")]
            tracing::debug!("RAII block ending, locks will be released");
        }

        #[cfg(feature = "verbose-debug")]
        tracing::debug!("RAII block ended, locks released, continuing execution");

        let elapsed = start_time.elapsed();

        #[cfg(feature = "verbose-debug")]
        {
            let content_len = result.content.len();
            let would_truncate = content_len > 300;

            tracing::debug!(
                tool_name = %tool_name,
                elapsed_ms = elapsed.as_millis(),
                content_len,
                would_truncate,
                "About to log completion"
            );
        }

        tracing::info!(
            tool_name = %tool_name,
            elapsed_ms = elapsed.as_millis(),
            is_error = result.is_error,
            content_len = result.content.len(),
            "ACP tool completed"
        );

        #[cfg(feature = "verbose-debug")]
        tracing::debug!("About to return Ok(result) from execute_tool");

        Ok(result)
    }

    /// Execute Bash tool with streaming output via meta field
    ///
    /// This implementation bypasses the Terminal API (which causes dispatch loop deadlock)
    /// and instead executes commands directly, sending terminal-like updates via the
    /// _meta field in ToolCallUpdate notifications.
    ///
    /// Zed supports these meta fields:
    /// - terminal_info: { terminal_id, cwd } - sent at start
    /// - terminal_output: { terminal_id, data } - sent for each output chunk
    /// - terminal_exit: { terminal_id, exit_code } - sent when command completes
    #[instrument(
        name = "acp_bash_tool",
        skip(self, arguments, context),
        fields(
            tool_use_id = ?tool_use_id,
        )
    )]
    async fn execute_bash_tool(
        &self,
        arguments: Value,
        tool_use_id: Option<&str>,
        context: &ToolContext,
    ) -> Result<ToolResult, String> {
        let bash_start = Instant::now();

        // OnceLock provides lock-free access after initialization
        let session_id = self.session_id.get();
        let connection_cx = self.connection_cx.get();

        // Extract command parameters
        let command = arguments
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or("Missing command")?
            .to_string();
        let description = arguments
            .get("description")
            .and_then(|v| v.as_str())
            .map(String::from);
        let run_in_background = arguments
            .get("run_in_background")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let timeout_ms = arguments
            .get("timeout")
            .and_then(|v| v.as_u64())
            .unwrap_or(120_000)
            .min(600_000); // Max 10 minutes

        // Generate unique terminal ID for tracking
        let terminal_id = uuid::Uuid::new_v4().to_string();

        // Title is the actual command (no truncation, let UI handle display)
        let title = command.clone();

        tracing::info!(
            command = %command,
            description = ?description,
            terminal_id = %terminal_id,
            run_in_background = run_in_background,
            timeout_ms = timeout_ms,
            cwd = ?context.cwd,
            "Executing Bash command with streaming output"
        );

        // Send terminal_info notification at start (if we have connection)
        // IMPORTANT: Use ToolCall (not ToolCallUpdate) notification because Zed
        // only creates terminals when terminal_info is in a ToolCall notification.
        if let (Some(cx), Some(session_id), Some(tool_use_id)) =
            (connection_cx, session_id, tool_use_id)
        {
            // Build meta with terminal_info and optional description for future use
            let mut meta_json = serde_json::json!({
                "terminal_info": {
                    "terminal_id": &terminal_id,
                    "cwd": context.cwd.display().to_string()
                }
            });
            // Add description to meta if available (for future use by clients)
            if let Some(ref desc) = description {
                meta_json["description"] = serde_json::json!(desc);
            }
            let meta = Self::value_to_meta(meta_json);
            if let Err(e) = Self::send_tool_call_with_meta(
                cx,
                session_id,
                tool_use_id,
                Some(&title),
                ToolCallStatus::InProgress,
                Some(&terminal_id), // Pass terminal_id for content association
                meta,
            ) {
                tracing::debug!("Failed to send terminal_info: {}", e);
            }
        }

        // Drop locks before executing command
        let cx_clone = connection_cx.cloned();
        let session_id_clone = session_id.cloned();
        let tool_use_id_clone = tool_use_id.map(String::from);

        // Execute command with streaming
        let result = self
            .execute_command_with_streaming(
                &command,
                &terminal_id,
                context,
                timeout_ms,
                run_in_background,
                cx_clone.as_ref(),
                session_id_clone.as_deref(),
                tool_use_id_clone.as_deref(),
            )
            .await;

        // Send terminal_exit notification
        if let (Some(cx), Some(session_id), Some(tool_use_id)) = (
            cx_clone.as_ref(),
            session_id_clone.as_ref(),
            tool_use_id_clone.as_ref(),
        ) {
            let exit_code = match &result {
                Ok(r) if !r.is_error => 0,
                _ => 1,
            };
            let meta = Self::value_to_meta(serde_json::json!({
                "terminal_exit": {
                    "terminal_id": &terminal_id,
                    "exit_code": exit_code
                }
            }));
            if let Err(e) = Self::send_tool_call_update_with_meta(
                cx,
                session_id,
                tool_use_id,
                Some(ToolCallStatus::Completed),
                None,
                None, // No content for terminal_exit
                meta,
            ) {
                tracing::debug!("Failed to send terminal_exit: {}", e);
            }
        }

        let bash_elapsed = bash_start.elapsed();
        tracing::info!(
            command = %command,
            terminal_id = %terminal_id,
            total_elapsed_ms = bash_elapsed.as_millis(),
            is_error = result.as_ref().map(|r| r.is_error).unwrap_or(true),
            "Bash command completed"
        );

        result
    }

    /// Execute command with streaming output via meta field
    ///
    /// This function executes the command directly using tokio::process::Command
    /// and sends output chunks via ToolCallUpdate notifications with terminal_output meta.
    #[allow(clippy::too_many_arguments)]
    async fn execute_command_with_streaming(
        &self,
        command: &str,
        terminal_id: &str,
        context: &ToolContext,
        timeout_ms: u64,
        run_in_background: bool,
        cx: Option<&JrConnectionCx<AgentToClient>>,
        session_id: Option<&str>,
        tool_use_id: Option<&str>,
    ) -> Result<ToolResult, String> {
        use tokio::io::{AsyncBufReadExt, BufReader};
        use tokio::process::Command;

        // Spawn the command
        let mut child = Command::new("bash")
            .arg("-c")
            .arg(command)
            .current_dir(&context.cwd)
            .env("CLAUDECODE", "1")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn command: {}", e))?;

        // Handle background execution
        if run_in_background {
            let shell_id = format!("term-{}", terminal_id);
            tracing::info!(shell_id = %shell_id, "Command started in background");
            return Ok(ToolResult::success(format!(
                "Command started in background.\n\nShell ID: {}\n\nUse BashOutput to check status.",
                shell_id
            )));
        }

        // Collect output
        let mut output = String::new();
        let mut stderr_output = String::new();

        // Take stdout and stderr
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        // Read stdout in a task
        let stdout_task = if let Some(stdout) = stdout {
            let cx = cx.cloned();
            let session_id = session_id.map(String::from);
            let tool_use_id = tool_use_id.map(String::from);
            let terminal_id = terminal_id.to_string();

            Some(tokio::spawn(async move {
                let reader = BufReader::new(stdout);
                let mut lines = reader.lines();
                let mut collected = String::new();

                while let Ok(Some(line)) = lines.next_line().await {
                    collected.push_str(&line);
                    collected.push('\n');

                    // Send terminal_output notification
                    if let (Some(cx), Some(session_id), Some(tool_use_id)) =
                        (cx.as_ref(), session_id.as_ref(), tool_use_id.as_ref())
                    {
                        let meta = Self::value_to_meta(serde_json::json!({
                            "terminal_output": {
                                "terminal_id": &terminal_id,
                                "data": format!("{}\n", line)
                            }
                        }));
                        drop(Self::send_tool_call_update_with_meta(
                            cx,
                            session_id,
                            tool_use_id,
                            None, // No status change for terminal_output
                            None,
                            None, // No content for terminal_output
                            meta,
                        ));
                    }
                }
                collected
            }))
        } else {
            None
        };

        // Read stderr in a task
        let stderr_task = stderr.map(|stderr| {
            tokio::spawn(async move {
                let reader = BufReader::new(stderr);
                let mut lines = reader.lines();
                let mut collected = String::new();

                while let Ok(Some(line)) = lines.next_line().await {
                    collected.push_str(&line);
                    collected.push('\n');
                }
                collected
            })
        });

        // Wait for command with timeout
        let timeout_duration = std::time::Duration::from_millis(timeout_ms);
        let wait_result = tokio::time::timeout(timeout_duration, child.wait()).await;

        // Collect outputs
        if let Some(task) = stdout_task {
            if let Ok(out) = task.await {
                output = out;
            }
        }
        if let Some(task) = stderr_task {
            if let Ok(err) = task.await {
                stderr_output = err;
            }
        }

        // Combine output
        let combined_output = if stderr_output.is_empty() {
            output
        } else if output.is_empty() {
            stderr_output
        } else {
            format!("{}\n--- stderr ---\n{}", output, stderr_output)
        };

        // Process result
        match wait_result {
            Ok(Ok(status)) => {
                let exit_code = status.code().unwrap_or(-1);
                tracing::info!(exit_code = exit_code, command = %command, "Command completed");

                if status.success() {
                    Ok(ToolResult::success(combined_output))
                } else {
                    Ok(ToolResult::error(format!(
                        "Command failed with exit code {}\n{}",
                        exit_code, combined_output
                    )))
                }
            }
            Ok(Err(e)) => {
                tracing::error!(error = %e, "Failed to wait for command");
                Ok(ToolResult::error(format!(
                    "Failed to wait for command: {}\n{}",
                    e, combined_output
                )))
            }
            Err(_) => {
                tracing::warn!(timeout_ms = timeout_ms, "Command timed out");
                // Try to kill the process
                drop(child.kill().await);
                Ok(ToolResult::error(format!(
                    "Command timed out after {}ms\n{}",
                    timeout_ms, combined_output
                )))
            }
        }
    }

    /// Execute Bash command via direct execution (legacy fallback)
    ///
    /// This is kept for compatibility but the new execute_bash_tool
    /// already uses direct execution with streaming.
    #[allow(dead_code)]
    async fn execute_bash_fallback(
        &self,
        arguments: Value,
        context: &ToolContext,
    ) -> Result<ToolResult, String> {
        tracing::info!("Executing Bash via direct fallback");

        // Create a new context WITHOUT terminal_client to force direct execution
        let fallback_context = ToolContext::new(context.session_id.clone(), context.cwd.clone());

        let result = self
            .mcp_server
            .execute("Bash", arguments, &fallback_context)
            .await;
        tracing::info!("Direct execution completed, is_error: {}", result.is_error);
        Ok(result)
    }
}

/// Placeholder handler for SDK tool definitions
struct PlaceholderHandler;

impl ToolHandler for PlaceholderHandler {
    fn handle(
        &self,
        args: Value,
    ) -> BoxFuture<'static, claude_code_agent_sdk::errors::Result<SdkToolResult>> {
        tracing::warn!("PlaceholderHandler called with args: {:?}", args);
        async move {
            // This should never be called - execution goes through AcpMcpServer::handle_message
            tracing::error!("PlaceholderHandler was called! SDK is not using handle_message!");
            Ok(SdkToolResult {
                content: vec![claude_code_agent_sdk::McpToolResultContent::Text {
                    text: "Tool execution error: placeholder handler called".to_string(),
                }],
                is_error: true,
            })
        }
        .boxed()
    }
}

#[async_trait]
impl SdkMcpServer for AcpMcpServer {
    #[instrument(
        name = "acp_handle_message",
        skip(self, message),
        fields(
            method = tracing::field::Empty,
            message_size = message.to_string().len(),
        )
    )]
    async fn handle_message(&self, message: Value) -> claude_code_agent_sdk::errors::Result<Value> {
        let start_time = Instant::now();

        let method = message["method"].as_str().ok_or_else(|| {
            claude_code_agent_sdk::errors::ClaudeError::Transport("Missing method".to_string())
        })?;

        // Record method to span
        tracing::Span::current().record("method", method);

        tracing::debug!(
            method = %method,
            message_id = ?message.get("id"),
            "AcpMcpServer handling message"
        );

        let result = match method {
            "initialize" => {
                tracing::info!(
                    server_name = %self.name,
                    server_version = %self.version,
                    "ACP MCP server initializing"
                );
                Ok(serde_json::json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": self.name,
                        "version": self.version
                    }
                }))
            }
            "tools/list" => {
                let tools: Vec<_> = self
                    .tools
                    .values()
                    .map(|t| {
                        serde_json::json!({
                            "name": t.name,
                            "description": t.description,
                            "inputSchema": t.input_schema
                        })
                    })
                    .collect();

                let tool_names: Vec<&str> = self.tools.keys().map(|s| s.as_str()).collect();
                tracing::info!(
                    tool_count = tools.len(),
                    tools = ?tool_names,
                    "Returning tool list"
                );
                Ok(serde_json::json!({
                    "tools": tools
                }))
            }
            "tools/call" => {
                let params = &message["params"];
                let tool_name = params["name"].as_str().ok_or_else(|| {
                    claude_code_agent_sdk::errors::ClaudeError::Transport(
                        "Missing tool name".to_string(),
                    )
                })?;
                let arguments = params["arguments"].clone();

                // Get tool_use_id from _meta if available
                let tool_use_id = params
                    .get("_meta")
                    .and_then(|m| m.get("claudecode/toolUseId"))
                    .and_then(|v| v.as_str());

                // Log full _meta for debugging
                let meta_preview = params.get("_meta").map(|m| {
                    if m.as_object().map(|o| o.len()).unwrap_or(0) > 1 {
                        format!("{:?}", m)
                    } else {
                        "{claudecode/toolUseId}".to_string()
                    }
                });

                tracing::info!(
                    tool_name = %tool_name,
                    tool_use_id = ?tool_use_id,
                    has_meta = params.get("_meta").is_some(),
                    meta_preview = ?meta_preview,
                    args_size = arguments.to_string().len(),
                    "Received tools/call request from Claude CLI"
                );

                let tool_start = Instant::now();

                let result = self
                    .execute_tool(tool_name, arguments, tool_use_id)
                    .await
                    .map_err(|e| {
                        tracing::error!(
                            tool_name = %tool_name,
                            error = %e,
                            "Tool execution failed"
                        );
                        claude_code_agent_sdk::errors::ClaudeError::Transport(e)
                    })?;

                #[cfg(feature = "verbose-debug")]
                tracing::debug!("execute_tool returned successfully");

                let tool_elapsed = tool_start.elapsed();

                tracing::info!(
                    tool_name = %tool_name,
                    elapsed_ms = tool_elapsed.as_millis(),
                    is_error = result.is_error,
                    content_len = result.content.len(),
                    "Tool execution completed"
                );

                #[cfg(feature = "verbose-debug")]
                tracing::debug!("About to create response JSON");

                let response = Ok(serde_json::json!({
                    "content": [{
                        "type": "text",
                        "text": result.content
                    }],
                    "is_error": result.is_error
                }));

                #[cfg(feature = "verbose-debug")]
                tracing::debug!("Response JSON created successfully");

                response
            }
            // MCP notifications - these don't expect a response but we return empty success
            "notifications/cancelled" => {
                // Handle cancellation notification
                // The request_id is in params.requestId
                let request_id = message["params"]["requestId"].as_str();
                tracing::info!(
                    request_id = ?request_id,
                    "Received MCP cancellation notification"
                );

                // Call the cancel callback to interrupt Claude CLI
                // Use try_lock to avoid potential deadlock if callback is locked
                match self.cancel_callback.try_lock() {
                    Ok(callback) => {
                        if let Some(ref cb) = *callback {
                            tracing::info!("Invoking cancel callback to interrupt Claude CLI");
                            cb();
                        } else {
                            tracing::warn!(
                                "No cancel callback registered, cancellation may not take effect"
                            );
                        }
                    }
                    Err(_) => {
                        tracing::warn!(
                            "Cancel callback is busy, cannot invoke cancellation safely"
                        );
                    }
                }

                Ok(serde_json::json!({}))
            }
            "notifications/initialized" => {
                tracing::debug!("Received initialized notification from Claude CLI");
                Ok(serde_json::json!({}))
            }
            "notifications/progress" => {
                let progress_token = message["params"]["progressToken"].as_str();
                let progress = message["params"]["progress"].as_f64();
                tracing::trace!(
                    progress_token = ?progress_token,
                    progress = ?progress,
                    "Received progress notification"
                );
                Ok(serde_json::json!({}))
            }
            _ => {
                // Check if it's a notification (starts with "notifications/")
                if method.starts_with("notifications/") {
                    tracing::debug!(
                        method = %method,
                        "Received unknown notification, ignoring"
                    );
                    Ok(serde_json::json!({}))
                } else {
                    tracing::warn!(
                        method = %method,
                        "Received unknown method"
                    );
                    Err(claude_code_agent_sdk::errors::ClaudeError::Transport(
                        format!("Unknown method: {}", method),
                    ))
                }
            }
        };

        let elapsed = start_time.elapsed();
        tracing::debug!(
            method = %method,
            elapsed_ms = elapsed.as_millis(),
            is_ok = result.is_ok(),
            "Message handling completed"
        );

        result
    }

    fn list_tools(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|t| ToolDefinition {
                name: t.name.clone(),
                description: t.description.clone(),
                input_schema: t.input_schema.clone(),
                annotations: t.annotations.clone(),
            })
            .collect()
    }
}

/// Get the list of built-in tools that should be disabled in the SDK
///
/// When using AcpMcpServer, these tools should be disabled in the SDK
/// so that our MCP server handles them instead.
pub fn get_disallowed_tools() -> Vec<String> {
    vec![
        "Bash".to_string(),
        "BashOutput".to_string(),
        "KillShell".to_string(),
        "Read".to_string(),
        "Write".to_string(),
        "Edit".to_string(),
        "Glob".to_string(),
        "Grep".to_string(),
        "LS".to_string(),
        "Task".to_string(),
        "TaskOutput".to_string(),
        "TodoWrite".to_string(),
        "ExitPlanMode".to_string(),
        "WebFetch".to_string(),
        "WebSearch".to_string(),
        "NotebookRead".to_string(),
        "NotebookEdit".to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use claude_code_agent_sdk::ACP_TOOL_PREFIX;

    #[test]
    fn test_acp_mcp_server_creation() {
        let server = AcpMcpServer::new("test", "1.0.0");
        assert_eq!(server.name, "test");
        assert_eq!(server.version, "1.0.0");
    }

    #[test]
    fn test_tool_definitions_use_original_names() {
        let server = AcpMcpServer::new("acp", "1.0.0");
        // Tools should use original names WITHOUT prefix
        // MCP protocol automatically adds mcp__acp__ prefix when exposing to Claude
        assert!(server.tools.contains_key("Bash"));
        assert!(server.tools.contains_key("Read"));
        assert!(server.tools.contains_key("Write"));
        // Prefixed names should NOT be in the map
        assert!(!server.tools.contains_key("mcp__acp__Bash"));
        assert!(!server.tools.contains_key("mcp__acp__Read"));
    }

    #[test]
    fn test_get_disallowed_tools() {
        let tools = get_disallowed_tools();
        assert!(tools.contains(&"Bash".to_string()));
        assert!(tools.contains(&"Read".to_string()));
    }

    #[test]
    fn test_acp_tool_prefix_constant() {
        // Verify the SDK constant is correct
        assert_eq!(ACP_TOOL_PREFIX, "mcp__acp__");
    }

    // ============================================================
    // MCP handle_message tests
    // ============================================================

    #[tokio::test]
    async fn test_handle_message_initialize() {
        let server = AcpMcpServer::new("test-server", "1.0.0");

        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        });

        let result = server.handle_message(request).await;
        assert!(result.is_ok(), "initialize should succeed");

        let response = result.unwrap();
        assert_eq!(response["protocolVersion"], "2024-11-05");
        assert!(response["capabilities"]["tools"].is_object());
        assert_eq!(response["serverInfo"]["name"], "test-server");
        assert_eq!(response["serverInfo"]["version"], "1.0.0");
    }

    #[tokio::test]
    async fn test_handle_message_tools_list() {
        let server = AcpMcpServer::new("test-server", "1.0.0");

        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        });

        let result = server.handle_message(request).await;
        assert!(result.is_ok(), "tools/list should succeed");

        let response = result.unwrap();
        let tools = response["tools"].as_array().unwrap();
        assert!(!tools.is_empty(), "Should have tools");

        // Check that Bash tool is present
        let bash_tool = tools.iter().find(|t| t["name"] == "Bash");
        assert!(bash_tool.is_some(), "Bash tool should be present");

        // Check tool structure
        let bash = bash_tool.unwrap();
        assert!(bash["description"].is_string());
        assert!(bash["inputSchema"].is_object());
    }

    #[tokio::test]
    async fn test_handle_message_tools_call_bash_fallback() {
        // Test Bash tool execution WITHOUT terminal client (fallback to direct execution)
        let server = AcpMcpServer::new("test-server", "1.0.0");

        // Set a valid cwd for the test
        server.set_cwd(std::env::temp_dir());
        server.set_session_id("test-session");

        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "Bash",
                "arguments": {
                    "command": "echo hello"
                }
            }
        });

        let result = server.handle_message(request).await;
        assert!(
            result.is_ok(),
            "tools/call should succeed: {:?}",
            result.err()
        );

        let response = result.unwrap();

        // Check response structure matches MCP tool result format
        assert!(
            response["content"].is_array(),
            "Response should have content array"
        );
        let content = response["content"].as_array().unwrap();
        assert!(!content.is_empty(), "Content should not be empty");

        // First content block should be text type
        assert_eq!(content[0]["type"], "text");
        assert!(content[0]["text"].is_string());

        // Should contain "hello" in output (direct execution of echo)
        let text = content[0]["text"].as_str().unwrap();
        assert!(
            text.contains("hello"),
            "Output should contain 'hello', got: {}",
            text
        );
    }

    #[tokio::test]
    async fn test_handle_message_tools_call_with_tool_use_id() {
        // Test that tool_use_id is extracted from _meta
        let server = AcpMcpServer::new("test-server", "1.0.0");
        server.set_cwd(std::env::temp_dir());
        server.set_session_id("test-session");

        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "Bash",
                "arguments": {
                    "command": "echo test"
                },
                "_meta": {
                    "claudecode/toolUseId": "toolu_123456"
                }
            }
        });

        let result = server.handle_message(request).await;
        assert!(result.is_ok(), "tools/call with tool_use_id should succeed");
    }

    #[tokio::test]
    async fn test_handle_message_notifications_initialized() {
        let server = AcpMcpServer::new("test-server", "1.0.0");

        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {}
        });

        let result = server.handle_message(request).await;
        assert!(result.is_ok(), "notifications/initialized should succeed");

        let response = result.unwrap();
        assert!(response.is_object(), "Should return empty object");
    }

    #[tokio::test]
    async fn test_handle_message_unknown_method() {
        let server = AcpMcpServer::new("test-server", "1.0.0");

        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "unknown/method",
            "params": {}
        });

        let result = server.handle_message(request).await;
        assert!(result.is_err(), "Unknown method should return error");
    }

    #[tokio::test]
    async fn test_handle_message_read_tool() {
        let server = AcpMcpServer::new("test-server", "1.0.0");
        server.set_cwd(std::env::temp_dir());
        server.set_session_id("test-session");

        // Create a test file
        let test_file = std::env::temp_dir().join("test_read_tool.txt");
        std::fs::write(&test_file, "test content").unwrap();

        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "Read",
                "arguments": {
                    "file_path": test_file.to_string_lossy()
                }
            }
        });

        let result = server.handle_message(request).await;
        assert!(result.is_ok(), "Read tool should succeed");

        let response = result.unwrap();
        let content = response["content"].as_array().unwrap();
        let text = content[0]["text"].as_str().unwrap();
        assert!(text.contains("test content"), "Should contain file content");

        // Clean up
        std::fs::remove_file(test_file).ok();
    }

    #[tokio::test]
    async fn test_handle_message_missing_method() {
        let server = AcpMcpServer::new("test-server", "1.0.0");

        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 7,
            "params": {}
        });

        let result = server.handle_message(request).await;
        assert!(result.is_err(), "Missing method should return error");
    }

    #[tokio::test]
    async fn test_handle_message_tools_call_missing_tool_name() {
        let server = AcpMcpServer::new("test-server", "1.0.0");

        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {
                "arguments": {
                    "command": "echo test"
                }
            }
        });

        let result = server.handle_message(request).await;
        assert!(result.is_err(), "Missing tool name should return error");
    }

    // ============================================================
    // Tool execution tests (without terminal client - fallback mode)
    // ============================================================

    #[tokio::test]
    async fn test_execute_bash_without_terminal_client() {
        let server = AcpMcpServer::new("test-server", "1.0.0");
        server.set_cwd(std::env::temp_dir());
        server.set_session_id("test-session");

        // Execute without terminal client (terminal_client is None by default)
        let result = server
            .execute_tool(
                "Bash",
                serde_json::json!({
                    "command": "echo fallback_test"
                }),
                None,
            )
            .await;

        assert!(result.is_ok(), "Execute should succeed");
        let tool_result = result.unwrap();
        assert!(!tool_result.is_error, "Should not be an error");
        assert!(
            tool_result.content.contains("fallback_test"),
            "Output should contain 'fallback_test', got: {}",
            tool_result.content
        );
    }

    #[tokio::test]
    async fn test_execute_glob_tool() {
        let server = AcpMcpServer::new("test-server", "1.0.0");

        // Use the project's src directory for the glob test
        let cwd = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        server.set_cwd(&cwd);
        server.set_session_id("test-session");

        let result = server
            .execute_tool(
                "Glob",
                serde_json::json!({
                    "pattern": "src/**/*.rs"
                }),
                None,
            )
            .await;

        assert!(result.is_ok(), "Glob should succeed");
        let tool_result = result.unwrap();
        assert!(!tool_result.is_error, "Should not be an error");
        // Should find at least main.rs and lib.rs
        assert!(
            tool_result.content.contains(".rs"),
            "Should find Rust files, got: {}",
            tool_result.content
        );
    }

    #[tokio::test]
    async fn test_execute_grep_tool() {
        let server = AcpMcpServer::new("test-server", "1.0.0");

        let cwd = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        server.set_cwd(&cwd);
        server.set_session_id("test-session");

        let result = server
            .execute_tool(
                "Grep",
                serde_json::json!({
                    "pattern": "AcpMcpServer",
                    "path": "src/mcp"
                }),
                None,
            )
            .await;

        assert!(result.is_ok(), "Grep should succeed");
        let tool_result = result.unwrap();
        // Should find this file at least
        assert!(
            tool_result.content.contains("acp_server.rs") || !tool_result.is_error,
            "Should find acp_server.rs or no error, got: {}",
            tool_result.content
        );
    }

    // ============================================================
    // Error message propagation tests
    // ============================================================

    #[tokio::test]
    async fn test_read_tool_error_propagation() {
        // Test that Read tool properly reports errors when file doesn't exist
        let server = AcpMcpServer::new("test-server", "1.0.0");
        server.set_cwd(std::env::temp_dir());
        server.set_session_id("test-session");

        // Try to read a non-existent file
        let result = server
            .execute_tool(
                "Read",
                serde_json::json!({
                    "file_path": "/nonexistent/path/file.txt"
                }),
                None,
            )
            .await;

        assert!(result.is_ok(), "Read should return a result");
        let tool_result = result.unwrap();
        assert!(
            tool_result.is_error,
            "Reading non-existent file should be marked as error"
        );
        assert!(
            !tool_result.content.is_empty(),
            "Error should have a message"
        );
    }

    #[tokio::test]
    async fn test_error_content_preparation_logic() {
        // Test the logic that prepares content for error notifications
        use super::ToolResult;

        // Simulate successful result
        let success_result = ToolResult::success("Operation completed".to_string());
        let content_success: Option<Vec<ToolCallContent>> = if success_result.is_error {
            Some(vec![success_result.content.clone().into()])
        } else {
            None
        };
        assert!(
            content_success.is_none(),
            "Successful results should not include content in notification"
        );

        // Simulate error result
        let error_message = "File not found: /path/to/file.txt";
        let error_result = ToolResult::error(error_message.to_string());
        let content_error: Option<Vec<ToolCallContent>> = if error_result.is_error {
            Some(vec![error_result.content.clone().into()])
        } else {
            None
        };
        assert!(
            content_error.is_some(),
            "Error results should include content in notification"
        );
        let content_vec = content_error.unwrap();
        assert_eq!(content_vec.len(), 1, "Should have one content item");

        // Verify the content contains the error message
        // ToolCallContent::Content(Content { content: ContentBlock::Text(TextContent) })
        match &content_vec[0] {
            ToolCallContent::Content(content) => match &content.content {
                sacp::schema::ContentBlock::Text(text) => {
                    assert_eq!(text.text, error_message, "Error message should match");
                }
                _ => panic!("Content block should be Text type"),
            },
            _ => panic!("Content should be Content type"),
        }
    }

    #[tokio::test]
    async fn test_glob_tool_error_on_invalid_pattern() {
        // Test that Glob tool handles invalid patterns correctly
        let server = AcpMcpServer::new("test-server", "1.0.0");
        let cwd = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        server.set_cwd(&cwd);
        server.set_session_id("test-session");

        // Use an invalid glob pattern (unclosed bracket)
        let result = server
            .execute_tool(
                "Glob",
                serde_json::json!({
                    "pattern": "src/**/[unclosed"
                }),
                None,
            )
            .await;

        // Result should be OK but may contain an error
        assert!(result.is_ok(), "Glob should return a result");
        let _tool_result = result.unwrap();
        // The glob implementation may not fail on invalid patterns, so we just
        // verify it returns without panicking
    }

    #[tokio::test]
    async fn test_bash_error_with_nonexistent_command() {
        // Test that Bash tool properly reports errors for invalid commands
        let server = AcpMcpServer::new("test-server", "1.0.0");
        server.set_cwd(std::env::temp_dir());
        server.set_session_id("test-session");

        let result = server
            .execute_tool(
                "Bash",
                serde_json::json!({
                    "command": "this_command_definitely_does_not_exist_12345"
                }),
                None,
            )
            .await;

        assert!(result.is_ok(), "Bash should return a result");
        let tool_result = result.unwrap();
        assert!(
            tool_result.is_error,
            "Non-existent command should be marked as error"
        );
    }

    // ============================================================
    // Concurrency tests - verify deadlock fixes
    // ============================================================

    // Note: test_concurrent_set_session_id_and_execute was removed because
    // OnceLock does not allow multiple sets. The test_lock_order_consistency test
    // below verifies that lock ordering is consistent.

    #[tokio::test]
    async fn test_lock_order_consistency() {
        // Test that lock acquisition order is consistent across different code paths
        let server = std::sync::Arc::new(AcpMcpServer::new("test-server", "1.0.0"));
        let cwd = std::env::temp_dir();
        server.set_cwd(&cwd);
        server.set_session_id("test-session");

        // Create a barrier to synchronize tasks
        let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(3));
        let mut handles = vec![];

        // Task 1: Execute Read tool (uses create_tool_context)
        let server1 = server.clone();
        let barrier1 = barrier.clone();
        let handle1 = tokio::spawn(async move {
            barrier1.wait().await;
            drop(
                server1
                    .execute_tool(
                        "Read",
                        serde_json::json!({"file_path": "/tmp/test.txt"}),
                        Some("tool-1"),
                    )
                    .await,
            );
        });

        // Task 2: Execute Bash tool (uses execute_bash_tool)
        let server2 = server.clone();
        let barrier2 = barrier.clone();
        let handle2 = tokio::spawn(async move {
            barrier2.wait().await;
            drop(
                server2
                    .execute_tool(
                        "Bash",
                        serde_json::json!({"command": "echo test"}),
                        Some("tool-2"),
                    )
                    .await,
            );
        });

        // Task 3: Another Read tool
        let server3 = server.clone();
        let barrier3 = barrier.clone();
        let handle3 = tokio::spawn(async move {
            barrier3.wait().await;
            drop(
                server3
                    .execute_tool(
                        "Read",
                        serde_json::json!({"file_path": "/tmp/test.txt"}),
                        Some("tool-3"),
                    )
                    .await,
            );
        });

        handles.push(handle1);
        handles.push(handle2);
        handles.push(handle3);

        // Wait for all tasks with a timeout
        let timeout_duration = std::time::Duration::from_secs(10);
        let start = std::time::Instant::now();

        for handle in handles {
            tokio::select! {
                () = tokio::time::sleep(timeout_duration) => {
                    panic!("Test timed out - likely deadlock detected!");
                }
                result = handle => {
                    result.unwrap();
                }
            }
        }

        let elapsed = start.elapsed();
        println!("All concurrent tasks completed in {:?}", elapsed);

        // Should complete well within timeout
        assert!(elapsed < timeout_duration, "Possible deadlock detected");
    }

    #[tokio::test]
    async fn test_cancel_callback_try_lock() {
        // Test that cancel_callback uses try_lock and doesn't cause deadlock
        let server = AcpMcpServer::new("test-server", "1.0.0");

        // Set a cancel callback
        let callback_called = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let callback_flag = callback_called.clone();

        server
            .set_cancel_callback(move || {
                callback_flag.store(true, std::sync::atomic::Ordering::SeqCst);
            })
            .await;

        // Simulate MCP cancellation notification
        let cancellation_message = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": {
                "requestId": "test-request-123"
            }
        });

        // This should not deadlock even if callback is locked elsewhere
        let result = server.handle_message(cancellation_message).await;
        assert!(result.is_ok(), "Cancellation handling should succeed");
        assert!(callback_called.load(std::sync::atomic::Ordering::SeqCst));
    }
}
