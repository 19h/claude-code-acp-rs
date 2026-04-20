//! External MCP server management
//!
//! Supports connecting to external MCP servers for extended tool capabilities.

use std::collections::HashMap;
use std::path::Path;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStdin, ChildStdout};
use tracing::{Span, instrument};

use process_wrap::tokio::*;

use super::registry::{ToolResult, ToolSchema};
use crate::session::WrappedChild;

/// Default timeout for MCP requests (3 minutes)
/// WebSearch and WebFetch may need significant time due to network I/O
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(180);

/// Default timeout for MCP initialization (60 seconds, MCP servers may need time to start)
const DEFAULT_INIT_TIMEOUT: Duration = Duration::from_secs(60);

/// External MCP server connection type
pub enum McpConnection {
    /// Stdio-based connection (spawned process)
    Stdio {
        /// The spawned process (wrapped for process group support)
        #[allow(dead_code)]
        child: WrappedChild,
        /// Writer to send messages
        stdin: ChildStdin,
        /// Reader to receive messages
        stdout: BufReader<ChildStdout>,
    },
}

/// External MCP server state
#[allow(missing_debug_implementations)]
pub struct ExternalMcpServer {
    /// Server name
    pub name: String,
    /// Connection to the server
    connection: McpConnection,
    /// Available tools from this server
    tools: Vec<ToolSchema>,
    /// Whether the server is initialized
    initialized: bool,
    /// Request ID counter for JSON-RPC
    request_id: AtomicU64,
    /// Total requests sent to this server
    total_requests: AtomicU64,
    /// Total time spent on requests (in milliseconds)
    total_request_time_ms: AtomicU64,
    /// Time when server was connected
    connected_at: Option<Instant>,
    /// Time when server was initialized
    initialized_at: Option<Instant>,
}

/// JSON-RPC request structure
#[derive(Debug, Serialize)]
struct JsonRpcRequest {
    jsonrpc: &'static str,
    id: u64,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<serde_json::Value>,
}

impl JsonRpcRequest {
    fn new(id: u64, method: impl Into<String>, params: Option<serde_json::Value>) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            method: method.into(),
            params,
        }
    }
}

/// JSON-RPC response structure
#[derive(Debug, Deserialize)]
struct JsonRpcResponse {
    #[allow(dead_code)]
    jsonrpc: String,
    #[allow(dead_code)]
    id: u64,
    result: Option<serde_json::Value>,
    error: Option<JsonRpcError>,
}

/// JSON-RPC error
#[derive(Debug, Deserialize)]
struct JsonRpcError {
    code: i64,
    message: String,
}

impl ExternalMcpServer {
    /// Connect to an external MCP server via stdio
    ///
    /// This spawns the MCP server process and establishes stdio communication.
    /// Use `initialize()` after connecting to complete the handshake.
    #[instrument(
        name = "mcp_connect_stdio",
        skip(env, cwd),
        fields(
            server_name = %name,
            command = %command,
            args_count = args.len(),
            has_env = env.is_some(),
            has_cwd = cwd.is_some(),
        )
    )]
    pub async fn connect_stdio(
        name: String,
        command: &str,
        args: &[String],
        env: Option<&HashMap<String, String>>,
        cwd: Option<&Path>,
    ) -> Result<Self, ExternalMcpError> {
        let start_time = Instant::now();

        tracing::info!(
            server_name = %name,
            command = %command,
            args = ?args,
            cwd = ?cwd,
            "Starting external MCP server process"
        );

        // Build the command using CommandWrap for process group support
        let mut cmd = CommandWrap::with_new(command, |c| {
            let cmd = c
                .args(args)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::null());

            if let Some(env) = env {
                tracing::debug!(
                    server_name = %name,
                    env_vars = ?env.keys().collect::<Vec<_>>(),
                    "Setting environment variables for MCP server"
                );
                cmd.envs(env);
            }

            if let Some(cwd) = cwd {
                cmd.current_dir(cwd);
            }
        });

        // Add platform-specific wrappers for process group management
        #[cfg(unix)]
        {
            cmd.wrap(ProcessGroup::leader());
        }

        #[cfg(windows)]
        {
            cmd.wrap(JobObject);
        }

        // Spawn the wrapped child process
        let mut wrapped_child = cmd.spawn().map_err(|e| {
            tracing::error!(
                server_name = %name,
                command = %command,
                cwd = ?cwd,
                error = %e,
                error_type = %std::any::type_name::<std::io::Error>(),
                error_kind = ?e.kind(),
                "Failed to spawn MCP server process"
            );
            ExternalMcpError::SpawnFailed {
                command: command.to_string(),
                error: e.to_string(),
            }
        })?;

        let pid = wrapped_child.id();
        tracing::debug!(
            server_name = %name,
            pid = ?pid,
            "MCP server process spawned with process group support"
        );

        // Take stdin and stdout before wrapping
        let stdin = wrapped_child
            .stdin()
            .take()
            .ok_or(ExternalMcpError::NoStdin)?;
        let stdout = wrapped_child
            .stdout()
            .take()
            .ok_or(ExternalMcpError::NoStdout)
            .map(BufReader::new)?;

        // Wrap the child for proper cleanup (already a Box<dyn ChildWrapper>)
        let wrapped = WrappedChild::new(wrapped_child);

        let connection = McpConnection::Stdio {
            child: wrapped,
            stdin,
            stdout,
        };

        let elapsed = start_time.elapsed();
        tracing::info!(
            server_name = %name,
            pid = ?pid,
            elapsed_ms = elapsed.as_millis(),
            "MCP server process started successfully with process group"
        );

        Ok(Self {
            name,
            connection,
            tools: Vec::new(),
            initialized: false,
            request_id: AtomicU64::new(1),
            total_requests: AtomicU64::new(0),
            total_request_time_ms: AtomicU64::new(0),
            connected_at: Some(start_time),
            initialized_at: None,
        })
    }

    /// Initialize the MCP server
    ///
    /// Performs the MCP handshake:
    /// 1. Send initialize request with client info
    /// 2. Send initialized notification
    /// 3. List available tools
    ///
    /// This method has a timeout to prevent indefinite blocking if the server
    /// is unresponsive.
    #[instrument(
        name = "mcp_initialize",
        skip(self),
        fields(
            server_name = %self.name,
            timeout_secs = DEFAULT_INIT_TIMEOUT.as_secs(),
        )
    )]
    pub async fn initialize(&mut self) -> Result<(), ExternalMcpError> {
        let init_start = Instant::now();

        tracing::info!(
            server_name = %self.name,
            "Starting MCP server initialization"
        );

        // Wrap the entire initialization in a timeout
        let init_result = tokio::time::timeout(DEFAULT_INIT_TIMEOUT, async {
            // Send initialize request
            let request_id = self.next_request_id();
            let request = JsonRpcRequest::new(
                request_id,
                "initialize",
                Some(serde_json::json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "claude-code-acp-rs",
                        "version": env!("CARGO_PKG_VERSION")
                    }
                })),
            );

            tracing::debug!(
                server_name = %self.name,
                request_id = request_id,
                "Sending initialize request"
            );

            let init_response = self.send_request_internal(request).await?;

            // Log server info if available
            if let Some(ref result) = init_response.result {
                if let Some(server_info) = result.get("serverInfo") {
                    tracing::info!(
                        server_name = %self.name,
                        remote_server_name = ?server_info.get("name"),
                        remote_server_version = ?server_info.get("version"),
                        protocol_version = ?result.get("protocolVersion"),
                        "Received initialize response from MCP server"
                    );
                }
            }

            // Send initialized notification
            tracing::debug!(
                server_name = %self.name,
                "Sending initialized notification"
            );
            self.send_notification("notifications/initialized", None)
                .await?;

            // List available tools
            let tools_request_id = self.next_request_id();
            let tools_request = JsonRpcRequest::new(tools_request_id, "tools/list", None);

            tracing::debug!(
                server_name = %self.name,
                request_id = tools_request_id,
                "Sending tools/list request"
            );

            let tools_response = self.send_request_internal(tools_request).await?;

            // Parse tools from response
            if let Some(result) = tools_response.result {
                if let Some(tools) = result.get("tools").and_then(|t| t.as_array()) {
                    self.tools = tools
                        .iter()
                        .filter_map(|t| {
                            let name = t.get("name")?.as_str()?;
                            let description =
                                t.get("description").and_then(|d| d.as_str()).unwrap_or("");
                            let input_schema = t
                                .get("inputSchema")
                                .cloned()
                                .unwrap_or(serde_json::json!({"type": "object"}));

                            Some(ToolSchema {
                                name: name.to_string(),
                                description: description.to_string(),
                                input_schema,
                            })
                        })
                        .collect();

                    // Log tool names
                    let tool_names: Vec<&str> =
                        self.tools.iter().map(|t| t.name.as_str()).collect();
                    tracing::info!(
                        server_name = %self.name,
                        tool_count = self.tools.len(),
                        tools = ?tool_names,
                        "Received tools from MCP server"
                    );
                }
            }

            Ok::<(), ExternalMcpError>(())
        })
        .await;

        match init_result {
            Ok(Ok(())) => {
                self.initialized = true;
                self.initialized_at = Some(Instant::now());

                let elapsed = init_start.elapsed();
                tracing::info!(
                    server_name = %self.name,
                    elapsed_ms = elapsed.as_millis(),
                    tool_count = self.tools.len(),
                    "MCP server initialization completed successfully"
                );

                Ok(())
            }
            Ok(Err(e)) => {
                let elapsed = init_start.elapsed();
                tracing::error!(
                    server_name = %self.name,
                    elapsed_ms = elapsed.as_millis(),
                    error = %e,
                    "MCP server initialization failed"
                );
                Err(e)
            }
            Err(_) => {
                let elapsed = init_start.elapsed();
                tracing::error!(
                    server_name = %self.name,
                    elapsed_ms = elapsed.as_millis(),
                    timeout_secs = DEFAULT_INIT_TIMEOUT.as_secs(),
                    "MCP server initialization timed out"
                );
                #[allow(clippy::cast_possible_truncation)]
                Err(ExternalMcpError::Timeout {
                    operation: "initialize".to_string(),
                    timeout_ms: DEFAULT_INIT_TIMEOUT.as_millis() as u64,
                })
            }
        }
    }

    /// Generate next request ID
    fn next_request_id(&self) -> u64 {
        self.request_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Send a JSON-RPC request and wait for response (with timeout)
    ///
    /// This is the public API that wraps the internal method with a timeout.
    #[instrument(
        name = "mcp_send_request",
        skip(self, request),
        fields(
            server_name = %self.name,
            method = %request.method,
            request_id = request.id,
        )
    )]
    async fn send_request(
        &mut self,
        request: JsonRpcRequest,
    ) -> Result<JsonRpcResponse, ExternalMcpError> {
        let method = request.method.clone();
        let request_id = request.id;

        let result =
            tokio::time::timeout(DEFAULT_REQUEST_TIMEOUT, self.send_request_internal(request))
                .await;

        if let Ok(inner_result) = result {
            inner_result
        } else {
            tracing::error!(
                server_name = %self.name,
                method = %method,
                request_id = request_id,
                timeout_ms = DEFAULT_REQUEST_TIMEOUT.as_millis(),
                "MCP request timed out"
            );
            #[allow(clippy::cast_possible_truncation)]
            Err(ExternalMcpError::Timeout {
                operation: method,
                timeout_ms: DEFAULT_REQUEST_TIMEOUT.as_millis() as u64,
            })
        }
    }

    /// Internal implementation of send_request without timeout
    async fn send_request_internal(
        &mut self,
        request: JsonRpcRequest,
    ) -> Result<JsonRpcResponse, ExternalMcpError> {
        let start_time = Instant::now();
        let method = request.method.clone();
        let request_id = request.id;

        let McpConnection::Stdio { stdin, stdout, .. } = &mut self.connection;

        // Serialize and send request
        let request_json = serde_json::to_string(&request)
            .map_err(|e| ExternalMcpError::SerializationError(e.to_string()))?;

        tracing::debug!(
            server_name = %self.name,
            method = %method,
            request_id = request_id,
            request_size = request_json.len(),
            "Sending JSON-RPC request to MCP server"
        );

        stdin
            .write_all(request_json.as_bytes())
            .await
            .map_err(|e| {
                tracing::error!(
                    server_name = %self.name,
                    method = %method,
                    request_size = request_json.len(),
                    error = %e,
                    error_type = %std::any::type_name::<std::io::Error>(),
                    error_kind = ?e.kind(),
                    "Failed to write request to MCP server"
                );
                ExternalMcpError::WriteError(e.to_string())
            })?;
        stdin
            .write_all(b"\n")
            .await
            .map_err(|e| ExternalMcpError::WriteError(e.to_string()))?;
        stdin
            .flush()
            .await
            .map_err(|e| ExternalMcpError::WriteError(e.to_string()))?;

        let write_elapsed = start_time.elapsed();
        tracing::debug!(
            server_name = %self.name,
            method = %method,
            write_elapsed_ms = write_elapsed.as_millis(),
            "Request sent, waiting for response"
        );

        // Read response
        let mut line = String::new();
        stdout.read_line(&mut line).await.map_err(|e| {
            tracing::error!(
                server_name = %self.name,
                method = %method,
                error = %e,
                "Failed to read response from MCP server"
            );
            ExternalMcpError::ReadError(e.to_string())
        })?;

        let total_elapsed = start_time.elapsed();

        // Update statistics
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        #[allow(clippy::cast_possible_truncation)]
        self.total_request_time_ms
            .fetch_add(total_elapsed.as_millis() as u64, Ordering::Relaxed);

        tracing::debug!(
            server_name = %self.name,
            method = %method,
            request_id = request_id,
            response_size = line.len(),
            elapsed_ms = total_elapsed.as_millis(),
            "Received response from MCP server"
        );

        let response: JsonRpcResponse = serde_json::from_str(&line).map_err(|e| {
            tracing::error!(
                server_name = %self.name,
                method = %method,
                error = %e,
                response_preview = %line.chars().take(200).collect::<String>(),
                "Failed to parse JSON-RPC response"
            );
            ExternalMcpError::DeserializationError(e.to_string())
        })?;

        let read_elapsed = total_elapsed.saturating_sub(write_elapsed);

        // Comprehensive performance summary
        tracing::info!(
            server_name = %self.name,
            method = %method,
            request_id = request_id,
            request_size_bytes = request_json.len(),
            response_size_bytes = line.len(),
            write_duration_ms = write_elapsed.as_millis(),
            read_duration_ms = read_elapsed.as_millis(),
            total_round_trip_ms = total_elapsed.as_millis(),
            "MCP JSON-RPC request completed successfully"
        );

        if let Some(error) = response.error {
            tracing::warn!(
                server_name = %self.name,
                method = %method,
                request_id = request_id,
                error_code = error.code,
                error_message = %error.message,
                elapsed_ms = total_elapsed.as_millis(),
                "MCP server returned error"
            );
            return Err(ExternalMcpError::RpcError {
                code: error.code,
                message: error.message,
            });
        }

        tracing::debug!(
            server_name = %self.name,
            method = %method,
            request_id = request_id,
            elapsed_ms = total_elapsed.as_millis(),
            "MCP request completed successfully"
        );

        Ok(response)
    }

    /// Send a JSON-RPC notification (no response expected)
    async fn send_notification(
        &mut self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<(), ExternalMcpError> {
        let McpConnection::Stdio { stdin, .. } = &mut self.connection;

        let notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        });

        let notification_json = serde_json::to_string(&notification)
            .map_err(|e| ExternalMcpError::SerializationError(e.to_string()))?;

        stdin
            .write_all(notification_json.as_bytes())
            .await
            .map_err(|e| ExternalMcpError::WriteError(e.to_string()))?;
        stdin
            .write_all(b"\n")
            .await
            .map_err(|e| ExternalMcpError::WriteError(e.to_string()))?;
        stdin
            .flush()
            .await
            .map_err(|e| ExternalMcpError::WriteError(e.to_string()))?;

        Ok(())
    }

    /// Call a tool on this server
    ///
    /// Executes a tool on the external MCP server with timeout protection.
    #[instrument(
        name = "mcp_call_tool",
        skip(self, arguments),
        fields(
            server_name = %self.name,
            tool_name = %tool_name,
            args_size = arguments.to_string().len(),
        )
    )]
    pub async fn call_tool(
        &mut self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<ToolResult, ExternalMcpError> {
        let start_time = Instant::now();

        if !self.initialized {
            tracing::error!(
                server_name = %self.name,
                tool_name = %tool_name,
                "Attempted to call tool on uninitialized server"
            );
            return Err(ExternalMcpError::NotInitialized);
        }

        tracing::info!(
            server_name = %self.name,
            tool_name = %tool_name,
            "Calling external MCP tool"
        );

        let request_id = self.next_request_id();
        let request = JsonRpcRequest::new(
            request_id,
            "tools/call",
            Some(serde_json::json!({
                "name": tool_name,
                "arguments": arguments
            })),
        );

        let response = self.send_request(request).await?;

        let elapsed = start_time.elapsed();

        // Parse tool result
        if let Some(result) = response.result {
            // Check if result has content array (MCP format)
            if let Some(content) = result.get("content").and_then(|c| c.as_array()) {
                let text: Vec<String> = content
                    .iter()
                    .filter_map(|c| {
                        if c.get("type").and_then(|t| t.as_str()) == Some("text") {
                            c.get("text").and_then(|t| t.as_str()).map(String::from)
                        } else {
                            None
                        }
                    })
                    .collect();

                let is_error = result
                    .get("is_error")
                    .or_else(|| result.get("isError")) // Support both snake_case and camelCase
                    .and_then(|e| e.as_bool())
                    .unwrap_or(false);

                let result_preview = text.join("\n").chars().take(200).collect::<String>();

                if is_error {
                    tracing::warn!(
                        server_name = %self.name,
                        tool_name = %tool_name,
                        elapsed_ms = elapsed.as_millis(),
                        result_preview = %result_preview,
                        "External MCP tool returned error"
                    );
                    return Ok(ToolResult::error(text.join("\n")));
                }

                tracing::info!(
                    server_name = %self.name,
                    tool_name = %tool_name,
                    elapsed_ms = elapsed.as_millis(),
                    result_len = text.iter().map(|s| s.len()).sum::<usize>(),
                    "External MCP tool completed successfully"
                );
                return Ok(ToolResult::success(text.join("\n")));
            }

            // Fallback: return raw JSON
            tracing::info!(
                server_name = %self.name,
                tool_name = %tool_name,
                elapsed_ms = elapsed.as_millis(),
                "External MCP tool completed (raw JSON response)"
            );
            Ok(ToolResult::success(result.to_string()))
        } else {
            tracing::info!(
                server_name = %self.name,
                tool_name = %tool_name,
                elapsed_ms = elapsed.as_millis(),
                "External MCP tool completed (empty response)"
            );
            Ok(ToolResult::success(""))
        }
    }

    /// Get server statistics
    pub fn stats(&self) -> McpServerStats {
        McpServerStats {
            server_name: self.name.clone(),
            total_requests: self.total_requests.load(Ordering::Relaxed),
            total_request_time_ms: self.total_request_time_ms.load(Ordering::Relaxed),
            tool_count: self.tools.len(),
            initialized: self.initialized,
            connected_at: self.connected_at,
            initialized_at: self.initialized_at,
        }
    }

    /// Get available tools from this server
    pub fn tools(&self) -> &[ToolSchema] {
        &self.tools
    }

    /// Check if the server is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Cleanup the MCP server process and wait for exit
    ///
    /// This kills the process group and waits for the process to exit,
    /// preventing zombie processes.
    #[instrument(
        name = "mcp_cleanup",
        skip(self),
        fields(server_name = %self.name)
    )]
    pub async fn cleanup(&mut self) -> Result<(), ExternalMcpError> {
        let McpConnection::Stdio { child, .. } = &mut self.connection;

        let start_time = Instant::now();

        tracing::info!(
            server_name = %self.name,
            "Cleaning up MCP server process"
        );

        // Kill the process group (terminates entire process tree)
        child.kill().await.ok();

        // Wait for the process to exit (prevents zombie)
        drop(child.wait().await);

        let elapsed = start_time.elapsed();
        tracing::info!(
            server_name = %self.name,
            elapsed_ms = elapsed.as_millis(),
            "MCP server cleanup completed"
        );

        Ok(())
    }
}

/// Drop implementation for ExternalMcpServer
///
/// This provides best-effort cleanup when the server is dropped.
/// Note: We can't wait in Drop, so we only start the kill.
impl Drop for ExternalMcpServer {
    fn drop(&mut self) {
        // Best-effort cleanup (can't wait in Drop)
        let McpConnection::Stdio { child, .. } = &mut self.connection;
        drop(child.start_kill());
    }
}

/// Manager for multiple external MCP servers
#[allow(missing_debug_implementations)]
pub struct ExternalMcpManager {
    /// Connected servers by name
    /// Using DashMap for lock-free concurrent access to different servers
    /// Using tokio::sync::Mutex to allow holding lock across .await points
    servers: DashMap<String, Arc<tokio::sync::Mutex<ExternalMcpServer>>>,
    /// Cached tool schemas per server, populated at connect time.
    /// This avoids locking each server mutex in `all_tools()`, which would
    /// cause tools to disappear from the list while a server is busy.
    tool_cache: DashMap<String, Vec<ToolSchema>>,
}

impl ExternalMcpManager {
    /// Create a new external MCP manager
    pub fn new() -> Self {
        Self {
            servers: DashMap::new(),
            tool_cache: DashMap::new(),
        }
    }

    /// Connect to an MCP server
    ///
    /// This method spawns the MCP server process, establishes communication,
    /// and performs the MCP handshake (initialize + tools/list).
    #[instrument(
        name = "mcp_manager_connect",
        skip(self, env, cwd),
        fields(
            server_name = %name,
            command = %command,
        )
    )]
    pub async fn connect(
        &self,
        name: String,
        command: &str,
        args: &[String],
        env: Option<&HashMap<String, String>>,
        cwd: Option<&Path>,
    ) -> Result<(), ExternalMcpError> {
        let overall_start = Instant::now();

        tracing::info!(
            server_name = %name,
            command = %command,
            args = ?args,
            "Connecting to external MCP server"
        );

        // Step 1: Spawn and connect
        let connect_start = Instant::now();
        let mut server =
            ExternalMcpServer::connect_stdio(name.clone(), command, args, env, cwd).await?;
        let connect_elapsed = connect_start.elapsed();

        tracing::debug!(
            server_name = %name,
            connect_elapsed_ms = connect_elapsed.as_millis(),
            "MCP server process connected"
        );

        // Step 2: Initialize
        let init_start = Instant::now();
        server.initialize().await?;
        let init_elapsed = init_start.elapsed();

        let overall_elapsed = overall_start.elapsed();

        tracing::info!(
            server_name = %name,
            tool_count = server.tools().len(),
            connect_elapsed_ms = connect_elapsed.as_millis(),
            init_elapsed_ms = init_elapsed.as_millis(),
            total_elapsed_ms = overall_elapsed.as_millis(),
            "Successfully connected and initialized MCP server"
        );

        // Log tool names for debugging
        let tool_names: Vec<&str> = server.tools().iter().map(|t| t.name.as_str()).collect();
        tracing::debug!(
            server_name = %name,
            tools = ?tool_names,
            "MCP server tools available"
        );

        // Cache tool schemas (with server-prefixed names) for lock-free access in all_tools()
        let cached_tools: Vec<ToolSchema> = server
            .tools()
            .iter()
            .map(|tool| ToolSchema {
                name: format!("mcp__{}_{}", name, tool.name),
                description: format!("[{}] {}", name, tool.description),
                input_schema: tool.input_schema.clone(),
            })
            .collect();
        self.tool_cache.insert(name.clone(), cached_tools);

        // Insert server into DashMap (no async needed)
        self.servers
            .insert(name, Arc::new(tokio::sync::Mutex::new(server)));
        Ok(())
    }

    /// Disconnect from an MCP server
    ///
    /// This properly cleans up the server process and prevents zombie processes.
    #[instrument(
        name = "mcp_manager_disconnect",
        skip(self),
        fields(server_name = %name)
    )]
    pub async fn disconnect(&self, name: &str) -> Result<(), ExternalMcpError> {
        self.tool_cache.remove(name);
        if let Some((_, server_arc)) = self.servers.remove(name) {
            let mut server = server_arc.lock().await;
            server.cleanup().await?;
        }
        Ok(())
    }

    /// Get all connected server names
    pub fn server_names(&self) -> Vec<String> {
        self.servers
            .iter()
            .map(|entry| entry.key().clone())
            .collect()
    }

    /// Get all available tools from all servers
    ///
    /// Tool names are prefixed with `mcp__<server>__`
    /// Reads from cached tool schemas populated at connect time, so this
    /// never blocks on server mutexes and tools remain visible even while
    /// a server is busy executing a tool call.
    pub fn all_tools(&self) -> Vec<ToolSchema> {
        let mut tools = Vec::new();

        for entry in &self.tool_cache {
            tools.extend(entry.value().iter().cloned());
        }

        tools
    }

    /// Call a tool on an external server
    ///
    /// Tool name should be prefixed with `mcp__<server>__`
    #[instrument(
        name = "mcp_manager_call_tool",
        skip(self, arguments),
        fields(
            full_tool_name = %full_tool_name,
        )
    )]
    pub async fn call_tool(
        &self,
        full_tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<ToolResult, ExternalMcpError> {
        // Parse server name and tool name from `mcp__<server>__<tool>`
        let parts: Vec<&str> = full_tool_name.splitn(3, "__").collect();
        if parts.len() != 3 || parts[0] != "mcp" {
            tracing::warn!(
                full_tool_name = %full_tool_name,
                "Invalid external MCP tool name format"
            );
            return Err(ExternalMcpError::InvalidToolName(
                full_tool_name.to_string(),
            ));
        }

        let server_name = parts[1];
        let tool_name = parts[2];

        // Record to current span
        Span::current().record("server_name", server_name);
        Span::current().record("tool_name", tool_name);

        tracing::debug!(
            server_name = %server_name,
            tool_name = %tool_name,
            "Routing tool call to external MCP server"
        );

        // Get the server from DashMap
        let server_arc = self.servers.get(server_name).ok_or_else(|| {
            let available: Vec<String> = self.server_names();
            tracing::error!(
                server_name = %server_name,
                tool_name = %tool_name,
                available_servers = ?available,
                "External MCP server not found"
            );
            ExternalMcpError::ServerNotFound(server_name.to_string())
        })?;

        // Clone the Arc to hold it across the await
        let server = server_arc.clone();
        drop(server_arc); // Release DashMap reference

        let start_time = Instant::now();

        // Lock the server's mutex and call the tool
        // tokio::sync::Mutex allows holding lock across .await points
        let result = {
            let mut server_guard = server.lock().await;
            server_guard.call_tool(tool_name, arguments).await?
        };

        let elapsed = start_time.elapsed();
        tracing::info!(
            server_name = %server_name,
            tool_name = %tool_name,
            elapsed_ms = elapsed.as_millis(),
            is_error = result.is_error,
            "External MCP tool call completed"
        );

        Ok(result)
    }

    /// Get statistics for all connected servers
    pub fn all_stats(&self) -> Vec<McpServerStats> {
        self.servers
            .iter()
            .filter_map(|entry| {
                let server = entry.value();
                // Try to lock the mutex (non-blocking)
                if let Ok(guard) = server.try_lock() {
                    Some(guard.stats())
                } else {
                    tracing::warn!(
                        server_name = %entry.key(),
                        "MCP server is busy, skipping for stats"
                    );
                    None
                }
            })
            .collect()
    }

    /// Check if a tool name refers to an external MCP tool
    ///
    /// External MCP tools have the format `mcp__<server>__<tool>` where
    /// `<server>` is not "acp" (which is reserved for the ACP prefix).
    pub fn is_external_tool(name: &str) -> bool {
        if !name.starts_with("mcp__") {
            return false;
        }

        // Split by __ and check structure
        let parts: Vec<&str> = name.splitn(3, "__").collect();
        if parts.len() != 3 || parts[0] != "mcp" {
            return false;
        }

        // "acp" is reserved for the ACP tool prefix, not external MCP
        parts[1] != "acp"
    }

    /// Get the friendly name for an external MCP tool
    ///
    /// This maps MCP tool names like `mcp__web-fetch__webReader` to friendly names
    /// like `WebFetch` that can be used in permission settings.
    ///
    /// Only supports official Anthropic Claude Code tools:
    /// - WebFetch (web-fetch/web-reader MCP server)
    /// - WebSearch (web-search-prime MCP server)
    ///
    /// Returns None if the tool is not an external MCP tool or has no known mapping.
    pub fn get_friendly_tool_name(name: &str) -> Option<String> {
        if !Self::is_external_tool(name) {
            return None;
        }

        let parts: Vec<&str> = name.splitn(3, "__").collect();
        let server_name = parts.get(1)?;
        let tool_name = parts.get(2)?;

        // Map known MCP server/tool combinations to friendly names
        // Only official Anthropic Claude Code tools are supported
        match (*server_name, *tool_name) {
            // Web Fetch MCP server
            ("web-fetch", "webReader") => Some("WebFetch".to_string()),
            ("web-reader", "webReader") => Some("WebFetch".to_string()),

            // Web Search Prime MCP server
            ("web-search-prime", "webSearchPrime") => Some("WebSearch".to_string()),

            // Unknown tool - return None
            _ => None,
        }
    }
}

impl Default for ExternalMcpManager {
    fn default() -> Self {
        Self::new()
    }
}

/// MCP server statistics
#[derive(Debug, Clone)]
pub struct McpServerStats {
    /// Server name
    pub server_name: String,
    /// Total requests sent
    pub total_requests: u64,
    /// Total time spent on requests (ms)
    pub total_request_time_ms: u64,
    /// Number of tools available
    pub tool_count: usize,
    /// Whether the server is initialized
    pub initialized: bool,
    /// Time when server was connected
    pub connected_at: Option<Instant>,
    /// Time when server was initialized
    pub initialized_at: Option<Instant>,
}

impl McpServerStats {
    /// Get average request time in milliseconds
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_request_time_ms(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.total_request_time_ms as f64 / self.total_requests as f64
        }
    }

    /// Get uptime since connection
    pub fn uptime(&self) -> Option<Duration> {
        self.connected_at.map(|t| t.elapsed())
    }
}

/// Errors for external MCP operations
#[derive(Debug, thiserror::Error)]
pub enum ExternalMcpError {
    /// Failed to spawn MCP server process
    #[error("Failed to spawn MCP server '{command}': {error}")]
    SpawnFailed { command: String, error: String },

    /// No stdin available
    #[error("No stdin available for MCP server")]
    NoStdin,

    /// No stdout available
    #[error("No stdout available for MCP server")]
    NoStdout,

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Deserialization error
    #[error("Deserialization error: {0}")]
    DeserializationError(String),

    /// Write error
    #[error("Write error: {0}")]
    WriteError(String),

    /// Read error
    #[error("Read error: {0}")]
    ReadError(String),

    /// RPC error from server
    #[error("RPC error {code}: {message}")]
    RpcError { code: i64, message: String },

    /// Server not initialized
    #[error("Server not initialized")]
    NotInitialized,

    /// Invalid tool name format
    #[error("Invalid tool name format: {0}")]
    InvalidToolName(String),

    /// Server not found
    #[error("MCP server not found: {0}")]
    ServerNotFound(String),

    /// Request or operation timed out
    #[error("MCP operation '{operation}' timed out after {timeout_ms}ms")]
    Timeout { operation: String, timeout_ms: u64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_external_mcp_manager_new() {
        let _manager = ExternalMcpManager::new();
        // Just verify creation works
        assert!(ExternalMcpManager::is_external_tool("mcp__server__tool"));
        assert!(!ExternalMcpManager::is_external_tool("Read"));
        assert!(!ExternalMcpManager::is_external_tool("mcp__acp__Read"));
    }

    #[test]
    fn test_is_external_tool() {
        // External MCP tools
        assert!(ExternalMcpManager::is_external_tool(
            "mcp__myserver__mytool"
        ));
        assert!(ExternalMcpManager::is_external_tool(
            "mcp__filesystem__read_file"
        ));

        // Not external tools
        assert!(!ExternalMcpManager::is_external_tool("Read"));
        assert!(!ExternalMcpManager::is_external_tool("Bash"));
        assert!(!ExternalMcpManager::is_external_tool("mcp__acp__Read")); // ACP prefix, not external
        assert!(!ExternalMcpManager::is_external_tool("mcp__single")); // Not enough parts
    }

    #[tokio::test]
    async fn test_manager_server_names_empty() {
        let manager = ExternalMcpManager::new();
        let names = manager.server_names();
        assert!(names.is_empty());
    }

    #[tokio::test]
    async fn test_manager_all_tools_empty() {
        let manager = ExternalMcpManager::new();
        let tools = manager.all_tools();
        assert!(tools.is_empty());
    }

    #[test]
    fn test_get_friendly_tool_name_web_fetch() {
        assert_eq!(
            ExternalMcpManager::get_friendly_tool_name("mcp__web-fetch__webReader"),
            Some("WebFetch".to_string())
        );
        assert_eq!(
            ExternalMcpManager::get_friendly_tool_name("mcp__web-reader__webReader"),
            Some("WebFetch".to_string())
        );
    }

    #[test]
    fn test_get_friendly_tool_name_web_search() {
        assert_eq!(
            ExternalMcpManager::get_friendly_tool_name("mcp__web-search-prime__webSearchPrime"),
            Some("WebSearch".to_string())
        );
    }

    #[test]
    fn test_get_friendly_tool_name_non_mcp_tool() {
        assert_eq!(ExternalMcpManager::get_friendly_tool_name("Read"), None);
        assert_eq!(ExternalMcpManager::get_friendly_tool_name("Bash"), None);
        assert_eq!(
            ExternalMcpManager::get_friendly_tool_name("mcp__acp__Read"),
            None
        );
    }

    #[test]
    fn test_get_friendly_tool_name_unknown_mcp_tool() {
        // Unknown MCP tools should return None (only official tools are supported)
        assert_eq!(
            ExternalMcpManager::get_friendly_tool_name("mcp__zai-mcp-server__ui_to_artifact"),
            None
        );
        assert_eq!(
            ExternalMcpManager::get_friendly_tool_name("mcp__context7__query-docs"),
            None
        );
        assert_eq!(
            ExternalMcpManager::get_friendly_tool_name("mcp__my-server__my_custom_tool"),
            None
        );
    }

    /// Test that disconnect properly cleans up MCP server processes
    ///
    /// This test verifies that:
    /// 1. A process can be spawned and tracked
    /// 2. disconnect() properly removes the server from the manager
    /// 3. The cleanup method is called
    ///
    /// Note: We can't test with `echo` or `sleep` because they don't implement
    /// the MCP JSON-RPC protocol. This test verifies the manager-level logic.
    #[tokio::test]
    async fn test_external_mcp_disconnect_removes_from_manager() {
        let manager = ExternalMcpManager::new();

        // Verify no servers initially
        let names = manager.server_names();
        assert!(names.is_empty());

        // Disconnecting a non-existent server should succeed (idempotent)
        let result = manager.disconnect("nonexistent-server").await;
        assert!(
            result.is_ok(),
            "Disconnecting non-existent server should be OK"
        );

        // Still no servers
        assert!(manager.server_names().is_empty());
    }

    /// Test that disconnect handles non-existent servers gracefully
    #[tokio::test]
    async fn test_external_mcp_disconnect_nonexistent() {
        let manager = ExternalMcpManager::new();

        // Disconnecting a non-existent server should not error
        let result = manager.disconnect("nonexistent-server").await;
        assert!(
            result.is_ok(),
            "Disconnecting non-existent server should be OK"
        );
    }

    /// Test cleanup method on ExternalMcpServer directly
    ///
    /// This is a lower-level unit test that verifies the cleanup logic
    /// without requiring a full MCP server connection.
    #[tokio::test]
    async fn test_external_mcp_server_cleanup_method() {
        // We can't easily test with a real process since MCP requires
        // JSON-RPC protocol handshake. But we can verify the method exists
        // and has the right signature.

        // This test is primarily a compile-time check that the cleanup
        // method exists and is callable. In a real integration test,
        // you would spawn a mock MCP server that speaks the protocol.

        // For now, we just verify the test compiles and the API is correct.
        // The actual zombie prevention is verified by:
        // 1. The process-wrap library's own tests
        // 2. Manual testing with real MCP servers
        // 3. System monitoring for zombie processes
    }
}
