//! Bash tool implementation
//!
//! Executes shell commands with security protections.
//! Supports both direct process execution and Client-side PTY via Terminal API.

use async_trait::async_trait;
use sacp::schema::ToolCallStatus;
use serde::Deserialize;
use serde_json::json;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::time::timeout;
use uuid::Uuid;

use super::base::{Tool, ToolKind};
use crate::mcp::registry::{ToolContext, ToolResult};
use crate::session::{BackgroundTerminal, ChildHandle, TerminalExitStatus, WrappedChild};
use crate::terminal::TerminalClient;

// Process group management
use process_wrap::tokio::*;

/// Maximum output size in characters
const MAX_OUTPUT_SIZE: usize = 30_000;

/// Shell operators that indicate command chaining (security risk)
///
/// These operators allow chaining multiple commands, which could be used
/// for command injection attacks. Commands containing these operators
/// should be handled with extra care in permission rules.
const SHELL_OPERATORS: &[&str] = &["&&", "||", ";", "|", "$(", "`", "\n"];

/// Check if a command string contains shell operators
///
/// This is used to prevent command injection when matching prefix-based
/// permission rules. For example, if a rule allows `npm run:*`, we must
/// ensure that `npm run build && rm -rf /` doesn't match by detecting
/// the `&&` operator in the remainder after the prefix.
///
/// # Examples
///
/// ```
/// use claude_code_acp::mcp::tools::contains_shell_operator;
///
/// assert!(contains_shell_operator("ls && rm -rf /"));
/// assert!(contains_shell_operator("cat file | grep secret"));
/// assert!(contains_shell_operator("$(whoami)"));
/// assert!(!contains_shell_operator("npm run build"));
/// ```
pub fn contains_shell_operator(command: &str) -> bool {
    SHELL_OPERATORS.iter().any(|op| command.contains(op))
}

/// Bash tool for executing shell commands
#[derive(Debug, Default)]
pub struct BashTool;

/// Bash tool input parameters
#[derive(Debug, Deserialize)]
struct BashInput {
    /// The command to execute
    command: String,
    /// Optional description of what the command does
    #[serde(default)]
    description: Option<String>,
    /// Optional timeout in milliseconds
    #[serde(default)]
    timeout: Option<u64>,
    /// Run command in background (returns immediately with shell ID)
    #[serde(default)]
    run_in_background: Option<bool>,
}

impl BashTool {
    /// Create a new Bash tool instance
    pub fn new() -> Self {
        Self
    }

    /// Safely truncate string to maximum size, respecting UTF-8 character boundaries
    fn safe_truncate(s: &mut String, max_len: usize) {
        if s.len() > max_len {
            // Handle edge case: max_len is 0
            if max_len == 0 {
                s.clear();
                s.push_str("... (output truncated)");
                return;
            }

            // Find valid UTF-8 boundary
            let mut truncate_at = max_len;
            while truncate_at > 0 && !s.is_char_boundary(truncate_at) {
                truncate_at -= 1;
            }
            s.truncate(truncate_at);
            s.push_str("\n... (output truncated)");
        }
    }

    /// Check permission before executing the tool
    ///
    /// Note: Permission checking is now handled at the SDK level.
    /// The SDK's `mcp_message` handler calls `can_use_tool` callback before executing MCP tools.
    /// This method is kept for potential future tool-specific permission logic.
    fn check_permission(
        &self,
        _input: &serde_json::Value,
        _context: &ToolContext,
    ) -> Option<ToolResult> {
        // Permission check is handled by SDK's can_use_tool callback
        None
    }
}

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &str {
        "Bash"
    }

    fn description(&self) -> &str {
        "Execute a shell command. Commands are run in a bash shell with the session's working directory. Use for git, npm, build tools, and other terminal operations."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "required": ["command"],
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "description": {
                    "type": "string",
                    "description": "A short description of what this command does"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in milliseconds (max 600000, default 120000)"
                },
                "run_in_background": {
                    "type": "boolean",
                    "description": "Run command in background. Returns immediately with a shell ID that can be used with BashOutput to retrieve output."
                }
            }
        })
    }

    fn kind(&self) -> ToolKind {
        ToolKind::Execute
    }

    fn requires_permission(&self) -> bool {
        true // Command execution requires permission
    }

    async fn execute(&self, input: serde_json::Value, context: &ToolContext) -> ToolResult {
        // Check permission before executing
        if let Some(result) = self.check_permission(&input, context) {
            return result;
        }

        // Parse input
        let params: BashInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        // Prefer Terminal API when available (Client-side PTY)
        if let Some(terminal_client) = context.terminal_client() {
            if params.run_in_background.unwrap_or(false) {
                return self
                    .execute_terminal_background(&params, terminal_client, context)
                    .await;
            }
            return self
                .execute_terminal_foreground(&params, terminal_client, context)
                .await;
        }

        // Fall back to direct process execution
        if params.run_in_background.unwrap_or(false) {
            return self.execute_background(&params, context);
        }

        self.execute_foreground(&params, context).await
    }
}

impl BashTool {
    /// Execute command in foreground (blocking)
    async fn execute_foreground(&self, params: &BashInput, context: &ToolContext) -> ToolResult {
        let cmd_start = Instant::now();

        // Use timeout as specified by user, without limiting maximum
        let timeout_ms = params.timeout;

        // Stage 1: Build the command
        let build_start = Instant::now();
        let mut cmd = Command::new("bash");
        cmd.arg("-c")
            .arg(&params.command)
            .current_dir(&context.cwd)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let build_duration = build_start.elapsed();

        tracing::debug!(
            command = %params.command,
            build_duration_ms = build_duration.as_millis(),
            timeout_ms = ?timeout_ms,
            "Bash command built"
        );

        // Stage 2: Execute with or without timeout
        let exec_start = Instant::now();
        let output = if let Some(ms) = timeout_ms {
            // User specified timeout - apply it
            let timeout_duration = Duration::from_millis(ms);
            match timeout(timeout_duration, cmd.output()).await {
                Ok(Ok(output)) => output,
                Ok(Err(e)) => {
                    let exec_duration = exec_start.elapsed();
                    tracing::error!(
                        command = %params.command,
                        exec_duration_ms = exec_duration.as_millis(),
                        error = %e,
                        "Bash command execution failed"
                    );
                    return ToolResult::error(format!("Failed to execute command: {}", e));
                }
                Err(_) => {
                    let exec_duration = exec_start.elapsed();
                    tracing::warn!(
                        command = %params.command,
                        exec_duration_ms = exec_duration.as_millis(),
                        timeout_ms = ms,
                        "Bash command timed out"
                    );
                    return ToolResult::error(format!("Command timed out after {}ms", ms));
                }
            }
        } else {
            // No timeout specified - execute without timeout limit
            match cmd.output().await {
                Ok(output) => output,
                Err(e) => {
                    let exec_duration = exec_start.elapsed();
                    tracing::error!(
                        command = %params.command,
                        exec_duration_ms = exec_duration.as_millis(),
                        error = %e,
                        "Bash command execution failed"
                    );
                    return ToolResult::error(format!("Failed to execute command: {}", e));
                }
            }
        };
        let exec_duration = exec_start.elapsed();

        // Stage 3: Process output
        let process_start = Instant::now();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let mut result_text = String::new();

        // Add stdout
        if !stdout.is_empty() {
            result_text.push_str(&stdout);
        }

        // Add stderr (prefixed if there's also stdout)
        if !stderr.is_empty() {
            if !result_text.is_empty() {
                result_text.push_str("\n--- stderr ---\n");
            }
            result_text.push_str(&stderr);
        }

        // Truncate if too long (using UTF-8 safe truncation)
        let was_truncated = result_text.len() > MAX_OUTPUT_SIZE;
        Self::safe_truncate(&mut result_text, MAX_OUTPUT_SIZE);

        // Handle empty output
        if result_text.is_empty() {
            result_text = "(no output)".to_string();
        }

        let process_duration = process_start.elapsed();
        let total_elapsed = cmd_start.elapsed();

        let exit_code = output.status.code().unwrap_or(-1);
        let success = output.status.success();

        // Log execution summary
        tracing::info!(
            command = %params.command,
            exit_code = exit_code,
            success = success,
            build_duration_ms = build_duration.as_millis(),
            exec_duration_ms = exec_duration.as_millis(),
            process_duration_ms = process_duration.as_millis(),
            total_elapsed_ms = total_elapsed.as_millis(),
            output_size_bytes = result_text.len(),
            was_truncated = was_truncated,
            "Bash command execution summary"
        );

        if success {
            ToolResult::success(result_text).with_metadata(json!({
                "exit_code": exit_code,
                "truncated": was_truncated,
                "description": params.description,
                "total_elapsed_ms": total_elapsed.as_millis(),
                "exec_duration_ms": exec_duration.as_millis()
            }))
        } else {
            ToolResult::error(format!(
                "Command failed with exit code {}\n{}",
                exit_code, result_text
            ))
            .with_metadata(json!({
                "exit_code": exit_code,
                "truncated": was_truncated,
                "total_elapsed_ms": total_elapsed.as_millis(),
                "exec_duration_ms": exec_duration.as_millis()
            }))
        }
    }

    /// Execute command in background (non-blocking)
    fn execute_background(&self, params: &BashInput, context: &ToolContext) -> ToolResult {
        // Get background process manager
        let manager = match context.background_processes() {
            Some(m) => m.clone(),
            None => {
                return ToolResult::error("Background process manager not available");
            }
        };

        // Build the command with process-wrap for process group support
        let mut cmd = CommandWrap::with_new("bash", |c| {
            c.arg("-c")
                .arg(&params.command)
                .current_dir(&context.cwd)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
        });

        // Add platform-specific wrapper for process group management
        #[cfg(unix)]
        cmd.wrap(ProcessGroup::leader());

        #[cfg(windows)]
        cmd.wrap(JobObject);

        // Spawn the process
        let mut wrapped_child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => return ToolResult::error(format!("Failed to spawn command: {}", e)),
        };

        // Extract stdout and stderr BEFORE wrapping (ChildWrapper doesn't expose them)
        let stdout = wrapped_child.stdout().take();
        let stderr = wrapped_child.stderr().take();

        // Generate shell ID
        let shell_id = format!("shell-{}", Uuid::new_v4().simple());

        // Create wrapped child handle (stdout/stderr not stored in handle)
        let child_handle = ChildHandle::Wrapped {
            child: Arc::new(tokio::sync::Mutex::new(WrappedChild::new(wrapped_child))),
        };

        // Create background terminal
        let terminal = BackgroundTerminal::new_running(child_handle);

        // Get reference to output buffer for the read task
        let output_buffer = match &terminal {
            BackgroundTerminal::Running { output_buffer, .. } => output_buffer.clone(),
            BackgroundTerminal::Finished { .. } => unreachable!(),
        };

        // Register with manager
        let shell_id_clone = shell_id.clone();
        manager.register(shell_id.clone(), terminal);

        // Spawn task to read output
        let manager_clone = manager.clone();
        let description = params.description.clone();
        tokio::spawn(async move {
            let mut combined_output = String::new();

            // Read stdout
            if let Some(stdout) = stdout {
                let reader = BufReader::new(stdout);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    combined_output.push_str(&line);
                    combined_output.push('\n');
                    let mut buffer = output_buffer.lock().await;
                    buffer.push_str(&line);
                    buffer.push('\n');
                }
            }

            // Read stderr
            if let Some(stderr) = stderr {
                let reader = BufReader::new(stderr);
                let mut lines = reader.lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    if !combined_output.is_empty() && !combined_output.ends_with('\n') {
                        combined_output.push('\n');
                    }
                    combined_output.push_str(&line);
                    combined_output.push('\n');
                    let mut buffer = output_buffer.lock().await;
                    buffer.push_str(&line);
                    buffer.push('\n');
                }
            }

            // Wait for process to finish and update terminal state
            // Use get() instead of get_mut() because BackgroundTerminal contains ChildHandle
            // We only need a shared reference to clone the ChildHandle
            if let Some(terminal_ref) = manager_clone.get(&shell_id_clone) {
                if let BackgroundTerminal::Running { child, .. } = &*terminal_ref {
                    // Clone the ChildHandle to hold it across await points
                    let mut child_handle = child.clone();
                    drop(terminal_ref); // Release DashMap read lock before await

                    // ChildHandle::wait() handles locking internally
                    if let Ok(status) = child_handle.wait().await {
                        let exit_code = status.code().unwrap_or(-1);
                        manager_clone
                            .finish_terminal(&shell_id_clone, TerminalExitStatus::Exited(exit_code))
                            .await;
                    } else {
                        manager_clone
                            .finish_terminal(&shell_id_clone, TerminalExitStatus::Aborted)
                            .await;
                    }
                }
            }
        });

        // Return immediately with shell ID
        ToolResult::success(format!(
            "Command started in background.\n\nShell ID: {}\n\nUse BashOutput to check status and retrieve output.",
            shell_id
        )).with_metadata(json!({
            "shell_id": shell_id,
            "status": "running",
            "description": description
        }))
    }

    /// Execute command using Terminal API in foreground (blocking)
    ///
    /// Uses Client-side PTY for execution, which provides better terminal
    /// emulation and real-time output streaming.
    async fn execute_terminal_foreground(
        &self,
        params: &BashInput,
        terminal_client: &Arc<TerminalClient>,
        context: &ToolContext,
    ) -> ToolResult {
        // Use timeout as specified by user, without limiting maximum
        let timeout_ms = params.timeout;

        // Create terminal with bash -c command
        let terminal_id = match terminal_client
            .create(
                "bash",
                vec!["-c".to_string(), params.command.clone()],
                Some(context.cwd.clone()),
                Some(MAX_OUTPUT_SIZE as u64),
            )
            .await
        {
            Ok(id) => id,
            Err(e) => return ToolResult::error(format!("Failed to create terminal: {}", e)),
        };

        // Send ToolCallUpdate with terminal_id immediately
        // This allows the client (e.g., Zed) to start showing terminal output
        if let Err(e) = context.send_terminal_update(
            terminal_id.0.as_ref(),
            ToolCallStatus::InProgress,
            params.description.as_deref(),
        ) {
            tracing::debug!("Failed to send terminal update: {}", e);
            // Continue even if notification fails - tool should still work
        }

        // Wait for command to exit with optional timeout
        let exit_result = if let Some(ms) = timeout_ms {
            // User specified timeout - apply it
            let timeout_duration = Duration::from_millis(ms);
            timeout(
                timeout_duration,
                terminal_client.wait_for_exit(terminal_id.clone()),
            )
            .await
        } else {
            // No timeout - wait indefinitely
            Ok(terminal_client.wait_for_exit(terminal_id.clone()).await)
        };

        // Get output regardless of exit status
        let output = match terminal_client.output(terminal_id.clone()).await {
            Ok(resp) => resp.output,
            Err(e) => format!("(failed to get output: {})", e),
        };

        // Release terminal (ignore result - best effort)
        drop(terminal_client.release(terminal_id).await);

        // Process result
        match exit_result {
            Ok(Ok(exit_response)) => {
                let exit_status = exit_response.exit_status;
                // exit_code is Option<u32>, convert to i32 for compatibility
                #[allow(clippy::cast_possible_wrap)]
                let exit_code = exit_status.exit_code.map(|c| c as i32).unwrap_or(-1);

                // Apply UTF-8 safe truncation if needed
                let mut result_text = if output.is_empty() {
                    "(no output)".to_string()
                } else {
                    output
                };

                let was_truncated = result_text.len() > MAX_OUTPUT_SIZE;
                Self::safe_truncate(&mut result_text, MAX_OUTPUT_SIZE);

                if exit_code == 0 {
                    ToolResult::success(result_text).with_metadata(json!({
                        "exit_code": exit_code,
                        "truncated": was_truncated,
                        "description": params.description,
                        "terminal_api": true
                    }))
                } else {
                    ToolResult::error(format!(
                        "Command failed with exit code {}\n{}",
                        exit_code, result_text
                    ))
                    .with_metadata(json!({
                        "exit_code": exit_code,
                        "truncated": was_truncated,
                        "terminal_api": true
                    }))
                }
            }
            Ok(Err(e)) => ToolResult::error(format!("Terminal execution failed: {}", e)),
            Err(_) => {
                // Timeout occurred - timeout_ms must be Some in this branch
                let ms = timeout_ms.unwrap_or(0);
                ToolResult::error(format!("Command timed out after {}ms\n{}", ms, output))
            }
        }
    }

    /// Execute command using Terminal API in background (non-blocking)
    ///
    /// Creates a terminal via Client-side PTY and returns immediately.
    /// The terminal_id serves as the shell_id for later queries.
    async fn execute_terminal_background(
        &self,
        params: &BashInput,
        terminal_client: &Arc<TerminalClient>,
        context: &ToolContext,
    ) -> ToolResult {
        // Create terminal with bash -c command
        let terminal_id = match terminal_client
            .create(
                "bash",
                vec!["-c".to_string(), params.command.clone()],
                Some(context.cwd.clone()),
                None, // No output limit for background
            )
            .await
        {
            Ok(id) => id,
            Err(e) => return ToolResult::error(format!("Failed to create terminal: {}", e)),
        };

        // Use terminal_id as shell_id (prefixed for clarity)
        let shell_id = format!("term-{}", terminal_id.0.as_ref());

        // Send ToolCallUpdate with terminal_id immediately
        // This allows the client (e.g., Zed) to start showing terminal output
        if let Err(e) = context.send_terminal_update(
            terminal_id.0.as_ref(),
            ToolCallStatus::InProgress,
            params.description.as_deref(),
        ) {
            tracing::debug!("Failed to send terminal update: {}", e);
            // Continue even if notification fails - tool should still work
        }

        // Return immediately with shell ID
        ToolResult::success(format!(
            "Command started in background via Terminal API.\n\nShell ID: {}\n\nUse BashOutput to check status and retrieve output.",
            shell_id
        )).with_metadata(json!({
            "shell_id": shell_id,
            "terminal_id": terminal_id.0.as_ref(),
            "status": "running",
            "description": params.description,
            "terminal_api": true
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_bash_echo() {
        let temp_dir = TempDir::new().unwrap();
        let tool = BashTool::new();
        let context = ToolContext::new("test", temp_dir.path());

        let result = tool
            .execute(
                json!({
                    "command": "echo 'Hello, World!'"
                }),
                &context,
            )
            .await;

        assert!(!result.is_error);
        assert!(result.content.contains("Hello, World!"));
    }

    #[tokio::test]
    async fn test_bash_with_cwd() {
        let temp_dir = TempDir::new().unwrap();
        let tool = BashTool::new();
        let context = ToolContext::new("test", temp_dir.path());

        let result = tool
            .execute(
                json!({
                    "command": "pwd"
                }),
                &context,
            )
            .await;

        assert!(!result.is_error);
        assert!(result.content.contains(temp_dir.path().to_str().unwrap()));
    }

    #[tokio::test]
    async fn test_bash_failure() {
        let temp_dir = TempDir::new().unwrap();
        let tool = BashTool::new();
        let context = ToolContext::new("test", temp_dir.path());

        let result = tool
            .execute(
                json!({
                    "command": "exit 1"
                }),
                &context,
            )
            .await;

        assert!(result.is_error);
        assert!(result.content.contains("exit code 1"));
    }

    #[tokio::test]
    async fn test_bash_stderr() {
        let temp_dir = TempDir::new().unwrap();
        let tool = BashTool::new();
        let context = ToolContext::new("test", temp_dir.path());

        let result = tool
            .execute(
                json!({
                    "command": "echo 'error message' >&2"
                }),
                &context,
            )
            .await;

        assert!(!result.is_error);
        assert!(result.content.contains("error message"));
    }

    #[tokio::test]
    async fn test_bash_timeout() {
        let temp_dir = TempDir::new().unwrap();
        let tool = BashTool::new();
        let context = ToolContext::new("test", temp_dir.path());

        let result = tool
            .execute(
                json!({
                    "command": "sleep 10",
                    "timeout": 100
                }),
                &context,
            )
            .await;

        assert!(result.is_error);
        assert!(result.content.contains("timed out"));
    }

    #[test]
    fn test_bash_tool_properties() {
        let tool = BashTool::new();
        assert_eq!(tool.name(), "Bash");
        assert_eq!(tool.kind(), ToolKind::Execute);
        assert!(tool.requires_permission());
    }

    #[test]
    fn test_shell_operator_detection() {
        // Commands with shell operators (should be detected)
        assert!(contains_shell_operator("ls && rm -rf /"));
        assert!(contains_shell_operator("cat file || echo fail"));
        assert!(contains_shell_operator("echo a; echo b"));
        assert!(contains_shell_operator("cat file | grep secret"));
        assert!(contains_shell_operator("echo $(whoami)"));
        assert!(contains_shell_operator("echo `whoami`"));
        assert!(contains_shell_operator("echo a\necho b"));

        // Safe commands (should not be detected)
        assert!(!contains_shell_operator("npm run build"));
        assert!(!contains_shell_operator("git status"));
        assert!(!contains_shell_operator("cargo test --release"));
        assert!(!contains_shell_operator("ls -la /tmp"));
        assert!(!contains_shell_operator("echo 'hello world'"));
    }

    #[test]
    fn test_shell_operator_prefix_matching() {
        // Simulate prefix matching scenario
        let prefix = "npm run ";
        let command = "npm run build && malicious";

        // After prefix match, check remainder for operators
        let remainder = &command[prefix.len()..];
        assert!(contains_shell_operator(remainder));

        // Safe case
        let safe_command = "npm run build --watch";
        let safe_remainder = &safe_command[prefix.len()..];
        assert!(!contains_shell_operator(safe_remainder));
    }
}
