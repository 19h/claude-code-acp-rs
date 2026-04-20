//! Terminal API client
//!
//! Provides a client interface for sending Terminal API requests to the ACP Client.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use sacp::Client;
use sacp::ConnectionTo;
use sacp::schema::{
    CreateTerminalRequest, CreateTerminalResponse, EnvVariable, KillTerminalRequest,
    KillTerminalResponse, ReleaseTerminalRequest, ReleaseTerminalResponse, SessionId, TerminalId,
    TerminalOutputRequest, TerminalOutputResponse, WaitForTerminalExitRequest,
    WaitForTerminalExitResponse,
};
use tracing::instrument;

use crate::types::AgentError;

/// Terminal API client for sending terminal requests to the ACP Client
///
/// The Client (editor like Zed) manages the actual PTY, and this client
/// sends requests through the ACP protocol to create, manage, and interact
/// with terminals.
#[derive(Debug, Clone)]
pub struct TerminalClient {
    /// Connection context for sending requests
    connection_cx: ConnectionTo<Client>,
    /// Session ID for this client
    session_id: SessionId,
}

impl TerminalClient {
    /// Create a new Terminal API client
    pub fn new(connection_cx: ConnectionTo<Client>, session_id: impl Into<SessionId>) -> Self {
        Self {
            connection_cx,
            session_id: session_id.into(),
        }
    }

    /// Create a new terminal and execute a command
    ///
    /// Returns a `TerminalId` that can be used with other terminal methods.
    /// The terminal will execute the specified command and capture output.
    ///
    /// # Arguments
    ///
    /// * `command` - The command to execute
    /// * `args` - Command arguments
    /// * `cwd` - Optional working directory (uses session cwd if not specified)
    /// * `output_byte_limit` - Optional limit on output bytes to retain
    #[instrument(
        name = "terminal_create",
        skip(self, command, args, cwd),
        fields(
            session_id = %self.session_id.0,
            args_count = args.len(),
            has_cwd = cwd.is_some(),
        )
    )]
    pub async fn create(
        &self,
        command: impl Into<String>,
        args: Vec<String>,
        cwd: Option<PathBuf>,
        output_byte_limit: Option<u64>,
    ) -> Result<TerminalId, AgentError> {
        let start_time = Instant::now();
        let cmd: String = command.into();

        tracing::info!(
            command = %cmd,
            args = ?args,
            cwd = ?cwd,
            output_byte_limit = ?output_byte_limit,
            "Creating terminal and executing command"
        );

        let mut request = CreateTerminalRequest::new(self.session_id.clone(), cmd.clone());
        request = request.args(args.clone());

        // Set CLAUDECODE environment variable (required by some clients like Zed)
        request = request.env(vec![EnvVariable::new("CLAUDECODE", "1")]);

        if let Some(cwd_path) = cwd.clone() {
            request = request.cwd(cwd_path);
        }

        if let Some(limit) = output_byte_limit {
            request = request.output_byte_limit(limit);
        }

        tracing::debug!("Sending terminal/create request to ACP client");

        let response: CreateTerminalResponse = self
            .connection_cx
            .send_request(request)
            .block_task()
            .await
            .map_err(|e| {
                let elapsed = start_time.elapsed();
                tracing::error!(
                    session_id = %self.session_id.0,
                    command = %cmd,
                    error = %e,
                    error_type = %std::any::type_name::<sacp::Error>(),
                    elapsed_ms = elapsed.as_millis(),
                    "Terminal create request failed"
                );
                AgentError::Internal(format!("Terminal create failed: {}", e))
            })?;

        let elapsed = start_time.elapsed();
        tracing::info!(
            terminal_id = %response.terminal_id.0,
            command = %cmd,
            elapsed_ms = elapsed.as_millis(),
            "Terminal created successfully"
        );

        Ok(response.terminal_id)
    }

    /// Get the current output and status of a terminal
    ///
    /// Returns the output captured so far and the exit status if completed.
    #[instrument(
        name = "terminal_output",
        skip(self, terminal_id),
        fields(session_id = %self.session_id.0)
    )]
    pub async fn output(
        &self,
        terminal_id: impl Into<TerminalId>,
    ) -> Result<TerminalOutputResponse, AgentError> {
        let start_time = Instant::now();
        let tid: TerminalId = terminal_id.into();

        tracing::debug!(
            terminal_id = %tid.0,
            "Getting terminal output"
        );

        let request = TerminalOutputRequest::new(self.session_id.clone(), tid.clone());

        let response = self
            .connection_cx
            .send_request(request)
            .block_task()
            .await
            .map_err(|e| {
                let elapsed = start_time.elapsed();
                tracing::error!(
                    terminal_id = %tid.0,
                    error = %e,
                    elapsed_ms = elapsed.as_millis(),
                    "Terminal output request failed"
                );
                AgentError::Internal(format!("Terminal output failed: {}", e))
            })?;

        let elapsed = start_time.elapsed();
        tracing::debug!(
            terminal_id = %tid.0,
            elapsed_ms = elapsed.as_millis(),
            output_len = response.output.len(),
            exit_status = ?response.exit_status,
            "Terminal output retrieved"
        );

        Ok(response)
    }

    /// Wait for a terminal command to exit
    ///
    /// Blocks until the command completes and returns the exit status.
    #[instrument(
        name = "terminal_wait_for_exit",
        skip(self, terminal_id),
        fields(session_id = %self.session_id.0)
    )]
    pub async fn wait_for_exit(
        &self,
        terminal_id: impl Into<TerminalId>,
    ) -> Result<WaitForTerminalExitResponse, AgentError> {
        let start_time = Instant::now();
        let tid: TerminalId = terminal_id.into();

        tracing::info!(
            terminal_id = %tid.0,
            "Waiting for terminal command to exit"
        );

        let request = WaitForTerminalExitRequest::new(self.session_id.clone(), tid.clone());

        let response = self
            .connection_cx
            .send_request(request)
            .block_task()
            .await
            .map_err(|e| {
                let elapsed = start_time.elapsed();
                tracing::error!(
                    terminal_id = %tid.0,
                    error = %e,
                    elapsed_ms = elapsed.as_millis(),
                    "Terminal wait_for_exit failed"
                );
                AgentError::Internal(format!("Terminal wait_for_exit failed: {}", e))
            })?;

        let elapsed = start_time.elapsed();
        tracing::info!(
            terminal_id = %tid.0,
            elapsed_ms = elapsed.as_millis(),
            exit_status = ?response.exit_status,
            "Terminal command exited"
        );

        Ok(response)
    }

    /// Kill a terminal command
    ///
    /// Sends SIGTERM to terminate the command. The terminal remains valid
    /// and can be queried for output or released.
    #[instrument(
        name = "terminal_kill",
        skip(self, terminal_id),
        fields(session_id = %self.session_id.0)
    )]
    pub async fn kill(
        &self,
        terminal_id: impl Into<TerminalId>,
    ) -> Result<KillTerminalResponse, AgentError> {
        let start_time = Instant::now();
        let tid: TerminalId = terminal_id.into();

        tracing::info!(
            terminal_id = %tid.0,
            "Killing terminal command"
        );

        let request = KillTerminalRequest::new(self.session_id.clone(), tid.clone());

        let response = self
            .connection_cx
            .send_request(request)
            .block_task()
            .await
            .map_err(|e| {
                let elapsed = start_time.elapsed();
                tracing::error!(
                    terminal_id = %tid.0,
                    error = %e,
                    elapsed_ms = elapsed.as_millis(),
                    "Terminal kill failed"
                );
                AgentError::Internal(format!("Terminal kill failed: {}", e))
            })?;

        let elapsed = start_time.elapsed();
        tracing::info!(
            terminal_id = %tid.0,
            elapsed_ms = elapsed.as_millis(),
            "Terminal command killed"
        );

        Ok(response)
    }

    /// Release a terminal and free its resources
    ///
    /// After release, the `TerminalId` can no longer be used.
    /// Any unretrieved output will be lost.
    #[instrument(
        name = "terminal_release",
        skip(self, terminal_id),
        fields(session_id = %self.session_id.0)
    )]
    pub async fn release(
        &self,
        terminal_id: impl Into<TerminalId>,
    ) -> Result<ReleaseTerminalResponse, AgentError> {
        let start_time = Instant::now();
        let tid: TerminalId = terminal_id.into();

        tracing::debug!(
            terminal_id = %tid.0,
            "Releasing terminal"
        );

        let request = ReleaseTerminalRequest::new(self.session_id.clone(), tid.clone());

        let response = self
            .connection_cx
            .send_request(request)
            .block_task()
            .await
            .map_err(|e| {
                let elapsed = start_time.elapsed();
                tracing::error!(
                    terminal_id = %tid.0,
                    error = %e,
                    elapsed_ms = elapsed.as_millis(),
                    "Terminal release failed"
                );
                AgentError::Internal(format!("Terminal release failed: {}", e))
            })?;

        let elapsed = start_time.elapsed();
        tracing::debug!(
            terminal_id = %tid.0,
            elapsed_ms = elapsed.as_millis(),
            "Terminal released"
        );

        Ok(response)
    }

    /// Get the session ID
    pub fn session_id(&self) -> &SessionId {
        &self.session_id
    }

    /// Create an Arc-wrapped client for sharing
    pub fn into_arc(self) -> Arc<Self> {
        Arc::new(self)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_terminal_client_session_id() {
        // We can't easily test without a real connection, but we can verify the struct compiles
        // and the session_id method works (would need mock connection for full test)
    }
}
