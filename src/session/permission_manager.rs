//! Permission Manager - Async permission handling for MCP tools
//!
//! Based on Zed's permission system pattern:
//! - Hook sends permission request and returns immediately
//! - Background task handles the request
//! - Uses unbounded channels (never block)
//! - Uses one-shot channels for request/response

use std::sync::Arc;

use sacp::JrConnectionCx;
use sacp::link::AgentToClient;
use sacp::schema::{
    PermissionOption, PermissionOptionId, PermissionOptionKind, RequestPermissionOutcome,
    RequestPermissionRequest, SessionId, ToolCallUpdate, ToolCallUpdateFields,
};

use crate::types::AgentError;

/// Permission decision result
#[derive(Debug, Clone, PartialEq)]
pub enum PermissionManagerDecision {
    /// User allowed this tool call (one-time)
    AllowOnce,
    /// User allowed this tool call and wants to always allow this pattern
    AllowAlways,
    /// User rejected this tool call
    Rejected,
    /// Permission request was cancelled
    Cancelled,
}

/// Pending permission request from hook
pub struct PendingPermissionRequest {
    pub tool_name: String,
    pub tool_input: serde_json::Value,
    pub tool_call_id: String,
    pub session_id: String,
    pub response_tx: tokio::sync::oneshot::Sender<PermissionManagerDecision>,
}

impl std::fmt::Debug for PendingPermissionRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingPermissionRequest")
            .field("tool_name", &self.tool_name)
            .field("tool_input", &self.tool_input)
            .field("tool_call_id", &self.tool_call_id)
            .field("session_id", &self.session_id)
            .field("response_tx", &"<oneshot::Sender>")
            .finish()
    }
}

/// Permission Manager - handles permission requests in background tasks
///
/// # Architecture
///
/// Based on Zed's async permission pattern:
/// 1. Hook sends request via unbounded channel (never blocks)
/// 2. Background task processes request
/// 3. One-shot channel returns result to caller
///
/// # Example
///
/// ```rust,ignore
/// let manager = PermissionManager::new(connection_cx);
/// let rx = manager.request_permission("Edit", input, "call_123", "session_456");
/// let decision = rx.await?;
/// ```
pub struct PermissionManager {
    /// Pending permission requests (unbounded, never blocks on send)
    pending_requests: tokio::sync::mpsc::UnboundedSender<PendingPermissionRequest>,
}

impl std::fmt::Debug for PermissionManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PermissionManager")
            .field("pending_requests", &"<mpsc::UnboundedSender>")
            .finish()
    }
}

impl PermissionManager {
    /// Create a new PermissionManager
    pub fn new(connection_cx: Arc<JrConnectionCx<AgentToClient>>) -> Self {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        // Spawn background task to handle permission requests
        let cx = connection_cx.clone();
        tokio::spawn(async move {
            Self::handle_permission_requests(rx, cx).await;
        });

        Self {
            pending_requests: tx,
        }
    }

    /// Request permission (non-blocking)
    ///
    /// Returns a receiver that will resolve when user responds.
    ///
    /// This never blocks - it immediately sends to the background task
    /// and returns a receiver for the result.
    pub fn request_permission(
        &self,
        tool_name: String,
        tool_input: serde_json::Value,
        tool_call_id: String,
        session_id: String,
    ) -> tokio::sync::oneshot::Receiver<PermissionManagerDecision> {
        let (tx, rx) = tokio::sync::oneshot::channel();

        let request = PendingPermissionRequest {
            tool_name,
            tool_input,
            tool_call_id,
            session_id,
            response_tx: tx,
        };

        // Send to background task (unbounded channel never blocks)
        if let Err(err) = self.pending_requests.send(request) {
            tracing::error!("Permission request channel closed, background task may have panicked");
            // Return Cancelled via oneshot so the caller doesn't hang forever
            let _ = err.0.response_tx.send(PermissionManagerDecision::Cancelled);
        }

        rx
    }

    /// Background task: handle permission requests
    async fn handle_permission_requests(
        mut receiver: tokio::sync::mpsc::UnboundedReceiver<PendingPermissionRequest>,
        connection_cx: Arc<JrConnectionCx<AgentToClient>>,
    ) {
        while let Some(request) = receiver.recv().await {
            tracing::info!(
                tool_name = %request.tool_name,
                tool_call_id = %request.tool_call_id,
                "Processing permission request in background task"
            );

            // Send permission request to client via SACP and wait for response
            let decision = match Self::send_permission_request_to_client(
                &connection_cx,
                &request.tool_name,
                &request.tool_input,
                &request.tool_call_id,
                &request.session_id,
            )
            .await
            {
                Ok(decision) => {
                    tracing::info!(
                        tool_name = %request.tool_name,
                        tool_call_id = %request.tool_call_id,
                        decision = ?decision,
                        "Permission decision received from client"
                    );
                    decision
                }
                Err(e) => {
                    tracing::error!(
                        tool_name = %request.tool_name,
                        tool_call_id = %request.tool_call_id,
                        error = %e,
                        "Failed to send permission request to client, defaulting to Cancelled"
                    );
                    PermissionManagerDecision::Cancelled
                }
            };

            let _ = request.response_tx.send(decision);
        }
    }

    /// Send permission request to client via SACP
    async fn send_permission_request_to_client(
        connection_cx: &JrConnectionCx<AgentToClient>,
        tool_name: &str,
        tool_input: &serde_json::Value,
        tool_call_id: &str,
        session_id: &str,
    ) -> Result<PermissionManagerDecision, AgentError> {
        // Build the permission options
        let options = vec![
            PermissionOption::new(
                PermissionOptionId::new("allow_always"),
                "Always Allow",
                PermissionOptionKind::AllowAlways,
            ),
            PermissionOption::new(
                PermissionOptionId::new("allow_once"),
                "Allow",
                PermissionOptionKind::AllowOnce,
            ),
            PermissionOption::new(
                PermissionOptionId::new("reject_once"),
                "Reject",
                PermissionOptionKind::RejectOnce,
            ),
        ];

        // Build the tool call update with title
        let tool_call_update = ToolCallUpdate::new(
            tool_call_id.to_string(),
            ToolCallUpdateFields::new()
                .title(format_tool_title(tool_name, tool_input))
                .raw_input(tool_input.clone()),
        );

        // Build the request
        let request =
            RequestPermissionRequest::new(SessionId::new(session_id), tool_call_update, options);

        // Send request and wait for response
        let response = connection_cx
            .send_request(request)
            .block_task()
            .await
            .map_err(|e| AgentError::Internal(format!("Permission request failed: {}", e)))?;

        // Parse the response
        Ok(parse_permission_response(response.outcome))
    }
}

/// Parse a permission response outcome into our decision type
#[allow(dead_code)]
fn parse_permission_response(outcome: RequestPermissionOutcome) -> PermissionManagerDecision {
    match outcome {
        RequestPermissionOutcome::Selected(selected) => {
            match selected.option_id.0.as_ref() {
                "allow_always" => PermissionManagerDecision::AllowAlways,
                "allow_once" => PermissionManagerDecision::AllowOnce,
                "reject_once" => PermissionManagerDecision::Rejected,
                _ => PermissionManagerDecision::Rejected, // Unknown option, treat as reject
            }
        }
        RequestPermissionOutcome::Cancelled => PermissionManagerDecision::Cancelled,
        // Handle any future variants (non_exhaustive enum)
        _ => PermissionManagerDecision::Cancelled,
    }
}

/// Format a title for the permission dialog based on tool name and input
fn format_tool_title(tool_name: &str, input: &serde_json::Value) -> String {
    // Strip mcp__acp__ prefix for display
    let display_name = tool_name.strip_prefix("mcp__acp__").unwrap_or(tool_name);

    match display_name {
        "Read" => {
            let path = input
                .get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or("file");
            format!("Read {}", path)
        }
        "Write" => {
            let path = input
                .get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or("file");
            format!("Write to {}", path)
        }
        "Edit" => {
            let path = input
                .get("file_path")
                .and_then(|v| v.as_str())
                .unwrap_or("file");
            format!("Edit {}", path)
        }
        "Bash" => {
            let cmd = input.get("command").and_then(|v| v.as_str()).unwrap_or("");
            let desc = input.get("description").and_then(|v| v.as_str());
            desc.map(String::from)
                .unwrap_or_else(|| format!("Run: {}", truncate_string(cmd, 50)))
        }
        _ => display_name.to_string(),
    }
}

/// Truncate a string to max length, adding "..." if truncated
///
/// Uses char_indices for UTF-8 safe truncation.
fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        // Find the last char boundary at or before max_len - 3 (for "...")
        let boundary = s
            .char_indices()
            .map(|(i, _)| i)
            .take_while(|&i| i <= max_len.saturating_sub(3))
            .last()
            .unwrap_or(0);
        format!("{}...", &s[..boundary])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_tool_title_read() {
        let title = format_tool_title("Read", &serde_json::json!({"file_path": "/tmp/test.txt"}));
        assert_eq!(title, "Read /tmp/test.txt");
    }

    #[test]
    fn test_format_tool_title_edit() {
        let title = format_tool_title("Edit", &serde_json::json!({"file_path": "/tmp/file.txt"}));
        assert_eq!(title, "Edit /tmp/file.txt");
    }

    #[test]
    fn test_format_tool_title_mcp_prefix() {
        let title = format_tool_title(
            "mcp__acp__Read",
            &serde_json::json!({"file_path": "/tmp/test.txt"}),
        );
        assert_eq!(title, "Read /tmp/test.txt");
    }

    #[test]
    fn test_truncate_string() {
        assert_eq!(truncate_string("hello", 10), "hello");
        assert_eq!(truncate_string("hello world", 8), "hello...");
        assert_eq!(truncate_string("hi", 2), "hi");
    }

    #[test]
    fn test_truncate_string_utf8() {
        // Chinese characters are 3 bytes each; truncation must not panic
        let chinese = "你好世界测试数据";
        let result = truncate_string(chinese, 10);
        // Should not panic and should end with "..."
        assert!(result.ends_with("..."));
        assert!(result.len() <= 13); // at most 10 bytes of chars + 3 for "..."
    }

    #[test]
    fn test_parse_permission_response_selected() {
        // This would require constructing RequestPermissionOutcome
        // For now, just verify the function compiles
        let _ = parse_permission_response;
    }
}
