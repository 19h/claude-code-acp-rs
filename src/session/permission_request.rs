//! Interactive permission request handling
//!
//! Implements the ACP permission request/response protocol for asking users
//! whether to allow tool execution.

use sacp::Client;
use sacp::ConnectionTo;
use sacp::schema::{
    PermissionOption, PermissionOptionId, PermissionOptionKind, RequestPermissionOutcome,
    RequestPermissionRequest, SessionId, ToolCallUpdate, ToolCallUpdateFields,
};

use crate::types::AgentError;

/// Permission request outcome after user interaction
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PermissionOutcome {
    /// User allowed this tool call (one-time)
    AllowOnce,
    /// User allowed this tool call and wants to always allow this pattern
    AllowAlways,
    /// User rejected this tool call
    Rejected,
    /// Permission request was cancelled
    Cancelled,
}

/// Builder for creating permission requests
#[derive(Debug)]
pub struct PermissionRequestBuilder {
    session_id: String,
    tool_call_id: String,
    title: String,
    tool_name: String,
    tool_input: serde_json::Value,
}

impl PermissionRequestBuilder {
    /// Create a new permission request builder
    pub fn new(
        session_id: impl Into<String>,
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        tool_input: serde_json::Value,
    ) -> Self {
        let tool_name_str: String = tool_name.into();
        let title = format_tool_title(&tool_name_str, &tool_input);
        Self {
            session_id: session_id.into(),
            tool_call_id: tool_call_id.into(),
            title,
            tool_name: tool_name_str,
            tool_input,
        }
    }

    /// Set a custom title for the permission dialog
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Build the request and send it to the client
    ///
    /// Returns the user's decision as a `PermissionOutcome`.
    pub async fn request(
        self,
        connection_cx: &ConnectionTo<Client>,
    ) -> Result<PermissionOutcome, AgentError> {
        // Build the options
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
            self.tool_call_id.clone(),
            ToolCallUpdateFields::new()
                .title(&self.title)
                .raw_input(self.tool_input.clone()),
        );

        // Debug: Log the tool call update being sent
        tracing::debug!(
            tool_call_id = %self.tool_call_id,
            title = %self.title,
            tool_name = %self.tool_name,
            "Building permission request with ToolCallUpdate"
        );

        // Build the request
        let request = RequestPermissionRequest::new(
            SessionId::new(self.session_id.clone()),
            tool_call_update,
            options,
        );

        // Debug: Log the serialized request for protocol debugging
        if let Ok(json) = serde_json::to_string_pretty(&request) {
            tracing::trace!(
                session_id = %self.session_id,
                request_json = %json,
                "Sending session/request_permission"
            );
        }

        // Send request and wait for response
        tracing::info!(
            tool_call_id = %self.tool_call_id,
            session_id = %self.session_id,
            "Sending permission request, waiting for user response..."
        );

        let response = connection_cx
            .send_request(request)
            .block_task()
            .await
            .map_err(|e| {
                tracing::error!(
                    tool_call_id = %self.tool_call_id,
                    error = %e,
                    "Permission request failed"
                );
                AgentError::Internal(format!("Permission request failed: {}", e))
            })?;

        tracing::info!(
            tool_call_id = %self.tool_call_id,
            "Received permission response"
        );

        // Parse the response
        Ok(parse_permission_response(response.outcome))
    }

    /// Get the tool name
    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }
}

/// Parse a permission response outcome into our outcome type
fn parse_permission_response(outcome: RequestPermissionOutcome) -> PermissionOutcome {
    match outcome {
        RequestPermissionOutcome::Selected(selected) => {
            match selected.option_id.0.as_ref() {
                "allow_always" => PermissionOutcome::AllowAlways,
                "allow_once" => PermissionOutcome::AllowOnce,
                "reject_once" => PermissionOutcome::Rejected,
                _ => PermissionOutcome::Rejected, // Unknown option, treat as reject
            }
        }
        RequestPermissionOutcome::Cancelled => PermissionOutcome::Cancelled,
        // Handle any future variants (non_exhaustive enum)
        _ => PermissionOutcome::Cancelled,
    }
}

/// Format a title for the permission dialog based on tool name and input
fn format_tool_title(tool_name: &str, input: &serde_json::Value) -> String {
    // Strip mcp__acp__ prefix if present
    let stripped_name = tool_name.strip_prefix("mcp__acp__").unwrap_or(tool_name);

    match stripped_name {
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
        "Grep" => {
            let pattern = input.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
            format!("Search: {}", pattern)
        }
        "Glob" => {
            let pattern = input.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
            format!("Find files: {}", pattern)
        }
        _ => stripped_name.to_string(),
    }
}

/// Truncate a string to max length, adding "..." if truncated
fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sacp::schema::SelectedPermissionOutcome;
    use serde_json::json;

    #[test]
    fn test_format_tool_title_read() {
        let title = format_tool_title("Read", &json!({"file_path": "/tmp/test.txt"}));
        assert_eq!(title, "Read /tmp/test.txt");
    }

    #[test]
    fn test_format_tool_title_bash() {
        let title = format_tool_title("Bash", &json!({"command": "ls -la"}));
        assert_eq!(title, "Run: ls -la");

        let title = format_tool_title(
            "Bash",
            &json!({"command": "ls -la", "description": "List files"}),
        );
        assert_eq!(title, "List files");
    }

    #[test]
    fn test_format_tool_title_long_command() {
        let long_cmd = "echo 'this is a very long command that should be truncated'";
        let title = format_tool_title("Bash", &json!({"command": long_cmd}));
        assert!(title.len() <= 60); // "Run: " + 50 chars + "..."
        assert!(title.ends_with("..."));
    }

    #[test]
    fn test_truncate_string() {
        assert_eq!(truncate_string("hello", 10), "hello");
        assert_eq!(truncate_string("hello world", 8), "hello...");
        assert_eq!(truncate_string("hi", 2), "hi");
    }

    #[test]
    fn test_permission_outcome_selected() {
        // Test Selected outcomes
        let selected_always = RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
            PermissionOptionId::new("allow_always"),
        ));
        assert_eq!(
            parse_permission_response(selected_always),
            PermissionOutcome::AllowAlways
        );

        let selected_once = RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
            PermissionOptionId::new("allow_once"),
        ));
        assert_eq!(
            parse_permission_response(selected_once),
            PermissionOutcome::AllowOnce
        );

        let selected_reject = RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
            PermissionOptionId::new("reject_once"),
        ));
        assert_eq!(
            parse_permission_response(selected_reject),
            PermissionOutcome::Rejected
        );
    }

    #[test]
    fn test_permission_outcome_cancelled() {
        let cancelled = RequestPermissionOutcome::Cancelled;
        assert_eq!(
            parse_permission_response(cancelled),
            PermissionOutcome::Cancelled
        );
    }

    #[test]
    fn test_permission_outcome_unknown() {
        // Unknown option should be treated as rejected
        let unknown = RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
            PermissionOptionId::new("unknown_option"),
        ));
        assert_eq!(
            parse_permission_response(unknown),
            PermissionOutcome::Rejected
        );
    }
}
