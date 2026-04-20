//! Tool registry for managing MCP tools

use std::collections::HashMap;
use std::sync::Arc;

use sacp::Client;
use sacp::ConnectionTo;
use sacp::schema::{
    SessionId, SessionNotification, SessionUpdate, Terminal, ToolCallContent, ToolCallId,
    ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields,
};
use serde::{Deserialize, Serialize};

use super::tools::Tool;
use crate::session::BackgroundProcessManager;
use crate::settings::PermissionChecker;
use crate::terminal::TerminalClient;

/// Tool execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// Result status
    pub status: ToolStatus,
    /// Output content
    pub content: String,
    /// Whether this is an error
    pub is_error: bool,
    /// Additional metadata
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

impl ToolResult {
    /// Create a successful result
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            status: ToolStatus::Success,
            content: content.into(),
            is_error: false,
            metadata: None,
        }
    }

    /// Create an error result
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            status: ToolStatus::Error,
            content: message.into(),
            is_error: true,
            metadata: None,
        }
    }

    /// Create a result with metadata
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// Tool execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolStatus {
    /// Tool executed successfully
    Success,
    /// Tool execution failed
    Error,
    /// Tool execution was cancelled
    Cancelled,
    /// Tool is still running (for async operations)
    Running,
}

/// Tool execution context
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// Session ID
    pub session_id: String,
    /// Working directory
    pub cwd: std::path::PathBuf,
    /// Whether to allow dangerous operations
    pub allow_dangerous: bool,
    /// Background process manager
    background_processes: Option<Arc<BackgroundProcessManager>>,
    /// Terminal client for executing commands via Client PTY
    terminal_client: Option<Arc<TerminalClient>>,
    /// Current tool use ID (for sending mid-execution updates)
    tool_use_id: Option<String>,
    /// Connection context for sending notifications
    connection_cx: Option<ConnectionTo<Client>>,
    /// Permission checker for tool-level permission checks
    pub permission_checker: Option<Arc<tokio::sync::RwLock<PermissionChecker>>>,
}

impl ToolContext {
    /// Create a new tool context
    pub fn new(session_id: impl Into<String>, cwd: impl Into<std::path::PathBuf>) -> Self {
        Self {
            session_id: session_id.into(),
            cwd: cwd.into(),
            allow_dangerous: false,
            background_processes: None,
            terminal_client: None,
            tool_use_id: None,
            connection_cx: None,
            permission_checker: None,
        }
    }

    /// Set whether dangerous operations are allowed
    pub fn with_dangerous(mut self, allow: bool) -> Self {
        self.allow_dangerous = allow;
        self
    }

    /// Set the background process manager
    pub fn with_background_processes(mut self, manager: Arc<BackgroundProcessManager>) -> Self {
        self.background_processes = Some(manager);
        self
    }

    /// Set the terminal client
    pub fn with_terminal_client(mut self, client: Arc<TerminalClient>) -> Self {
        self.terminal_client = Some(client);
        self
    }

    /// Set the current tool use ID
    pub fn with_tool_use_id(mut self, id: impl Into<String>) -> Self {
        self.tool_use_id = Some(id.into());
        self
    }

    /// Set the connection context for sending notifications
    pub fn with_connection_cx(mut self, cx: ConnectionTo<Client>) -> Self {
        self.connection_cx = Some(cx);
        self
    }

    /// Set the permission checker for tool-level permission checks
    pub fn with_permission_checker(
        mut self,
        checker: Arc<tokio::sync::RwLock<PermissionChecker>>,
    ) -> Self {
        self.permission_checker = Some(checker);
        self
    }

    /// Get the background process manager
    pub fn background_processes(&self) -> Option<&Arc<BackgroundProcessManager>> {
        self.background_processes.as_ref()
    }

    /// Get the terminal client
    ///
    /// When available, tools can use this to execute commands via the Client's PTY
    /// instead of directly spawning processes.
    pub fn terminal_client(&self) -> Option<&Arc<TerminalClient>> {
        self.terminal_client.as_ref()
    }

    /// Get the current tool use ID
    pub fn tool_use_id(&self) -> Option<&str> {
        self.tool_use_id.as_deref()
    }

    /// Send a ToolCallUpdate notification with Terminal content
    ///
    /// This is used by tools like Bash to send terminal ID immediately after
    /// creating a terminal, so the client can start showing terminal output.
    ///
    /// # Arguments
    ///
    /// * `terminal_id` - The terminal ID from CreateTerminalResponse
    /// * `status` - The tool call status (usually InProgress)
    /// * `title` - Optional title/description for the tool call
    ///
    /// # Returns
    ///
    /// `Ok(())` if notification was sent, `Err` if context doesn't have connection
    pub fn send_terminal_update(
        &self,
        terminal_id: impl Into<String>,
        status: ToolCallStatus,
        title: Option<&str>,
    ) -> Result<(), String> {
        let Some(connection_cx) = &self.connection_cx else {
            return Err("No connection context available".to_string());
        };

        let Some(tool_use_id) = &self.tool_use_id else {
            return Err("No tool use ID available".to_string());
        };

        // Build terminal content
        let terminal = Terminal::new(terminal_id.into());
        let content = vec![ToolCallContent::Terminal(terminal)];

        // Build update fields
        let mut update_fields = ToolCallUpdateFields::new().status(status).content(content);

        if let Some(title) = title {
            update_fields = update_fields.title(title);
        }

        // Build and send notification
        let tool_call_id = ToolCallId::new(tool_use_id.clone());
        let update = ToolCallUpdate::new(tool_call_id, update_fields);
        let notification = SessionNotification::new(
            SessionId::new(self.session_id.as_str()),
            SessionUpdate::ToolCallUpdate(update),
        );

        connection_cx
            .send_notification(notification)
            .map_err(|e| format!("Failed to send notification: {}", e))
    }
}

/// ACP tool prefix for compatibility with TypeScript implementation
pub const ACP_TOOL_PREFIX: &str = "mcp__acp__";

/// Tool registry for managing available tools
#[derive(Debug, Default)]
pub struct ToolRegistry {
    /// Registered tools by name
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool
    pub fn register<T: Tool + 'static>(&mut self, tool: T) {
        let name = tool.name().to_string();
        self.tools.insert(name, Arc::new(tool));
    }

    /// Register a tool as Arc
    pub fn register_arc(&mut self, tool: Arc<dyn Tool>) {
        let name = tool.name().to_string();
        self.tools.insert(name, tool);
    }

    /// Get a tool by name, supporting ACP prefix
    ///
    /// If the tool name starts with `mcp__acp__`, it will try to find
    /// the tool with the prefix stripped.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        // Try direct lookup first
        if let Some(tool) = self.tools.get(name) {
            return Some(tool.clone());
        }

        // Try stripping ACP prefix
        if let Some(stripped) = name.strip_prefix(ACP_TOOL_PREFIX) {
            if let Some(tool) = self.tools.get(stripped) {
                return Some(tool.clone());
            }
        }

        None
    }

    /// Check if a tool exists, supporting ACP prefix
    pub fn contains(&self, name: &str) -> bool {
        if self.tools.contains_key(name) {
            return true;
        }

        // Try stripping ACP prefix
        if let Some(stripped) = name.strip_prefix(ACP_TOOL_PREFIX) {
            return self.tools.contains_key(stripped);
        }

        false
    }

    /// Normalize a tool name by stripping ACP prefix if present
    pub fn normalize_name(name: &str) -> &str {
        name.strip_prefix(ACP_TOOL_PREFIX).unwrap_or(name)
    }

    /// Get all tool names
    pub fn names(&self) -> Vec<&str> {
        self.tools.keys().map(String::as_str).collect()
    }

    /// Get the number of registered tools
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Execute a tool by name
    pub async fn execute(
        &self,
        name: &str,
        input: serde_json::Value,
        context: &ToolContext,
    ) -> ToolResult {
        match self.get(name) {
            Some(tool) => tool.execute(input, context).await,
            None => ToolResult::error(format!("Tool not found: {}", name)),
        }
    }

    /// Get tool schemas for all registered tools
    pub fn schemas(&self) -> Vec<ToolSchema> {
        self.tools
            .values()
            .map(|tool| ToolSchema {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                input_schema: tool.input_schema(),
            })
            .collect()
    }
}

/// Tool schema for registration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: String,
    /// JSON Schema for input
    pub input_schema: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_tool_result_success() {
        let result = ToolResult::success("Hello, World!");
        assert_eq!(result.status, ToolStatus::Success);
        assert_eq!(result.content, "Hello, World!");
        assert!(!result.is_error);
    }

    #[test]
    fn test_tool_result_error() {
        let result = ToolResult::error("Something went wrong");
        assert_eq!(result.status, ToolStatus::Error);
        assert_eq!(result.content, "Something went wrong");
        assert!(result.is_error);
    }

    #[test]
    fn test_tool_result_with_metadata() {
        let result = ToolResult::success("data").with_metadata(json!({"lines": 10}));
        assert!(result.metadata.is_some());
    }

    #[test]
    fn test_tool_context() {
        let ctx = ToolContext::new("session-1", "/tmp").with_dangerous(true);
        assert_eq!(ctx.session_id, "session-1");
        assert_eq!(ctx.cwd, std::path::PathBuf::from("/tmp"));
        assert!(ctx.allow_dangerous);
    }

    #[test]
    fn test_registry_new() {
        let registry = ToolRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_acp_prefix_normalize() {
        // Without prefix
        assert_eq!(ToolRegistry::normalize_name("Read"), "Read");
        assert_eq!(ToolRegistry::normalize_name("Bash"), "Bash");

        // With prefix
        assert_eq!(ToolRegistry::normalize_name("mcp__acp__Read"), "Read");
        assert_eq!(ToolRegistry::normalize_name("mcp__acp__Bash"), "Bash");
        assert_eq!(
            ToolRegistry::normalize_name("mcp__acp__TodoWrite"),
            "TodoWrite"
        );
    }

    #[test]
    fn test_acp_prefix_constant() {
        assert_eq!(ACP_TOOL_PREFIX, "mcp__acp__");
    }
}
