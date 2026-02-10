//! Claude SDK Message to ACP SessionNotification conversion
//!
//! Converts SDK messages (assistant, system, result, stream events)
//! into ACP session notifications for the client.

use std::time::Instant;

use claude_code_agent_sdk::{
    AssistantMessage, ContentBlock as SdkContentBlock, ImageBlock, ImageSource, Message,
    ResultMessage, StreamEvent, ToolResultBlock, ToolResultContent, ToolUseBlock,
};
use dashmap::DashMap;
use regex::Regex;
use sacp::schema::{
    ContentBlock as AcpContentBlock, ContentChunk, Diff, ImageContent, Plan, PlanEntry,
    PlanEntryPriority, PlanEntryStatus, SessionId, SessionNotification, SessionUpdate, Terminal,
    TextContent, ToolCall, ToolCallContent, ToolCallId, ToolCallLocation, ToolCallStatus,
    ToolCallUpdate, ToolCallUpdateFields, ToolKind as AcpToolKind,
};

use crate::types::{ToolKind, ToolUseEntry};

use super::extract_tool_info;

/// Static regex for finding backtick sequences at start of lines
/// Used by markdown_escape to determine the appropriate escape sequence
static BACKTICK_REGEX: std::sync::LazyLock<Regex> =
    std::sync::LazyLock::new(|| Regex::new(r"(?m)^```+").expect("valid backtick regex"));

/// Static regex for removing SYSTEM_REMINDER blocks
/// Matches <system-reminder>...</system-reminder> including multiline content
static SYSTEM_REMINDER_REGEX: std::sync::LazyLock<Regex> = std::sync::LazyLock::new(|| {
    Regex::new(r"(?s)<system-reminder>.*?</system-reminder>").expect("valid system-reminder regex")
});

/// Wrap text in markdown code block with appropriate number of backticks
///
/// Ensures the text is safely wrapped by using more backticks than any sequence
/// found in the text itself.
///
/// Reference: vendors/claude-code-acp/src/tools.ts:591-599
fn markdown_escape(text: &str) -> String {
    let mut escape = "```".to_string();

    // Find all sequences of backticks at the start of lines
    for cap in BACKTICK_REGEX.captures_iter(text) {
        let m = cap.get(0).expect("match exists").as_str();
        while m.len() >= escape.len() {
            escape.push('`');
        }
    }

    // Build the final string
    let needs_newline = !text.ends_with('\n');
    format!(
        "{}\n{}{}{}",
        escape,
        text,
        if needs_newline { "\n" } else { "" },
        escape
    )
}

/// Remove SYSTEM_REMINDER tags and their content from text
///
/// Reference: vendors/claude-code-acp/src/tools.ts:430-431
fn remove_system_reminders(text: &str) -> String {
    SYSTEM_REMINDER_REGEX.replace_all(text, "").to_string()
}

/// Notification converter for transforming SDK messages to ACP notifications
///
/// Maintains a cache of tool uses to correlate tool_use blocks with their results.
#[derive(Debug)]
pub struct NotificationConverter {
    /// Cache of tool use entries, keyed by tool_use_id
    tool_use_cache: DashMap<String, ToolUseEntry>,
    /// Current working directory for relative path display
    cwd: Option<std::path::PathBuf>,
    /// Optional request_id for tracking prompt requests
    request_id: Option<String>,
}

impl Default for NotificationConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl NotificationConverter {
    /// Create a new notification converter
    pub fn new() -> Self {
        Self {
            tool_use_cache: DashMap::new(),
            cwd: None,
            request_id: None,
        }
    }

    /// Create a new notification converter with working directory
    ///
    /// # Arguments
    ///
    /// * `cwd` - The current working directory for computing relative paths
    pub fn with_cwd(cwd: std::path::PathBuf) -> Self {
        Self {
            tool_use_cache: DashMap::new(),
            cwd: Some(cwd),
            request_id: None,
        }
    }

    /// Set the request_id for this converter
    ///
    /// The request_id will be attached to all SessionNotification instances
    /// created by this converter, allowing clients to track which responses
    /// correspond to which requests.
    ///
    /// # Arguments
    ///
    /// * `request_id` - The unique request identifier
    pub fn set_request_id(&mut self, request_id: String) {
        self.request_id = Some(request_id);
    }

    /// Clear the request_id
    pub fn clear_request_id(&mut self) {
        self.request_id = None;
    }

    /// Attach request_id to a notification if one is set
    fn attach_request_id(&self, notification: SessionNotification) -> SessionNotification {
        if let Some(ref req_id) = self.request_id {
            // Build Meta (serde_json::Map) with request_id
            let mut meta = serde_json::Map::new();
            meta.insert("request_id".to_string(), serde_json::json!(req_id));
            notification.meta(meta)
        } else {
            notification
        }
    }

    /// Convert a SDK Message to ACP session update notifications
    ///
    /// # Arguments
    ///
    /// * `message` - The SDK message to convert
    /// * `session_id` - The session ID for the notifications
    ///
    /// # Returns
    ///
    /// A vector of ACP SessionNotification objects
    pub fn convert_message(&self, message: &Message, session_id: &str) -> Vec<SessionNotification> {
        let start_time = Instant::now();

        // Determine message type for logging
        let message_type = match message {
            Message::Assistant(_) => "Assistant",
            Message::StreamEvent(_) => "StreamEvent",
            Message::Result(_) => "Result",
            Message::System(_) => "System",
            Message::User(_) => "User",
            Message::ControlCancelRequest(_) => "ControlCancelRequest",
        };

        let sid = SessionId::new(session_id.to_string());
        let notifications = match message {
            Message::Assistant(assistant) => self.convert_assistant_message(assistant, &sid),
            Message::StreamEvent(event) => self.convert_stream_event(event, &sid),
            Message::Result(result) => self.convert_result_message(result, &sid),
            Message::System(_) => {
                // System messages are typically internal, not sent as notifications
                vec![]
            }
            Message::User(_) => {
                // User messages are echoed back, usually not needed
                vec![]
            }
            Message::ControlCancelRequest(_) => {
                // Internal control messages
                vec![]
            }
        };

        let elapsed = start_time.elapsed();
        let output_count = notifications.len();

        tracing::trace!(
            message_type = %message_type,
            session_id = %session_id,
            output_count = output_count,
            conversion_duration_us = elapsed.as_micros(),
            "Message conversion completed"
        );

        notifications
    }

    /// Convert an assistant message
    ///
    /// Note: In streaming mode, Text and Thinking blocks are delivered via
    /// content_block_delta events (StreamEvent), so we skip them here to avoid
    /// sending the same content twice. Only ToolUse and ToolResult blocks are
    /// processed from non-streamed messages.
    fn convert_assistant_message(
        &self,
        assistant: &AssistantMessage,
        session_id: &SessionId,
    ) -> Vec<SessionNotification> {
        let mut notifications = Vec::new();

        for block in &assistant.message.content {
            match block {
                // Skip Text and Thinking blocks in streaming mode
                // They are delivered via StreamEvent::content_block_delta
                SdkContentBlock::Text(_) => {
                    // Skip - handled by stream events
                }
                SdkContentBlock::Thinking(_) => {
                    // Skip - handled by stream events
                }
                SdkContentBlock::ToolUse(tool_use) => {
                    // Cache the tool use for later correlation with result
                    self.cache_tool_use(tool_use);
                    // Special handling for TodoWrite: send Plan instead of ToolCall
                    // Reference: vendors/claude-code-acp/src/acp-agent.ts lines 1051-1058
                    let effective_name = tool_use
                        .name
                        .strip_prefix("mcp__acp__")
                        .unwrap_or(&tool_use.name);
                    if effective_name == "TodoWrite" {
                        if let Some(notification) =
                            self.make_plan_from_todo_write(session_id, tool_use)
                        {
                            notifications.push(notification);
                            continue;
                        }
                    }
                    notifications.push(self.make_tool_call(session_id, tool_use));
                }
                SdkContentBlock::ToolResult(tool_result) => {
                    notifications.extend(self.make_tool_result(session_id, tool_result));
                }
                SdkContentBlock::Image(image) => {
                    // Convert SDK Image to ACP Image notification
                    // Reference: vendors/claude-code-acp/src/acp-agent.ts lines 1027-1035
                    notifications.push(self.make_image_message(session_id, image));
                }
            }
        }

        notifications
    }

    /// Convert a stream event (incremental updates)
    #[allow(clippy::unused_self)]
    fn convert_stream_event(
        &self,
        event: &StreamEvent,
        session_id: &SessionId,
    ) -> Vec<SessionNotification> {
        let event_type = event.event.get("type").and_then(|v| v.as_str());

        match event_type {
            Some("content_block_start") => {
                // Handle content_block_start - important for tool calls and results
                // When streaming is enabled, tool_use and tool_result blocks arrive via content_block_start
                if let Some(content_block) = event.event.get("content_block") {
                    if let Some(block_type) = content_block.get("type").and_then(|v| v.as_str()) {
                        // Handle tool_use types
                        // Reference: vendors/claude-code-acp/src/acp-agent.ts lines 1047-1049
                        if matches!(block_type, "tool_use" | "server_tool_use" | "mcp_tool_use") {
                            match serde_json::from_value::<ToolUseBlock>(content_block.clone()) {
                                Ok(tool_use) => {
                                    self.cache_tool_use(&tool_use);
                                    // Special handling for TodoWrite: send Plan instead of ToolCall
                                    // Reference: vendors/claude-code-acp/src/acp-agent.ts lines 1051-1058
                                    let effective_name = tool_use
                                        .name
                                        .strip_prefix("mcp__acp__")
                                        .unwrap_or(&tool_use.name);
                                    if effective_name == "TodoWrite" {
                                        if let Some(notification) =
                                            self.make_plan_from_todo_write(session_id, &tool_use)
                                        {
                                            return vec![notification];
                                        }
                                    }
                                    return vec![self.make_tool_call(session_id, &tool_use)];
                                }
                                Err(e) => {
                                    tracing::error!(
                                        session_id = %session_id.0,
                                        block_type = %block_type,
                                        error = %e,
                                        "Failed to parse tool_use block"
                                    );
                                }
                            }
                        }
                        // Handle tool_result types
                        // Reference: vendors/claude-code-acp/src/acp-agent.ts lines 1109-1116
                        else if matches!(
                            block_type,
                            "tool_result"
                                | "mcp_tool_result"
                                | "tool_search_tool_result"
                                | "web_fetch_tool_result"
                                | "web_search_tool_result"
                                | "code_execution_tool_result"
                                | "bash_code_execution_tool_result"
                                | "text_editor_code_execution_tool_result"
                        ) {
                            match serde_json::from_value::<ToolResultBlock>(content_block.clone()) {
                                Ok(tool_result) => {
                                    return self.make_tool_result(session_id, &tool_result);
                                }
                                Err(e) => {
                                    tracing::error!(
                                        session_id = %session_id.0,
                                        block_type = %block_type,
                                        error = %e,
                                        "Failed to parse tool_result block"
                                    );
                                }
                            }
                        }
                        // Handle image type
                        // Reference: vendors/claude-code-acp/src/acp-agent.ts lines 1027-1035
                        else if block_type == "image" {
                            match serde_json::from_value::<ImageBlock>(content_block.clone()) {
                                Ok(image) => {
                                    return vec![self.make_image_message(session_id, &image)];
                                }
                                Err(e) => {
                                    tracing::error!(
                                        session_id = %session_id.0,
                                        block_type = %block_type,
                                        error = %e,
                                        "Failed to parse image block"
                                    );
                                }
                            }
                        }
                        // Skip known non-notification types
                        // Reference: vendors/claude-code-acp/src/acp-agent.ts lines 1141-1148
                        else if matches!(
                            block_type,
                            "text"
                                | "thinking"
                                | "document"
                                | "search_result"
                                | "redacted_thinking"
                                | "input_json_delta"
                                | "citations_delta"
                                | "signature_delta"
                                | "container_upload"
                        ) {
                            // These are handled elsewhere or not needed as notifications
                        }
                        // Log unknown block types (like TS's unreachable)
                        else {
                            tracing::warn!(
                                session_id = %session_id.0,
                                block_type = %block_type,
                                content_block = ?content_block,
                                "Unknown content_block type in content_block_start"
                            );
                        }
                    }
                }
                vec![]
            }
            Some("content_block_delta") => {
                if let Some(delta) = event.event.get("delta") {
                    if let Some(delta_type) = delta.get("type").and_then(|v| v.as_str()) {
                        match delta_type {
                            "text_delta" => {
                                if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                                    return vec![self.make_agent_message_chunk(session_id, text)];
                                }
                            }
                            "thinking_delta" => {
                                if let Some(thinking) =
                                    delta.get("thinking").and_then(|v| v.as_str())
                                {
                                    return vec![
                                        self.make_agent_thought_chunk(session_id, thinking),
                                    ];
                                }
                            }
                            // Skip known delta types that don't need notifications
                            "input_json_delta" | "citations_delta" | "signature_delta" => {}
                            // Log unknown delta types
                            _ => {
                                tracing::debug!(
                                    session_id = %session_id.0,
                                    delta_type = %delta_type,
                                    "Unknown delta type in content_block_delta"
                                );
                            }
                        }
                    } else {
                        // Fallback for delta without explicit type field
                        if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                            return vec![self.make_agent_message_chunk(session_id, text)];
                        }
                        if let Some(thinking) = delta.get("thinking").and_then(|v| v.as_str()) {
                            return vec![self.make_agent_thought_chunk(session_id, thinking)];
                        }
                    }
                }
                vec![]
            }
            // No content needed for these events
            Some("content_block_stop" | "message_start" | "message_delta" | "message_stop") => {
                vec![]
            }
            // Log unknown event types (like TS's unreachable)
            Some(unknown_type) => {
                tracing::warn!(
                    session_id = %session_id.0,
                    event_type = %unknown_type,
                    "Unknown stream event type"
                );
                vec![]
            }
            None => vec![],
        }
    }

    /// Convert a result message
    fn convert_result_message(
        &self,
        _result: &ResultMessage,
        _session_id: &SessionId,
    ) -> Vec<SessionNotification> {
        // Result messages update usage statistics but don't typically
        // generate notifications (the prompt response handles completion)
        vec![]
    }

    /// Cache a tool use entry
    fn cache_tool_use(&self, tool_use: &ToolUseBlock) {
        let entry = ToolUseEntry::new(
            tool_use.id.clone(),
            tool_use.name.clone(),
            tool_use.input.clone(),
        );
        self.tool_use_cache.insert(tool_use.id.clone(), entry);
    }

    /// Get a cached tool use entry
    pub fn get_tool_use(&self, tool_use_id: &str) -> Option<ToolUseEntry> {
        self.tool_use_cache.get(tool_use_id).map(|r| r.clone())
    }

    /// Remove a cached tool use entry
    pub fn remove_tool_use(&self, tool_use_id: &str) -> Option<ToolUseEntry> {
        self.tool_use_cache.remove(tool_use_id).map(|(_, v)| v)
    }

    /// Clear all cached tool uses
    pub fn clear_cache(&self) {
        self.tool_use_cache.clear();
    }

    // === Notification builders ===

    /// Make an agent message notification (full text as chunk)
    ///
    /// Currently unused because Text blocks are skipped in convert_assistant_message
    /// to avoid duplication with stream events.
    #[allow(dead_code, clippy::unused_self)]
    fn make_agent_message(&self, session_id: &SessionId, text: &str) -> SessionNotification {
        // Use AgentMessageChunk since there's no AgentMessage variant
        let notification = SessionNotification::new(
            session_id.clone(),
            SessionUpdate::AgentMessageChunk(ContentChunk::new(AcpContentBlock::Text(
                TextContent::new(text),
            ))),
        );
        self.attach_request_id(notification)
    }

    /// Make an agent message chunk notification (incremental)
    #[allow(clippy::unused_self)]
    fn make_agent_message_chunk(&self, session_id: &SessionId, chunk: &str) -> SessionNotification {
        let notification = SessionNotification::new(
            session_id.clone(),
            SessionUpdate::AgentMessageChunk(ContentChunk::new(AcpContentBlock::Text(
                TextContent::new(chunk),
            ))),
        );
        self.attach_request_id(notification)
    }

    /// Make an agent thought notification (full thought as chunk)
    ///
    /// Currently unused because Thinking blocks are skipped in convert_assistant_message
    /// to avoid duplication with stream events.
    #[allow(dead_code, clippy::unused_self)]
    fn make_agent_thought(&self, session_id: &SessionId, thought: &str) -> SessionNotification {
        // Use AgentThoughtChunk since there's no separate thought variant
        let notification = SessionNotification::new(
            session_id.clone(),
            SessionUpdate::AgentThoughtChunk(ContentChunk::new(AcpContentBlock::Text(
                TextContent::new(thought),
            ))),
        );
        self.attach_request_id(notification)
    }

    /// Make an agent thought chunk notification (incremental)
    #[allow(clippy::unused_self)]
    fn make_agent_thought_chunk(&self, session_id: &SessionId, chunk: &str) -> SessionNotification {
        let notification = SessionNotification::new(
            session_id.clone(),
            SessionUpdate::AgentThoughtChunk(ContentChunk::new(AcpContentBlock::Text(
                TextContent::new(chunk),
            ))),
        );
        self.attach_request_id(notification)
    }

    /// Make an image message notification
    ///
    /// Converts SDK ImageBlock to ACP ImageContent and wraps in AgentMessageChunk.
    /// Reference: vendors/claude-code-acp/src/acp-agent.ts lines 1027-1035
    #[allow(clippy::unused_self)]
    fn make_image_message(
        &self,
        session_id: &SessionId,
        image: &ImageBlock,
    ) -> SessionNotification {
        let (data, mime_type, uri) = match &image.source {
            ImageSource::Base64 { media_type, data } => (data.clone(), media_type.clone(), None),
            ImageSource::Url { url } => {
                // For URL-based images, data is empty and uri is set
                (String::new(), String::new(), Some(url.clone()))
            }
        };

        let image_content = ImageContent::new(data, mime_type).uri(uri);

        let notification = SessionNotification::new(
            session_id.clone(),
            SessionUpdate::AgentMessageChunk(ContentChunk::new(AcpContentBlock::Image(
                image_content,
            ))),
        );
        self.attach_request_id(notification)
    }

    /// Map local ToolKind to ACP ToolKind
    fn map_tool_kind(kind: ToolKind) -> AcpToolKind {
        match kind {
            ToolKind::Read => AcpToolKind::Read,
            ToolKind::Edit => AcpToolKind::Edit,
            ToolKind::Execute => AcpToolKind::Execute,
            ToolKind::Search => AcpToolKind::Search,
            ToolKind::Think => AcpToolKind::Think,
            ToolKind::Fetch => AcpToolKind::Fetch,
            ToolKind::SwitchMode | ToolKind::Other => AcpToolKind::default(),
        }
    }

    /// Make a tool call notification
    #[allow(clippy::unused_self)]
    fn make_tool_call(
        &self,
        session_id: &SessionId,
        tool_use: &ToolUseBlock,
    ) -> SessionNotification {
        let tool_info = extract_tool_info(&tool_use.name, &tool_use.input, self.cwd.as_ref());

        let tool_call_id = ToolCallId::new(tool_use.id.clone());
        let tool_kind = Self::map_tool_kind(tool_info.kind);

        // For Bash tool, include command in title if description is not available
        let title = if tool_use.name == "Bash" {
            // Get description or command
            let description = tool_use.input.get("description").and_then(|v| v.as_str());
            let command = tool_use.input.get("command").and_then(|v| v.as_str());

            match (description, command) {
                (Some(desc), _) => desc.to_string(),
                (None, Some(cmd)) => {
                    // Truncate long commands for display (UTF-8 safe)
                    if cmd.len() > 80 {
                        let boundary = cmd
                            .char_indices()
                            .map(|(i, _)| i)
                            .take_while(|&i| i <= 77)
                            .last()
                            .unwrap_or(0);
                        format!("{}...", &cmd[..boundary])
                    } else {
                        cmd.to_string()
                    }
                }
                _ => tool_info.title.clone(),
            }
        } else {
            tool_info.title.clone()
        };

        // Debug: Log tool call creation
        tracing::debug!(
            tool_call_id = %tool_use.id,
            tool_name = %tool_use.name,
            title = %title,
            session_id = %session_id.0,
            "Creating ToolCall notification for session/update"
        );

        let mut tool_call = ToolCall::new(tool_call_id, &title)
            .kind(tool_kind)
            .status(ToolCallStatus::InProgress) // Tool is being executed
            .raw_input(tool_use.input.clone());

        // Add locations if present
        if let Some(ref locations) = tool_info.locations
            && !locations.is_empty()
        {
            let acp_locations: Vec<ToolCallLocation> = locations
                .iter()
                .map(|loc| {
                    let mut location = ToolCallLocation::new(&loc.path);
                    if let Some(line) = loc.line {
                        location = location.line(line);
                    }
                    location
                })
                .collect();
            tool_call = tool_call.locations(acp_locations);
        }

        let notification =
            SessionNotification::new(session_id.clone(), SessionUpdate::ToolCall(tool_call));
        self.attach_request_id(notification)
    }

    /// Make tool result notifications
    ///
    /// Returns a vector of notifications:
    /// - ToolCallUpdate for all tools
    /// - Plan notification for TodoWrite tool (when successful)
    fn make_tool_result(
        &self,
        session_id: &SessionId,
        tool_result: &ToolResultBlock,
    ) -> Vec<SessionNotification> {
        let Some(entry) = self.get_tool_use(&tool_result.tool_use_id) else {
            // Tool call not found in cache - this can happen if:
            // 1. Tool was never cached (bug in streaming handling)
            // 2. Tool was already processed and removed
            // 3. Duplicate tool result received
            tracing::warn!(
                session_id = %session_id.0,
                tool_use_id = %tool_result.tool_use_id,
                "Tool call not found in cache, skipping tool result notification"
            );
            return vec![];
        };

        tracing::debug!(
            session_id = %session_id.0,
            tool_use_id = %tool_result.tool_use_id,
            tool_name = %entry.name,
            "Processing tool result notification"
        );

        let output = match &tool_result.content {
            Some(ToolResultContent::Text(text)) => text.clone(),
            Some(ToolResultContent::Blocks(blocks)) => {
                serde_json::to_string(blocks).unwrap_or_default()
            }
            None => String::new(),
        };

        let is_error = tool_result.is_error.unwrap_or(false);
        let status = if is_error {
            ToolCallStatus::Failed
        } else {
            ToolCallStatus::Completed
        };

        // Build raw_output JSON
        let raw_output = serde_json::json!({
            "content": output,
            "is_error": is_error
        });

        // Build content based on tool type
        let content = self.build_tool_result_content(&entry, &output, is_error);

        let tool_call_id = ToolCallId::new(tool_result.tool_use_id.clone());
        let update_fields = ToolCallUpdateFields::new()
            .status(status)
            .content(content)
            .raw_output(raw_output);
        let update = ToolCallUpdate::new(tool_call_id, update_fields);

        let notification =
            SessionNotification::new(session_id.clone(), SessionUpdate::ToolCallUpdate(update));
        let notifications = vec![self.attach_request_id(notification)];

        // Note: Plan notification for TodoWrite is now sent at tool_use time
        // (in make_plan_from_todo_write), so we don't send it here anymore.
        // This matches TypeScript behavior: acp-agent.ts lines 1051-1058

        notifications
    }

    /// Make a Plan notification from TodoWrite tool_use
    ///
    /// This is called when we receive a TodoWrite tool_use, to send the Plan
    /// immediately (instead of waiting for tool_result).
    /// Reference: vendors/claude-code-acp/src/acp-agent.ts lines 1051-1058
    #[allow(clippy::unused_self)]
    fn make_plan_from_todo_write(
        &self,
        session_id: &SessionId,
        tool_use: &ToolUseBlock,
    ) -> Option<SessionNotification> {
        // Extract todos from input
        let todos = tool_use.input.get("todos")?.as_array()?;

        let plan_entries: Vec<PlanEntry> = todos
            .iter()
            .filter_map(|todo| {
                let content = todo.get("content")?.as_str()?;
                let status_str = todo.get("status")?.as_str()?;

                // Convert TodoWrite status to PlanEntryStatus
                let status = match status_str {
                    "in_progress" => PlanEntryStatus::InProgress,
                    "completed" => PlanEntryStatus::Completed,
                    _ => PlanEntryStatus::Pending,
                };

                // TodoWrite doesn't have priority, default to Medium
                Some(PlanEntry::new(content, PlanEntryPriority::Medium, status))
            })
            .collect();

        if plan_entries.is_empty() {
            return None;
        }

        let plan = Plan::new(plan_entries);
        let notification = SessionNotification::new(session_id.clone(), SessionUpdate::Plan(plan));
        Some(self.attach_request_id(notification))
    }

    /// Build tool result content based on tool type
    ///
    /// For Edit/Write tools, returns Diff content.
    /// For Read tool, removes SYSTEM_REMINDER and wraps with markdown.
    /// For errors, wraps with markdown code block.
    /// Reference: vendors/claude-code-acp/src/tools.ts toolUpdateFromToolResult
    fn build_tool_result_content(
        &self,
        entry: &ToolUseEntry,
        output: &str,
        is_error: bool,
    ) -> Vec<ToolCallContent> {
        // Strip mcp__acp__ prefix for matching
        let effective_name = entry.name.strip_prefix("mcp__acp__").unwrap_or(&entry.name);

        match effective_name {
            "Edit" if !is_error => {
                // Extract file_path, old_string, new_string from input
                let file_path = entry
                    .input
                    .get("file_path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let old_string = entry
                    .input
                    .get("old_string")
                    .and_then(|v| v.as_str())
                    .map(String::from);
                let new_string = entry
                    .input
                    .get("new_string")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                if !file_path.is_empty() && !new_string.is_empty() {
                    // Create structured Diff content for Zed to render as visual diff
                    // Reference: vendors/claude-code-acp/src/tools.ts:460-507
                    let diff = Diff::new(file_path, new_string).old_text(old_string);
                    vec![ToolCallContent::Diff(diff)]
                } else {
                    vec![output.to_string().into()]
                }
            }
            "Write" if !is_error => {
                // Extract file_path and content from input
                let file_path = entry
                    .input
                    .get("file_path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let content = entry
                    .input
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                if file_path.is_empty() {
                    vec![output.to_string().into()]
                } else {
                    // For Write, old_text is None (new file) or we don't have it
                    let diff = Diff::new(file_path, content);
                    vec![ToolCallContent::Diff(diff)]
                }
            }
            "Read" if !is_error => {
                // Remove SYSTEM_REMINDER and wrap with markdown
                // Reference: vendors/claude-code-acp/src/tools.ts:430-431
                let cleaned = remove_system_reminders(output);
                let wrapped = markdown_escape(&cleaned);
                vec![wrapped.into()]
            }
            _ if is_error => {
                // Wrap errors with markdown code block
                // Reference: vendors/claude-code-acp/src/tools.ts:553-556
                let wrapped = format!("```\n{}\n```", output);
                vec![wrapped.into()]
            }
            _ => {
                // Default: text content
                vec![output.to_string().into()]
            }
        }
    }

    /// Build Terminal content for embedding a terminal in tool result
    ///
    /// This is used when a tool (like Bash) uses the Terminal API to execute commands.
    /// The terminal_id is obtained from the `terminal/create` response.
    ///
    /// # Arguments
    ///
    /// * `terminal_id` - The terminal ID from CreateTerminalResponse
    ///
    /// # Returns
    ///
    /// A `ToolCallContent::Terminal` that can be included in tool results
    pub fn build_terminal_content(terminal_id: impl Into<String>) -> ToolCallContent {
        let terminal = Terminal::new(terminal_id.into());
        ToolCallContent::Terminal(terminal)
    }

    /// Build a ToolCallUpdate notification with Terminal content
    ///
    /// This is used when a Bash command is executed via the Terminal API.
    /// The client will embed the terminal output based on the terminal_id.
    ///
    /// # Arguments
    ///
    /// * `session_id` - The session ID
    /// * `tool_use_id` - The tool use ID for the Bash call
    /// * `terminal_id` - The terminal ID from terminal/create
    /// * `status` - The tool call status
    ///
    /// # Returns
    ///
    /// A SessionNotification with the ToolCallUpdate
    pub fn make_terminal_result(
        &self,
        session_id: &SessionId,
        tool_use_id: &str,
        terminal_id: impl Into<String>,
        status: ToolCallStatus,
    ) -> SessionNotification {
        let terminal_content = Self::build_terminal_content(terminal_id);
        let tool_call_id = ToolCallId::new(tool_use_id.to_string());
        let update_fields = ToolCallUpdateFields::new()
            .status(status)
            .content(vec![terminal_content]);
        let update = ToolCallUpdate::new(tool_call_id, update_fields);

        let notification =
            SessionNotification::new(session_id.clone(), SessionUpdate::ToolCallUpdate(update));
        self.attach_request_id(notification)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_converter_new() {
        let converter = NotificationConverter::new();
        assert!(converter.tool_use_cache.is_empty());
    }

    #[test]
    fn test_cache_tool_use() {
        let converter = NotificationConverter::new();
        let tool_use = ToolUseBlock {
            id: "tool_123".to_string(),
            name: "Read".to_string(),
            input: json!({"file_path": "/test.txt"}),
        };

        converter.cache_tool_use(&tool_use);

        let cached = converter.get_tool_use("tool_123");
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().name, "Read");
    }

    #[test]
    fn test_make_agent_message() {
        let converter = NotificationConverter::new();
        let session_id = SessionId::new("session-1");
        let notification = converter.make_agent_message(&session_id, "Hello!");

        assert_eq!(notification.session_id.0.as_ref(), "session-1");
        // Check that it's an AgentMessageChunk update
        assert!(matches!(
            notification.update,
            SessionUpdate::AgentMessageChunk(_)
        ));
    }

    #[test]
    fn test_make_agent_message_chunk() {
        let converter = NotificationConverter::new();
        let session_id = SessionId::new("session-1");
        let notification = converter.make_agent_message_chunk(&session_id, "chunk");

        // Check that it's an AgentMessageChunk update
        assert!(matches!(
            notification.update,
            SessionUpdate::AgentMessageChunk(_)
        ));
    }

    #[test]
    fn test_make_agent_thought() {
        let converter = NotificationConverter::new();
        let session_id = SessionId::new("session-1");
        let notification = converter.make_agent_thought(&session_id, "thinking...");

        // Check that it's an AgentThoughtChunk update
        assert!(matches!(
            notification.update,
            SessionUpdate::AgentThoughtChunk(_)
        ));
    }

    #[test]
    fn test_make_tool_call() {
        let converter = NotificationConverter::new();
        let session_id = SessionId::new("session-1");
        let tool_use = ToolUseBlock {
            id: "tool_456".to_string(),
            name: "Bash".to_string(),
            input: json!({"command": "ls", "description": "List files"}),
        };

        let notification = converter.make_tool_call(&session_id, &tool_use);

        // Check that it's a ToolCall update
        assert!(matches!(notification.update, SessionUpdate::ToolCall(_)));
        if let SessionUpdate::ToolCall(tool_call) = &notification.update {
            assert_eq!(tool_call.tool_call_id.0.as_ref(), "tool_456");
        }
    }

    #[test]
    fn test_remove_tool_use() {
        let converter = NotificationConverter::new();
        let tool_use = ToolUseBlock {
            id: "tool_789".to_string(),
            name: "Edit".to_string(),
            input: json!({}),
        };

        converter.cache_tool_use(&tool_use);
        assert!(converter.get_tool_use("tool_789").is_some());

        let removed = converter.remove_tool_use("tool_789");
        assert!(removed.is_some());
        assert!(converter.get_tool_use("tool_789").is_none());
    }

    #[test]
    fn test_map_tool_kind() {
        assert!(matches!(
            NotificationConverter::map_tool_kind(ToolKind::Read),
            AcpToolKind::Read
        ));
        assert!(matches!(
            NotificationConverter::map_tool_kind(ToolKind::Edit),
            AcpToolKind::Edit
        ));
        assert!(matches!(
            NotificationConverter::map_tool_kind(ToolKind::Execute),
            AcpToolKind::Execute
        ));
    }

    #[test]
    fn test_make_plan_from_todo_write() {
        let converter = NotificationConverter::new();
        let session_id = SessionId::new("session-1");

        // Create a TodoWrite tool use
        let tool_use = ToolUseBlock {
            id: "todo_123".to_string(),
            name: "TodoWrite".to_string(),
            input: json!({
                "todos": [
                    {
                        "content": "Implement feature",
                        "status": "in_progress",
                        "activeForm": "Implementing feature"
                    },
                    {
                        "content": "Write tests",
                        "status": "pending",
                        "activeForm": "Writing tests"
                    },
                    {
                        "content": "Setup project",
                        "status": "completed",
                        "activeForm": "Setting up project"
                    }
                ]
            }),
        };

        let notification = converter.make_plan_from_todo_write(&session_id, &tool_use);

        assert!(notification.is_some());
        let notification = notification.unwrap();

        if let SessionUpdate::Plan(plan) = &notification.update {
            assert_eq!(plan.entries.len(), 3);
            assert_eq!(plan.entries[0].content, "Implement feature");
            assert_eq!(plan.entries[0].status, PlanEntryStatus::InProgress);
            assert_eq!(plan.entries[1].content, "Write tests");
            assert_eq!(plan.entries[1].status, PlanEntryStatus::Pending);
            assert_eq!(plan.entries[2].content, "Setup project");
            assert_eq!(plan.entries[2].status, PlanEntryStatus::Completed);
        } else {
            panic!("Expected Plan update");
        }
    }

    #[test]
    fn test_make_tool_result_todowrite_no_duplicate_plan() {
        // Since Plan is now sent at tool_use time, tool_result should NOT include Plan
        let converter = NotificationConverter::new();
        let session_id = SessionId::new("session-1");

        // Cache a TodoWrite tool use
        let tool_use = ToolUseBlock {
            id: "todo_456".to_string(),
            name: "TodoWrite".to_string(),
            input: json!({
                "todos": [
                    {
                        "content": "Task 1",
                        "status": "pending",
                        "activeForm": "Doing task 1"
                    }
                ]
            }),
        };
        converter.cache_tool_use(&tool_use);

        // Create a tool result
        let tool_result = ToolResultBlock {
            tool_use_id: "todo_456".to_string(),
            content: Some(ToolResultContent::Text("Todos updated".to_string())),
            is_error: Some(false),
        };

        let notifications = converter.make_tool_result(&session_id, &tool_result);

        // Should have only 1 notification: ToolCallUpdate (no Plan, since Plan is sent at tool_use time)
        assert_eq!(notifications.len(), 1);
        assert!(matches!(
            notifications[0].update,
            SessionUpdate::ToolCallUpdate(_)
        ));
    }

    #[test]
    fn test_build_terminal_content() {
        let content = NotificationConverter::build_terminal_content("term-123");
        match content {
            ToolCallContent::Terminal(terminal) => {
                assert_eq!(terminal.terminal_id.0.as_ref(), "term-123");
            }
            _ => panic!("Expected Terminal content"),
        }
    }

    #[test]
    fn test_make_terminal_result() {
        let converter = NotificationConverter::new();
        let session_id = SessionId::new("session-1");

        let notification = converter.make_terminal_result(
            &session_id,
            "tool_789",
            "term-456",
            ToolCallStatus::Completed,
        );

        assert_eq!(notification.session_id.0.as_ref(), "session-1");
        if let SessionUpdate::ToolCallUpdate(update) = &notification.update {
            assert_eq!(update.tool_call_id.0.as_ref(), "tool_789");
            // Check content contains Terminal
            let fields = &update.fields;
            let content = fields.content.as_ref().expect("content should exist");
            assert_eq!(content.len(), 1);
            match &content[0] {
                ToolCallContent::Terminal(terminal) => {
                    assert_eq!(terminal.terminal_id.0.as_ref(), "term-456");
                }
                _ => panic!("Expected Terminal content"),
            }
        } else {
            panic!("Expected ToolCallUpdate");
        }
    }

    #[test]
    fn test_request_id_propagation() {
        let mut converter = NotificationConverter::new();
        let session_id = SessionId::new("session-1");

        // Without request_id, notification should not have meta
        let notification = converter.make_agent_message_chunk(&session_id, "test");
        assert!(notification.meta.is_none());

        // Set request_id
        converter.set_request_id("req-123".to_string());

        // Now notifications should have request_id in meta
        let notification = converter.make_agent_message_chunk(&session_id, "test");
        assert!(notification.meta.is_some());
        if let Some(meta) = &notification.meta {
            assert_eq!(meta.get("request_id"), Some(&serde_json::json!("req-123")));
        }
    }

    #[test]
    fn test_request_id_clear() {
        let mut converter = NotificationConverter::new();
        let session_id = SessionId::new("session-1");

        // Set request_id
        converter.set_request_id("req-456".to_string());
        let notification = converter.make_agent_message_chunk(&session_id, "test");
        assert!(notification.meta.is_some());

        // Clear request_id
        converter.clear_request_id();
        let notification = converter.make_agent_message_chunk(&session_id, "test");
        assert!(notification.meta.is_none());
    }

    #[test]
    fn test_request_id_propagation_all_notification_types() {
        let mut converter = NotificationConverter::new();
        let session_id = SessionId::new("session-1");

        // Set request_id
        let test_request_id = "test-req-789";
        converter.set_request_id(test_request_id.to_string());

        // Verify all notification types include request_id
        let notifications = vec![
            converter.make_agent_message_chunk(&session_id, "test"),
            converter.make_agent_thought_chunk(&session_id, "thinking"),
            converter.make_tool_call(
                &session_id,
                &ToolUseBlock {
                    id: "tool-1".to_string(),
                    name: "TestTool".to_string(),
                    input: serde_json::json!({}),
                },
            ),
        ];

        for notification in notifications {
            assert!(
                notification.meta.is_some(),
                "Notification should have meta when request_id is set"
            );
            if let Some(meta) = &notification.meta {
                assert_eq!(
                    meta.get("request_id"),
                    Some(&serde_json::json!(test_request_id)),
                    "Meta should contain the correct request_id"
                );
            }
        }
    }

    #[test]
    fn test_request_id_with_converter_with_cwd() {
        let mut converter = NotificationConverter::with_cwd(std::path::PathBuf::from("/test"));
        let session_id = SessionId::new("session-1");

        // Initially no request_id
        let notification = converter.make_agent_message_chunk(&session_id, "test");
        assert!(notification.meta.is_none());

        // Set request_id
        converter.set_request_id("req-cwd-test".to_string());

        // Now should have request_id
        let notification = converter.make_agent_message_chunk(&session_id, "test");
        assert!(notification.meta.is_some());
        if let Some(meta) = &notification.meta {
            assert_eq!(
                meta.get("request_id"),
                Some(&serde_json::json!("req-cwd-test"))
            );
        }
    }

    #[test]
    fn test_request_id_default() {
        let converter = NotificationConverter::new();
        // Default request_id should be None
        assert!(converter.request_id.is_none());
    }

    #[test]
    fn test_request_id_with_cwd_default() {
        let converter = NotificationConverter::with_cwd(std::path::PathBuf::from("/test"));
        // Default request_id should be None even with cwd
        assert!(converter.request_id.is_none());
    }
}
