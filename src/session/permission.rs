//! Permission handling for tool execution
//!
//! This module provides permission checking using a strategy pattern,
//! where each permission mode has its own strategy implementation.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::permissions::strategies::{
    AcceptEditsModeStrategy, BypassPermissionsModeStrategy, DefaultModeStrategy,
    DontAskModeStrategy, PermissionModeStrategy, PlanModeStrategy,
};
use crate::settings::{PermissionChecker, PermissionDecision};
use claude_code_agent_sdk::PermissionMode as SdkPermissionMode;

/// Permission mode for tool execution
///
/// Controls how tool calls are approved during a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PermissionMode {
    /// Default mode - prompt for dangerous operations
    Default,
    /// Auto-approve file edits
    AcceptEdits,
    /// Planning mode - read-only operations
    Plan,
    /// Don't ask mode - deny if not pre-approved
    DontAsk,
    /// Bypass all permission checks (default mode for development)
    #[default]
    BypassPermissions,
}

impl PermissionMode {
    /// Parse from string (ACP setMode request)
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "default" => Some(Self::Default),
            "acceptEdits" => Some(Self::AcceptEdits),
            "plan" => Some(Self::Plan),
            "dontAsk" => Some(Self::DontAsk),
            "bypassPermissions" => Some(Self::BypassPermissions),
            _ => None,
        }
    }

    /// Convert to string for SDK
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::AcceptEdits => "acceptEdits",
            Self::Plan => "plan",
            Self::DontAsk => "dontAsk",
            Self::BypassPermissions => "bypassPermissions",
        }
    }

    /// Convert to SDK PermissionMode
    ///
    /// Note: SDK doesn't support DontAsk mode yet, so we map it to Default
    pub fn to_sdk_mode(&self) -> SdkPermissionMode {
        match self {
            PermissionMode::Default => SdkPermissionMode::Default,
            PermissionMode::AcceptEdits => SdkPermissionMode::AcceptEdits,
            PermissionMode::Plan => SdkPermissionMode::Plan,
            PermissionMode::DontAsk => {
                // SDK doesn't support DontAsk yet, treat as Default
                SdkPermissionMode::Default
            }
            PermissionMode::BypassPermissions => SdkPermissionMode::BypassPermissions,
        }
    }

    /// Check if this mode allows write operations
    pub fn allows_writes(&self) -> bool {
        matches!(
            self,
            Self::Default | Self::AcceptEdits | Self::BypassPermissions
        )
    }

    /// Check if this mode auto-approves edits
    pub fn auto_approve_edits(&self) -> bool {
        matches!(self, Self::AcceptEdits | Self::BypassPermissions)
    }
}

/// Permission check result from the handler
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolPermissionResult {
    /// Tool execution is allowed (auto-approved or by rule)
    Allowed,
    /// Tool execution is blocked (by rule or mode)
    Blocked { reason: String },
    /// User should be asked for permission
    NeedsPermission,
}

/// Permission handler for tool execution
///
/// Uses a strategy pattern where each permission mode has its own strategy.
pub struct PermissionHandler {
    mode: PermissionMode,
    /// Strategy for current permission mode
    strategy: Arc<dyn PermissionModeStrategy>,
    /// Shared permission checker from settings (shared with hook)
    checker: Option<Arc<RwLock<PermissionChecker>>>,
}

impl fmt::Debug for PermissionHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PermissionHandler")
            .field("mode", &self.mode)
            .field("strategy", &"<strategy>")
            .field("checker", &self.checker)
            .finish()
    }
}

impl Default for PermissionHandler {
    fn default() -> Self {
        Self {
            mode: PermissionMode::Default,
            strategy: Arc::new(DefaultModeStrategy),
            checker: None,
        }
    }
}

impl PermissionHandler {
    /// Create a new permission handler
    ///
    /// Uses Default mode (standard behavior with permission prompts).
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a specific mode
    pub fn with_mode(mode: PermissionMode) -> Self {
        Self {
            mode,
            strategy: Self::create_strategy(mode),
            checker: None,
        }
    }

    /// Create with settings-based checker
    ///
    /// Uses Default mode (standard behavior with permission prompts).
    pub fn with_checker(checker: Arc<RwLock<PermissionChecker>>) -> Self {
        Self {
            mode: PermissionMode::Default,
            strategy: Arc::new(DefaultModeStrategy),
            checker: Some(checker),
        }
    }

    /// Create with settings-based checker (non-async, for convenience)
    ///
    /// Uses Default mode (standard behavior with permission prompts).
    pub fn with_checker_owned(checker: PermissionChecker) -> Self {
        Self {
            mode: PermissionMode::Default,
            strategy: Arc::new(DefaultModeStrategy),
            checker: Some(Arc::new(RwLock::new(checker))),
        }
    }

    /// Create strategy for a given mode
    fn create_strategy(mode: PermissionMode) -> Arc<dyn PermissionModeStrategy> {
        match mode {
            PermissionMode::Default => Arc::new(DefaultModeStrategy),
            PermissionMode::AcceptEdits => Arc::new(AcceptEditsModeStrategy),
            PermissionMode::Plan => Arc::new(PlanModeStrategy),
            PermissionMode::DontAsk => Arc::new(DontAskModeStrategy),
            PermissionMode::BypassPermissions => Arc::new(BypassPermissionsModeStrategy),
        }
    }

    /// Get current permission mode
    pub fn mode(&self) -> PermissionMode {
        self.mode
    }

    /// Set permission mode
    pub fn set_mode(&mut self, mode: PermissionMode) {
        self.mode = mode;
        self.strategy = Self::create_strategy(mode);
    }

    /// Set the permission checker
    pub fn set_checker(&mut self, checker: Arc<RwLock<PermissionChecker>>) {
        self.checker = Some(checker);
    }

    /// Get mutable reference to checker (for adding runtime rules)
    pub async fn checker_mut(
        &mut self,
    ) -> Option<tokio::sync::RwLockWriteGuard<'_, PermissionChecker>> {
        if let Some(ref checker) = self.checker {
            Some(checker.write().await)
        } else {
            None
        }
    }

    /// Check if a tool operation should be auto-approved
    ///
    /// Returns true if the operation should proceed without user prompt.
    ///
    /// Delegates to the current strategy.
    pub fn should_auto_approve(&self, tool_name: &str, input: &serde_json::Value) -> bool {
        self.strategy.should_auto_approve(tool_name, input)
    }

    /// Check if a tool is blocked in current mode
    ///
    /// Returns true if the tool is blocked.
    ///
    /// Note: This method doesn't take tool_input, so it's less precise than
    /// the strategy method. For plan mode, it conservatively blocks all writes
    /// since it can't check if the file is in the plans directory.
    pub fn is_tool_blocked(&self, tool_name: &str) -> bool {
        self.strategy
            .is_tool_blocked(tool_name, &serde_json::Value::Null)
            .is_some()
    }

    /// Check permission for a tool with full context
    ///
    /// Combines strategy-based checking with settings rules.
    /// Returns the permission result.
    pub async fn check_permission(
        &self,
        tool_name: &str,
        tool_input: &serde_json::Value,
    ) -> ToolPermissionResult {
        // Check settings rules first (if available)
        if let Some(ref checker) = self.checker {
            let checker_read = checker.read().await;
            let result = checker_read.check_permission(tool_name, tool_input);
            match result.decision {
                PermissionDecision::Deny => {
                    return ToolPermissionResult::Blocked {
                        reason: result
                            .rule
                            .map(|r| format!("Denied by rule: {}", r))
                            .unwrap_or_else(|| "Denied by settings".to_string()),
                    };
                }
                PermissionDecision::Allow => {
                    return ToolPermissionResult::Allowed;
                }
                PermissionDecision::Ask => {
                    // Fall through to strategy-based check
                }
            }
        }

        // Use strategy for mode-specific logic
        let strategy_result = self.strategy.check_permission(tool_name, tool_input);

        // Special handling for DontAsk mode: convert NeedsPermission to Blocked
        if self.mode == PermissionMode::DontAsk {
            if strategy_result == ToolPermissionResult::NeedsPermission {
                return ToolPermissionResult::Blocked {
                    reason: "Tool not pre-approved by settings rules in DontAsk mode".to_string(),
                };
            }
        }

        // User interaction tools should always be allowed
        if matches!(
            tool_name,
            "AskUserQuestion" | "Task" | "TodoWrite" | "SlashCommand"
        ) {
            return ToolPermissionResult::Allowed;
        }

        strategy_result
    }

    /// Add a runtime allow rule (e.g., from user's "Always Allow" choice)
    pub async fn add_allow_rule(&self, tool_name: &str) {
        if let Some(ref checker) = self.checker {
            let mut checker_write = checker.write().await;
            checker_write.add_allow_rule(tool_name);
        }
    }

    /// Add a fine-grained allow rule based on tool call details
    /// This is used for "Always Allow" with specific parameters
    pub async fn add_allow_rule_for_tool_call(
        &self,
        tool_name: &str,
        tool_input: &serde_json::Value,
    ) {
        if let Some(ref checker) = self.checker {
            let mut checker_write = checker.write().await;
            checker_write.add_allow_rule_for_tool_call(tool_name, tool_input);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_permission_mode_parse() {
        assert_eq!(
            PermissionMode::parse("default"),
            Some(PermissionMode::Default)
        );
        assert_eq!(
            PermissionMode::parse("acceptEdits"),
            Some(PermissionMode::AcceptEdits)
        );
        assert_eq!(PermissionMode::parse("plan"), Some(PermissionMode::Plan));
        assert_eq!(
            PermissionMode::parse("bypassPermissions"),
            Some(PermissionMode::BypassPermissions)
        );
        assert_eq!(PermissionMode::parse("invalid"), None);
    }

    #[test]
    fn test_permission_mode_str() {
        assert_eq!(PermissionMode::Default.as_str(), "default");
        assert_eq!(PermissionMode::AcceptEdits.as_str(), "acceptEdits");
        assert_eq!(PermissionMode::Plan.as_str(), "plan");
        assert_eq!(
            PermissionMode::BypassPermissions.as_str(),
            "bypassPermissions"
        );
    }

    #[test]
    fn test_permission_handler_default() {
        let handler = PermissionHandler::new();
        let input = json!({});

        // Default mode auto-approves reads
        assert!(handler.should_auto_approve("Read", &input));
        assert!(handler.should_auto_approve("Glob", &input));
        assert!(handler.should_auto_approve("Grep", &input));
        assert!(handler.should_auto_approve("LS", &input));
        assert!(handler.should_auto_approve("NotebookRead", &input));
        // But not writes - these require permission
        assert!(!handler.should_auto_approve("Edit", &input));
        assert!(!handler.should_auto_approve("Bash", &input));
    }

    #[test]
    fn test_permission_handler_accept_edits() {
        let handler = PermissionHandler::with_mode(PermissionMode::AcceptEdits);
        let input = json!({});

        // AcceptEdits now auto-approves ALL tools (same as BypassPermissions)
        // This is needed for root user compatibility
        assert!(handler.should_auto_approve("Read", &input));
        assert!(handler.should_auto_approve("Edit", &input));
        assert!(handler.should_auto_approve("Write", &input));
        assert!(handler.should_auto_approve("Bash", &input));
    }

    #[test]
    fn test_permission_handler_bypass() {
        let handler = PermissionHandler::with_mode(PermissionMode::BypassPermissions);
        let input = json!({});

        // Everything auto-approved
        assert!(handler.should_auto_approve("Read", &input));
        assert!(handler.should_auto_approve("Edit", &input));
        assert!(handler.should_auto_approve("Bash", &input));
    }

    #[test]
    fn test_permission_handler_plan_mode() {
        let handler = PermissionHandler::with_mode(PermissionMode::Plan);
        let input = json!({});

        // Only reads auto-approved
        assert!(handler.should_auto_approve("Read", &input));
        assert!(handler.should_auto_approve("Glob", &input));
        assert!(handler.should_auto_approve("Grep", &input));
        assert!(handler.should_auto_approve("LS", &input));
        assert!(handler.should_auto_approve("NotebookRead", &input));
        assert!(!handler.should_auto_approve("Edit", &input));

        // Writes are blocked
        assert!(handler.is_tool_blocked("Edit"));
        assert!(handler.is_tool_blocked("Bash"));
        assert!(!handler.is_tool_blocked("Read"));
        assert!(!handler.is_tool_blocked("LS"));
    }

    #[tokio::test]
    async fn test_plan_mode_strategy_allows_plan_file_writes() {
        let handler = PermissionHandler::with_mode(PermissionMode::Plan);
        let home = dirs::home_dir().unwrap();
        let plan_file = home.join(".claude").join("plans").join("test.md");

        match handler
            .check_permission(
                "Write",
                &json!({"file_path": plan_file.to_str().unwrap(), "content": "test"}),
            )
            .await
        {
            ToolPermissionResult::Allowed => {}
            _ => panic!("Expected Allowed for plan file writes"),
        }
    }

    #[tokio::test]
    async fn test_plan_mode_strategy_blocks_non_plan_writes() {
        let handler = PermissionHandler::with_mode(PermissionMode::Plan);

        match handler
            .check_permission(
                "Write",
                &json!({"file_path": "/tmp/test.txt", "content": "test"}),
            )
            .await
        {
            ToolPermissionResult::Blocked { .. } => {}
            _ => panic!("Expected Blocked for non-plan file writes"),
        }
    }

    #[tokio::test]
    async fn test_default_mode_strategy() {
        let handler = PermissionHandler::new();

        // Reads are auto-approved
        match handler.check_permission("Read", &json!({})).await {
            ToolPermissionResult::Allowed => {}
            _ => panic!("Expected Allowed for Read"),
        }

        // Writes need permission
        match handler.check_permission("Write", &json!({})).await {
            ToolPermissionResult::NeedsPermission => {}
            _ => panic!("Expected NeedsPermission for Write"),
        }
    }

    #[tokio::test]
    async fn test_bypass_permissions_strategy() {
        let handler = PermissionHandler::with_mode(PermissionMode::BypassPermissions);

        // Everything is allowed
        match handler
            .check_permission("Bash", &json!({"command": "rm -rf /"}))
            .await
        {
            ToolPermissionResult::Allowed => {}
            _ => panic!("Expected Allowed for Bash in BypassPermissions mode"),
        }
    }

    #[tokio::test]
    async fn test_accept_edits_strategy() {
        let handler = PermissionHandler::with_mode(PermissionMode::AcceptEdits);

        // Everything is allowed
        match handler.check_permission("Write", &json!({})).await {
            ToolPermissionResult::Allowed => {}
            _ => panic!("Expected Allowed for Write in AcceptEdits mode"),
        }
    }
}
