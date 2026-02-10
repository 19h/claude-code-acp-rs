//! Default mode strategy
//!
//! This strategy provides standard permission checking:
//! - Auto-approves read operations
//! - Auto-approves known safe Bash commands
//! - Requires user permission for other operations

use crate::command_safety::is_known_safe_command;
use crate::permissions::strategies::PermissionModeStrategy;
use crate::session::{PermissionMode, ToolPermissionResult};
use serde_json::Value;

/// Strategy for Default mode - standard permission prompts
#[derive(Debug)]
pub struct DefaultModeStrategy;

impl PermissionModeStrategy for DefaultModeStrategy {
    fn mode(&self) -> PermissionMode {
        PermissionMode::Default
    }

    fn should_auto_approve(&self, tool_name: &str, tool_input: &Value) -> bool {
        // Auto-approve read operations
        if matches!(tool_name, "Read" | "Glob" | "Grep" | "LS" | "NotebookRead") {
            return true;
        }

        // Auto-approve known safe Bash commands
        if tool_name == "Bash" {
            if let Some(cmd) = tool_input.get("command").and_then(|v| v.as_str()) {
                return is_known_safe_command(cmd);
            }
        }

        false
    }

    fn is_tool_blocked(&self, _tool_name: &str, _tool_input: &Value) -> Option<String> {
        // Default mode doesn't block tools explicitly
        // Tools that aren't auto-approved will fall through to permission prompts
        None
    }

    fn check_permission(&self, tool_name: &str, tool_input: &Value) -> ToolPermissionResult {
        // Auto-approve if strategy allows
        if self.should_auto_approve(tool_name, tool_input) {
            return ToolPermissionResult::Allowed;
        }

        // Otherwise, need to ask user (or check settings rules)
        ToolPermissionResult::NeedsPermission
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_mode() {
        let strategy = DefaultModeStrategy;
        assert_eq!(strategy.mode(), PermissionMode::Default);
    }

    #[test]
    fn test_auto_approves_reads() {
        let strategy = DefaultModeStrategy;
        assert!(strategy.should_auto_approve("Read", &json!({})));
        assert!(strategy.should_auto_approve("Glob", &json!({})));
        assert!(strategy.should_auto_approve("Grep", &json!({})));
        assert!(strategy.should_auto_approve("LS", &json!({})));
        assert!(strategy.should_auto_approve("NotebookRead", &json!({})));
    }

    #[test]
    fn test_does_not_auto_approve_writes() {
        let strategy = DefaultModeStrategy;
        assert!(!strategy.should_auto_approve("Write", &json!({})));
        assert!(!strategy.should_auto_approve("Edit", &json!({})));
    }

    #[test]
    fn test_auto_approves_safe_bash_commands() {
        let strategy = DefaultModeStrategy;
        assert!(strategy.should_auto_approve("Bash", &json!({"command": "cat file.txt"})));
        assert!(strategy.should_auto_approve("Bash", &json!({"command": "echo test"})));
        assert!(!strategy.should_auto_approve("Bash", &json!({"command": "rm -rf /"})));
    }

    #[test]
    fn test_never_blocks_explicitly() {
        let strategy = DefaultModeStrategy;
        assert!(strategy.is_tool_blocked("AnyTool", &json!({})).is_none());
    }

    #[test]
    fn test_check_permission_auto_approves_reads() {
        let strategy = DefaultModeStrategy;
        match strategy.check_permission("Read", &json!({})) {
            ToolPermissionResult::Allowed => {}
            _ => panic!("Expected Allowed for Read"),
        }
    }

    #[test]
    fn test_check_permission_needs_permission_for_writes() {
        let strategy = DefaultModeStrategy;
        match strategy.check_permission("Write", &json!({})) {
            ToolPermissionResult::NeedsPermission => {}
            _ => panic!("Expected NeedsPermission for Write"),
        }
    }
}
