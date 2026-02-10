//! AcceptEdits mode strategy
//!
//! This strategy auto-approves ALL tools, behaving identically to BypassPermissions.
//! It's compatible with root user environments where BypassPermissions cannot be used.

use crate::permissions::strategies::PermissionModeStrategy;
use crate::session::{PermissionMode, ToolPermissionResult};
use serde_json::Value;

/// Strategy for AcceptEdits mode - auto-approve all tools
///
/// This behaves identically to BypassPermissions but is compatible
/// with root user environments where BypassPermissions cannot be used.
#[derive(Debug)]
pub struct AcceptEditsModeStrategy;

impl PermissionModeStrategy for AcceptEditsModeStrategy {
    fn mode(&self) -> PermissionMode {
        PermissionMode::AcceptEdits
    }

    fn should_auto_approve(&self, _tool_name: &str, _tool_input: &Value) -> bool {
        // Auto-approve ALL tools (same as BypassPermissions)
        true
    }

    fn is_tool_blocked(&self, _tool_name: &str, _tool_input: &Value) -> Option<String> {
        // Nothing is blocked in AcceptEdits mode
        None
    }

    fn check_permission(&self, _tool_name: &str, _tool_input: &Value) -> ToolPermissionResult {
        // Allow everything
        ToolPermissionResult::Allowed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_mode() {
        let strategy = AcceptEditsModeStrategy;
        assert_eq!(strategy.mode(), PermissionMode::AcceptEdits);
    }

    #[test]
    fn test_always_auto_approves() {
        let strategy = AcceptEditsModeStrategy;
        assert!(strategy.should_auto_approve("Write", &json!({})));
        assert!(strategy.should_auto_approve("Bash", &json!({"command": "rm -rf /"})));
    }

    #[test]
    fn test_never_blocks() {
        let strategy = AcceptEditsModeStrategy;
        assert!(strategy.is_tool_blocked("AnyTool", &json!({})).is_none());
    }

    #[test]
    fn test_always_allows() {
        let strategy = AcceptEditsModeStrategy;
        match strategy.check_permission("AnyTool", &json!({})) {
            ToolPermissionResult::Allowed => {}
            _ => panic!("Expected Allowed"),
        }
    }
}
