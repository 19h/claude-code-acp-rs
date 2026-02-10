//! BypassPermissions mode strategy
//!
//! This strategy allows all tools without any permission checks.

use crate::permissions::strategies::PermissionModeStrategy;
use crate::session::{PermissionMode, ToolPermissionResult};
use serde_json::Value;

/// Strategy for BypassPermissions mode - allow everything
#[derive(Debug)]
pub struct BypassPermissionsModeStrategy;

impl PermissionModeStrategy for BypassPermissionsModeStrategy {
    fn mode(&self) -> PermissionMode {
        PermissionMode::BypassPermissions
    }

    fn should_auto_approve(&self, _tool_name: &str, _tool_input: &Value) -> bool {
        true
    }

    fn is_tool_blocked(&self, _tool_name: &str, _tool_input: &Value) -> Option<String> {
        None
    }

    fn check_permission(&self, _tool_name: &str, _tool_input: &Value) -> ToolPermissionResult {
        ToolPermissionResult::Allowed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_mode() {
        let strategy = BypassPermissionsModeStrategy;
        assert_eq!(strategy.mode(), PermissionMode::BypassPermissions);
    }

    #[test]
    fn test_always_auto_approves() {
        let strategy = BypassPermissionsModeStrategy;
        assert!(strategy.should_auto_approve("AnyTool", &json!({})));
        assert!(strategy.should_auto_approve("Bash", &json!({"command": "rm -rf /"})));
    }

    #[test]
    fn test_never_blocks() {
        let strategy = BypassPermissionsModeStrategy;
        assert!(strategy.is_tool_blocked("AnyTool", &json!({})).is_none());
    }

    #[test]
    fn test_always_allows() {
        let strategy = BypassPermissionsModeStrategy;
        match strategy.check_permission("AnyTool", &json!({})) {
            ToolPermissionResult::Allowed => {}
            _ => panic!("Expected Allowed"),
        }
    }
}
