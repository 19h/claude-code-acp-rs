//! DontAsk mode strategy
//!
//! This strategy denies tools that aren't pre-approved by settings rules.
//! No user prompts are shown.

use crate::permissions::strategies::PermissionModeStrategy;
use crate::session::{PermissionMode, ToolPermissionResult};
use serde_json::Value;

/// Strategy for DontAsk mode - deny unless pre-approved by settings
#[derive(Debug)]
pub struct DontAskModeStrategy;

impl PermissionModeStrategy for DontAskModeStrategy {
    fn mode(&self) -> PermissionMode {
        PermissionMode::DontAsk
    }

    fn should_auto_approve(&self, _tool_name: &str, _tool_input: &Value) -> bool {
        // DontAsk mode: no auto-approval
        // Tools must be explicitly allowed by settings rules
        false
    }

    fn is_tool_blocked(&self, _tool_name: &str, _tool_input: &Value) -> Option<String> {
        // DontAsk doesn't block tools explicitly
        // Tools without allow rules will fall through to "ask" which becomes "deny"
        None
    }

    fn check_permission(&self, _tool_name: &str, _tool_input: &Value) -> ToolPermissionResult {
        // In DontAsk mode, we defer entirely to settings rules
        // If no rule matches, the tool should be denied (not asked)
        // This is handled by the PermissionHandler which checks settings first
        ToolPermissionResult::NeedsPermission
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_mode() {
        let strategy = DontAskModeStrategy;
        assert_eq!(strategy.mode(), PermissionMode::DontAsk);
    }

    #[test]
    fn test_never_auto_approves() {
        let strategy = DontAskModeStrategy;
        assert!(!strategy.should_auto_approve("Read", &json!({})));
        assert!(!strategy.should_auto_approve("AnyTool", &json!({})));
    }

    #[test]
    fn test_never_blocks_explicitly() {
        let strategy = DontAskModeStrategy;
        assert!(strategy.is_tool_blocked("AnyTool", &json!({})).is_none());
    }

    #[test]
    fn test_always_needs_permission() {
        let strategy = DontAskModeStrategy;
        match strategy.check_permission("AnyTool", &json!({})) {
            ToolPermissionResult::NeedsPermission => {}
            _ => panic!("Expected NeedsPermission"),
        }
    }
}
