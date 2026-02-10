//! Plan mode strategy
//!
//! This strategy provides read-only access with an exception for writing to
//! the ~/.claude/plans/ directory. This is used during planning phases where
//! the user should be able to explore and write plans, but not make changes
//! to the codebase.

use crate::permissions::strategies::PermissionModeStrategy;
use crate::session::{PermissionMode, ToolPermissionResult};
use crate::utils::is_plans_directory_path;
use serde_json::Value;

/// Strategy for Plan mode - read-only with exceptions for plan files
#[derive(Debug)]
pub struct PlanModeStrategy;

impl PermissionModeStrategy for PlanModeStrategy {
    fn mode(&self) -> PermissionMode {
        PermissionMode::Plan
    }

    fn should_auto_approve(&self, tool_name: &str, _tool_input: &Value) -> bool {
        // Auto-approve read operations
        matches!(tool_name, "Read" | "Glob" | "Grep" | "LS" | "NotebookRead")
    }

    fn is_tool_blocked(&self, tool_name: &str, tool_input: &Value) -> Option<String> {
        let is_write_operation = matches!(tool_name, "Edit" | "Write" | "Bash" | "NotebookEdit");

        if !is_write_operation {
            return None; // Read operations are allowed
        }

        // Check if this is a write to the plans directory (exception)
        if matches!(tool_name, "Edit" | "Write" | "NotebookEdit") {
            let file_path = tool_input
                .get("file_path")
                .or_else(|| tool_input.get("path"))
                .or_else(|| tool_input.get("notebook_path"))
                .and_then(|v| v.as_str());

            if let Some(path) = file_path {
                if is_plans_directory_path(path) {
                    return None; // Allow plan file writes
                }
            }
        }

        // Block all other write operations
        Some(format!(
            "Tool {} is not allowed in Plan mode (only read operations and writing to ~/.claude/plans/ are allowed)",
            tool_name
        ))
    }

    fn check_permission(&self, tool_name: &str, tool_input: &Value) -> ToolPermissionResult {
        // Check if blocked first
        if let Some(reason) = self.is_tool_blocked(tool_name, tool_input) {
            return ToolPermissionResult::Blocked { reason };
        }

        // Auto-approve reads
        if self.should_auto_approve(tool_name, tool_input) {
            return ToolPermissionResult::Allowed;
        }

        // Plan file writes are allowed (checked in is_tool_blocked)
        // If we reach here, it's an allowed plan file write
        ToolPermissionResult::Allowed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn home_plans_path() -> String {
        let home = dirs::home_dir().unwrap();
        home.join(".claude")
            .join("plans")
            .join("test.md")
            .to_str()
            .unwrap()
            .to_string()
    }

    #[test]
    fn test_mode() {
        let strategy = PlanModeStrategy;
        assert_eq!(strategy.mode(), PermissionMode::Plan);
    }

    #[test]
    fn test_auto_approves_reads() {
        let strategy = PlanModeStrategy;
        assert!(strategy.should_auto_approve("Read", &json!({})));
        assert!(strategy.should_auto_approve("Glob", &json!({})));
        assert!(strategy.should_auto_approve("Grep", &json!({})));
        assert!(strategy.should_auto_approve("LS", &json!({})));
        assert!(strategy.should_auto_approve("NotebookRead", &json!({})));
    }

    #[test]
    fn test_does_not_auto_approve_writes() {
        let strategy = PlanModeStrategy;
        assert!(!strategy.should_auto_approve("Write", &json!({})));
        assert!(!strategy.should_auto_approve("Edit", &json!({})));
        assert!(!strategy.should_auto_approve("Bash", &json!({})));
    }

    #[test]
    fn test_blocks_non_plan_writes() {
        let strategy = PlanModeStrategy;
        let result = strategy.is_tool_blocked(
            "Write",
            &json!({"file_path": "/tmp/test.txt", "content": "test"}),
        );
        assert!(result.is_some());
        assert!(result.unwrap().contains("not allowed in Plan mode"));
    }

    #[test]
    fn test_blocks_bash() {
        let strategy = PlanModeStrategy;
        let result = strategy.is_tool_blocked("Bash", &json!({"command": "echo test"}));
        assert!(result.is_some());
        assert!(result.unwrap().contains("not allowed in Plan mode"));
    }

    #[test]
    fn test_allows_plan_file_writes() {
        let strategy = PlanModeStrategy;
        let plan_path = home_plans_path();
        let result = strategy.is_tool_blocked(
            "Write",
            &json!({"file_path": plan_path, "content": "# Plan"}),
        );
        assert!(result.is_none(), "Plan file writes should be allowed");
    }

    #[test]
    fn test_check_permission_allows_reads() {
        let strategy = PlanModeStrategy;
        match strategy.check_permission("Read", &json!({})) {
            ToolPermissionResult::Allowed => {}
            _ => panic!("Expected Allowed for Read"),
        }
    }

    #[test]
    fn test_check_permission_blocks_non_plan_writes() {
        let strategy = PlanModeStrategy;
        match strategy.check_permission("Write", &json!({"file_path": "/tmp/test.txt"})) {
            ToolPermissionResult::Blocked { .. } => {}
            _ => panic!("Expected Blocked for non-plan file writes"),
        }
    }

    #[test]
    fn test_check_permission_allows_plan_writes() {
        let strategy = PlanModeStrategy;
        let plan_path = home_plans_path();
        match strategy.check_permission("Write", &json!({"file_path": plan_path})) {
            ToolPermissionResult::Allowed => {}
            _ => panic!("Expected Allowed for plan file writes"),
        }
    }
}
