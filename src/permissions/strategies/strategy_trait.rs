//! Core strategy trait for permission modes

use crate::session::{PermissionMode, ToolPermissionResult};
use serde_json::Value;

/// Strategy trait for permission mode checking
///
/// Each strategy encapsulates the permission logic for a specific mode,
/// providing a single source of truth for how tools should be checked.
pub trait PermissionModeStrategy: Send + Sync {
    /// Get the permission mode this strategy handles
    fn mode(&self) -> PermissionMode;

    /// Check if a tool should be auto-approved without user interaction
    ///
    /// Returns true if the tool can proceed immediately, false if it needs
    /// further checking or user approval.
    fn should_auto_approve(&self, tool_name: &str, tool_input: &Value) -> bool;

    /// Check if a tool is explicitly blocked in this mode
    ///
    /// Returns Some(reason) if blocked, None if allowed.
    fn is_tool_blocked(&self, tool_name: &str, tool_input: &Value) -> Option<String>;

    /// Perform comprehensive permission check
    ///
    /// Returns the final permission decision for this tool invocation.
    /// This method is called after settings rules are checked.
    fn check_permission(&self, tool_name: &str, tool_input: &Value) -> ToolPermissionResult;
}

#[cfg(test)]
mod tests {
    // Trait is tested through implementing strategies
    // See individual strategy tests for details
}
