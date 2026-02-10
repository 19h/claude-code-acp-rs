//! Permission checker implementation
//!
//! Checks tool permissions against settings rules.

use std::path::{Path, PathBuf};

use super::manager::Settings;
use super::rule::{ParsedRule, PermissionCheckResult};
use crate::command_safety::extract_command_basename;

/// Permission checker that evaluates tool permissions against settings rules
#[derive(Debug)]
pub struct PermissionChecker {
    /// Merged settings with permission rules
    settings: Settings,
    /// Working directory for path resolution
    cwd: PathBuf,
    /// Parsed and cached allow rules
    allow_rules: Vec<(String, ParsedRule)>,
    /// Parsed and cached deny rules
    deny_rules: Vec<(String, ParsedRule)>,
    /// Parsed and cached ask rules
    ask_rules: Vec<(String, ParsedRule)>,
}

impl PermissionChecker {
    /// Create a new permission checker
    pub fn new(settings: Settings, cwd: impl AsRef<Path>) -> Self {
        let cwd = cwd.as_ref().to_path_buf();

        // Pre-parse rules for efficiency
        let allow_rules = Self::parse_rules(
            settings.permissions.as_ref().and_then(|p| p.allow.as_ref()),
            &cwd,
        );
        let deny_rules = Self::parse_rules(
            settings.permissions.as_ref().and_then(|p| p.deny.as_ref()),
            &cwd,
        );
        let ask_rules = Self::parse_rules(
            settings.permissions.as_ref().and_then(|p| p.ask.as_ref()),
            &cwd,
        );

        Self {
            settings,
            cwd,
            allow_rules,
            deny_rules,
            ask_rules,
        }
    }

    /// Parse a list of rule strings into ParsedRule objects
    fn parse_rules(rules: Option<&Vec<String>>, cwd: &Path) -> Vec<(String, ParsedRule)> {
        rules
            .map(|rules| {
                rules
                    .iter()
                    .map(|rule| (rule.clone(), ParsedRule::parse_with_glob(rule, cwd)))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Check permission for a tool invocation
    ///
    /// Priority: deny > allow > ask
    ///
    /// Returns the permission decision and matching rule (if any).
    pub fn check_permission(
        &self,
        tool_name: &str,
        tool_input: &serde_json::Value,
    ) -> PermissionCheckResult {
        // Check deny rules first (highest priority)
        for (rule_str, parsed) in &self.deny_rules {
            if parsed.matches(tool_name, tool_input, &self.cwd) {
                tracing::debug!("Tool {} denied by rule: {}", tool_name, rule_str);
                return PermissionCheckResult::deny(rule_str);
            }
        }

        // Check allow rules
        for (rule_str, parsed) in &self.allow_rules {
            if parsed.matches(tool_name, tool_input, &self.cwd) {
                tracing::debug!("Tool {} allowed by rule: {}", tool_name, rule_str);
                return PermissionCheckResult::allow(rule_str);
            }
        }

        // Check ask rules
        for (rule_str, parsed) in &self.ask_rules {
            if parsed.matches(tool_name, tool_input, &self.cwd) {
                tracing::debug!(
                    "Tool {} requires permission (ask rule): {}",
                    tool_name,
                    rule_str
                );
                return PermissionCheckResult::ask_with_rule(rule_str);
            }
        }

        // Default: ask
        tracing::debug!("Tool {} has no matching rule, defaulting to ask", tool_name);
        PermissionCheckResult::ask()
    }

    /// Get the settings
    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    /// Get the working directory
    pub fn cwd(&self) -> &Path {
        &self.cwd
    }

    /// Check if there are any permission rules configured
    pub fn has_rules(&self) -> bool {
        !self.allow_rules.is_empty() || !self.deny_rules.is_empty() || !self.ask_rules.is_empty()
    }

    /// Add a runtime allow rule (e.g., from user's "Always Allow" choice)
    pub fn add_allow_rule(&mut self, rule: &str) {
        let parsed = ParsedRule::parse_with_glob(rule, &self.cwd);
        self.allow_rules.push((rule.to_string(), parsed));
    }

    /// Add a runtime allow rule for "Always Allow" permission decision
    ///
    /// For Bash tool: extracts the command name (first word) and generates
    /// a rule like `Bash(find:*)` that allows all invocations of that command
    /// regardless of arguments or paths.
    ///
    /// For file operations (Read/Edit/Write): generates a directory-based rule
    /// that allows operations in the same directory tree.
    ///
    /// This provides reasonable granularity:
    /// - `find /path1` → rule `Bash(find:*)` → allows all `find` commands
    /// - `ls /path` → not matched → needs separate permission
    pub fn add_allow_rule_for_tool_call(
        &mut self,
        tool_name: &str,
        tool_input: &serde_json::Value,
    ) {
        // Strip mcp__acp__ prefix for consistent rule matching
        let stripped = tool_name.strip_prefix("mcp__acp__").unwrap_or(tool_name);

        let rule = match stripped {
            "Bash" => {
                // Extract command name (first word only) for Bash
                if let Some(cmd) = tool_input.get("command").and_then(|v| v.as_str()) {
                    let cmd_name = Self::extract_command_name(cmd);
                    if cmd_name.is_empty() {
                        stripped.to_string()
                    } else {
                        format!("Bash({}:*)", cmd_name) // e.g., "Bash(find:*)"
                    }
                } else {
                    stripped.to_string()
                }
            }
            "Read" | "Grep" | "Glob" | "LS" => {
                // For read operations, generate directory-based rule
                Self::generate_file_rule("Read", tool_input, &self.cwd)
            }
            "Edit" | "Write" => {
                // For write operations, generate directory-based rule
                Self::generate_file_rule(stripped, tool_input, &self.cwd)
            }
            _ => stripped.to_string(),
        };

        tracing::info!(
            tool_name = %tool_name,
            generated_rule = %rule,
            "Adding allow rule for Always Allow"
        );

        let parsed = ParsedRule::parse_with_glob(&rule, &self.cwd);
        self.allow_rules.push((rule, parsed));
    }

    /// Extract command name (basename only) from a shell command
    ///
    /// Supports both simple commands and full path commands:
    /// - `find /path -name "*.rs"` → `find`
    /// - `/usr/bin/find . -name "*.rs"` → `find`
    /// - `ls -la /tmp` → `ls`
    fn extract_command_name(cmd: &str) -> String {
        extract_command_basename(cmd).to_string()
    }

    /// Generate a file-based permission rule
    fn generate_file_rule(tool_name: &str, tool_input: &serde_json::Value, cwd: &Path) -> String {
        if let Some(path) = tool_input.get("file_path").and_then(|v| v.as_str()) {
            let path = Path::new(path);

            // Try to get the directory
            if let Some(dir) = path.parent() {
                // Make path relative to cwd if possible
                let dir_str = if let Ok(relative) = dir.strip_prefix(cwd) {
                    format!("./{}", relative.display())
                } else {
                    dir.to_string_lossy().to_string()
                };

                // Generate glob rule for the directory
                if dir_str.is_empty() || dir_str == "." {
                    format!("{}(./*)", tool_name)
                } else {
                    format!("{}({}/**)", tool_name, dir_str)
                }
            } else {
                tool_name.to_string()
            }
        } else {
            tool_name.to_string()
        }
    }

    /// Add a runtime deny rule
    pub fn add_deny_rule(&mut self, rule: &str) {
        let parsed = ParsedRule::parse_with_glob(rule, &self.cwd);
        self.deny_rules.push((rule.to_string(), parsed));
    }

    /// Get the default permission mode from settings
    pub fn default_mode(&self) -> Option<&str> {
        self.settings
            .permissions
            .as_ref()
            .and_then(|p| p.default_mode.as_deref())
    }

    /// Get additional directories from settings
    pub fn additional_directories(&self) -> Option<&Vec<String>> {
        self.settings
            .permissions
            .as_ref()
            .and_then(|p| p.additional_directories.as_ref())
    }
}

impl Default for PermissionChecker {
    fn default() -> Self {
        Self::new(Settings::default(), PathBuf::from("."))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::{PermissionDecision, PermissionSettings};
    use serde_json::json;

    fn settings_with_permissions(permissions: PermissionSettings) -> Settings {
        Settings {
            permissions: Some(permissions),
            ..Default::default()
        }
    }

    #[test]
    fn test_empty_rules_default_to_ask() {
        let checker = PermissionChecker::default();
        let result = checker.check_permission("Read", &json!({"file_path": "/tmp/test.txt"}));

        assert_eq!(result.decision, PermissionDecision::Ask);
        assert!(result.rule.is_none());
    }

    #[test]
    fn test_allow_rule() {
        let permissions = PermissionSettings {
            allow: Some(vec!["Read".to_string()]),
            ..Default::default()
        };
        let checker = PermissionChecker::new(settings_with_permissions(permissions), "/tmp");

        let result = checker.check_permission("Read", &json!({"file_path": "/tmp/test.txt"}));
        assert_eq!(result.decision, PermissionDecision::Allow);
        assert_eq!(result.rule, Some("Read".to_string()));
    }

    #[test]
    fn test_deny_rule() {
        let permissions = PermissionSettings {
            deny: Some(vec!["Bash".to_string()]),
            ..Default::default()
        };
        let checker = PermissionChecker::new(settings_with_permissions(permissions), "/tmp");

        let result = checker.check_permission("Bash", &json!({"command": "rm -rf /"}));
        assert_eq!(result.decision, PermissionDecision::Deny);
    }

    #[test]
    fn test_deny_takes_priority_over_allow() {
        let permissions = PermissionSettings {
            allow: Some(vec!["Bash".to_string()]),
            deny: Some(vec!["Bash".to_string()]),
            ..Default::default()
        };
        let checker = PermissionChecker::new(settings_with_permissions(permissions), "/tmp");

        let result = checker.check_permission("Bash", &json!({"command": "ls"}));
        assert_eq!(result.decision, PermissionDecision::Deny);
    }

    #[test]
    fn test_allow_takes_priority_over_ask() {
        let permissions = PermissionSettings {
            allow: Some(vec!["Read".to_string()]),
            ask: Some(vec!["Read".to_string()]),
            ..Default::default()
        };
        let checker = PermissionChecker::new(settings_with_permissions(permissions), "/tmp");

        let result = checker.check_permission("Read", &json!({}));
        assert_eq!(result.decision, PermissionDecision::Allow);
    }

    #[test]
    fn test_bash_wildcard_rule() {
        let permissions = PermissionSettings {
            allow: Some(vec!["Bash(npm run:*)".to_string()]),
            ..Default::default()
        };
        let checker = PermissionChecker::new(settings_with_permissions(permissions), "/tmp");

        // Should allow npm run commands
        assert_eq!(
            checker
                .check_permission("Bash", &json!({"command": "npm run build"}))
                .decision,
            PermissionDecision::Allow
        );

        // Should not allow npm install
        assert_eq!(
            checker
                .check_permission("Bash", &json!({"command": "npm install"}))
                .decision,
            PermissionDecision::Ask
        );

        // Should block command chaining
        assert_eq!(
            checker
                .check_permission("Bash", &json!({"command": "npm run build && rm -rf /"}))
                .decision,
            PermissionDecision::Ask
        );
    }

    #[test]
    fn test_read_group_matching() {
        let permissions = PermissionSettings {
            allow: Some(vec!["Read".to_string()]),
            ..Default::default()
        };
        let checker = PermissionChecker::new(settings_with_permissions(permissions), "/tmp");

        // Read rule should allow Read, Grep, Glob, LS
        assert_eq!(
            checker.check_permission("Read", &json!({})).decision,
            PermissionDecision::Allow
        );
        assert_eq!(
            checker.check_permission("Grep", &json!({})).decision,
            PermissionDecision::Allow
        );
        assert_eq!(
            checker.check_permission("Glob", &json!({})).decision,
            PermissionDecision::Allow
        );
        assert_eq!(
            checker.check_permission("LS", &json!({})).decision,
            PermissionDecision::Allow
        );

        // Should not allow Write
        assert_eq!(
            checker.check_permission("Write", &json!({})).decision,
            PermissionDecision::Ask
        );
    }

    #[test]
    fn test_add_runtime_rule() {
        let mut checker = PermissionChecker::default();

        // Initially should ask
        assert_eq!(
            checker.check_permission("Read", &json!({})).decision,
            PermissionDecision::Ask
        );

        // Add allow rule at runtime
        checker.add_allow_rule("Read");

        // Now should allow
        assert_eq!(
            checker.check_permission("Read", &json!({})).decision,
            PermissionDecision::Allow
        );
    }

    #[test]
    fn test_acp_prefix_stripped() {
        let permissions = PermissionSettings {
            allow: Some(vec!["Read".to_string()]),
            ..Default::default()
        };
        let checker = PermissionChecker::new(settings_with_permissions(permissions), "/tmp");

        // Should work with or without ACP prefix
        assert_eq!(
            checker.check_permission("Read", &json!({})).decision,
            PermissionDecision::Allow
        );
        assert_eq!(
            checker
                .check_permission("mcp__acp__Read", &json!({}))
                .decision,
            PermissionDecision::Allow
        );
    }

    #[test]
    fn test_has_rules() {
        let checker = PermissionChecker::default();
        assert!(!checker.has_rules());

        let permissions = PermissionSettings {
            allow: Some(vec!["Read".to_string()]),
            ..Default::default()
        };
        let checker = PermissionChecker::new(settings_with_permissions(permissions), "/tmp");
        assert!(checker.has_rules());
    }

    #[test]
    fn test_add_allow_rule_for_bash_command() {
        let mut checker = PermissionChecker::new(Settings::default(), "/tmp");

        // Add rule for specific bash command (find)
        checker
            .add_allow_rule_for_tool_call("Bash", &json!({"command": "find /path1 -name '*.rs'"}));

        // Should allow ANY find command (same command name)
        assert_eq!(
            checker
                .check_permission("Bash", &json!({"command": "find /different/path -type f"}))
                .decision,
            PermissionDecision::Allow
        );

        // Should allow find with different arguments
        assert_eq!(
            checker
                .check_permission("Bash", &json!({"command": "find . -name '*.txt' -delete"}))
                .decision,
            PermissionDecision::Allow
        );

        // Should NOT allow different commands (ls, rm, etc.)
        assert_eq!(
            checker
                .check_permission("Bash", &json!({"command": "ls -la /tmp"}))
                .decision,
            PermissionDecision::Ask
        );

        assert_eq!(
            checker
                .check_permission("Bash", &json!({"command": "rm -rf /"}))
                .decision,
            PermissionDecision::Ask
        );
    }

    #[test]
    fn test_add_allow_rule_for_file_operation() {
        let mut checker = PermissionChecker::new(Settings::default(), "/tmp");

        // Add rule for specific file
        checker.add_allow_rule_for_tool_call(
            "Read",
            &json!({"file_path": "/tmp/project/src/main.rs"}),
        );

        // Should allow files in the same directory
        assert_eq!(
            checker
                .check_permission("Read", &json!({"file_path": "/tmp/project/src/lib.rs"}))
                .decision,
            PermissionDecision::Allow
        );

        // Should allow subdirectories (glob **)
        assert_eq!(
            checker
                .check_permission(
                    "Read",
                    &json!({"file_path": "/tmp/project/src/utils/helper.rs"})
                )
                .decision,
            PermissionDecision::Allow
        );

        // Should NOT allow different directories
        assert_eq!(
            checker
                .check_permission("Read", &json!({"file_path": "/etc/passwd"}))
                .decision,
            PermissionDecision::Ask
        );
    }

    #[test]
    fn test_add_allow_rule_for_mcp_prefixed_tool() {
        let mut checker = PermissionChecker::new(Settings::default(), "/tmp");

        // Add rule with MCP prefix
        checker
            .add_allow_rule_for_tool_call("mcp__acp__Bash", &json!({"command": "npm run build"}));

        // Should work for both prefixed and non-prefixed tool names
        assert_eq!(
            checker
                .check_permission("Bash", &json!({"command": "npm run test"}))
                .decision,
            PermissionDecision::Allow
        );
        assert_eq!(
            checker
                .check_permission("mcp__acp__Bash", &json!({"command": "npm run lint"}))
                .decision,
            PermissionDecision::Allow
        );
    }

    #[test]
    fn test_extract_command_name() {
        // Should extract only the command name (basename)
        assert_eq!(
            PermissionChecker::extract_command_name("cargo build --release"),
            "cargo"
        );
        assert_eq!(
            PermissionChecker::extract_command_name("find /path -name '*.rs'"),
            "find"
        );
        assert_eq!(PermissionChecker::extract_command_name("ls -la /tmp"), "ls");
        assert_eq!(PermissionChecker::extract_command_name("npm"), "npm");
        assert_eq!(PermissionChecker::extract_command_name(""), "");
        // Full path commands should return just the basename
        assert_eq!(
            PermissionChecker::extract_command_name("/usr/bin/find . -name '*.rs'"),
            "find"
        );
        assert_eq!(
            PermissionChecker::extract_command_name("/usr/local/bin/cargo build"),
            "cargo"
        );
    }
}
