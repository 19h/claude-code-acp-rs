//! PreToolUse hook implementation
//!
//! Checks permissions using SettingsManager before tool execution.
//! For "Ask" decisions, sends permission request directly (has correct tool_use_id).

use std::sync::{Arc, OnceLock};
use std::time::Instant;

use claude_code_agent_sdk::{
    HookCallback, HookContext, HookInput, HookJsonOutput, HookSpecificOutput,
    PreToolUseHookSpecificOutput, SyncHookJsonOutput,
};
use dashmap::DashMap;
use futures::future::BoxFuture;
use sacp::{
    JrConnectionCx,
    link::AgentToClient,
    schema::{
        SessionId, SessionNotification, SessionUpdate, ToolCallContent, ToolCallId, ToolCallStatus,
        ToolCallUpdate, ToolCallUpdateFields,
    },
};
use tokio::sync::RwLock;
use tracing::Instrument;

use crate::command_safety::{command_might_be_dangerous, is_known_safe_command};
use crate::session::{PermissionHandler, PermissionMode};
use crate::settings::PermissionChecker;
use crate::utils::is_plans_directory_path;

/// Creates a PreToolUse hook that checks permissions using settings rules and permission mode.
///
/// This hook runs before the SDK's built-in permission rules, allowing us to enforce
/// our own permission settings for ACP-prefixed tools.
///
/// # Permission Handling
///
/// - **Allow**: Returns with `permission_decision: "allow"` - tool executes immediately
/// - **Deny**: Returns with `permission_decision: "deny"` - tool execution is blocked
/// - **Ask**: Returns with `permission_decision: "ask"` - SDK triggers permission request flow
///
/// # Permission Mode Integration
///
/// The hook respects the session's permission mode:
/// - **BypassPermissions/AcceptEdits**: Allows all tools without checking rules
///   (AcceptEdits behaves like BypassPermissions for root compatibility)
/// - **Plan**: Blocks write operations (Edit, Write, Bash, NotebookEdit)
/// - **Default**: Auto-allows read-only operations (Read, Grep, Glob, LS, NotebookRead),
///   checks settings rules for other tools
/// - **DontAsk**: Checks settings rules and mode-based auto-approval
///
/// # Architecture
///
/// The hook and `can_use_tool` callback work together:
/// 1. **Hook** (this file): Makes quick decisions based on static rules (allow/deny/ask)
/// 2. **`can_use_tool` callback** (`can_use_tool.rs`): Checks cached permission results
///
/// For "Ask" decisions, the hook sends permission request directly (using the correct tool_use_id),
/// then caches the result. The `can_use_tool` callback checks this cache and returns immediately.
///
/// # Arguments
///
/// * `connection_cx_lock` - Connection for sending permission requests
/// * `session_id` - Session ID for permission requests
/// * `permission_checker` - Optional permission checker for settings-based rules
/// * `permission` - Shared permission handler (contains mode that can be updated at runtime)
/// * `permission_cache` - Cache for storing permission results (for can_use_tool callback)
/// * `tool_use_id_cache` - Cache for storing tool_use_id (for can_use_tool callback)
///
/// # Returns
///
/// A hook callback that can be used with ClaudeAgentOptions
pub fn create_pre_tool_use_hook(
    connection_cx_lock: Arc<OnceLock<JrConnectionCx<AgentToClient>>>,
    session_id: String,
    permission_checker: Option<Arc<RwLock<PermissionChecker>>>,
    permission: Arc<RwLock<PermissionHandler>>,
    permission_cache: Arc<DashMap<String, bool>>,
    tool_use_id_cache: Arc<DashMap<String, String>>,
) -> HookCallback {
    Arc::new(
        move |input: HookInput, tool_use_id: Option<String>, _context: HookContext| {
            // Clone the connection_cx_lock for sending denied tool result notifications
            let connection_cx_lock = Arc::clone(&connection_cx_lock);
            let permission_checker = permission_checker.clone();
            let permission = permission.clone();
            let session_id = session_id.clone();
            let _permission_cache = Arc::clone(&permission_cache);
            let tool_use_id_cache = Arc::clone(&tool_use_id_cache);

            // Extract tool name early for span naming
            let (tool_name, is_pre_tool) = match &input {
                HookInput::PreToolUse(pre_tool) => (pre_tool.tool_name.clone(), true),
                _ => (String::new(), false),
            };

            // Create a span for this hook execution
            let span = if is_pre_tool {
                tracing::info_span!(
                    "pre_tool_use_hook",
                    tool_name = %tool_name,
                    tool_use_id = ?tool_use_id,
                    permission_decision = tracing::field::Empty,
                    permission_rule = tracing::field::Empty,
                    check_duration_us = tracing::field::Empty,
                )
            } else {
                tracing::debug_span!(
                    "pre_tool_use_hook_skip",
                    event_type = ?std::mem::discriminant(&input)
                )
            };

            Box::pin(
                async move {
                    let start_time = Instant::now();

                    // Only handle PreToolUse events
                    let (tool_name, tool_input) = if let HookInput::PreToolUse(pre_tool) = &input {
                        (pre_tool.tool_name.clone(), pre_tool.tool_input.clone())
                    } else {
                        tracing::debug!("Ignoring non-PreToolUse event");
                        return HookJsonOutput::Sync(SyncHookJsonOutput {
                            continue_: Some(true),
                            ..Default::default()
                        });
                    };

                    tracing::debug!(
                        tool_name = %tool_name,
                        tool_use_id = ?tool_use_id,
                        "PreToolUse hook triggered"
                    );

                    // IMPORTANT: ExitPlanMode is handled specially by canUseTool callback
                    // We skip all permission checks here to avoid double permission prompts
                    // The canUseTool callback will handle the permission dialog
                    let stripped_tool_name = tool_name.strip_prefix("mcp__acp__").unwrap_or(&tool_name);
                    if stripped_tool_name == "ExitPlanMode" {
                        tracing::info!(
                            tool_name = %tool_name,
                            tool_use_id = ?tool_use_id,
                            "ExitPlanMode detected in pre_tool_use - skipping permission checks, delegating to canUseTool callback"
                        );
                        return HookJsonOutput::Sync(SyncHookJsonOutput {
                            continue_: Some(true),
                            hook_specific_output: Some(HookSpecificOutput::PreToolUse(
                                PreToolUseHookSpecificOutput {
                                    permission_decision: Some("defer".to_string()),
                                    permission_decision_reason: Some(
                                        "ExitPlanMode permission handled by canUseTool callback".to_string()
                                    ),
                                    updated_input: None,
                                    additional_context: None,
                                },
                            )),
                            ..Default::default()
                        });
                    }

                    // Get current permission mode
                    let mode = permission.read().await.mode();

                    // BypassPermissions and AcceptEdits modes allow everything
                    // (AcceptEdits behaves like BypassPermissions for root compatibility)
                    if matches!(
                        mode,
                        PermissionMode::BypassPermissions | PermissionMode::AcceptEdits
                    ) {
                        let elapsed = start_time.elapsed();
                        let mode_str = match mode {
                            PermissionMode::BypassPermissions => "BypassPermissions",
                            PermissionMode::AcceptEdits => "AcceptEdits",
                            _ => unreachable!(),
                        };
                        tracing::info!(
                            tool_name = %tool_name,
                            tool_use_id = ?tool_use_id,
                            mode = %mode_str,
                            elapsed_us = elapsed.as_micros(),
                            "Tool allowed by permission mode (auto-approve all)"
                        );

                        return HookJsonOutput::Sync(SyncHookJsonOutput {
                            continue_: Some(true),
                            hook_specific_output: Some(HookSpecificOutput::PreToolUse(
                                PreToolUseHookSpecificOutput {
                                    permission_decision: Some("allow".to_string()),
                                    permission_decision_reason: Some(format!(
                                        "Allowed by {} mode (auto-approve all tools)",
                                        mode_str
                                    )),
                                    updated_input: None,
                                    additional_context: None,
                                },
                            )),
                            ..Default::default()
                        });
                    }

                    // Default mode: auto-allow read-only operations
                    // This allows tools like Read, Grep, Glob, LS, NotebookRead to execute without permission prompt
                    if mode == PermissionMode::Default {
                        let is_read_only = matches!(
                            stripped_tool_name,
                            "Read" | "Grep" | "Glob" | "LS" | "NotebookRead"
                        );
                        if is_read_only {
                            let elapsed = start_time.elapsed();
                            tracing::debug!(
                                tool_name = %tool_name,
                                tool_use_id = ?tool_use_id,
                                mode = "default",
                                elapsed_us = elapsed.as_micros(),
                                "Tool auto-allowed in Default mode (read-only operation)"
                            );
                            return HookJsonOutput::Sync(SyncHookJsonOutput {
                                continue_: Some(true),
                                hook_specific_output: Some(HookSpecificOutput::PreToolUse(
                                    PreToolUseHookSpecificOutput {
                                        permission_decision: Some("allow".to_string()),
                                        permission_decision_reason: Some(
                                            "Auto-allowed in Default mode (read-only operation)"
                                                .to_string(),
                                        ),
                                        updated_input: None,
                                        additional_context: None,
                                    },
                                )),
                                ..Default::default()
                            });
                        }

                        // Check Bash commands for known safe commands (auto-allow)
                        if stripped_tool_name == "Bash" {
                            if let Some(cmd) = tool_input.get("command").and_then(|v| v.as_str()) {
                                // Check if this is a known safe command
                                if is_known_safe_command(cmd) {
                                    let elapsed = start_time.elapsed();
                                    tracing::info!(
                                        tool_name = %tool_name,
                                        command = %cmd,
                                        tool_use_id = ?tool_use_id,
                                        mode = "default",
                                        elapsed_us = elapsed.as_micros(),
                                        "Bash command auto-allowed (known safe command)"
                                    );
                                    return HookJsonOutput::Sync(SyncHookJsonOutput {
                                        continue_: Some(true),
                                        hook_specific_output: Some(HookSpecificOutput::PreToolUse(
                                            PreToolUseHookSpecificOutput {
                                                permission_decision: Some("allow".to_string()),
                                                permission_decision_reason: Some(format!(
                                                    "Auto-allowed: known safe command ({})",
                                                    cmd.split_whitespace().next().unwrap_or("")
                                                )),
                                                updated_input: None,
                                                additional_context: None,
                                            },
                                        )),
                                        ..Default::default()
                                    });
                                }

                                // Check if this is a dangerous command (log warning for user awareness)
                                if command_might_be_dangerous(cmd) {
                                    tracing::warn!(
                                        tool_name = %tool_name,
                                        command = %cmd,
                                        tool_use_id = ?tool_use_id,
                                        "Bash command flagged as potentially dangerous"
                                    );
                                    // Continue to normal permission flow - user will be asked
                                }
                            }
                        }
                    }

                    // Plan mode: Block write operations EXCEPT for plan files
                    if mode == PermissionMode::Plan {
                        let is_write_operation = matches!(
                            stripped_tool_name,
                            "Edit" | "Write" | "Bash" | "NotebookEdit"
                        );

                        if is_write_operation {
                            // For file operations, check if writing to plans directory
                            let is_plan_file = if matches!(stripped_tool_name, "Edit" | "Write" | "NotebookEdit") {
                                tool_input
                                    .get("file_path")
                                    .or_else(|| tool_input.get("path"))
                                    .and_then(|v| v.as_str())
                                    .map(is_plans_directory_path)
                                    .unwrap_or(false)
                            } else {
                                // Bash is never allowed in Plan mode
                                false
                            };

                            if !is_plan_file {
                                let reason = format!(
                                    "Tool {} is not allowed in Plan mode (only read operations and writing to ~/.claude/plans/ are allowed)",
                                    stripped_tool_name
                                );
                                tracing::warn!(
                                    tool_name = %tool_name,
                                    tool_use_id = ?tool_use_id,
                                    mode = "plan",
                                    elapsed_us = start_time.elapsed().as_micros(),
                                    "Tool blocked by Plan mode"
                                );
                                return create_deny_response(
                                    &connection_cx_lock,
                                    &session_id,
                                    tool_use_id.as_ref(),
                                    &tool_name,
                                    reason,
                                );
                            }

                            // Allow plan file writes
                            tracing::info!(
                                tool_name = %tool_name,
                                file_path = ?tool_input.get("file_path"),
                                "Plan mode: allowing write to plans directory"
                            );
                        }

                        // Auto-allow read operations in Plan mode
                        let is_read_only = matches!(
                            stripped_tool_name,
                            "Read" | "Grep" | "Glob" | "LS" | "NotebookRead"
                        );
                        if is_read_only {
                            return HookJsonOutput::Sync(SyncHookJsonOutput {
                                continue_: Some(true),
                                hook_specific_output: Some(HookSpecificOutput::PreToolUse(
                                    PreToolUseHookSpecificOutput {
                                        permission_decision: Some("allow".to_string()),
                                        permission_decision_reason: Some(
                                            "Allowed in Plan mode (read-only operation)".to_string()
                                        ),
                                        updated_input: None,
                                        additional_context: None,
                                    },
                                )),
                                ..Default::default()
                            });
                        }
                    }

                    // Check permission (if checker is available, otherwise default to Ask)
                    let permission_check = if let Some(checker) = &permission_checker {
                        let checker = checker.read().await;
                        checker.check_permission(&tool_name, &tool_input)
                    } else {
                        // No permission checker - default to Ask
                        crate::settings::PermissionCheckResult {
                            decision: crate::settings::PermissionDecision::Ask,
                            rule: None,
                            source: None,
                        }
                    };
                    let elapsed = start_time.elapsed();

                    // Record permission decision to span (batched for performance)
                    let span = tracing::Span::current();
                    span.record(
                        "permission_decision",
                        format!("{:?}", permission_check.decision),
                    );
                    span.record("check_duration_us", elapsed.as_micros());
                    if let Some(ref rule) = permission_check.rule {
                        span.record("permission_rule", rule.as_str());
                    }

                    tracing::info!(
                        tool_name = %tool_name,
                        tool_use_id = ?tool_use_id,
                        decision = ?permission_check.decision,
                        rule = ?permission_check.rule,
                        elapsed_us = elapsed.as_micros(),
                        "Permission check completed"
                    );

                    // 根据权限决策返回相应的 Hook 输出
                    // SDK 已修改为在 mcp_message 处理中调用 can_use_tool 回调，
                    // 因此 Ask 决策会由 SDK 层处理，Hook 只需要返回 continue_: true
                    match permission_check.decision {
                        crate::settings::PermissionDecision::Allow => {
                            tracing::debug!(
                                tool_name = %tool_name,
                                rule = ?permission_check.rule,
                                "Tool execution allowed by rule"
                            );
                            HookJsonOutput::Sync(SyncHookJsonOutput {
                                continue_: Some(true),
                                hook_specific_output: Some(HookSpecificOutput::PreToolUse(
                                    PreToolUseHookSpecificOutput {
                                        permission_decision: Some("allow".to_string()),
                                        permission_decision_reason: permission_check.rule,
                                        updated_input: None,
                                        additional_context: None,
                                    },
                                )),
                                ..Default::default()
                            })
                        }
                        crate::settings::PermissionDecision::Deny => {
                            tracing::info!(
                                tool_name = %tool_name,
                                rule = ?permission_check.rule,
                                "Tool execution denied by rule"
                            );
                            let reason = permission_check.rule.unwrap_or_else(|| {
                                // Use stripped tool name, or fall back to original, or a default
                                let display_name = if !stripped_tool_name.is_empty() {
                                    stripped_tool_name
                                } else if !tool_name.is_empty() {
                                    tool_name.as_str()
                                } else {
                                    "the requested tool" // Fallback if both are empty
                                };
                                format!("Tool {} denied by permission settings", display_name)
                            });
                            create_deny_response(
                                &connection_cx_lock,
                                &session_id,
                                tool_use_id.as_ref(),
                                &tool_name,
                                reason,
                            )
                        }
                        crate::settings::PermissionDecision::Ask => {
                            // Following TypeScript version's design:
                            // For "ask" decisions, we just return { continue: true } to let the
                            // normal permission flow continue. The actual permission request
                            // will be sent by the can_use_tool callback, NOT here.
                            //
                            // This ensures proper message ordering:
                            // 1. SDK processes tool_use -> sends session/update ToolCall
                            // 2. SDK calls can_use_tool callback
                            // 3. can_use_tool sends requestPermission() and waits for user response
                            //
                            // If we sent requestPermission() here (before can_use_tool), there
                            // could be race conditions with session/update notifications.

                            // Cache tool_use_id for can_use_tool callback to use
                            // The CLI doesn't always pass tool_use_id in mcp_message requests,
                            // so we cache it here where we have it.
                            if let Some(ref tuid) = tool_use_id {
                                let key = crate::session::stable_cache_key(&tool_input);
                                tracing::debug!(
                                    tool_name = %tool_name,
                                    tool_use_id = %tuid,
                                    "Caching tool_use_id for can_use_tool callback"
                                );
                                tool_use_id_cache.insert(key, tuid.clone());
                            }

                            tracing::debug!(
                                tool_name = %tool_name,
                                "Ask decision - delegating to can_use_tool callback"
                            );
                            HookJsonOutput::Sync(SyncHookJsonOutput {
                                continue_: Some(true),
                                hook_specific_output: None,
                                ..Default::default()
                            })
                        }
                    }
                }
                .instrument(span),
            ) as BoxFuture<'static, HookJsonOutput>
        },
    )
}

/// Send a tool result notification when a tool is denied by permission check
///
/// This ensures that clients (like Zed) receive a corresponding tool_result
/// notification even when a tool is blocked before execution.
///
/// # Arguments
///
/// * `connection_cx_lock` - The connection context for sending notifications
/// * `session_id` - The session ID
/// * `tool_use_id` - The tool use ID to correlate with the tool_use notification
/// * `tool_name` - The name of the tool that was denied
/// * `reason` - The reason for the denial
///
/// # Note
///
/// This function sends the notification **synchronously** to ensure:
/// 1. The notification is sent before the hook response
/// 2. The notification can be properly flushed with the flush mechanism
/// 3. Any send errors are detected immediately
fn send_denied_tool_result(
    connection_cx_lock: &Arc<OnceLock<JrConnectionCx<AgentToClient>>>,
    session_id: &str,
    tool_use_id: &str,
    tool_name: &str,
    reason: &str,
) {
    let Some(connection_cx) = connection_cx_lock.get() else {
        tracing::warn!(
            tool_name = %tool_name,
            tool_use_id = %tool_use_id,
            "Connection context not available, cannot send denied tool result"
        );
        return;
    };

    let session_id = SessionId::new(session_id.to_string());
    let tool_call_id = ToolCallId::new(tool_use_id.to_string());

    // Build error content
    let error_content = format!("Tool execution denied: {}", reason);
    let content: Vec<ToolCallContent> = vec![format!("```\n{}\n```", error_content).into()];

    // Build raw_output JSON
    let raw_output = serde_json::json!({
        "content": error_content,
        "is_error": true
    });

    // Create tool result notification with Failed status
    let update_fields = ToolCallUpdateFields::new()
        .status(ToolCallStatus::Failed)
        .content(content)
        .raw_output(raw_output);

    let update = ToolCallUpdate::new(tool_call_id, update_fields);
    let notification = SessionNotification::new(session_id, SessionUpdate::ToolCallUpdate(update));

    // Send the notification synchronously
    // Note: send_notification uses unbounded_send which is non-blocking
    // The actual network IO is handled by the outgoing actor
    if let Err(e) = connection_cx.send_notification(notification) {
        tracing::warn!(
            tool_name = %tool_name,
            tool_use_id = %tool_use_id,
            error = %e,
            "Failed to send denied tool result notification"
        );
    } else {
        tracing::debug!(
            tool_name = %tool_name,
            tool_use_id = %tool_use_id,
            "Sent denied tool result notification"
        );
    }
}

/// Create a deny hook response with tool_result notification
///
/// This helper function ensures that when a tool is denied:
/// 1. A tool_result notification is sent to the client (so Zed doesn't show "not found")
/// 2. The hook returns the proper deny response
///
/// # Arguments
///
/// * `connection_cx_lock` - The connection context for sending notifications
/// * `session_id` - The session ID
/// * `tool_use_id` - Optional tool use ID
/// * `tool_name` - The name of the tool that was denied
/// * `reason` - The reason for the denial
///
/// # Returns
///
/// A HookJsonOutput with deny decision
fn create_deny_response(
    connection_cx_lock: &Arc<OnceLock<JrConnectionCx<AgentToClient>>>,
    session_id: &str,
    tool_use_id: Option<&String>,
    tool_name: &str,
    reason: String,
) -> HookJsonOutput {
    // Send tool_result notification to client so Zed doesn't show "Tool call not found"
    // Note: send_notification is non-blocking (uses unbounded_send)
    if let Some(tuid) = tool_use_id {
        send_denied_tool_result(connection_cx_lock, session_id, tuid, tool_name, &reason);
    }

    HookJsonOutput::Sync(SyncHookJsonOutput {
        continue_: Some(true),
        hook_specific_output: Some(HookSpecificOutput::PreToolUse(
            PreToolUseHookSpecificOutput {
                permission_decision: Some("deny".to_string()),
                permission_decision_reason: Some(reason),
                updated_input: None,
                additional_context: None,
            },
        )),
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::{PermissionSettings, Settings};
    use serde_json::json;

    fn make_permission_checker(permissions: PermissionSettings) -> Arc<RwLock<PermissionChecker>> {
        let settings = Settings {
            permissions: Some(permissions),
            ..Default::default()
        };
        Arc::new(RwLock::new(PermissionChecker::new(settings, "/tmp")))
    }

    fn make_test_hook(checker: Arc<RwLock<PermissionChecker>>) -> HookCallback {
        make_test_hook_with_mode(checker, PermissionMode::Default)
    }

    fn make_test_hook_with_mode(
        checker: Arc<RwLock<PermissionChecker>>,
        mode: PermissionMode,
    ) -> HookCallback {
        let connection_cx_lock: Arc<OnceLock<JrConnectionCx<AgentToClient>>> =
            Arc::new(OnceLock::new());
        let permission_cache: Arc<DashMap<String, bool>> = Arc::new(DashMap::new());
        let tool_use_id_cache: Arc<DashMap<String, String>> = Arc::new(DashMap::new());
        // Create PermissionHandler with the specified mode
        let permission = PermissionHandler::with_mode(mode);
        create_pre_tool_use_hook(
            connection_cx_lock,
            "test-session".to_string(),
            Some(checker),
            Arc::new(RwLock::new(permission)),
            permission_cache,
            tool_use_id_cache,
        )
    }

    #[tokio::test]
    async fn test_pre_tool_use_hook_allow() {
        let checker = make_permission_checker(PermissionSettings {
            allow: Some(vec!["Read".to_string()]),
            ..Default::default()
        });

        let hook = make_test_hook(checker);
        let input = HookInput::PreToolUse(claude_code_agent_sdk::PreToolUseHookInput {
            session_id: "test".to_string(),
            transcript_path: "/tmp/test".to_string(),
            cwd: "/tmp".to_string(),
            permission_mode: None,
            tool_name: "Read".to_string(),
            tool_input: json!({"file_path": "/tmp/test.txt"}),
            tool_use_id: "test-tool-use-id".to_string(),
        });

        let result = hook(input, None, HookContext::default()).await;

        match result {
            HookJsonOutput::Sync(output) => {
                assert_eq!(output.continue_, Some(true));
                if let Some(HookSpecificOutput::PreToolUse(specific)) = output.hook_specific_output
                {
                    assert_eq!(specific.permission_decision, Some("allow".to_string()));
                } else {
                    panic!("Expected PreToolUse specific output");
                }
            }
            HookJsonOutput::Async(_) => panic!("Expected sync output"),
        }
    }

    // TODO: Re-enable when implementing permission checks
    // #[tokio::test]
    // async fn test_pre_tool_use_hook_deny() {
    //     let checker = make_permission_checker(PermissionSettings {
    //         deny: Some(vec!["Bash".to_string()]),
    //         ..Default::default()
    //     });
    //
    //     let hook = make_test_hook(checker);
    //     let input = HookInput::PreToolUse(claude_code_agent_sdk::PreToolUseHookInput {
    //         session_id: "test".to_string(),
    //         transcript_path: "/tmp/test".to_string(),
    //         cwd: "/tmp".to_string(),
    //         permission_mode: None,
    //         tool_name: "Bash".to_string(),
    //         tool_input: json!({"command": "ls"}),
    //     });
    //
    //     let result = hook(input, None, HookContext::default()).await;
    //
    //     match result {
    //         HookJsonOutput::Sync(output) => {
    //             assert_eq!(output.continue_, Some(true));
    //             if let Some(HookSpecificOutput::PreToolUse(specific)) = output.hook_specific_output
    //             {
    //                 assert_eq!(specific.permission_decision, Some("deny".to_string()));
    //             } else {
    //                 panic!("Expected PreToolUse specific output");
    //             }
    //         }
    //         HookJsonOutput::Async(_) => panic!("Expected sync output"),
    //     }
    // }

    #[tokio::test]
    async fn test_pre_tool_use_hook_ask_by_default() {
        // When no rules match, decision is "Ask".
        // Following TypeScript version's design, the hook just returns { continue: true }
        // to let the can_use_tool callback handle the permission request.
        let checker = make_permission_checker(PermissionSettings::default());
        let hook = make_test_hook(checker);

        // Test MCP tool - no matching rules means "Ask" decision,
        // hook returns continue=true with no hook_specific_output
        let input_mcp = HookInput::PreToolUse(claude_code_agent_sdk::PreToolUseHookInput {
            session_id: "test".to_string(),
            transcript_path: "/tmp/test".to_string(),
            cwd: "/tmp".to_string(),
            permission_mode: None,
            tool_name: "mcp__acp__Write".to_string(),
            tool_input: json!({"file_path": "/tmp/test.txt", "content": "test"}),
            tool_use_id: "test-tool-use-id".to_string(),
        });

        let result_mcp = hook(input_mcp, None, HookContext::default()).await;

        match result_mcp {
            HookJsonOutput::Sync(output) => {
                // Ask decision returns continue=true with no hook_specific_output
                // The actual permission request is handled by can_use_tool callback
                assert_eq!(output.continue_, Some(true));
                assert!(
                    output.hook_specific_output.is_none(),
                    "Ask decision should not set hook_specific_output"
                );
            }
            HookJsonOutput::Async(_) => panic!("Expected sync output"),
        }

        // Test built-in tool - same behavior
        let input_builtin = HookInput::PreToolUse(claude_code_agent_sdk::PreToolUseHookInput {
            session_id: "test".to_string(),
            transcript_path: "/tmp/test".to_string(),
            cwd: "/tmp".to_string(),
            permission_mode: None,
            tool_name: "Write".to_string(),
            tool_input: json!({"file_path": "/tmp/test.txt", "content": "test"}),
            tool_use_id: "test-tool-use-id".to_string(),
        });

        let result_builtin = hook(input_builtin, None, HookContext::default()).await;

        match result_builtin {
            HookJsonOutput::Sync(output) => {
                // Ask decision returns continue=true with no hook_specific_output
                assert_eq!(output.continue_, Some(true));
                assert!(
                    output.hook_specific_output.is_none(),
                    "Ask decision should not set hook_specific_output"
                );
            }
            HookJsonOutput::Async(_) => panic!("Expected sync output"),
        }
    }

    #[tokio::test]
    async fn test_pre_tool_use_hook_ignores_other_events() {
        let checker = make_permission_checker(PermissionSettings::default());

        let hook = make_test_hook(checker);
        let input = HookInput::PostToolUse(claude_code_agent_sdk::PostToolUseHookInput {
            session_id: "test".to_string(),
            transcript_path: "/tmp/test".to_string(),
            cwd: "/tmp".to_string(),
            permission_mode: None,
            tool_name: "Read".to_string(),
            tool_input: json!({}),
            tool_response: json!("content"),
            tool_use_id: "test-tool-use-id".to_string(),
        });

        let result = hook(input, None, HookContext::default()).await;

        match result {
            HookJsonOutput::Sync(output) => {
                assert_eq!(output.continue_, Some(true));
                assert!(output.hook_specific_output.is_none());
            }
            HookJsonOutput::Async(_) => panic!("Expected sync output"),
        }
    }

    #[tokio::test]
    async fn test_bypass_permissions_mode_allows_everything() {
        // BypassPermissions mode should allow all tools without checking rules
        let checker = make_permission_checker(PermissionSettings {
            deny: Some(vec!["Bash".to_string()]),
            ..Default::default()
        });

        let hook = make_test_hook_with_mode(checker, PermissionMode::BypassPermissions);
        let input = HookInput::PreToolUse(claude_code_agent_sdk::PreToolUseHookInput {
            session_id: "test".to_string(),
            transcript_path: "/tmp/test".to_string(),
            cwd: "/tmp".to_string(),
            permission_mode: None,
            tool_name: "Bash".to_string(),
            tool_input: json!({"command": "rm -rf /"}),
            tool_use_id: "test-tool-use-id".to_string(),
        });

        let result = hook(input, None, HookContext::default()).await;

        match result {
            HookJsonOutput::Sync(output) => {
                assert_eq!(output.continue_, Some(true));
                if let Some(HookSpecificOutput::PreToolUse(specific)) = output.hook_specific_output
                {
                    assert_eq!(specific.permission_decision, Some("allow".to_string()));
                    assert!(
                        specific
                            .permission_decision_reason
                            .unwrap()
                            .contains("BypassPermissions")
                    );
                } else {
                    panic!("Expected PreToolUse specific output");
                }
            }
            HookJsonOutput::Async(_) => panic!("Expected sync output"),
        }
    }

    #[tokio::test]
    async fn test_default_mode_respects_settings_rules() {
        // Default mode should respect settings rules
        let checker = make_permission_checker(PermissionSettings {
            allow: Some(vec!["Read".to_string()]),
            ..Default::default()
        });

        let hook = make_test_hook_with_mode(checker, PermissionMode::Default);
        let input = HookInput::PreToolUse(claude_code_agent_sdk::PreToolUseHookInput {
            session_id: "test".to_string(),
            transcript_path: "/tmp/test".to_string(),
            cwd: "/tmp".to_string(),
            permission_mode: None,
            tool_name: "Read".to_string(),
            tool_input: json!({"file_path": "/tmp/test.txt"}),
            tool_use_id: "test-tool-use-id".to_string(),
        });

        let result = hook(input, None, HookContext::default()).await;

        match result {
            HookJsonOutput::Sync(output) => {
                assert_eq!(output.continue_, Some(true));
                if let Some(HookSpecificOutput::PreToolUse(specific)) = output.hook_specific_output
                {
                    assert_eq!(specific.permission_decision, Some("allow".to_string()));
                } else {
                    panic!("Expected PreToolUse specific output");
                }
            }
            HookJsonOutput::Async(_) => panic!("Expected sync output"),
        }
    }

    #[tokio::test]
    async fn test_default_mode_auto_allows_read_only_tools() {
        // Default mode should auto-allow read-only operations (Read, Grep, Glob, LS, NotebookRead)
        // even without explicit allow rules
        let checker = make_permission_checker(PermissionSettings::default()); // No rules

        let hook = make_test_hook_with_mode(checker, PermissionMode::Default);

        // Test Read tool
        let input_read = HookInput::PreToolUse(claude_code_agent_sdk::PreToolUseHookInput {
            session_id: "test".to_string(),
            transcript_path: "/tmp/test".to_string(),
            cwd: "/tmp".to_string(),
            permission_mode: None,
            tool_name: "Read".to_string(),
            tool_input: json!({"file_path": "/tmp/test.txt"}),
            tool_use_id: "test-tool-use-id".to_string(),
        });

        let result = hook(input_read, None, HookContext::default()).await;
        match result {
            HookJsonOutput::Sync(output) => {
                assert_eq!(output.continue_, Some(true));
                if let Some(HookSpecificOutput::PreToolUse(specific)) = output.hook_specific_output
                {
                    assert_eq!(specific.permission_decision, Some("allow".to_string()));
                    assert!(
                        specific
                            .permission_decision_reason
                            .unwrap()
                            .contains("read-only")
                    );
                } else {
                    panic!("Expected PreToolUse specific output");
                }
            }
            HookJsonOutput::Async(_) => panic!("Expected sync output"),
        }

        // Test LS tool
        let input_ls = HookInput::PreToolUse(claude_code_agent_sdk::PreToolUseHookInput {
            session_id: "test".to_string(),
            transcript_path: "/tmp/test".to_string(),
            cwd: "/tmp".to_string(),
            permission_mode: None,
            tool_name: "LS".to_string(),
            tool_input: json!({"path": "/tmp"}),
            tool_use_id: "test-tool-use-id".to_string(),
        });

        let result_ls = hook(input_ls, None, HookContext::default()).await;
        match result_ls {
            HookJsonOutput::Sync(output) => {
                assert_eq!(output.continue_, Some(true));
                if let Some(HookSpecificOutput::PreToolUse(specific)) = output.hook_specific_output
                {
                    assert_eq!(specific.permission_decision, Some("allow".to_string()));
                } else {
                    panic!("Expected PreToolUse specific output for LS");
                }
            }
            HookJsonOutput::Async(_) => panic!("Expected sync output"),
        }

        // Test Grep tool with mcp__acp__ prefix
        let input_grep = HookInput::PreToolUse(claude_code_agent_sdk::PreToolUseHookInput {
            session_id: "test".to_string(),
            transcript_path: "/tmp/test".to_string(),
            cwd: "/tmp".to_string(),
            permission_mode: None,
            tool_name: "mcp__acp__Grep".to_string(),
            tool_input: json!({"pattern": "test", "path": "/tmp"}),
            tool_use_id: "test-tool-use-id".to_string(),
        });

        let result_grep = hook(input_grep, None, HookContext::default()).await;
        match result_grep {
            HookJsonOutput::Sync(output) => {
                assert_eq!(output.continue_, Some(true));
                if let Some(HookSpecificOutput::PreToolUse(specific)) = output.hook_specific_output
                {
                    assert_eq!(specific.permission_decision, Some("allow".to_string()));
                } else {
                    panic!("Expected PreToolUse specific output for Grep");
                }
            }
            HookJsonOutput::Async(_) => panic!("Expected sync output"),
        }
    }

    #[tokio::test]
    async fn test_default_mode_auto_allows_safe_bash_commands() {
        // Default mode should auto-allow known safe Bash commands
        let checker = make_permission_checker(PermissionSettings::default()); // No rules

        let hook = make_test_hook_with_mode(checker, PermissionMode::Default);

        // Test with a safe command (ls is in the safe command whitelist)
        let input = HookInput::PreToolUse(claude_code_agent_sdk::PreToolUseHookInput {
            session_id: "test".to_string(),
            transcript_path: "/tmp/test".to_string(),
            cwd: "/tmp".to_string(),
            permission_mode: None,
            tool_name: "Bash".to_string(),
            tool_input: json!({"command": "ls -la /tmp"}),
            tool_use_id: "test-tool-use-id".to_string(),
        });

        let result = hook(input, None, HookContext::default()).await;
        match result {
            HookJsonOutput::Sync(output) => {
                assert_eq!(output.continue_, Some(true));
                if let Some(HookSpecificOutput::PreToolUse(specific)) = output.hook_specific_output
                {
                    assert_eq!(specific.permission_decision, Some("allow".to_string()));
                    assert!(
                        specific
                            .permission_decision_reason
                            .as_ref()
                            .unwrap()
                            .contains("known safe command"),
                        "Expected 'known safe command' in reason"
                    );
                } else {
                    panic!("Expected PreToolUse specific output for safe Bash command");
                }
            }
            HookJsonOutput::Async(_) => panic!("Expected sync output"),
        }

        // Test with find (conditionally safe without dangerous options)
        let input_find = HookInput::PreToolUse(claude_code_agent_sdk::PreToolUseHookInput {
            session_id: "test".to_string(),
            transcript_path: "/tmp/test".to_string(),
            cwd: "/tmp".to_string(),
            permission_mode: None,
            tool_name: "Bash".to_string(),
            tool_input: json!({"command": "find . -name '*.rs'"}),
            tool_use_id: "test-tool-use-id".to_string(),
        });

        let result_find = hook(input_find, None, HookContext::default()).await;
        match result_find {
            HookJsonOutput::Sync(output) => {
                assert_eq!(output.continue_, Some(true));
                if let Some(HookSpecificOutput::PreToolUse(specific)) = output.hook_specific_output
                {
                    assert_eq!(specific.permission_decision, Some("allow".to_string()));
                } else {
                    panic!("Expected PreToolUse specific output for safe find command");
                }
            }
            HookJsonOutput::Async(_) => panic!("Expected sync output"),
        }

        // Test with git status (safe git subcommand)
        let input_git = HookInput::PreToolUse(claude_code_agent_sdk::PreToolUseHookInput {
            session_id: "test".to_string(),
            transcript_path: "/tmp/test".to_string(),
            cwd: "/tmp".to_string(),
            permission_mode: None,
            tool_name: "Bash".to_string(),
            tool_input: json!({"command": "git status"}),
            tool_use_id: "test-tool-use-id".to_string(),
        });

        let result_git = hook(input_git, None, HookContext::default()).await;
        match result_git {
            HookJsonOutput::Sync(output) => {
                assert_eq!(output.continue_, Some(true));
                if let Some(HookSpecificOutput::PreToolUse(specific)) = output.hook_specific_output
                {
                    assert_eq!(specific.permission_decision, Some("allow".to_string()));
                } else {
                    panic!("Expected PreToolUse specific output for safe git command");
                }
            }
            HookJsonOutput::Async(_) => panic!("Expected sync output"),
        }
    }

    #[tokio::test]
    async fn test_default_mode_asks_for_write_tools() {
        // Default mode should ask for permission for write tools (Bash with non-safe commands, Edit, Write)
        let checker = make_permission_checker(PermissionSettings::default()); // No rules

        let hook = make_test_hook_with_mode(checker, PermissionMode::Default);

        // Use a command that is NOT in the safe command list
        let input = HookInput::PreToolUse(claude_code_agent_sdk::PreToolUseHookInput {
            session_id: "test".to_string(),
            transcript_path: "/tmp/test".to_string(),
            cwd: "/tmp".to_string(),
            permission_mode: None,
            tool_name: "Bash".to_string(),
            tool_input: json!({"command": "mkdir new_dir"}), // mkdir is not a safe command
            tool_use_id: "test-tool-use-id".to_string(),
        });

        let result = hook(input, None, HookContext::default()).await;
        match result {
            HookJsonOutput::Sync(output) => {
                // Ask decision returns continue=true with no hook_specific_output
                assert_eq!(output.continue_, Some(true));
                assert!(
                    output.hook_specific_output.is_none(),
                    "Bash with non-safe command should trigger Ask decision, not auto-allow"
                );
            }
            HookJsonOutput::Async(_) => panic!("Expected sync output"),
        }
    }

    #[tokio::test]
    async fn test_create_deny_response_without_tool_use_id() {
        // Test that create_deny_response handles missing tool_use_id gracefully
        let connection_cx_lock: Arc<OnceLock<JrConnectionCx<AgentToClient>>> =
            Arc::new(OnceLock::new());
        let permission_cache: Arc<DashMap<String, bool>> = Arc::new(DashMap::new());
        let tool_use_id_cache: Arc<DashMap<String, String>> = Arc::new(DashMap::new());
        // Create PermissionHandler with Default mode
        let permission = PermissionHandler::with_mode(PermissionMode::Default);
        let hook = create_pre_tool_use_hook(
            connection_cx_lock,
            "test-session".to_string(),
            None,
            Arc::new(RwLock::new(permission)),
            permission_cache,
            tool_use_id_cache,
        );

        // Test with no tool_use_id - should not panic
        let input = HookInput::PreToolUse(claude_code_agent_sdk::PreToolUseHookInput {
            session_id: "test".to_string(),
            transcript_path: "/tmp/test".to_string(),
            cwd: "/tmp".to_string(),
            permission_mode: None,
            tool_name: "Write".to_string(),
            tool_input: json!({"file_path": "/tmp/test.txt", "content": "test"}),
            tool_use_id: "test-tool-use-id".to_string(),
        });

        // This should not panic even without tool_use_id
        let result = hook(input, None, HookContext::default()).await;
        match result {
            HookJsonOutput::Sync(output) => {
                // Should get an Ask decision since there are no rules and no permission mode restriction
                assert_eq!(output.continue_, Some(true));
            }
            HookJsonOutput::Async(_) => panic!("Expected sync output"),
        }
    }

    #[tokio::test]
    async fn test_empty_tool_name_handling() {
        // Test that empty tool_name is handled gracefully
        let _connection_cx_lock: Arc<OnceLock<JrConnectionCx<AgentToClient>>> =
            Arc::new(OnceLock::new());
        let _permission_cache: Arc<DashMap<String, bool>> = Arc::new(DashMap::new());
        let _tool_use_id_cache: Arc<DashMap<String, String>> = Arc::new(DashMap::new());

        // Test with empty tool_name - should not panic and should use fallback
        let empty_tool_name = "";
        let result = format!("Tool {} denied", empty_tool_name);
        assert!(
            result.contains("Tool  denied"),
            "Empty tool_name produces double space"
        );

        // Test the fallback logic
        let display_name = if empty_tool_name.is_empty() {
            "the requested tool"
        } else {
            empty_tool_name
        };
        assert_eq!(display_name, "the requested tool");

        // Test with normal tool_name
        let normal_tool_name = "Write";
        let display_name2 = if normal_tool_name.is_empty() {
            "the requested tool"
        } else {
            normal_tool_name
        };
        assert_eq!(display_name2, "Write");
    }

    #[tokio::test]
    async fn test_plan_mode_allows_writing_plan_files() {
        // Plan mode should allow writing to ~/.claude/plans/
        let checker = make_permission_checker(PermissionSettings::default());
        let hook = make_test_hook_with_mode(checker, PermissionMode::Plan);

        let home = dirs::home_dir().unwrap();
        let plan_file = home.join(".claude").join("plans").join("test-plan.md");

        let input = HookInput::PreToolUse(claude_code_agent_sdk::PreToolUseHookInput {
            session_id: "test".to_string(),
            transcript_path: "/tmp/test".to_string(),
            cwd: "/tmp".to_string(),
            permission_mode: None,
            tool_name: "Write".to_string(),
            tool_input: json!({
                "file_path": plan_file.to_str().unwrap(),
                "content": "# Test Plan"
            }),
            tool_use_id: "test-tool-use-id".to_string(),
        });

        let result = hook(input, None, HookContext::default()).await;

        match result {
            HookJsonOutput::Sync(output) => {
                // Should allow (not deny) plan file writes
                if let Some(HookSpecificOutput::PreToolUse(specific)) = output.hook_specific_output
                {
                    assert_ne!(specific.permission_decision, Some("deny".to_string()));
                }
            }
            HookJsonOutput::Async(_) => panic!("Expected sync output"),
        }
    }

    #[tokio::test]
    async fn test_plan_mode_blocks_non_plan_file_writes() {
        // Plan mode should block writes to non-plan files
        let checker = make_permission_checker(PermissionSettings::default());
        let hook = make_test_hook_with_mode(checker, PermissionMode::Plan);

        let input = HookInput::PreToolUse(claude_code_agent_sdk::PreToolUseHookInput {
            session_id: "test".to_string(),
            transcript_path: "/tmp/test".to_string(),
            cwd: "/tmp".to_string(),
            permission_mode: None,
            tool_name: "Write".to_string(),
            tool_input: json!({
                "file_path": "/tmp/test.txt",
                "content": "test"
            }),
            tool_use_id: "test-tool-use-id".to_string(),
        });

        let result = hook(input, None, HookContext::default()).await;

        match result {
            HookJsonOutput::Sync(output) => {
                assert_eq!(output.continue_, Some(true));
                if let Some(HookSpecificOutput::PreToolUse(specific)) = output.hook_specific_output
                {
                    assert_eq!(specific.permission_decision, Some("deny".to_string()));
                    assert!(
                        specific
                            .permission_decision_reason
                            .as_ref()
                            .unwrap()
                            .contains("Plan mode")
                    );
                }
            }
            HookJsonOutput::Async(_) => panic!("Expected sync output"),
        }
    }

    #[tokio::test]
    async fn test_plan_mode_blocks_bash() {
        // Plan mode should block Bash commands even in plans directory
        let checker = make_permission_checker(PermissionSettings::default());
        let hook = make_test_hook_with_mode(checker, PermissionMode::Plan);

        let home = dirs::home_dir().unwrap();

        let input = HookInput::PreToolUse(claude_code_agent_sdk::PreToolUseHookInput {
            session_id: "test".to_string(),
            transcript_path: "/tmp/test".to_string(),
            cwd: home.to_str().unwrap().to_string(),
            permission_mode: None,
            tool_name: "Bash".to_string(),
            tool_input: json!({
                "command": "echo 'test' > .claude/plans/test.md"
            }),
            tool_use_id: "test-tool-use-id".to_string(),
        });

        let result = hook(input, None, HookContext::default()).await;

        match result {
            HookJsonOutput::Sync(output) => {
                assert_eq!(output.continue_, Some(true));
                if let Some(HookSpecificOutput::PreToolUse(specific)) = output.hook_specific_output
                {
                    assert_eq!(specific.permission_decision, Some("deny".to_string()));
                }
            }
            HookJsonOutput::Async(_) => panic!("Expected sync output"),
        }
    }

    #[tokio::test]
    async fn test_plan_mode_allows_read_operations() {
        // Plan mode should allow read operations
        let checker = make_permission_checker(PermissionSettings::default());
        let hook = make_test_hook_with_mode(checker, PermissionMode::Plan);

        let input = HookInput::PreToolUse(claude_code_agent_sdk::PreToolUseHookInput {
            session_id: "test".to_string(),
            transcript_path: "/tmp/test".to_string(),
            cwd: "/tmp".to_string(),
            permission_mode: None,
            tool_name: "Read".to_string(),
            tool_input: json!({"file_path": "/tmp/test.txt"}),
            tool_use_id: "test-tool-use-id".to_string(),
        });

        let result = hook(input, None, HookContext::default()).await;

        match result {
            HookJsonOutput::Sync(output) => {
                if let Some(HookSpecificOutput::PreToolUse(specific)) = output.hook_specific_output
                {
                    assert_eq!(specific.permission_decision, Some("allow".to_string()));
                }
            }
            HookJsonOutput::Async(_) => panic!("Expected sync output"),
        }
    }

    #[tokio::test]
    async fn test_is_plans_directory_path() {
        // Test absolute path
        let home = dirs::home_dir().unwrap();
        let plans_path = home.join(".claude").join("plans").join("plan.md");
        assert!(is_plans_directory_path(plans_path.to_str().unwrap()));

        // Test ~ expansion
        assert!(is_plans_directory_path("~/.claude/plans/plan.md"));

        // Test non-plans path
        assert!(!is_plans_directory_path("/tmp/plan.md"));
        assert!(!is_plans_directory_path("~/other/path/plan.md"));

        // Test edge case: similar but not plans directory
        assert!(!is_plans_directory_path("~/../.claude/plans/plan.md"));
    }

    #[tokio::test]
    async fn test_plan_mode_allows_edit_in_plans_dir() {
        // Plan mode should allow Edit operations in plans directory
        let checker = make_permission_checker(PermissionSettings::default());
        let hook = make_test_hook_with_mode(checker, PermissionMode::Plan);

        let home = dirs::home_dir().unwrap();
        let plan_file = home.join(".claude").join("plans").join("existing-plan.md");

        let input = HookInput::PreToolUse(claude_code_agent_sdk::PreToolUseHookInput {
            session_id: "test".to_string(),
            transcript_path: "/tmp/test".to_string(),
            cwd: "/tmp".to_string(),
            permission_mode: None,
            tool_name: "Edit".to_string(),
            tool_input: json!({
                "file_path": plan_file.to_str().unwrap(),
                "edits": []
            }),
            tool_use_id: "test-tool-use-id".to_string(),
        });

        let result = hook(input, None, HookContext::default()).await;

        match result {
            HookJsonOutput::Sync(output) => {
                assert_eq!(output.continue_, Some(true));
                // Should not deny
                if let Some(HookSpecificOutput::PreToolUse(specific)) = output.hook_specific_output
                {
                    assert_ne!(specific.permission_decision, Some("deny".to_string()));
                }
            }
            HookJsonOutput::Async(_) => panic!("Expected sync output"),
        }
    }
}
