//! can_use_tool callback implementation for SDK
//!
//! This module implements the permission checking callback that the SDK calls
//! before executing tools. Following the TypeScript version's design, this callback
//! is responsible for sending permission requests when needed.

use claude_code_agent_sdk::types::permissions::{
    CanUseToolCallback, PermissionResult, PermissionResultAllow, PermissionResultDeny,
    PermissionUpdate, PermissionUpdateDestination, PermissionUpdateType, ToolPermissionContext,
};
use sacp::schema::{
    Content, ContentBlock, PermissionOption, PermissionOptionId, PermissionOptionKind,
    RequestPermissionOutcome, RequestPermissionRequest, SessionId, TextContent, ToolCallContent,
    ToolCallUpdate, ToolCallUpdateFields,
};
use sacp::{JrConnectionCx, link::AgentToClient};
use std::sync::{Arc, OnceLock};
use tracing::{debug, info, warn};

use crate::session::{
    PermissionMode, PermissionOutcome, PermissionRequestBuilder, Session, ToolPermissionResult,
};
use crate::types::AgentError;
use std::fs;
use std::path::PathBuf;

/// ExitPlanMode specific permission outcome
#[derive(Debug, Clone, PartialEq, Eq)]
enum ExitPlanModeOutcome {
    /// User approved to exit plan mode with specified permission mode
    Approve(PermissionMode),
    /// User chose to keep planning
    KeepPlanning,
}

/// Read the most recent plan file from ~/.claude/plans/
///
/// Returns Ok(Some(content)) if plan file is found and readable,
/// Ok(None) if no plan file exists or file is too large (>20MB),
/// or Err if there's an error reading.
fn read_plan_file() -> Result<Option<String>, std::io::Error> {
    // Maximum plan file size: 20MB
    // Plan files are typically small (a few KB), but we add a safety limit
    const MAX_PLAN_FILE_SIZE: u64 = 20 * 1024 * 1024; // 20MB

    // Get the home directory
    let Some(home) = dirs::home_dir() else {
        return Ok(None);
    };

    let plans_dir = home.join(".claude").join("plans");

    // Check if plans directory exists
    if !plans_dir.exists() {
        return Ok(None);
    }

    // Read the directory and find the most recently modified .md file
    let entries = fs::read_dir(&plans_dir)?;
    let mut most_recent_file: Option<PathBuf> = None;
    let mut most_recent_mtime: std::time::SystemTime = std::time::SystemTime::UNIX_EPOCH;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("md") {
            if let Ok(metadata) = entry.metadata() {
                if let Ok(mtime) = metadata.modified() {
                    if mtime > most_recent_mtime {
                        most_recent_mtime = mtime;
                        most_recent_file = Some(path);
                    }
                }
            }
        }
    }

    // Read the most recent plan file
    if let Some(file_path) = most_recent_file {
        // Check file size before reading
        let metadata = fs::metadata(&file_path)?;
        let file_size = metadata.len();

        if file_size > MAX_PLAN_FILE_SIZE {
            warn!(
                "Plan file too large ({} bytes > {} limit), skipping: {:?}",
                file_size, MAX_PLAN_FILE_SIZE, file_path
            );
            return Ok(None);
        }

        match fs::read_to_string(&file_path) {
            Ok(content) => {
                info!(
                    "Read plan file: {:?} (size: {} bytes)",
                    file_path, file_size
                );
                Ok(Some(content))
            }
            Err(e) => {
                warn!("Failed to read plan file {:?}: {}", file_path, e);
                Err(e)
            }
        }
    } else {
        Ok(None)
    }
}

/// Send ExitPlanMode permission request with custom options
async fn send_exit_plan_mode_request(
    session_id: &str,
    tool_use_id: &str,
    tool_input: &serde_json::Value,
    connection_cx: &JrConnectionCx<AgentToClient>,
) -> Result<ExitPlanModeOutcome, AgentError> {
    // ExitPlanMode specific options matching TypeScript implementation
    let options = vec![
        PermissionOption::new(
            PermissionOptionId::new("acceptEdits"),
            "Yes, and auto-accept edits",
            PermissionOptionKind::AllowAlways,
        ),
        PermissionOption::new(
            PermissionOptionId::new("default"),
            "Yes, and manually approve edits",
            PermissionOptionKind::AllowOnce,
        ),
        PermissionOption::new(
            PermissionOptionId::new("plan"),
            "No, keep planning",
            PermissionOptionKind::RejectOnce,
        ),
    ];

    // Determine the raw input to display
    // Priority: 1. Use 'plan' field from tool_input if provided
    //           2. Try to read from plan file
    //           3. Fall back to original tool_input
    let (raw_input, plan_content_for_display) =
        if let Some(plan_content) = tool_input.get("plan").and_then(|v| v.as_str()) {
            // Agent provided plan content
            (
                serde_json::json!({"plan": plan_content}),
                Some(plan_content.to_string()),
            )
        } else {
            // Try to read the most recent plan file
            match read_plan_file() {
                Ok(Some(plan_content)) => {
                    info!("Read plan file content for ExitPlanMode display");
                    (
                        serde_json::json!({"plan": plan_content}),
                        Some(plan_content),
                    )
                }
                Ok(None) => {
                    warn!("No plan file found, using original tool_input");
                    (tool_input.clone(), None)
                }
                Err(e) => {
                    warn!("Failed to read plan file: {}, using original tool_input", e);
                    (tool_input.clone(), None)
                }
            }
        };

    // Build content array for plan display
    // This follows TypeScript implementation: content: [{ type: "content", content: { type: "text", text: plan } }]
    let content = if let Some(plan_text) = plan_content_for_display {
        vec![ToolCallContent::Content(Content::new(ContentBlock::Text(
            TextContent::new(plan_text),
        )))]
    } else {
        vec![]
    };

    // Build tool call update with title, content, and raw input
    // Following TypeScript implementation: toolInfoFromToolUse for ExitPlanMode
    let tool_call_update = ToolCallUpdate::new(
        tool_use_id.to_string(),
        ToolCallUpdateFields::new()
            .title("Ready to code?")
            .content(content)
            .raw_input(raw_input),
    );

    // Build the request
    let request = RequestPermissionRequest::new(
        SessionId::new(session_id.to_string()),
        tool_call_update,
        options,
    );

    tracing::info!(
        session_id = %session_id,
        tool_use_id = %tool_use_id,
        "Sending ExitPlanMode permission request"
    );

    // Send request and wait for response
    let response = connection_cx
        .send_request(request)
        .block_task()
        .await
        .map_err(|e| {
            tracing::error!(
                session_id = %session_id,
                error = %e,
                "ExitPlanMode permission request failed"
            );
            AgentError::Internal(format!("Permission request failed: {}", e))
        })?;

    // Parse the response
    match response.outcome {
        RequestPermissionOutcome::Selected(selected) => match &*selected.option_id.0 {
            "acceptEdits" => {
                info!("User selected: Yes, and auto-accept edits");
                Ok(ExitPlanModeOutcome::Approve(PermissionMode::AcceptEdits))
            }
            "default" => {
                info!("User selected: Yes, and manually approve edits");
                Ok(ExitPlanModeOutcome::Approve(PermissionMode::Default))
            }
            "plan" => {
                info!("User selected: No, keep planning");
                Ok(ExitPlanModeOutcome::KeepPlanning)
            }
            _ => {
                warn!(
                    "Unknown option_id: {}, treating as keep planning",
                    &*selected.option_id.0
                );
                Ok(ExitPlanModeOutcome::KeepPlanning)
            }
        },
        RequestPermissionOutcome::Cancelled => {
            info!("ExitPlanMode permission request was cancelled");
            Ok(ExitPlanModeOutcome::KeepPlanning)
        }
        _ => {
            // Handle non_exhaustive enum - treat any new variants as keep planning
            info!("ExitPlanMode permission request got unexpected response, keeping planning");
            Ok(ExitPlanModeOutcome::KeepPlanning)
        }
    }
}

/// Handle ExitPlanMode tool with special permission dialog
async fn handle_exit_plan_mode(
    session: &Session,
    tool_use_id: &str,
    tool_input: &serde_json::Value,
) -> PermissionResult {
    // Clone tool_input once to avoid double cloning
    // This is reused for both the permission request and the return value
    let tool_input = tool_input.clone();

    // Get connection_cx from session
    let Some(connection_cx) = session.get_connection_cx() else {
        warn!(
            session_id = %session.session_id,
            "Connection not ready for ExitPlanMode"
        );
        return PermissionResult::Deny(PermissionResultDeny {
            message: "Connection not ready for ExitPlanMode".to_string(),
            interrupt: false,
        });
    };

    // Send ExitPlanMode permission request
    match send_exit_plan_mode_request(&session.session_id, tool_use_id, &tool_input, connection_cx)
        .await
    {
        Ok(ExitPlanModeOutcome::Approve(mode)) => {
            info!(
                session_id = %session.session_id,
                mode = ?mode,
                "ExitPlanMode approved, switching to new mode"
            );

            // Update session permission mode
            session.set_permission_mode(mode).await;

            // Send session/update notification
            session.send_mode_update(mode.as_str());

            // Return Allow with updated_permissions (matching TypeScript implementation)
            // This tells the SDK:
            // 1. Allow the ExitPlanMode tool to execute
            // 2. Apply the permission mode update
            // 3. Subsequent tools will use the new mode
            PermissionResult::Allow(PermissionResultAllow {
                updated_input: Some(tool_input),
                updated_permissions: Some(vec![PermissionUpdate {
                    type_: PermissionUpdateType::SetMode,
                    rules: None,
                    behavior: None,
                    mode: Some(mode.to_sdk_mode()),
                    directories: None,
                    destination: Some(PermissionUpdateDestination::Session),
                }]),
            })
        }
        Ok(ExitPlanModeOutcome::KeepPlanning) => {
            info!(
                session_id = %session.session_id,
                "ExitPlanMode rejected, staying in Plan mode"
            );
            PermissionResult::Deny(PermissionResultDeny {
                message: "Plan mode continued. You can keep working on your plan.".to_string(),
                interrupt: false,
            })
        }
        Err(e) => {
            warn!(
                session_id = %session.session_id,
                error = %e,
                "ExitPlanMode request failed"
            );
            PermissionResult::Deny(PermissionResultDeny {
                message: format!("ExitPlanMode failed: {}", e),
                interrupt: false,
            })
        }
    }
}

/// Create a can_use_tool callback that receives Session via OnceLock
///
/// Following TypeScript version's design, this callback:
/// 1. Checks permission rules first
/// 2. For "ask" decisions, sends requestPermission() and waits for user response
/// 3. Returns the appropriate allow/deny result
pub fn create_can_use_tool_callback(
    session_lock: Arc<OnceLock<Arc<Session>>>,
) -> CanUseToolCallback {
    Arc::new(
        move |tool_name: String, tool_input: serde_json::Value, context: ToolPermissionContext| {
            let session_lock = Arc::clone(&session_lock);

            Box::pin(async move {
                debug!(
                    tool_name = %tool_name,
                    tool_use_id = ?context.tool_use_id,
                    "can_use_tool callback called"
                );

                // Try to get session from OnceLock
                let Some(session) = session_lock.get() else {
                    warn!(
                        tool_name = %tool_name,
                        "Session not ready in callback - denying"
                    );
                    return PermissionResult::Deny(PermissionResultDeny {
                        message: "Session not initialized yet".to_string(),
                        interrupt: false,
                    });
                };

                // Special handling for ExitPlanMode - show custom permission dialog
                // This must be done before the permission check, as ExitPlanMode
                // needs to show a "Ready to code?" prompt regardless of current mode
                if tool_name == "ExitPlanMode" || tool_name == "mcp__acp__ExitPlanMode" {
                    info!(
                        tool_name = %tool_name,
                        "ExitPlanMode detected - handling with special permission request"
                    );

                    // Get tool_use_id from context or cache
                    let tool_use_id = match context.tool_use_id {
                        Some(id) => id,
                        None => {
                            if let Some(cached_id) = session.get_cached_tool_use_id(&tool_input) {
                                cached_id
                            } else {
                                warn!("No tool_use_id available for ExitPlanMode");
                                return PermissionResult::Deny(PermissionResultDeny {
                                    message: "No tool_use_id available for ExitPlanMode"
                                        .to_string(),
                                    interrupt: false,
                                });
                            }
                        }
                    };

                    return handle_exit_plan_mode(session, &tool_use_id, &tool_input).await;
                }

                // Check permission handler first
                let handler_guard = session.permission().await;
                let result = handler_guard
                    .check_permission(&tool_name, &tool_input)
                    .await;
                drop(handler_guard); // Release the lock before async operations

                match result {
                    ToolPermissionResult::Allowed => {
                        info!(
                            tool_name = %tool_name,
                            "Permission allowed by handler"
                        );
                        PermissionResult::Allow(PermissionResultAllow::default())
                    }
                    ToolPermissionResult::Blocked { reason } => {
                        info!(
                            tool_name = %tool_name,
                            reason = %reason,
                            "Permission blocked by handler"
                        );
                        PermissionResult::Deny(PermissionResultDeny {
                            message: reason,
                            interrupt: false,
                        })
                    }
                    ToolPermissionResult::NeedsPermission => {
                        // This is the "ask" case - send permission request to client
                        // Following TypeScript version's design
                        info!(
                            tool_name = %tool_name,
                            "Permission needed - sending request to client"
                        );

                        // Get tool_use_id from context, or from cache if not provided
                        // The cache is populated by pre_tool_use hook when Ask decision is made
                        let tool_use_id = match context.tool_use_id {
                            Some(id) => id,
                            None => {
                                // Try to get from cache (populated by pre_tool_use hook)
                                if let Some(cached_id) = session.get_cached_tool_use_id(&tool_input)
                                {
                                    debug!(
                                        tool_name = %tool_name,
                                        cached_tool_use_id = %cached_id,
                                        "Using cached tool_use_id from hook"
                                    );
                                    cached_id
                                } else {
                                    warn!(
                                        tool_name = %tool_name,
                                        "No tool_use_id in context or cache - denying for security"
                                    );
                                    return PermissionResult::Deny(PermissionResultDeny {
                                        message: "No tool_use_id available for permission request"
                                            .to_string(),
                                        interrupt: false,
                                    });
                                }
                            }
                        };

                        // Get connection_cx from session
                        let Some(connection_cx) = session.get_connection_cx() else {
                            warn!(
                                tool_name = %tool_name,
                                "Connection not ready - denying for security"
                            );
                            return PermissionResult::Deny(PermissionResultDeny {
                                message: "Connection not ready for permission request".to_string(),
                                interrupt: false,
                            });
                        };

                        // Send permission request and wait for response
                        let outcome = PermissionRequestBuilder::new(
                            &session.session_id,
                            &tool_use_id,
                            &tool_name,
                            tool_input.clone(),
                        )
                        .request(connection_cx)
                        .await;

                        match outcome {
                            Ok(PermissionOutcome::AllowOnce) => {
                                info!(tool_name = %tool_name, "Permission allowed once by user");
                                PermissionResult::Allow(PermissionResultAllow::default())
                            }
                            Ok(PermissionOutcome::AllowAlways) => {
                                info!(tool_name = %tool_name, "Permission allowed always by user");
                                // Add rule to permission checker for future invocations
                                let handler_guard = session.permission().await;
                                handler_guard
                                    .add_allow_rule_for_tool_call(&tool_name, &tool_input)
                                    .await;
                                drop(handler_guard);
                                PermissionResult::Allow(PermissionResultAllow::default())
                            }
                            Ok(PermissionOutcome::Rejected | PermissionOutcome::Cancelled) => {
                                info!(tool_name = %tool_name, "Permission rejected/cancelled by user");
                                PermissionResult::Deny(PermissionResultDeny {
                                    message: "User denied permission".to_string(),
                                    interrupt: false,
                                })
                            }
                            Err(e) => {
                                warn!(
                                    tool_name = %tool_name,
                                    error = %e,
                                    "Permission request failed"
                                );
                                PermissionResult::Deny(PermissionResultDeny {
                                    message: format!("Permission request failed: {}", e),
                                    interrupt: false,
                                })
                            }
                        }
                    }
                }
            })
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: The callback now requires Arc<OnceLock<Arc<Session>>>
    // which requires a full Session setup to test.
    // The basic test below verifies the callback compiles correctly.
    // Functional tests require integration testing with a real session.

    #[test]
    fn test_callback_function_compiles() {
        // This test verifies the callback function signature is correct
        let session_lock: Arc<OnceLock<Arc<Session>>> = Arc::new(OnceLock::new());
        let _callback = create_can_use_tool_callback(session_lock);
        // If this compiles, the signature is correct
    }
}
