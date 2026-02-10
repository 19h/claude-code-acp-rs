//! PostToolUse hook implementation
//!
//! Executes registered callbacks after tool execution completes.

use std::sync::Arc;
use std::time::Instant;

use claude_code_agent_sdk::{
    HookCallback, HookContext, HookInput, HookJsonOutput, SyncHookJsonOutput,
};
use futures::future::BoxFuture;
use tracing::Instrument;

use super::callback_registry::HookCallbackRegistry;

/// Creates a PostToolUse hook that executes registered callbacks.
///
/// This hook runs after tool execution and invokes any callbacks registered
/// for the tool use ID. Callbacks can be used to send updates to the ACP client.
///
/// # Arguments
///
/// * `callback_registry` - The callback registry to use for looking up callbacks
///
/// # Returns
///
/// A hook callback that can be used with ClaudeAgentOptions
pub fn create_post_tool_use_hook(callback_registry: Arc<HookCallbackRegistry>) -> HookCallback {
    Arc::new(
        move |input: HookInput, tool_use_id: Option<String>, _context: HookContext| {
            let callback_registry = callback_registry.clone();

            // Extract tool name early for span naming
            let (tool_name, is_post_tool) = match &input {
                HookInput::PostToolUse(post_tool) => (post_tool.tool_name.clone(), true),
                _ => (String::new(), false),
            };

            // Create a span for this hook execution
            let span = if is_post_tool {
                tracing::info_span!(
                    "post_tool_use_hook",
                    tool_name = %tool_name,
                    tool_use_id = ?tool_use_id,
                    callback_executed = tracing::field::Empty,
                    callback_duration_us = tracing::field::Empty,
                    total_duration_us = tracing::field::Empty,
                )
            } else {
                tracing::debug_span!(
                    "post_tool_use_hook_skip",
                    event_type = ?std::mem::discriminant(&input)
                )
            };

            Box::pin(
                async move {
                    let start_time = Instant::now();

                    // Only handle PostToolUse events
                    let (tool_name, tool_input, tool_response) =
                        if let HookInput::PostToolUse(post_tool) = &input {
                            (
                                post_tool.tool_name.clone(),
                                post_tool.tool_input.clone(),
                                post_tool.tool_response.clone(),
                            )
                        } else {
                            tracing::debug!("Ignoring non-PostToolUse event");
                            return HookJsonOutput::Sync(SyncHookJsonOutput {
                                continue_: Some(true),
                                ..Default::default()
                            });
                        };

                    // Get response preview for logging
                    let response_preview = tool_response
                        .as_str()
                        .map(|s| s.chars().take(100).collect::<String>())
                        .unwrap_or_else(|| tool_response.to_string().chars().take(100).collect());

                    tracing::debug!(
                        tool_name = %tool_name,
                        tool_use_id = ?tool_use_id,
                        response_preview = %response_preview,
                        "PostToolUse hook triggered"
                    );

                    // Execute callback if registered
                    let callback_executed = if let Some(ref tool_use_id) = tool_use_id {
                        let callback_start = Instant::now();
                        let executed = callback_registry
                            .execute_post_tool_use(tool_use_id, tool_input.clone(), tool_response)
                            .await;
                        let callback_elapsed = callback_start.elapsed();

                        // Record callback execution to span (batched for performance)
                        let span = tracing::Span::current();
                        span.record("callback_executed", executed);
                        span.record("callback_duration_us", callback_elapsed.as_micros());

                        if executed {
                            tracing::info!(
                                tool_name = %tool_name,
                                tool_use_id = %tool_use_id,
                                callback_elapsed_us = callback_elapsed.as_micros(),
                                "PostToolUse callback executed"
                            );
                        } else {
                            tracing::trace!(
                                tool_name = %tool_name,
                                tool_use_id = %tool_use_id,
                                "No callback registered for tool"
                            );
                        }

                        executed
                    } else {
                        tracing::trace!(
                            tool_name = %tool_name,
                            "No tool_use_id provided for PostToolUse hook"
                        );
                        false
                    };

                    let elapsed = start_time.elapsed();
                    tracing::Span::current().record("total_duration_us", elapsed.as_micros());

                    tracing::debug!(
                        tool_name = %tool_name,
                        tool_use_id = ?tool_use_id,
                        callback_executed = callback_executed,
                        total_elapsed_us = elapsed.as_micros(),
                        "PostToolUse hook completed"
                    );

                    HookJsonOutput::Sync(SyncHookJsonOutput {
                        continue_: Some(true),
                        ..Default::default()
                    })
                }
                .instrument(span),
            ) as BoxFuture<'static, HookJsonOutput>
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::FutureExt;
    use serde_json::json;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[tokio::test]
    async fn test_post_tool_use_hook_executes_callback() {
        let registry = Arc::new(HookCallbackRegistry::new());
        let was_called = Arc::new(AtomicBool::new(false));
        let was_called_clone = was_called.clone();

        registry.register_post_tool_use(
            "test-id".to_string(),
            Box::new(move |_id, _input, _response| {
                let was_called = was_called_clone.clone();
                async move {
                    was_called.store(true, Ordering::SeqCst);
                }
                .boxed()
            }),
        );

        let hook = create_post_tool_use_hook(registry);
        let input = HookInput::PostToolUse(claude_code_agent_sdk::PostToolUseHookInput {
            session_id: "test".to_string(),
            transcript_path: "/tmp/test".to_string(),
            cwd: "/tmp".to_string(),
            permission_mode: None,
            tool_name: "Bash".to_string(),
            tool_input: json!({"command": "ls"}),
            tool_response: json!("file1\nfile2"),
            tool_use_id: "test-tool-use-id".to_string(),
        });

        let result = hook(input, Some("test-id".to_string()), HookContext::default()).await;

        match result {
            HookJsonOutput::Sync(output) => {
                assert_eq!(output.continue_, Some(true));
            }
            HookJsonOutput::Async(_) => panic!("Expected sync output"),
        }

        assert!(was_called.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_post_tool_use_hook_no_callback() {
        let registry = Arc::new(HookCallbackRegistry::new());
        let hook = create_post_tool_use_hook(registry);

        let input = HookInput::PostToolUse(claude_code_agent_sdk::PostToolUseHookInput {
            session_id: "test".to_string(),
            transcript_path: "/tmp/test".to_string(),
            cwd: "/tmp".to_string(),
            permission_mode: None,
            tool_name: "Read".to_string(),
            tool_input: json!({"file_path": "/tmp/test.txt"}),
            tool_response: json!("content"),
            tool_use_id: "test-tool-use-id".to_string(),
        });

        let result = hook(
            input,
            Some("nonexistent-id".to_string()),
            HookContext::default(),
        )
        .await;

        match result {
            HookJsonOutput::Sync(output) => {
                assert_eq!(output.continue_, Some(true));
            }
            HookJsonOutput::Async(_) => panic!("Expected sync output"),
        }
    }

    #[tokio::test]
    async fn test_post_tool_use_hook_no_tool_use_id() {
        let registry = Arc::new(HookCallbackRegistry::new());
        let hook = create_post_tool_use_hook(registry);

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
            }
            HookJsonOutput::Async(_) => panic!("Expected sync output"),
        }
    }

    #[tokio::test]
    async fn test_post_tool_use_hook_ignores_other_events() {
        let registry = Arc::new(HookCallbackRegistry::new());
        let hook = create_post_tool_use_hook(registry);

        let input = HookInput::PreToolUse(claude_code_agent_sdk::PreToolUseHookInput {
            session_id: "test".to_string(),
            transcript_path: "/tmp/test".to_string(),
            cwd: "/tmp".to_string(),
            permission_mode: None,
            tool_name: "Read".to_string(),
            tool_input: json!({}),
            tool_use_id: "test-tool-use-id".to_string(),
        });

        let result = hook(input, Some("test-id".to_string()), HookContext::default()).await;

        match result {
            HookJsonOutput::Sync(output) => {
                assert_eq!(output.continue_, Some(true));
            }
            HookJsonOutput::Async(_) => panic!("Expected sync output"),
        }
    }
}
