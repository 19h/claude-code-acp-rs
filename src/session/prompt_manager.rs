//! Prompt task management
//!
//! This module provides the PromptManager for tracking and cancelling
//! active prompts per session. It ensures that only one prompt runs at
//! a time per session, and automatically cancels old prompts when new ones arrive.

use dashmap::DashMap;
use std::time::Instant;
use tokio::task::JoinHandle;

/// Prompt task identifier
pub type PromptId = String;

/// Prompt task wrapper
///
/// Contains all the information needed to track and cancel a prompt task.
#[derive(Debug)]
pub struct PromptTask {
    /// Unique identifier for this prompt
    pub id: PromptId,
    /// JoinHandle for the task (used to wait for completion)
    pub handle: JoinHandle<()>,
    /// Cancellation token (used to signal cancellation)
    pub cancel_token: tokio_util::sync::CancellationToken,
    /// When this prompt was created
    pub created_at: Instant,
    /// Which session this prompt belongs to
    pub session_id: String,
}

/// Prompt manager
///
/// Tracks active prompts per session and ensures serialization:
/// - Only one prompt can run at a time per session
/// - New prompts automatically cancel old prompts
/// - Provides timeout protection for cancellation
#[derive(Debug)]
pub struct PromptManager {
    /// Map of session_id -> PromptTask
    /// Using DashMap for concurrent access without blocking
    active_prompts: DashMap<String, PromptTask>,
}

impl Default for PromptManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PromptManager {
    /// Create a new prompt manager
    pub fn new() -> Self {
        Self {
            active_prompts: DashMap::new(),
        }
    }

    /// Cancel any active prompt for the given session
    ///
    /// This will:
    /// 1. Send a cancellation signal via the token
    /// 2. Wait for the task to complete (with 5 second timeout)
    /// 3. Remove the task from tracking
    ///
    /// Returns `true` if an old prompt was cancelled, `false` if there was
    /// no active prompt for this session.
    pub async fn cancel_session_prompt(&self, session_id: &str) -> bool {
        use tokio::time::{Duration, timeout};

        const CANCEL_TIMEOUT: Duration = Duration::from_secs(5);

        // Remove the old prompt task from the map
        if let Some((_, task)) = self.active_prompts.remove(session_id) {
            tracing::info!(
                session_id = %session_id,
                prompt_id = %task.id,
                "Cancelling previous prompt"
            );

            // Send cancellation signal
            task.cancel_token.cancel();

            // Wait for task to complete (with timeout)
            let timeout_result = timeout(CANCEL_TIMEOUT, task.handle).await;

            match timeout_result {
                Ok(Ok(())) => {
                    tracing::info!("Previous prompt cancelled gracefully");
                    true
                }
                Ok(Err(e)) => {
                    tracing::warn!(error = ?e, "Previous prompt task failed");
                    true // Task is done, even if it failed
                }
                Err(_) => {
                    tracing::warn!(
                        "Previous prompt did not complete in {:?}, continuing anyway",
                        CANCEL_TIMEOUT
                    );
                    false // Task didn't complete in time
                }
            }
        } else {
            false // No active prompt for this session
        }
    }

    /// Register a new prompt task
    ///
    /// This should be called after spawning a prompt task.
    /// The prompt will be tracked and can be cancelled later.
    pub fn register_prompt(
        &self,
        session_id: String,
        handle: JoinHandle<()>,
        cancel_token: tokio_util::sync::CancellationToken,
    ) -> PromptId {
        // Generate a unique prompt ID
        let prompt_id = format!("{}-{}", session_id, uuid::Uuid::new_v4());

        let task = PromptTask {
            id: prompt_id.clone(),
            handle,
            cancel_token,
            created_at: Instant::now(),
            session_id: session_id.clone(),
        };

        // Insert into the map (this will replace any existing prompt for this session)
        self.active_prompts.insert(session_id.clone(), task);

        tracing::info!(
            session_id = %session_id,
            prompt_id = %prompt_id,
            "Registered new prompt task"
        );

        prompt_id
    }

    /// Mark a prompt as completed
    ///
    /// This should be called when a prompt finishes normally (not cancelled).
    /// It removes the prompt from tracking if the prompt_id matches.
    pub fn complete_prompt(&self, session_id: &str, prompt_id: &str) {
        // Only remove if the prompt ID matches
        // Use DashMap's try_remove to check and remove atomically
        if let Some((_, task)) = self.active_prompts.remove(session_id) {
            if task.id != prompt_id {
                // ID doesn't match, put it back
                self.active_prompts.insert(session_id.to_string(), task);
                return;
            }
        }

        tracing::info!(
            session_id = %session_id,
            prompt_id = %prompt_id,
            "Completed prompt task"
        );
    }

    /// Get the number of active prompts
    pub fn active_count(&self) -> usize {
        self.active_prompts.len()
    }

    /// Check if a session has an active prompt
    pub fn has_active_prompt(&self, session_id: &str) -> bool {
        self.active_prompts.contains_key(session_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::time::sleep;

    #[test]
    fn test_prompt_manager_default() {
        let manager = PromptManager::new();
        assert_eq!(manager.active_count(), 0);
        assert!(!manager.has_active_prompt("test-session"));
    }

    #[tokio::test]
    async fn test_register_prompt() {
        let manager = PromptManager::new();
        let cancel_token = tokio_util::sync::CancellationToken::new();

        // Create a simple task that completes immediately
        let handle = tokio::spawn(async move {
            // Task that does nothing
        });

        let prompt_id = manager.register_prompt("test-session".to_string(), handle, cancel_token);

        assert!(prompt_id.starts_with("test-session-"));
        assert_eq!(manager.active_count(), 1);
        assert!(manager.has_active_prompt("test-session"));

        // Clean up
        manager.complete_prompt("test-session", &prompt_id);
        assert_eq!(manager.active_count(), 0);
    }

    #[tokio::test]
    async fn test_cancel_session_prompt() {
        let manager = PromptManager::new();
        let cancel_token = tokio_util::sync::CancellationToken::new();
        let cancel_token_clone = cancel_token.clone();

        // Create a task that waits for cancellation
        let handle = tokio::spawn(async move {
            tokio::select! {
                () = cancel_token_clone.cancelled() => {
                    // Cancelled
                }
                () = sleep(Duration::from_secs(10)) => {
                    // Would timeout, but should be cancelled first
                }
            }
        });

        manager.register_prompt("test-session".to_string(), handle, cancel_token);

        // Cancel the prompt
        let cancelled = manager.cancel_session_prompt("test-session").await;
        assert!(cancelled);
        assert_eq!(manager.active_count(), 0);
    }

    #[tokio::test]
    async fn test_cancel_nonexistent_prompt() {
        let manager = PromptManager::new();
        let cancelled = manager.cancel_session_prompt("nonexistent").await;
        assert!(!cancelled);
    }

    #[tokio::test]
    async fn test_complete_prompt_only_if_id_matches() {
        let manager = PromptManager::new();
        let cancel_token = tokio_util::sync::CancellationToken::new();

        let handle = tokio::spawn(async move {
            sleep(Duration::from_millis(100)).await;
        });

        let session_id = "test-session";
        let prompt_id = manager.register_prompt(session_id.to_string(), handle, cancel_token);

        // Try to complete with wrong ID
        manager.complete_prompt(session_id, "wrong-id");
        // Should still be active
        assert!(manager.has_active_prompt(session_id));

        // Complete with correct ID
        manager.complete_prompt(session_id, &prompt_id);
        // Should be removed
        assert!(!manager.has_active_prompt(session_id));
    }

    #[tokio::test]
    async fn test_new_prompt_replaces_old() {
        let manager = PromptManager::new();

        // Register first prompt
        let cancel_token1 = tokio_util::sync::CancellationToken::new();
        let handle1 = tokio::spawn(async move {
            sleep(Duration::from_millis(100)).await;
        });

        let session_id = "test-session";
        manager.register_prompt(session_id.to_string(), handle1, cancel_token1);

        assert_eq!(manager.active_count(), 1);

        // Register second prompt (replaces first)
        let cancel_token2 = tokio_util::sync::CancellationToken::new();
        let handle2 = tokio::spawn(async move {
            // Immediate completion
        });

        manager.register_prompt(session_id.to_string(), handle2, cancel_token2);

        // Still only one active prompt
        assert_eq!(manager.active_count(), 1);
    }
}
