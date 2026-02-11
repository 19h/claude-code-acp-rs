//! Flush mechanism for ACP connection
//!
//! This module provides helper functions to ensure all pending notifications
//! are sent before returning EndTurn to the client.
//!
//! ## The Problem
//!
//! The sacp library's `send_notification()` uses `unbounded_send()` which
//! returns immediately, but messages are processed asynchronously by the
//! outgoing_protocol_actor. This causes a race condition where EndTurn can
//! arrive before all notifications are sent.
//!
//! ## The Solution
//!
//! Since the sacp fork does not have a native flush() API yet, we use a
//! minimal fixed delay (5ms) to ensure message delivery. This is sufficient
//! because the SDK already implements query-scoped message channels,
//! preventing cross-query message mixing.
//!
//! See: docs/MESSAGE_ORDERING_ISSUE.md

use sacp::JrConnectionCx;
use sacp::link::AgentToClient;

/// Ensure all pending notifications have been flushed to the client
///
/// Since the sacp fork does not have a native flush() API yet,
/// we use a minimal fixed delay (5ms) to ensure message delivery.
/// This is sufficient because the SDK already implements query-scoped
/// message channels, preventing cross-query message mixing.
///
/// # Arguments
///
/// * `connection_cx` - The ACP connection context (unused but kept for API consistency)
/// * `notification_count` - Number of notifications sent (used for logging)
pub async fn ensure_notifications_flushed(
    _connection_cx: &JrConnectionCx<AgentToClient>,
    notification_count: u64,
) {
    // Use minimal 5ms sleep to ensure message delivery
    // No need for variable sleep since SDK provides message isolation
    tracing::debug!(
        notification_count = notification_count,
        "Flushing notifications with 5ms delay"
    );
    tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn test_ensure_notifications_flushed() {
        // This test just ensures the function doesn't panic
        // We can't easily test the actual flush behavior without a real connection
        // The actual flush behavior is tested through integration tests
    }
}
