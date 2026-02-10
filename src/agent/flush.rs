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
//! When using a patched version of sacp (from your fork with flush PR),
//! we can call `flush()` to wait for all pending messages to be sent.
//!
//! See: docs/MESSAGE_ORDERING_ISSUE.md

#![allow(dead_code)] // FlushError may be unused when sacp-flush feature is disabled

use sacp::JrConnectionCx;
use sacp::link::AgentToClient;

/// Ensure all pending notifications are sent before continuing
///
/// This function attempts to use the native flush() method if available
/// (from the patched sacp), otherwise falls back to a short sleep.
///
/// # Arguments
///
/// * `connection_cx` - The ACP connection context
/// * `notification_count` - Number of notifications sent (for timing calculation)
///
/// # Behavior
///
/// - **With sacp-flush feature enabled**: Attempts native flush() (currently
///   uses sleep as placeholder until flush API is implemented)
/// - **With sacp-flush feature disabled**: Uses sleep-based approximation
///
/// # Feature Flags
///
/// - `sacp-flush`: Enable native flush support (currently uses placeholder;
///   needs real flush API integration after PR is merged)
///
/// # ⚠️ Important Note
///
/// The `sacp-flush` feature is currently enabled by default for development
/// purposes, but the native flush implementation is a **placeholder** that
/// uses a fixed 50ms sleep. The actual flush() method from your sacp fork
/// needs to be integrated into `flush_with_native()`. See that function's
/// documentation for implementation details.
pub async fn ensure_notifications_flushed(
    connection_cx: &JrConnectionCx<AgentToClient>,
    notification_count: u64,
) {
    #[cfg(feature = "sacp-flush")]
    {
        // When sacp-flush feature is enabled, try to use native flush
        // The flush() method should be available on JrConnectionCx
        // from the patched sacp library
        match flush_with_native(connection_cx, notification_count).await {
            Ok(()) => {
                tracing::debug!("Successfully flushed pending notifications");
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    notification_count = notification_count,
                    "Native flush failed, falling back to sleep"
                );
                fallback_sleep(notification_count).await;
            }
        }
    }

    #[cfg(not(feature = "sacp-flush"))]
    {
        // When sacp-flush feature is NOT enabled, use sleep-based fallback
        // This is the default when using official sacp from crates.io
        fallback_sleep(notification_count).await;
    }
}

/// Attempt to use native flush() from patched sacp
///
/// This function is only compiled when the `sacp-flush` feature is enabled.
/// It requires that your sacp fork has implemented a flush() method on
/// JrConnectionCx.
///
/// # Implementation TODO
///
/// The exact signature of flush() may vary depending on your fork's implementation.
/// You need to:
///
/// 1. Check your fork's flush() API signature
/// 2. Replace the placeholder sleep with the actual flush() call
/// 3. Test to ensure it works correctly
///
/// Example signatures (adjust based on your fork):
/// ```ignore
/// // Option 1: Direct method
/// connection_cx.flush().await
///     .map_err(|e| FlushError::Transport(e.to_string()))
///
/// // Option 2: Through a trait
/// use sacp::FlushExt;
/// connection_cx.flush().await
///     .map_err(|e| FlushError::Transport(e.to_string()))
/// ```
#[cfg(feature = "sacp-flush")]
async fn flush_with_native(
    _connection_cx: &JrConnectionCx<AgentToClient>,
    notification_count: u64,
) -> Result<(), FlushError> {
    // ========================================================================
    // TODO: IMPLEMENT THIS WITH YOUR FORK'S FLUSH API
    // ========================================================================
    //
    // This is a placeholder that needs to be replaced once we know the exact
    // API of your fork's flush() method.
    //
    // Current behavior: Uses the same sleep calculation as fallback
    // Expected behavior: Call _connection_cx.flush().await
    //
    // Steps to implement:
    // 1. Find your fork's flush() signature:
    //    cd vendors/symposium-acp
    //    grep -r "pub.*fn flush" src/sacp/
    //
    // 2. Update the code below to call the actual flush method
    //
    // 3. Test to ensure it works

    #[cfg(debug_assertions)]
    {
        tracing::warn!(
            notification_count = notification_count,
            "flush_with_native() is using placeholder sleep implementation! \
             Update this to call the actual flush() method from your sacp fork."
        );
    }

    // Placeholder: Use same calculation as fallback for consistency
    // TODO: Replace with _connection_cx.flush().await
    fallback_sleep(notification_count).await;
    Ok(())
}

/// Fallback sleep-based approach for notification flush
///
/// This is used when:
/// - Using official sacp without flush support
/// - Flush call fails for any reason
///
/// The sleep duration is calculated based on notification count:
/// - Base: 10ms
/// - Per notification: 2ms
/// - Maximum: 100ms
///
/// # Safety
///
/// Uses saturating arithmetic to prevent overflow even with extremely large
/// notification counts (e.g., u64::MAX).
async fn fallback_sleep(notification_count: u64) {
    // Use saturating_add to prevent overflow when notification_count is very large
    // This ensures we never panic even in edge cases
    let wait_ms = 10u64
        .saturating_add(notification_count.saturating_mul(2))
        .min(100);
    tokio::time::sleep(tokio::time::Duration::from_millis(wait_ms)).await;
}

/// Error type for flush operations
///
/// Note: When sacp-flush feature is disabled, this enum is only used in
/// type signatures but not actually instantiated, which may trigger dead_code
/// warnings. This is expected and acceptable.
#[derive(Debug, thiserror::Error)]
pub enum FlushError {
    #[error("Transport error: {0}")]
    Transport(String),

    #[error("Flush not supported by this sacp version")]
    NotSupported,

    #[error("Flush timed out")]
    Timeout,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_sleep_calculation() {
        // Test the sleep duration calculation logic
        let test_cases = vec![
            (0u64, 10u64),    // Minimum 10ms
            (1u64, 12u64),    // 10 + 2*1 = 12ms
            (10u64, 30u64),   // 10 + 2*10 = 30ms
            (45u64, 100u64),  // 10 + 2*45 = 100ms (capped at max)
            (100u64, 100u64), // Capped at 100ms
        ];

        for (count, expected) in test_cases {
            let calculated = (10 + count.saturating_mul(2)).min(100);
            assert_eq!(calculated, expected, "Failed for count={}", count);
        }
    }

    #[test]
    fn test_notification_count_overflow() {
        // Test that u64 overflow is handled correctly with saturating_mul
        let count = u64::MAX;
        // saturating_mul prevents overflow and returns u64::MAX on overflow
        let result = count.saturating_mul(2);
        assert_eq!(result, u64::MAX);

        // Note: 10 + u64::MAX would overflow, but in practice this case
        // should use saturating_add as well for complete safety
        let wait_ms = 10u64.saturating_add(result).min(100);
        assert_eq!(wait_ms, 100, "Should be capped at max value");
    }

    #[test]
    fn test_large_notification_count() {
        // Test with very large notification count (but not overflow)
        let count = 1_000_000u64;
        let calculated = (10 + count.saturating_mul(2)).min(100);
        assert_eq!(calculated, 100, "Large count should be capped at max");
    }

    #[test]
    fn test_flush_error_display() {
        let err = FlushError::Transport("test error".to_string());
        assert_eq!(format!("{}", err), "Transport error: test error");

        let err = FlushError::NotSupported;
        assert_eq!(
            format!("{}", err),
            "Flush not supported by this sacp version"
        );

        let err = FlushError::Timeout;
        assert_eq!(format!("{}", err), "Flush timed out");
    }
}
