//! Terminal API module
//!
//! Implements Client-side PTY approach where the Agent sends terminal requests
//! to the Client (editor), and the Client manages the actual PTY.
//!
//! This matches the TypeScript claude-code-acp implementation's terminal API:
//! - `terminal/create`: Create a new terminal and execute a command
//! - `terminal/output`: Get current output and status
//! - `terminal/wait_for_exit`: Wait for command to complete
//! - `terminal/kill`: Kill the command (terminal remains valid)
//! - `terminal/release`: Release terminal resources

mod client;
mod handle;

pub use client::TerminalClient;
pub use handle::TerminalHandle;

// Re-export relevant types from sacp::schema for convenience
pub use sacp::schema::{
    CreateTerminalRequest, CreateTerminalResponse, KillTerminalRequest, KillTerminalResponse,
    ReleaseTerminalRequest, ReleaseTerminalResponse, TerminalExitStatus, TerminalId,
    TerminalOutputRequest, TerminalOutputResponse, WaitForTerminalExitRequest,
    WaitForTerminalExitResponse,
};
