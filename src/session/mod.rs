//! Session management for ACP Agent
//!
//! This module handles:
//! - Session lifecycle (create, get, remove)
//! - Token usage tracking
//! - Permission handling
//! - Session state management
//! - Interactive permission requests
//! - Background process management

mod background_processes;
mod manager;
mod permission;
mod permission_manager;
mod permission_request;
mod prompt_manager;
#[allow(clippy::module_inception)]
mod session;
mod usage;
mod wrapped_child;

pub use background_processes::{
    BackgroundProcessManager, BackgroundTerminal, ChildHandle, TerminalExitStatus,
};
pub use manager::SessionManager;
pub use permission::{PermissionHandler, PermissionMode, ToolPermissionResult};
pub use permission_manager::{
    PendingPermissionRequest, PermissionManager, PermissionManagerDecision,
};
pub use permission_request::{PermissionOutcome, PermissionRequestBuilder};
pub use prompt_manager::{PromptId, PromptManager, PromptTask};
pub use session::{Session, stable_cache_key};
pub use usage::UsageTracker;
pub use wrapped_child::WrappedChild;
