//! Permission mode strategy implementations
//!
//! Each strategy encapsulates the permission logic for a specific mode,
//! providing a single source of truth for how tools should be checked.

mod accept_edits_mode;
mod bypass_permissions_mode;
mod default_mode;
mod dont_ask_mode;
mod plan_mode;
mod strategy_trait;

pub use accept_edits_mode::AcceptEditsModeStrategy;
pub use bypass_permissions_mode::BypassPermissionsModeStrategy;
pub use default_mode::DefaultModeStrategy;
pub use dont_ask_mode::DontAskModeStrategy;
pub use plan_mode::PlanModeStrategy;
pub use strategy_trait::PermissionModeStrategy;
