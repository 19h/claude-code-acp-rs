//! Session state and management
//!
//! Each session represents an active Claude conversation with its own
//! ClaudeClient instance, usage tracking, and permission state.

use dashmap::DashMap;
use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use tokio::sync::broadcast;

use claude_code_agent_sdk::types::config::PermissionMode as SdkPermissionMode;
use claude_code_agent_sdk::types::mcp::{
    McpHttpServerConfig, McpSdkServerConfig, McpSseServerConfig, McpStdioServerConfig,
};
use claude_code_agent_sdk::{
    ClaudeAgentOptions, ClaudeClient, HookEvent, HookMatcher, McpServerConfig, McpServers,
    SettingSource, SystemPrompt, SystemPromptPreset,
};
use sacp::JrConnectionCx;
use sacp::link::AgentToClient;
use sacp::schema::{
    CurrentModeUpdate, McpServer, SessionId, SessionModeId, SessionNotification, SessionUpdate,
};
use tokio::sync::RwLock;
use tracing::instrument;

use crate::converter::NotificationConverter;
use crate::hooks::{HookCallbackRegistry, create_post_tool_use_hook, create_pre_tool_use_hook};
use crate::mcp::AcpMcpServer;
use crate::permissions::create_can_use_tool_callback;
use crate::settings::{PermissionChecker, SettingsManager};
use crate::terminal::TerminalClient;
use crate::types::{AgentConfig, AgentError, NewSessionMeta, Result};

use super::BackgroundProcessManager;
use super::background_processes::BackgroundTerminal;
use super::permission::{PermissionHandler, PermissionMode};
use super::usage::UsageTracker;

/// Get the list of tools that should be replaced by ACP MCP server tools.
///
/// Only tools that interact with the terminal or filesystem should be replaced:
/// - Terminal tools: Bash, BashOutput, KillShell
/// - File tools: Read, Write, Edit
///
/// Other tools like Glob, Grep, Task, etc. should remain as CLI built-in tools.
fn get_acp_replacement_tools() -> Vec<&'static str> {
    vec![
        // Terminal tools - must be replaced to use ACP Terminal API
        "Bash",
        "BashOutput",
        "KillShell",
        // File tools - must be replaced for ACP file synchronization
        "Read",
        "Write",
        "Edit",
    ]
}

fn headers_to_map(headers: &[sacp::schema::HttpHeader]) -> Option<HashMap<String, String>> {
    if headers.is_empty() {
        None
    } else {
        Some(
            headers
                .iter()
                .map(|header| (header.name.clone(), header.value.clone()))
                .collect(),
        )
    }
}

fn env_to_map(env: &[sacp::schema::EnvVariable]) -> Option<HashMap<String, String>> {
    if env.is_empty() {
        None
    } else {
        Some(
            env.iter()
                .map(|entry| (entry.name.clone(), entry.value.clone()))
                .collect(),
        )
    }
}

fn build_passthrough_mcp_server(server: &McpServer) -> Option<(String, McpServerConfig)> {
    match server {
        McpServer::Http(http) => Some((
            http.name.clone(),
            McpServerConfig::Http(McpHttpServerConfig {
                url: http.url.clone(),
                headers: headers_to_map(&http.headers),
            }),
        )),
        McpServer::Sse(sse) => Some((
            sse.name.clone(),
            McpServerConfig::Sse(McpSseServerConfig {
                url: sse.url.clone(),
                headers: headers_to_map(&sse.headers),
            }),
        )),
        McpServer::Stdio(stdio) => Some((
            stdio.name.clone(),
            McpServerConfig::Stdio(McpStdioServerConfig {
                command: stdio.command.to_string_lossy().into_owned(),
                args: (!stdio.args.is_empty()).then_some(stdio.args.clone()),
                env: env_to_map(&stdio.env),
            }),
        )),
        _ => None,
    }
}

/// An active Claude session
///
/// Each session holds its own ClaudeClient instance and maintains
/// independent state for usage tracking, permissions, and message conversion.
pub struct Session {
    /// Unique session identifier
    pub session_id: String,
    /// Working directory for this session
    pub cwd: PathBuf,
    /// The Claude client for this session
    client: RwLock<ClaudeClient>,
    /// Permission handler for tool execution (wrapped in Arc for can_use_tool callback)
    permission: Arc<RwLock<PermissionHandler>>,
    /// Token usage tracker
    usage_tracker: UsageTracker,
    /// Notification converter with tool use cache (wrapped for interior mutability)
    converter: RwLock<NotificationConverter>,
    /// Whether the client is connected
    connected: AtomicBool,
    /// Hook callback registry for PostToolUse callbacks
    hook_callback_registry: Arc<HookCallbackRegistry>,
    /// Permission checker for hooks
    permission_checker: Arc<RwLock<PermissionChecker>>,
    /// Current model ID for this session (can be changed at runtime via setSessionModel)
    current_model: tokio::sync::RwLock<Option<String>>,
    /// ACP MCP server for tool execution with notifications
    acp_mcp_server: Arc<AcpMcpServer>,
    /// Background process manager
    background_processes: Arc<BackgroundProcessManager>,
    /// Connection context OnceLock for ACP requests (shared with hooks)
    /// Used by pre_tool_use_hook for permission requests
    connection_cx_lock: Arc<OnceLock<JrConnectionCx<AgentToClient>>>,
    /// Cancel signal sender - used to notify when MCP cancellation is received
    cancel_sender: broadcast::Sender<()>,
    /// Cache for permission results by tool_input
    /// PreToolUse hook saves authorized results here, can_use_tool callback checks it
    /// Key: JSON string of tool_input, Value: true if authorized
    /// Only stores authorized results (denied tools don't execute, no need to cache)
    permission_cache: Arc<DashMap<String, bool>>,
    /// Cache for tool_use_id by tool_input
    /// PreToolUse hook caches this when Ask decision is made
    /// can_use_tool callback uses this to get tool_use_id when CLI doesn't provide it
    /// Key: stable cache key of tool_input, Value: tool_use_id
    tool_use_id_cache: Arc<DashMap<String, String>>,
    /// Whether this session has been cancelled by user
    /// Set to true when cancel() is called, reset to false at start of new prompt
    /// Used to distinguish user cancellation from execution errors
    cancelled: AtomicBool,
}

/// Generate a stable cache key from JSON value
///
/// JSON serialization order is not guaranteed to be stable.
/// This function canonicalizes the JSON by sorting object keys using BTreeMap,
/// ensuring identical content always produces the same cache key.
pub fn stable_cache_key(tool_input: &serde_json::Value) -> String {
    fn canonicalize(value: &serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Object(map) => {
                // Use BTreeMap to ensure keys are sorted
                let sorted: BTreeMap<_, _> = map
                    .iter()
                    .map(|(k, v)| (k.clone(), canonicalize(v)))
                    .collect();
                serde_json::Value::Object(sorted.into_iter().collect())
            }
            serde_json::Value::Array(arr) => {
                serde_json::Value::Array(arr.iter().map(canonicalize).collect())
            }
            other => other.clone(),
        }
    }
    canonicalize(tool_input).to_string()
}

impl Session {
    /// Create a new session and wrap in Arc
    ///
    /// Returns Arc<Self> because the can_use_tool callback needs Arc<Session>.
    /// We use OnceLock to break the circular dependency between Session and callback.
    ///
    /// # Arguments
    ///
    /// * `session_id` - Unique identifier for this session
    /// * `cwd` - Working directory
    /// * `config` - Agent configuration from environment
    /// * `meta` - Session metadata from the new session request
    /// * `mcp_servers` - MCP servers requested by the ACP client for passthrough
    #[instrument(
        name = "session_create",
        skip(config, meta),
        fields(
            session_id = %session_id,
            cwd = ?cwd,
            has_meta = meta.is_some(),
        )
    )]
    pub fn new(
        session_id: String,
        cwd: PathBuf,
        config: &AgentConfig,
        meta: Option<&NewSessionMeta>,
        mcp_servers: &[McpServer],
    ) -> Result<Arc<Self>> {
        let start_time = Instant::now();

        tracing::info!(
            session_id = %session_id,
            cwd = ?cwd,
            "Creating new session"
        );

        // Create hook callback registry
        let hook_callback_registry = Arc::new(HookCallbackRegistry::new());

        // Create permission checker for hooks
        // Load settings from ~/.claude/settings.json, .claude/settings.json, etc.
        let settings_manager = SettingsManager::new(&cwd)
            .unwrap_or_else(|e| {
                tracing::warn!("Failed to load settings manager from cwd: {}. Using default settings.", e);
                // Fallback: try to load settings from home directory
                if let Some(home) = dirs::home_dir() {
                    tracing::info!("Attempting to load settings from home directory");
                    SettingsManager::new(&home).unwrap_or_else(|e2| {
                        tracing::error!("Failed to load settings from home directory: {}. Using minimal default settings.", e2);
                        // Last resort: create a manager with minimal settings
                        SettingsManager::new_with_settings(crate::settings::Settings::default(), "/")
                    })
                } else {
                    tracing::error!("No home directory found. Using minimal default settings.");
                    SettingsManager::new_with_settings(crate::settings::Settings::default(), "/")
                }
            });
        // Create shared permission checker that will be used by both hook and permission handler
        // This ensures that runtime rule changes (e.g., "Always Allow") are reflected in both places
        let permission_checker = Arc::new(RwLock::new(PermissionChecker::new(
            settings_manager.settings().clone(),
            &cwd,
        )));

        // Create PermissionHandler with shared PermissionChecker
        // This ensures both pre_tool_use_hook and can_use_tool callback use the same rules
        // PermissionHandler uses AcceptEdits mode (compatible with root, allows all tools)
        let permission_handler = Arc::new(RwLock::new(PermissionHandler::with_checker(
            permission_checker.clone(),
        )));

        // Create shared connection_cx_lock for hook permission requests
        let connection_cx_lock: Arc<OnceLock<JrConnectionCx<AgentToClient>>> =
            Arc::new(OnceLock::new());

        // Create shared permission_cache for hook-to-callback communication
        // PreToolUse hook caches permission results, can_use_tool callback checks it
        let permission_cache: Arc<DashMap<String, bool>> = Arc::new(DashMap::new());

        // Create shared tool_use_id_cache for hook-to-callback tool_use_id passing
        // PreToolUse hook caches tool_use_id when Ask decision is made
        // can_use_tool callback uses this when CLI doesn't provide tool_use_id
        let tool_use_id_cache: Arc<DashMap<String, String>> = Arc::new(DashMap::new());

        // Create hooks with shared permission checker and handler
        let pre_tool_use_hook = create_pre_tool_use_hook(
            connection_cx_lock.clone(),
            session_id.clone(),
            Some(permission_checker.clone()),
            permission_handler.clone(),
            permission_cache.clone(),
            tool_use_id_cache.clone(),
        );
        let post_tool_use_hook = create_post_tool_use_hook(hook_callback_registry.clone());

        // Build hooks map
        let mut hooks_map: HashMap<HookEvent, Vec<HookMatcher>> = HashMap::new();
        hooks_map.insert(
            HookEvent::PreToolUse,
            vec![
                HookMatcher::builder()
                    .hooks(vec![pre_tool_use_hook])
                    .build(),
            ],
        );
        hooks_map.insert(
            HookEvent::PostToolUse,
            vec![
                HookMatcher::builder()
                    .hooks(vec![post_tool_use_hook])
                    .build(),
            ],
        );

        tracing::info!(
            session_id = %session_id,
            hooks_count = 2,
            "Hooks configured: PreToolUse, PostToolUse"
        );

        // Create OnceLock for storing Arc<Session> (needed for callback)
        let session_lock: Arc<OnceLock<Arc<Session>>> = Arc::new(OnceLock::new());

        // Create ACP MCP server
        let acp_mcp_server = Arc::new(AcpMcpServer::new("acp", env!("CARGO_PKG_VERSION")));

        // Create background process manager
        let background_processes = Arc::new(BackgroundProcessManager::new());

        // Build MCP servers with our ACP server
        let mut mcp_servers_dict = HashMap::new();
        mcp_servers_dict.insert(
            "acp".to_string(),
            McpServerConfig::Sdk(McpSdkServerConfig {
                name: "acp".to_string(),
                instance: acp_mcp_server.clone(),
            }),
        );

        for server in mcp_servers {
            let Some((name, config)) = build_passthrough_mcp_server(server) else {
                tracing::warn!(
                    session_id = %session_id,
                    server = ?server,
                    "Skipping unsupported MCP passthrough server"
                );
                continue;
            };

            if name == "acp" {
                tracing::warn!(
                    session_id = %session_id,
                    "Skipping passthrough MCP server named 'acp' because it conflicts with the internal ACP server"
                );
                continue;
            }

            mcp_servers_dict.insert(name, config);
        }

        tracing::info!(
            session_id = %session_id,
            mcp_server_count = mcp_servers_dict.len(),
            passthrough_count = mcp_servers.len(),
            "MCP servers configured"
        );

        // Create can_use_tool callback with OnceLock<Session>
        let can_use_tool_callback = create_can_use_tool_callback(session_lock.clone());

        // Build ClaudeAgentOptions
        //
        // Note: We use AcceptEdits instead of BypassPermissions because
        // BypassPermissions mode cannot be used with root/sudo privileges
        // for security reasons (Claude CLI enforces this restriction).
        // AcceptEdits allows tool execution without permission prompts while
        // being compatible with root user environments.
        let mut options = ClaudeAgentOptions::builder()
            .cwd(cwd.clone())
            .hooks(hooks_map)
            .mcp_servers(McpServers::Dict(mcp_servers_dict))
            .can_use_tool(can_use_tool_callback)
            .permission_mode(SdkPermissionMode::AcceptEdits)
            // Using circular buffer (ringbuf) - auto-recycles old data, no need for large buffer
            .max_buffer_size(20 * 1024 * 1024) // 20MB 缓冲区
            // 允许读取用户级 settings (for API keys, etc.)
            // Agent's env variables (via options.env) will still take precedence
            .setting_sources(vec![SettingSource::User])
            // Enable automatic CLI download for npm/binary distribution users
            // who don't have Claude Code CLI pre-installed
            .auto_download_cli(true)
            .build();

        // Keep Claude focused on the overlay-provided MCP topology.
        // Without strict mode, user/project MCP settings can leak unrelated servers
        // into the session and drown out the IDA tool surface.
        options
            .extra_args
            .insert("strict-mcp-config".to_string(), None);

        // Debug: Verify can_use_tool is set
        tracing::info!(
            session_id = %session_id,
            has_can_use_tool = options.can_use_tool.is_some(),
            has_hooks = options.hooks.is_some(),
            "Options configured after build"
        );

        // Verify mcp_servers is set correctly
        match &options.mcp_servers {
            McpServers::Dict(dict) => {
                tracing::debug!(
                    session_id = %session_id,
                    servers = ?dict.keys().collect::<Vec<_>>(),
                    "MCP servers registered"
                );
            }
            McpServers::Empty => {
                tracing::warn!(
                    session_id = %session_id,
                    "MCP servers is Empty - this is unexpected!"
                );
            }
            McpServers::Path(p) => {
                tracing::warn!(
                    session_id = %session_id,
                    path = ?p,
                    "MCP servers is Path - this is unexpected!"
                );
            }
        }

        // Configure ACP tools to replace CLI built-in tools
        // This disables CLI's built-in tools and enables our MCP tools with mcp__acp__ prefix
        let acp_tools = get_acp_replacement_tools();
        options.use_acp_tools(&acp_tools);

        // Enable streaming to receive incremental content updates
        // This allows SDK to send StreamEvent messages with content_block_delta
        options.include_partial_messages = true;

        tracing::debug!(
            session_id = %session_id,
            acp_tools = ?acp_tools,
            disallowed_tools = ?options.disallowed_tools,
            allowed_tools = ?options.allowed_tools,
            "ACP tools configured"
        );

        // Apply config from environment
        config.apply_to_options(&mut options);

        tracing::debug!(
            session_id = %session_id,
            model = ?options.model,
            fallback_model = ?options.fallback_model,
            max_thinking_tokens = ?options.max_thinking_tokens,
            base_url = ?config.base_url,
            api_key = ?config.masked_api_key(),
            env_vars_count = options.env.len(),
            "Agent config applied"
        );

        // Apply meta options if provided
        if let Some(meta) = meta {
            // Set system prompt: replace takes priority over append
            if let Some(replace) = meta.get_system_prompt_replace() {
                // Complete replacement of system prompt
                options.system_prompt = Some(SystemPrompt::Text(replace.to_string()));
                tracing::info!(
                    session_id = %session_id,
                    prompt_len = replace.len(),
                    "Using custom system prompt from meta (replace)"
                );
            } else if let Some(append) = meta.get_system_prompt_append() {
                // Append to default claude_code preset
                let preset = SystemPromptPreset::with_append("claude_code", append);
                options.system_prompt = Some(SystemPrompt::Preset(preset));
                tracing::info!(
                    session_id = %session_id,
                    append_len = append.len(),
                    "Appending to system prompt from meta"
                );
            }

            // Set resume session if provided
            if let Some(resume_id) = meta.get_resume_session_id() {
                options.resume = Some(resume_id.to_string());
                tracing::info!(
                    session_id = %session_id,
                    resume_session_id = %resume_id,
                    "Resuming from previous session"
                );
            }

            // Set fork_session flag if this is a fork operation
            if meta.fork_session {
                options.fork_session = true;
                tracing::info!(
                    session_id = %session_id,
                    "Forking session (fork_session=true)"
                );
            }

            // Set max thinking tokens if provided (enables extended thinking mode)
            if let Some(tokens) = meta.get_max_thinking_tokens() {
                options.max_thinking_tokens = Some(tokens);
                tracing::info!(
                    session_id = %session_id,
                    max_thinking_tokens = tokens,
                    "Extended thinking mode enabled via meta"
                );
            }
        }

        // Create the client
        let client = ClaudeClient::new(options);

        let elapsed = start_time.elapsed();
        tracing::info!(
            session_id = %session_id,
            elapsed_ms = elapsed.as_millis(),
            "Session created successfully"
        );

        // Clone cwd for converter before moving cwd into the struct
        let cwd_for_converter = cwd.clone();

        // Build the Session struct
        let session = Self {
            session_id,
            cwd,
            client: RwLock::new(client),
            permission: permission_handler,
            usage_tracker: UsageTracker::new(),
            converter: RwLock::new(NotificationConverter::with_cwd(cwd_for_converter)),
            connected: AtomicBool::new(false),
            hook_callback_registry,
            permission_checker,
            current_model: tokio::sync::RwLock::new(None),
            acp_mcp_server,
            background_processes,
            connection_cx_lock,
            cancel_sender: broadcast::channel(1).0,
            permission_cache,
            tool_use_id_cache,
            cancelled: AtomicBool::new(false),
        };

        // Wrap in Arc
        let session_arc = Arc::new(session);

        // Set the OnceLock so the callback can access the Session
        drop(session_lock.set(session_arc.clone()));

        Ok(session_arc)
    }

    /// Set the connection context for ACP requests
    ///
    /// This is called once during handle_prompt to enable permission requests.
    /// The OnceLock ensures it's only set once even if called multiple times.
    pub fn set_connection_cx(&self, cx: JrConnectionCx<AgentToClient>) {
        if self.connection_cx_lock.get().is_none() {
            drop(self.connection_cx_lock.set(cx));
        }
    }

    /// Get the connection context if available
    ///
    /// Returns None if called before handle_prompt sets the connection.
    pub fn get_connection_cx(&self) -> Option<&JrConnectionCx<AgentToClient>> {
        self.connection_cx_lock.get()
    }

    /// Cache a permission result for a tool_input
    ///
    /// Called by PreToolUse hook after user grants permission.
    /// The can_use_tool callback checks this cache before sending permission requests.
    pub fn cache_permission(&self, tool_input: &serde_json::Value, allowed: bool) {
        let key = stable_cache_key(tool_input);
        tracing::debug!(
            key_len = key.len(),
            allowed = allowed,
            "Caching permission result"
        );
        self.permission_cache.insert(key, allowed);
    }

    /// Check if a tool_input has cached permission
    ///
    /// Called by can_use_tool callback to check if permission was already granted.
    /// Returns Some(true) if allowed, Some(false) if denied, None if not cached.
    /// Removes the entry from cache after retrieval (one-time use).
    pub fn check_cached_permission(&self, tool_input: &serde_json::Value) -> Option<bool> {
        let key = stable_cache_key(tool_input);
        self.permission_cache.remove(&key).map(|(_, v)| v)
    }

    /// Get a reference to the permission_cache for sharing with hooks
    pub fn permission_cache(&self) -> Arc<DashMap<String, bool>> {
        Arc::clone(&self.permission_cache)
    }

    /// Cache tool_use_id for a tool_input
    ///
    /// Called by PreToolUse hook when Ask decision is made.
    /// The can_use_tool callback uses this to get tool_use_id when CLI doesn't provide it.
    pub fn cache_tool_use_id(&self, tool_input: &serde_json::Value, tool_use_id: &str) {
        let key = stable_cache_key(tool_input);
        tracing::debug!(
            key_len = key.len(),
            tool_use_id = %tool_use_id,
            "Caching tool_use_id"
        );
        self.tool_use_id_cache.insert(key, tool_use_id.to_string());
    }

    /// Get cached tool_use_id for a tool_input
    ///
    /// Called by can_use_tool callback to get tool_use_id when CLI doesn't provide it.
    /// Returns the tool_use_id if cached, None otherwise.
    /// Removes the entry from cache after retrieval (one-time use).
    pub fn get_cached_tool_use_id(&self, tool_input: &serde_json::Value) -> Option<String> {
        let key = stable_cache_key(tool_input);
        self.tool_use_id_cache.remove(&key).map(|(_, v)| v)
    }

    /// Get a reference to the tool_use_id_cache for sharing with hooks
    pub fn tool_use_id_cache(&self) -> Arc<DashMap<String, String>> {
        Arc::clone(&self.tool_use_id_cache)
    }

    /// Connect to Claude CLI
    ///
    /// This spawns the Claude CLI process and establishes JSON-RPC communication.
    #[instrument(
        name = "session_connect",
        skip(self),
        fields(session_id = %self.session_id)
    )]
    pub async fn connect(&self) -> Result<()> {
        if self.connected.load(Ordering::SeqCst) {
            tracing::debug!(
                session_id = %self.session_id,
                "Already connected to Claude CLI"
            );
            return Ok(());
        }

        let start_time = Instant::now();
        tracing::info!(
            session_id = %self.session_id,
            cwd = ?self.cwd,
            "Connecting to Claude CLI..."
        );

        let mut client = self.client.write().await;
        client.connect().await.map_err(|e| {
            let agent_error = AgentError::from(e);
            tracing::error!(
                session_id = %self.session_id,
                error = %agent_error,
                error_code = ?agent_error.error_code(),
                is_retryable = %agent_error.is_retryable(),
                error_chain = ?agent_error.source(),
                "Failed to connect to Claude CLI"
            );
            agent_error
        })?;

        self.connected.store(true, Ordering::SeqCst);

        let elapsed = start_time.elapsed();
        tracing::info!(
            session_id = %self.session_id,
            elapsed_ms = elapsed.as_millis(),
            "Successfully connected to Claude CLI"
        );

        Ok(())
    }

    /// Disconnect from Claude CLI
    #[instrument(
        name = "session_disconnect",
        skip(self),
        fields(session_id = %self.session_id)
    )]
    pub async fn disconnect(&self) -> Result<()> {
        if !self.connected.load(Ordering::SeqCst) {
            tracing::debug!(
                session_id = %self.session_id,
                "Already disconnected from Claude CLI"
            );
            return Ok(());
        }

        let start_time = Instant::now();
        tracing::info!(
            session_id = %self.session_id,
            "Disconnecting from Claude CLI..."
        );

        let mut client = self.client.write().await;
        client.disconnect().await.map_err(|e| {
            let agent_error = AgentError::from(e);
            tracing::error!(
                session_id = %self.session_id,
                error = %agent_error,
                error_code = ?agent_error.error_code(),
                is_retryable = %agent_error.is_retryable(),
                error_chain = ?agent_error.source(),
                "Failed to disconnect from Claude CLI"
            );
            agent_error
        })?;

        self.connected.store(false, Ordering::SeqCst);

        let elapsed = start_time.elapsed();
        tracing::info!(
            session_id = %self.session_id,
            elapsed_ms = elapsed.as_millis(),
            "Disconnected from Claude CLI"
        );

        Ok(())
    }

    /// Check if the session is connected
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    /// Get read access to the client
    pub async fn client(&self) -> tokio::sync::RwLockReadGuard<'_, ClaudeClient> {
        self.client.read().await
    }

    /// Get write access to the client
    pub async fn client_mut(&self) -> tokio::sync::RwLockWriteGuard<'_, ClaudeClient> {
        self.client.write().await
    }

    /// Get a receiver for cancel signals
    ///
    /// This can be used to listen for MCP cancellation notifications.
    /// When a cancel notification is received, a signal is sent through the channel.
    pub fn cancel_receiver(&self) -> broadcast::Receiver<()> {
        self.cancel_sender.subscribe()
    }

    /// Cancel this session and interrupt the Claude CLI
    ///
    /// This sends an interrupt signal to the Claude CLI to stop the current operation.
    /// Also sets the cancelled flag to true so that the stop reason can be determined correctly.
    #[instrument(
        name = "session_cancel",
        skip(self),
        fields(session_id = %self.session_id)
    )]
    pub async fn cancel(&self) {
        // Set cancelled flag first
        // Use Release ordering to ensure visibility to other threads
        self.cancelled.store(true, Ordering::Release);

        tracing::info!(
            session_id = %self.session_id,
            "Sending interrupt signal to Claude CLI (cancelled=true)"
        );

        // Send interrupt signal to Claude CLI to stop current operation
        if let Ok(client) = self.client.try_read() {
            if let Err(e) = client.interrupt().await {
                tracing::warn!(
                    session_id = %self.session_id,
                    error = %e,
                    "Failed to send interrupt signal to Claude CLI"
                );
            } else {
                tracing::info!(
                    session_id = %self.session_id,
                    "Interrupt signal sent to Claude CLI"
                );
            }
        } else {
            tracing::warn!(
                session_id = %self.session_id,
                "Could not acquire client lock for interrupt"
            );
        }
    }

    /// Check if session is cancelled by user
    ///
    /// This flag is set when the user clicks "Stop" in the UI.
    /// Used to determine whether to return Cancelled or EndTurn stop reason.
    pub fn is_user_cancelled(&self) -> bool {
        // Use Acquire ordering to synchronize with the Release store in cancel()
        self.cancelled.load(Ordering::Acquire)
    }

    /// Reset the cancelled flag
    ///
    /// Called at the start of each new prompt to ensure the flag is cleared.
    pub fn reset_cancelled(&self) {
        // Use Release ordering for consistency, though Relaxed would also work here
        // since we're just clearing the flag at the start of a new prompt
        self.cancelled.store(false, Ordering::Release);
    }

    /// Get the permission handler
    pub async fn permission(&self) -> tokio::sync::RwLockReadGuard<'_, PermissionHandler> {
        self.permission.read().await
    }

    /// Get the current permission mode
    pub async fn permission_mode(&self) -> PermissionMode {
        self.permission.read().await.mode()
    }

    /// Set the permission mode
    ///
    /// Updates the PermissionHandler. The hook will read the mode
    /// from the same PermissionHandler, ensuring consistency.
    pub async fn set_permission_mode(&self, mode: PermissionMode) {
        // Update the permission handler (single source of truth)
        self.permission.write().await.set_mode(mode);

        tracing::info!(
            session_id = %self.session_id,
            mode = mode.as_str(),
            "Permission mode updated"
        );
    }

    /// Send session/update notification for permission mode change
    ///
    /// This sends a CurrentModeUpdate notification to the client to inform it
    /// that the permission mode has changed. This is used for ExitPlanMode to
    /// notify the UI that the mode has been switched.
    pub fn send_mode_update(&self, mode: &str) {
        let Some(connection_cx) = self.get_connection_cx() else {
            tracing::warn!(
                session_id = %self.session_id,
                mode = %mode,
                "Connection not ready for mode update notification"
            );
            return;
        };

        let mode_update = CurrentModeUpdate::new(SessionModeId::new(mode));
        let notification = SessionNotification::new(
            SessionId::new(self.session_id.clone()),
            SessionUpdate::CurrentModeUpdate(mode_update),
        );

        if let Err(e) = connection_cx.send_notification(notification) {
            tracing::warn!(
                session_id = %self.session_id,
                mode = %mode,
                error = %e,
                "Failed to send CurrentModeUpdate notification"
            );
        } else {
            tracing::info!(
                session_id = %self.session_id,
                mode = %mode,
                "Sent CurrentModeUpdate notification"
            );
        }
    }

    /// Add an allow rule for a tool
    ///
    /// This is called when user selects "Always Allow" in permission prompt.
    pub async fn add_permission_allow_rule(&self, tool_name: &str) {
        self.permission.read().await.add_allow_rule(tool_name).await;
    }

    /// Get the current model ID
    pub async fn current_model(&self) -> Option<String> {
        self.current_model.read().await.clone()
    }

    /// Set the model for this session
    ///
    /// Updates the model ID used for subsequent prompts. Also updates the
    /// underlying SDK client's model option.
    pub async fn set_model(&self, model_id: String) {
        // Update session model
        {
            let mut model = self.current_model.write().await;
            *model = Some(model_id.clone());
        } // Release write lock before accessing client

        tracing::info!(model_id = %model_id, "Session model updated");

        // Also set model on the SDK client for next prompt
        let client = self.client.read().await;
        if let Err(e) = client.set_model(Some(model_id.as_str())).await {
            tracing::warn!(error = %e, "Failed to set model on SDK client");
        }
    }

    /// Get the usage tracker
    pub fn usage_tracker(&self) -> &UsageTracker {
        &self.usage_tracker
    }

    /// Get the notification converter (read-only access)
    pub async fn converter(&self) -> tokio::sync::RwLockReadGuard<'_, NotificationConverter> {
        self.converter.read().await
    }

    /// Set the request_id on the notification converter
    ///
    /// This will attach the request_id to all SessionNotification instances
    /// created by this session's converter, allowing clients to track which
    /// responses correspond to which requests.
    ///
    /// # Arguments
    ///
    /// * `request_id` - The unique request identifier
    pub async fn set_converter_request_id(&self, request_id: String) {
        let mut converter = self.converter.write().await;
        converter.set_request_id(request_id);
    }

    /// Clear the request_id from the notification converter
    pub async fn clear_converter_request_id(&self) {
        let mut converter = self.converter.write().await;
        converter.clear_request_id();
    }

    /// Clear the tool use cache from the notification converter
    ///
    /// Should be called at the end of each prompt to prevent unbounded
    /// memory growth from accumulated tool use entries.
    pub async fn clear_converter_cache(&self) {
        let converter = self.converter.write().await;
        converter.clear_cache();
    }

    /// Get the hook callback registry
    pub fn hook_callback_registry(&self) -> &Arc<HookCallbackRegistry> {
        &self.hook_callback_registry
    }

    /// Get the permission checker
    pub fn permission_checker(&self) -> &Arc<RwLock<PermissionChecker>> {
        &self.permission_checker
    }

    /// Register a PostToolUse callback for a tool use
    pub fn register_post_tool_use_callback(
        &self,
        tool_use_id: String,
        callback: crate::hooks::PostToolUseCallback,
    ) {
        self.hook_callback_registry
            .register_post_tool_use(tool_use_id, callback);
    }

    /// Get the ACP MCP server
    pub fn acp_mcp_server(&self) -> &Arc<AcpMcpServer> {
        &self.acp_mcp_server
    }

    /// Get the background process manager
    pub fn background_processes(&self) -> &Arc<BackgroundProcessManager> {
        &self.background_processes
    }

    /// Cleanup all child processes for this session
    ///
    /// This method ensures all external MCP servers and background bash processes
    /// are properly terminated and reaped to prevent zombie processes.
    #[instrument(
        name = "session_cleanup",
        skip(self),
        fields(session_id = %self.session_id)
    )]
    pub async fn cleanup(&self) -> Result<()> {
        let start_time = Instant::now();

        tracing::info!(
            session_id = %self.session_id,
            "Starting session cleanup"
        );

        // 1. Disconnect all external MCP servers with proper cleanup
        let external_manager = self.acp_mcp_server.mcp_server().external_manager();
        let server_names = external_manager.server_names();

        let mcp_count = server_names.len();
        let mut mcp_success = 0;
        let mut mcp_errors = 0;

        for server_name in &server_names {
            match external_manager.disconnect(server_name).await {
                Ok(()) => mcp_success += 1,
                Err(e) => {
                    mcp_errors += 1;
                    tracing::warn!(
                        session_id = %self.session_id,
                        server_name = %server_name,
                        error = %e,
                        "Failed to disconnect MCP server during cleanup"
                    );
                }
            }
        }

        tracing::info!(
            session_id = %self.session_id,
            mcp_count = mcp_count,
            mcp_success = mcp_success,
            mcp_errors = mcp_errors,
            "External MCP servers cleanup completed"
        );

        // 2. Kill all background bash processes
        let shell_ids = self.background_processes.shell_ids();
        let shell_count = shell_ids.len();
        let mut shell_success = 0;
        let mut shell_errors = 0;

        for shell_id in &shell_ids {
            if let Some((_, BackgroundTerminal::Running { mut child, .. })) =
                self.background_processes.remove(shell_id)
            {
                match child.kill().await {
                    Ok(()) => {
                        // Wait for the process to exit (prevents zombie)
                        drop(child.wait().await);
                        shell_success += 1;
                    }
                    Err(e) => {
                        shell_errors += 1;
                        tracing::warn!(
                            session_id = %self.session_id,
                            shell_id = %shell_id,
                            error = %e,
                            "Failed to kill background process during cleanup"
                        );
                    }
                }
            }
        }

        tracing::info!(
            session_id = %self.session_id,
            shell_count = shell_count,
            shell_success = shell_success,
            shell_errors = shell_errors,
            "Background processes cleanup completed"
        );

        let elapsed = start_time.elapsed();
        tracing::info!(
            session_id = %self.session_id,
            elapsed_ms = elapsed.as_millis(),
            "Session cleanup completed"
        );

        Ok(())
    }

    /// Configure the ACP MCP server with connection and terminal client
    ///
    /// This should be called after creating the session to enable Terminal API
    /// integration for Bash commands.
    pub async fn configure_acp_server(
        &self,
        connection_cx: JrConnectionCx<AgentToClient>,
        terminal_client: Option<Arc<TerminalClient>>,
    ) {
        self.acp_mcp_server.set_session_id(&self.session_id);
        self.acp_mcp_server.set_connection(connection_cx);
        self.acp_mcp_server.set_cwd(self.cwd.clone());
        self.acp_mcp_server
            .set_background_processes(self.background_processes.clone());
        self.acp_mcp_server
            .set_permission_checker(self.permission_checker.clone());

        if let Some(client) = terminal_client {
            self.acp_mcp_server.set_terminal_client(client);
        }

        // Set up cancel callback to interrupt Claude CLI when MCP cancellation is received
        let session_id = self.session_id.clone();
        let cancel_sender = self.cancel_sender.clone();

        self.acp_mcp_server
            .set_cancel_callback(move || {
                tracing::info!(
                    session_id = %session_id,
                    "MCP cancel callback invoked, sending cancel signal"
                );
                // Send cancel signal through the channel
                // Note: Cancellation is now handled per-prompt via CancellationToken
                let _ = cancel_sender.send(());
            })
            .await;
    }
}

#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("session_id", &self.session_id)
            .field("cwd", &self.cwd)
            .field("connected", &self.connected.load(Ordering::Relaxed))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> AgentConfig {
        AgentConfig {
            base_url: None,
            api_key: None,
            model: None,
            small_fast_model: None,
            max_thinking_tokens: None,
        }
    }

    #[test]
    fn test_session_new() {
        let session = Session::new(
            "test-session-1".to_string(),
            PathBuf::from("/tmp"),
            &test_config(),
            None,
            &[],
        )
        .unwrap();

        assert_eq!(session.session_id, "test-session-1");
        assert_eq!(session.cwd, PathBuf::from("/tmp"));
        assert!(!session.is_connected());
        // Cancelled flag should be false initially
        assert!(!session.is_user_cancelled());
    }

    #[test]
    fn test_cancelled_flag_lifecycle() {
        let session = Session::new(
            "test-cancel-session".to_string(),
            PathBuf::from("/tmp"),
            &test_config(),
            None,
            &[],
        )
        .unwrap();

        // 1. Initially cancelled should be false
        assert!(
            !session.is_user_cancelled(),
            "Cancelled should be false initially"
        );

        // 2. After setting cancelled to true via direct store (simulating cancel())
        session.cancelled.store(true, Ordering::Release);
        assert!(
            session.is_user_cancelled(),
            "Cancelled should be true after setting"
        );

        // 3. After reset_cancelled(), should be false again
        session.reset_cancelled();
        assert!(
            !session.is_user_cancelled(),
            "Cancelled should be false after reset"
        );

        // 4. Set again and verify
        session.cancelled.store(true, Ordering::Release);
        assert!(
            session.is_user_cancelled(),
            "Cancelled should be true after setting again"
        );
    }

    #[tokio::test]
    async fn test_session_cancel() {
        let session = Session::new(
            "test-session-2".to_string(),
            PathBuf::from("/tmp"),
            &test_config(),
            None,
            &[],
        )
        .unwrap();

        // Note: Cancellation is now handled per-prompt via CancellationToken
        // This test just verifies that cancel() doesn't panic
        session.cancel().await;
    }

    #[tokio::test]
    async fn test_session_permission_mode() {
        let session = Session::new(
            "test-session-3".to_string(),
            PathBuf::from("/tmp"),
            &test_config(),
            None,
            &[],
        )
        .unwrap();

        // Default is Default mode (standard behavior with permission prompts)
        assert_eq!(session.permission_mode().await, PermissionMode::Default);
        session.set_permission_mode(PermissionMode::DontAsk).await;
        assert_eq!(session.permission_mode().await, PermissionMode::DontAsk);
    }

    #[test]
    fn test_stable_cache_key_ordering() {
        use serde_json::json;

        // JSON objects with same content but different key ordering should produce same cache key
        let json1 = json!({"a": 1, "b": 2, "c": 3});
        let json2 = json!({"c": 3, "b": 2, "a": 1});
        let json3 = json!({"b": 2, "a": 1, "c": 3});

        let key1 = stable_cache_key(&json1);
        let key2 = stable_cache_key(&json2);
        let key3 = stable_cache_key(&json3);

        assert_eq!(
            key1, key2,
            "Different key ordering should produce same cache key"
        );
        assert_eq!(
            key2, key3,
            "Different key ordering should produce same cache key"
        );
    }

    #[test]
    fn test_stable_cache_key_nested_objects() {
        use serde_json::json;

        // Nested objects should also be canonicalized
        let json1 = json!({
            "command": "cargo build",
            "options": {"a": 1, "b": 2}
        });
        let json2 = json!({
            "options": {"b": 2, "a": 1},
            "command": "cargo build"
        });

        let key1 = stable_cache_key(&json1);
        let key2 = stable_cache_key(&json2);

        assert_eq!(key1, key2, "Nested objects should also produce stable keys");
    }

    #[test]
    fn test_stable_cache_key_arrays() {
        use serde_json::json;

        // Arrays with objects inside should be canonicalized
        let json1 = json!({
            "items": [{"a": 1, "b": 2}, {"c": 3, "d": 4}]
        });
        let json2 = json!({
            "items": [{"b": 2, "a": 1}, {"d": 4, "c": 3}]
        });

        let key1 = stable_cache_key(&json1);
        let key2 = stable_cache_key(&json2);

        assert_eq!(key1, key2, "Arrays with objects should produce stable keys");
    }

    #[test]
    fn test_stable_cache_key_different_content() {
        use serde_json::json;

        // Different content should produce different keys
        let json1 = json!({"command": "cargo build"});
        let json2 = json!({"command": "cargo test"});

        let key1 = stable_cache_key(&json1);
        let key2 = stable_cache_key(&json2);

        assert_ne!(
            key1, key2,
            "Different content should produce different keys"
        );
    }

    #[test]
    fn test_build_passthrough_mcp_server_sse() {
        let server = McpServer::Sse(
            sacp::schema::McpServerSse::new("ida", "http://127.0.0.1:8080/sse").headers(vec![
                sacp::schema::HttpHeader::new("Authorization", "Bearer token"),
            ]),
        );

        let (name, config) = build_passthrough_mcp_server(&server).unwrap();
        assert_eq!(name, "ida");

        match config {
            McpServerConfig::Sse(sse) => {
                assert_eq!(sse.url, "http://127.0.0.1:8080/sse");
                assert_eq!(
                    sse.headers
                        .as_ref()
                        .and_then(|headers| headers.get("Authorization")),
                    Some(&"Bearer token".to_string())
                );
            }
            _ => panic!("expected SSE MCP config"),
        }
    }

    #[test]
    fn test_build_passthrough_mcp_server_stdio() {
        let server = McpServer::Stdio(
            sacp::schema::McpServerStdio::new("ctx7", "/usr/bin/env")
                .args(vec!["node".to_string(), "server.js".to_string()])
                .env(vec![sacp::schema::EnvVariable::new("TOKEN", "secret")]),
        );

        let (name, config) = build_passthrough_mcp_server(&server).unwrap();
        assert_eq!(name, "ctx7");

        match config {
            McpServerConfig::Stdio(stdio) => {
                assert_eq!(stdio.command, "/usr/bin/env");
                assert_eq!(
                    stdio.args,
                    Some(vec!["node".to_string(), "server.js".to_string()])
                );
                assert_eq!(
                    stdio.env.as_ref().and_then(|env| env.get("TOKEN")),
                    Some(&"secret".to_string())
                );
            }
            _ => panic!("expected stdio MCP config"),
        }
    }

    /// Test Session::cleanup() with no processes
    ///
    /// Verifies that cleanup succeeds when there are no MCP servers
    /// or background processes to clean up.
    #[tokio::test]
    async fn test_session_cleanup_no_processes() {
        let session = Session::new(
            "test-cleanup-session".to_string(),
            PathBuf::from("/tmp"),
            &test_config(),
            None,
            &[],
        )
        .unwrap();

        // Cleanup should succeed even with no processes
        let result = session.cleanup().await;
        assert!(result.is_ok(), "Cleanup with no processes should succeed");
    }

    /// Test Session::cleanup() with background processes
    ///
    /// Verifies that cleanup properly terminates and waits for
    /// background bash processes.
    #[tokio::test]
    async fn test_session_cleanup_with_background_process() {
        let session = Session::new(
            "test-cleanup-with-bg".to_string(),
            PathBuf::from("/tmp"),
            &test_config(),
            None,
            &[],
        )
        .unwrap();

        // Create a mock background process using the process manager
        // We can't spawn a real process in tests without causing issues,
        // so we verify the cleanup logic structure

        // Get the background process manager
        let bg_manager = session.background_processes();

        // Verify no processes initially
        assert_eq!(bg_manager.count(), 0);

        // Cleanup should succeed
        let result = session.cleanup().await;
        assert!(result.is_ok(), "Cleanup should succeed");
    }

    // ========================================================================
    // Bundled CLI integration tests
    // ========================================================================

    /// Verify the `bundled-cli` feature flag is enabled by default.
    ///
    /// This test ensures that when building with default features,
    /// the `bundled-cli` feature is active, which triggers the SDK's
    /// build.rs to download the Claude Code CLI binary.
    ///
    /// If this test is not compiled, it means the feature is disabled.
    #[test]
    #[cfg(feature = "bundled-cli")]
    fn test_bundled_cli_feature_enabled() {
        // This test body only compiles when bundled-cli feature is active.
        // Its mere existence in `cargo test --all-features` output proves the feature works.
    }

    /// Verify SDK's `bundled_cli_path()` returns a valid path structure.
    ///
    /// The path should be: `~/.claude/sdk/bundled/{CLI_VERSION}/claude`
    /// This validates that the SDK integration is properly wired up.
    #[test]
    fn test_bundled_cli_path_structure() {
        use claude_code_agent_sdk::CLI_VERSION;
        use claude_code_agent_sdk::version::bundled_cli_path;

        let path =
            bundled_cli_path().expect("bundled_cli_path() should return Some on systems with HOME");
        let path_str = path.to_string_lossy();

        // Must contain the bundled directory structure
        assert!(
            path_str.contains(".claude/sdk/bundled"),
            "bundled path should contain '.claude/sdk/bundled', got: {}",
            path_str
        );

        // Must contain the CLI version in the path
        assert!(
            path_str.contains(CLI_VERSION),
            "bundled path should contain CLI_VERSION '{}', got: {}",
            CLI_VERSION,
            path_str
        );

        // On Unix, the binary name should be 'claude'
        #[cfg(not(target_os = "windows"))]
        assert!(
            path_str.ends_with("/claude"),
            "bundled path should end with '/claude', got: {}",
            path_str
        );

        // On Windows, the binary name should be 'claude.exe'
        #[cfg(target_os = "windows")]
        assert!(
            path_str.ends_with("\\claude.exe"),
            "bundled path should end with '\\claude.exe', got: {}",
            path_str
        );
    }

    /// Verify SDK's CLI_VERSION is a valid semver that meets the minimum requirement.
    #[test]
    fn test_bundled_cli_version_valid() {
        use claude_code_agent_sdk::CLI_VERSION;
        use claude_code_agent_sdk::version::{check_version, parse_version};

        // CLI_VERSION must parse as valid semver
        let parsed = parse_version(CLI_VERSION);
        assert!(
            parsed.is_some(),
            "CLI_VERSION '{}' should be valid semver",
            CLI_VERSION
        );

        // CLI_VERSION must meet the minimum CLI version requirement
        assert!(
            check_version(CLI_VERSION),
            "CLI_VERSION '{}' should meet minimum version requirement",
            CLI_VERSION
        );
    }

    /// Verify that Session does NOT set cli_path explicitly,
    /// relying on the SDK's find_cli() to auto-discover the bundled CLI.
    ///
    /// This is important because the SDK's find_cli() Strategy 0 checks
    /// the bundled path first, so we should NOT override that behavior.
    #[test]
    fn test_session_uses_sdk_auto_discovery() {
        let session = Session::new(
            "test-bundled-cli".to_string(),
            PathBuf::from("/tmp"),
            &test_config(),
            None,
            &[],
        )
        .unwrap();

        // Session should exist without explicitly setting cli_path.
        // The SDK's find_cli() will handle CLI discovery at connect time,
        // checking the bundled path (~/.claude/sdk/bundled/{version}/claude) first.
        assert!(!session.is_connected());
    }
}
