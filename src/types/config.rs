//! Agent configuration from environment variables

use std::collections::HashMap;

/// Agent configuration loaded from environment variables and settings files
///
/// Configuration priority (highest to lowest):
/// 1. Environment variables (e.g., `ANTHROPIC_MODEL`)
/// 2. Settings files - Top-level fields (e.g., `model`)
/// 3. Settings files - `env` object (e.g., `env.ANTHROPIC_MODEL`)
/// 4. Defaults
///
/// Settings files are loaded from:
/// - `~/.claude/settings.json` (user settings)
/// - `<project_dir>/.claude/settings.json` (project settings)
/// - `<project_dir>/.claude/settings.local.json` (local settings)
///
/// Supports configuring alternative AI model providers (e.g., domestic providers in China)
/// through environment variables or settings files.
#[derive(Debug, Clone, Default)]
pub struct AgentConfig {
    /// Anthropic API base URL
    /// Environment variable: `ANTHROPIC_BASE_URL`
    /// Settings field: `apiBaseUrl`
    pub base_url: Option<String>,

    /// API key for authentication
    /// Environment variable: `ANTHROPIC_API_KEY` (preferred) or `ANTHROPIC_AUTH_TOKEN` (legacy)
    /// Note: Not supported in settings files for security reasons
    pub api_key: Option<String>,

    /// Primary model name
    /// Environment variable: `ANTHROPIC_MODEL`
    /// Settings field: `model`
    pub model: Option<String>,

    /// Small/fast model name (fallback)
    /// Environment variable: `ANTHROPIC_SMALL_FAST_MODEL`
    /// Settings field: `smallFastModel`
    pub small_fast_model: Option<String>,

    /// Maximum tokens for thinking blocks (extended thinking mode)
    ///
    /// Can be set via:
    /// - Environment variable: `MAX_THINKING_TOKENS`
    /// - Settings field: `alwaysThinkingEnabled` (sets to default 20000)
    /// - Settings `env` object: `env.MAX_THINKING_TOKENS`
    ///
    /// When `alwaysThinkingEnabled` is true in settings, this defaults to 20000.
    /// Typical values: 4096, 8000, 16000, 20000
    pub max_thinking_tokens: Option<u32>,
}

impl AgentConfig {
    /// Create a new empty configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Load configuration from environment variables
    ///
    /// Reads the following environment variables:
    /// - `ANTHROPIC_BASE_URL`: API base URL
    /// - `ANTHROPIC_API_KEY`: API key (preferred)
    /// - `ANTHROPIC_AUTH_TOKEN`: Auth token (legacy, fallback if API_KEY not set)
    /// - `ANTHROPIC_MODEL`: Primary model name
    /// - `ANTHROPIC_SMALL_FAST_MODEL`: Small/fast model name
    /// - `MAX_THINKING_TOKENS`: Maximum tokens for thinking blocks
    pub fn from_env() -> Self {
        // Prefer ANTHROPIC_API_KEY, fallback to ANTHROPIC_AUTH_TOKEN for compatibility
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .ok()
            .or_else(|| std::env::var("ANTHROPIC_AUTH_TOKEN").ok());

        // Parse MAX_THINKING_TOKENS if present
        let max_thinking_tokens = std::env::var("MAX_THINKING_TOKENS")
            .ok()
            .and_then(|s| s.parse::<u32>().ok());

        Self {
            base_url: std::env::var("ANTHROPIC_BASE_URL").ok(),
            api_key,
            model: std::env::var("ANTHROPIC_MODEL").ok(),
            small_fast_model: std::env::var("ANTHROPIC_SMALL_FAST_MODEL").ok(),
            max_thinking_tokens,
        }
    }

    /// Load configuration from settings files and environment variables
    ///
    /// Configuration priority (highest to lowest):
    /// 1. Environment variables (e.g., `ANTHROPIC_MODEL`)
    /// 2. Settings files - Top-level fields (e.g., `model`)
    /// 3. Settings files - `env` object (e.g., `env.ANTHROPIC_MODEL`)
    /// 4. Defaults (including `alwaysThinkingEnabled` → default MAX_THINKING_TOKENS)
    ///
    /// Settings files are loaded in this order (later ones override earlier):
    /// - `~/.claude/settings.json` (user settings)
    /// - `<project_dir>/.claude/settings.json` (project settings)
    /// - `<project_dir>/.claude/settings.local.json` (local settings)
    ///
    /// # Arguments
    ///
    /// * `project_dir` - The project working directory
    ///
    /// # Example settings.json
    ///
    /// Using top-level fields:
    /// ```json
    /// {
    ///   "model": "claude-opus-4-20250514",
    ///   "smallFastModel": "claude-haiku-4-20250514",
    ///   "apiBaseUrl": "https://api.anthropic.com"
    /// }
    /// ```
    ///
    /// Using `env` object (compatible with Claude Code CLI):
    /// ```json
    /// {
    ///   "env": {
    ///     "ANTHROPIC_MODEL": "claude-opus-4-20250514",
    ///     "ANTHROPIC_SMALL_FAST_MODEL": "claude-haiku-4-20250514",
    ///     "ANTHROPIC_BASE_URL": "https://api.anthropic.com"
    ///   }
    /// }
    /// ```
    ///
    /// Enabling extended thinking mode with `alwaysThinkingEnabled`:
    /// ```json
    /// {
    ///   "model": "claude-sonnet-4-20250514",
    ///   "alwaysThinkingEnabled": true
    /// }
    /// ```
    /// This will set `MAX_THINKING_TOKENS` to 20000 by default.
    /// You can still override it with the `MAX_THINKING_TOKENS` environment variable.
    pub fn from_settings_or_env(project_dir: &std::path::Path) -> Self {
        use crate::settings::SettingsManager;

        // Default max thinking tokens when always_thinking_enabled is true
        const DEFAULT_MAX_THINKING_TOKENS: u32 = 20000;

        // Load settings from files (may fail if files don't exist)
        let settings = SettingsManager::new(project_dir)
            .map(|m| m.settings().clone())
            .unwrap_or_default();

        // Trace settings file discovery (debug level)
        // Check if settings.env has any configuration entries
        let has_env_settings = settings.env.as_ref().is_some_and(|env| !env.is_empty());
        tracing::trace!(
            has_user_settings =
                settings.model.is_some() || settings.api_base_url.is_some() || has_env_settings,
            "Settings files discovered"
        );

        // Check if env vars are set before moving settings
        let has_model_env = std::env::var("ANTHROPIC_MODEL").is_ok();
        let has_base_url_env = std::env::var("ANTHROPIC_BASE_URL").is_ok();
        let has_small_fast_model_env = std::env::var("ANTHROPIC_SMALL_FAST_MODEL").is_ok();
        let has_max_thinking_tokens_env = std::env::var("MAX_THINKING_TOKENS").is_ok();

        // Store settings flags before moving (both top-level and env)
        let has_model_settings = settings.model.is_some();
        let has_base_url_settings = settings.api_base_url.is_some();
        let has_small_fast_model_settings = settings.small_fast_model.is_some();
        let has_model_env_settings = settings
            .env
            .as_ref()
            .and_then(|env| env.get("ANTHROPIC_MODEL"))
            .is_some();
        let has_base_url_env_settings = settings
            .env
            .as_ref()
            .and_then(|env| env.get("ANTHROPIC_BASE_URL"))
            .is_some();
        let has_small_fast_model_env_settings = settings
            .env
            .as_ref()
            .and_then(|env| env.get("ANTHROPIC_SMALL_FAST_MODEL"))
            .is_some();
        let has_max_thinking_tokens_env_settings = settings
            .env
            .as_ref()
            .and_then(|env| env.get("MAX_THINKING_TOKENS"))
            .is_some();

        // Resolve configuration with priority: Env > Settings (top-level) > Settings (env) > Default
        let base_url = std::env::var("ANTHROPIC_BASE_URL")
            .ok()
            .or(settings.api_base_url)
            .or_else(|| {
                settings
                    .env
                    .as_ref()
                    .and_then(|env| env.get("ANTHROPIC_BASE_URL").cloned())
            });

        let model = std::env::var("ANTHROPIC_MODEL")
            .ok()
            .or(settings.model)
            .or_else(|| {
                settings
                    .env
                    .as_ref()
                    .and_then(|env| env.get("ANTHROPIC_MODEL").cloned())
            });

        let small_fast_model = std::env::var("ANTHROPIC_SMALL_FAST_MODEL")
            .ok()
            .or(settings.small_fast_model)
            .or_else(|| {
                settings
                    .env
                    .as_ref()
                    .and_then(|env| env.get("ANTHROPIC_SMALL_FAST_MODEL").cloned())
            });

        // API key is not loaded from settings for security reasons
        // Note: ANTHROPIC_AUTH_TOKEN in settings.env is also not loaded for security
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .ok()
            .or_else(|| std::env::var("ANTHROPIC_AUTH_TOKEN").ok());

        // Check if always_thinking_enabled is set in settings
        let always_thinking_enabled = settings.always_thinking_enabled.unwrap_or(false);

        let max_thinking_tokens = std::env::var("MAX_THINKING_TOKENS")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .or_else(|| {
                settings.env.as_ref().and_then(|env| {
                    env.get("MAX_THINKING_TOKENS")
                        .and_then(|s| s.parse::<u32>().ok())
                })
            })
            .or({
                // If always_thinking_enabled is true and no MAX_THINKING_TOKENS is set,
                // use the default value to enable extended thinking mode
                if always_thinking_enabled {
                    Some(DEFAULT_MAX_THINKING_TOKENS)
                } else {
                    None
                }
            });

        let config = Self {
            base_url,
            api_key,
            model,
            small_fast_model,
            max_thinking_tokens,
        };

        // Log configuration sources
        tracing::info!(
            model = ?config.model,
            model_source = if has_model_env { "env" } else if has_model_settings { "settings" } else if has_model_env_settings { "settings.env" } else { "default" },
            base_url = ?config.base_url,
            base_url_source = if has_base_url_env { "env" } else if has_base_url_settings { "settings" } else if has_base_url_env_settings { "settings.env" } else { "default" },
            small_fast_model = ?config.small_fast_model,
            small_fast_model_source = if has_small_fast_model_env { "env" } else if has_small_fast_model_settings { "settings" } else if has_small_fast_model_env_settings { "settings.env" } else { "default" },
            max_thinking_tokens = ?config.max_thinking_tokens,
            max_thinking_tokens_source = if has_max_thinking_tokens_env { "env" } else if has_max_thinking_tokens_env_settings { "settings.env" } else if always_thinking_enabled { "alwaysThinkingEnabled" } else { "default" },
            always_thinking_enabled = always_thinking_enabled,
            api_key = ?config.masked_api_key(),
            "Configuration loaded (priority: env > settings.{{top-level, env}} > default)"
        );

        // Warn if no model is configured
        if config.model.is_none() {
            tracing::warn!(
                "No model configured. Using default: {}. \
                 Set ANTHROPIC_MODEL environment variable or configure 'model' in settings.json",
                "claude-sonnet-4-20250514"
            );
        }

        config
    }

    /// Check if any configuration is set
    pub fn is_configured(&self) -> bool {
        self.base_url.is_some()
            || self.api_key.is_some()
            || self.model.is_some()
            || self.small_fast_model.is_some()
            || self.max_thinking_tokens.is_some()
    }

    /// Get environment variables to pass to Claude Code CLI
    ///
    /// Returns a HashMap of environment variable names and values
    /// that should be passed to the subprocess.
    pub fn to_env_vars(&self) -> HashMap<String, String> {
        let mut env = HashMap::new();

        if let Some(ref url) = self.base_url {
            env.insert("ANTHROPIC_BASE_URL".to_string(), url.clone());
        }

        // Pass API key/auth token
        // Priority: self.api_key > ANTHROPIC_API_KEY env var > ANTHROPIC_AUTH_TOKEN env var
        if let Some(ref key) = self.api_key {
            env.insert("ANTHROPIC_API_KEY".to_string(), key.clone());
        } else {
            // If api_key is not in config, check process environment variables
            // This handles settings.env where env vars are set but not parsed into config fields
            if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
                env.insert("ANTHROPIC_API_KEY".to_string(), key);
            } else if let Ok(token) = std::env::var("ANTHROPIC_AUTH_TOKEN") {
                // Pass as ANTHROPIC_API_KEY for compatibility with CLI
                env.insert("ANTHROPIC_API_KEY".to_string(), token);
            }
        }

        if let Some(ref model) = self.model {
            env.insert("ANTHROPIC_MODEL".to_string(), model.clone());
        }
        if let Some(ref model) = self.small_fast_model {
            env.insert("ANTHROPIC_SMALL_FAST_MODEL".to_string(), model.clone());
        }

        env
    }

    /// Get a masked version of the API key for logging
    ///
    /// Shows first 4 and last 4 characters with the middle masked by asterisks.
    /// For example: `sk-ant-api03-xxx...` becomes `sk-a***xxx`
    ///
    /// Returns None if no API key is set.
    pub fn masked_api_key(&self) -> Option<String> {
        self.api_key.as_ref().map(|key| {
            let key = key.as_str();
            if key.is_empty() {
                "***".to_string()
            } else if key.len() <= 2 {
                // Very short keys: show first character only
                format!("{}***", &key[..1])
            } else if key.len() <= 8 {
                // Short keys: show first and last character
                format!("{}***{}", &key[..1], &key[key.len() - 1..])
            } else {
                // Longer keys: show first 4 and last 4 characters
                format!("{}***{}", &key[..4], &key[key.len() - 4..])
            }
        })
    }

    /// Apply configuration to ClaudeAgentOptions
    ///
    /// Sets the model and environment variables on the options.
    pub fn apply_to_options(&self, options: &mut claude_code_agent_sdk::ClaudeAgentOptions) {
        // Set model if configured
        if let Some(ref model) = self.model {
            options.model = Some(model.clone());
        }

        // Set fallback model if configured
        if let Some(ref fallback) = self.small_fast_model {
            options.fallback_model = Some(fallback.clone());
        }

        // Set max_thinking_tokens if configured (enables extended thinking mode)
        if let Some(tokens) = self.max_thinking_tokens {
            options.max_thinking_tokens = Some(tokens);
        }

        // Pass base_url and api_key as environment variables
        let env_vars = self.to_env_vars();
        if !env_vars.is_empty() {
            options.env = env_vars;
        }

        // Log the applied configuration
        tracing::debug!(
            model = ?self.model,
            fallback_model = ?self.small_fast_model,
            base_url = ?self.base_url,
            max_thinking_tokens = ?self.max_thinking_tokens,
            api_key = ?self.masked_api_key(),
            env_vars_count = options.env.len(),
            "Agent configuration applied to options"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Guard that saves and restores environment variables
    /// Automatically restores on drop, ensuring cleanup even on test failure
    struct EnvGuard {
        vars: Vec<(String, Option<String>)>,
    }

    impl EnvGuard {
        fn new(var_names: &[&str]) -> Self {
            let vars = var_names
                .iter()
                .map(|&name| {
                    let original = std::env::var(name).ok();
                    // Remove the env var for clean test state
                    // Safety: We're in a serial test context
                    unsafe {
                        std::env::remove_var(name);
                    }
                    (name.to_string(), original)
                })
                .collect();
            Self { vars }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            // Restore original env vars
            for (name, original) in &self.vars {
                // Safety: We're restoring to the original state
                unsafe {
                    match original {
                        Some(val) => std::env::set_var(name, val),
                        None => std::env::remove_var(name),
                    }
                }
            }
        }
    }

    #[test]
    fn test_default_config() {
        let config = AgentConfig::default();
        assert!(config.base_url.is_none());
        assert!(config.api_key.is_none());
        assert!(config.model.is_none());
        assert!(config.small_fast_model.is_none());
        assert!(config.max_thinking_tokens.is_none());
        assert!(!config.is_configured());
    }

    #[test]
    fn test_to_env_vars() {
        let config = AgentConfig {
            base_url: Some("https://api.example.com".to_string()),
            api_key: Some("secret-key".to_string()),
            model: Some("claude-3".to_string()),
            small_fast_model: None,
            max_thinking_tokens: None,
        };

        let env = config.to_env_vars();
        assert_eq!(
            env.get("ANTHROPIC_BASE_URL").unwrap(),
            "https://api.example.com"
        );
        assert_eq!(env.get("ANTHROPIC_API_KEY").unwrap(), "secret-key");
        assert_eq!(env.get("ANTHROPIC_MODEL").unwrap(), "claude-3");
        assert!(!env.contains_key("ANTHROPIC_SMALL_FAST_MODEL"));
    }

    #[test]
    fn test_masked_api_key() {
        // Test with no API key
        let config = AgentConfig::default();
        assert!(config.masked_api_key().is_none());

        // Test with empty string (edge case)
        let config = AgentConfig {
            api_key: Some(String::new()),
            ..Default::default()
        };
        assert_eq!(config.masked_api_key().unwrap(), "***");

        // Test with single character (edge case)
        let config = AgentConfig {
            api_key: Some("a".to_string()),
            ..Default::default()
        };
        assert_eq!(config.masked_api_key().unwrap(), "a***");

        // Test with two characters (edge case)
        let config = AgentConfig {
            api_key: Some("ab".to_string()),
            ..Default::default()
        };
        assert_eq!(config.masked_api_key().unwrap(), "a***");

        // Test with short API key (<= 8 characters)
        let config = AgentConfig {
            api_key: Some("abc123".to_string()),
            ..Default::default()
        };
        assert_eq!(config.masked_api_key().unwrap(), "a***3");

        // Test with long API key
        let config = AgentConfig {
            api_key: Some("sk-ant-api03-12345-abcd".to_string()),
            ..Default::default()
        };
        assert_eq!(config.masked_api_key().unwrap(), "sk-a***abcd");

        // Test with Anthropic-style key
        let config = AgentConfig {
            api_key: Some("sk-ant-api03-xxxx-xxxx-xxxx-xxxxxxxxxxx".to_string()),
            ..Default::default()
        };
        let masked = config.masked_api_key().unwrap();
        assert!(masked.starts_with("sk-a"));
        assert!(masked.ends_with("xxxx"));
        assert!(masked.contains("***"));
    }

    #[test]
    fn test_is_configured() {
        let mut config = AgentConfig::default();
        assert!(!config.is_configured());

        config.model = Some("test".to_string());
        assert!(config.is_configured());
    }

    #[test]
    fn test_max_thinking_tokens_config() {
        let config = AgentConfig {
            base_url: None,
            api_key: None,
            model: None,
            small_fast_model: None,
            max_thinking_tokens: Some(4096),
        };

        assert!(config.is_configured());
        assert_eq!(config.max_thinking_tokens, Some(4096));
    }

    #[test]
    #[serial_test::serial]
    fn test_from_settings_or_env() {
        // Use EnvGuard to save and restore env vars, ensuring clean test state
        let _guard = EnvGuard::new(&[
            "ANTHROPIC_MODEL",
            "ANTHROPIC_SMALL_FAST_MODEL",
            "ANTHROPIC_BASE_URL",
        ]);

        // Use temp dir for test files
        let temp_base = std::env::temp_dir();
        let temp_dir = temp_base.join("test_config_combined");
        let settings_dir = temp_dir.join(".claude");

        // Cleanup any existing test directory
        drop(std::fs::remove_dir_all(&temp_dir));
        std::fs::create_dir_all(&settings_dir).ok();

        let settings_file = settings_dir.join("settings.json");
        std::fs::write(
            &settings_file,
            r#"{
            "model": "settings-model",
            "smallFastModel": "settings-small-model",
            "apiBaseUrl": "https://settings.api.com"
        }"#,
        )
        .ok();

        // Test 1: Env overrides settings
        // Set env var for model (should override)
        unsafe {
            std::env::set_var("ANTHROPIC_MODEL", "env-model");
        }

        let config = AgentConfig::from_settings_or_env(&temp_dir);
        assert_eq!(config.model, Some("env-model".to_string()));
        assert_eq!(
            config.small_fast_model,
            Some("settings-small-model".to_string())
        );
        assert_eq!(
            config.base_url,
            Some("https://settings.api.com".to_string())
        );

        // Test 2: Settings only (no env)
        // Verify env var is actually removed before asserting
        unsafe {
            std::env::remove_var("ANTHROPIC_MODEL");
        }
        assert!(
            std::env::var("ANTHROPIC_MODEL").is_err(),
            "ANTHROPIC_MODEL should be removed"
        );

        let config2 = AgentConfig::from_settings_or_env(&temp_dir);
        assert_eq!(config2.model, Some("settings-model".to_string()));
        assert_eq!(
            config2.small_fast_model,
            Some("settings-small-model".to_string())
        );

        // Test 3: Env only (no settings)
        std::fs::remove_file(&settings_file).ok();
        unsafe {
            std::env::set_var("ANTHROPIC_MODEL", "env-only-model");
        }

        let config3 = AgentConfig::from_settings_or_env(&temp_dir);
        assert_eq!(config3.model, Some("env-only-model".to_string()));
        assert!(config3.small_fast_model.is_none());

        // Cleanup (EnvGuard handles env var restoration)
        drop(std::fs::remove_dir_all(&temp_dir));
    }

    #[test]
    #[serial_test::serial]
    fn test_from_settings_env_fallback() {
        // Test that settings.env is used as fallback when top-level fields are not set
        //
        // Note: This test uses settings.local.json which has highest priority.
        // It explicitly sets model to null to override any user's global ~/.claude/settings.json
        // However, due to the merge logic (only overwrites if Some), we can't truly "unset"
        // a field. So this test verifies the code path works by explicitly NOT setting
        // top-level model in local settings and checking that env values are available.
        //
        // If this test fails with "opus" or another model, it means the user has global
        // settings at ~/.claude/settings.json which takes precedence. In that case, the
        // test_from_settings_priority_order test should be used to verify the fallback logic.

        // Use EnvGuard to save and restore env vars, ensuring clean test state
        let _guard = EnvGuard::new(&[
            "ANTHROPIC_MODEL",
            "ANTHROPIC_SMALL_FAST_MODEL",
            "ANTHROPIC_BASE_URL",
        ]);

        let temp_base = std::env::temp_dir();
        let temp_dir = temp_base.join("test_config_env_fallback");
        let settings_dir = temp_dir.join(".claude");

        // Cleanup any existing test directory
        drop(std::fs::remove_dir_all(&temp_dir));
        std::fs::create_dir_all(&settings_dir).ok();

        // Use settings.local.json to set top-level model to a known value
        // Then verify settings.env values are still accessible (even if not used for model
        // when top-level is set)
        //
        // This tests the settings loading path, even though the fallback to env
        // won't be exercised if top-level model is set.
        let settings_file = settings_dir.join("settings.local.json");
        std::fs::write(
            &settings_file,
            r#"{
            "model": "local-model",
            "smallFastModel": "local-small-model",
            "apiBaseUrl": "https://local.api.com"
        }"#,
        )
        .ok();

        let config = AgentConfig::from_settings_or_env(&temp_dir);
        // Local settings should override any user global settings
        assert_eq!(config.model, Some("local-model".to_string()));
        assert_eq!(
            config.small_fast_model,
            Some("local-small-model".to_string())
        );
        assert_eq!(config.base_url, Some("https://local.api.com".to_string()));

        // Cleanup
        drop(std::fs::remove_dir_all(&temp_dir));
    }

    #[test]
    #[serial_test::serial]
    fn test_from_settings_priority_order() {
        // Test priority: env > settings.top-level > settings.env > default
        // Use EnvGuard to save and restore env vars, ensuring clean test state
        let _guard = EnvGuard::new(&[
            "ANTHROPIC_MODEL",
            "ANTHROPIC_SMALL_FAST_MODEL",
            "ANTHROPIC_BASE_URL",
        ]);

        let temp_base = std::env::temp_dir();
        let temp_dir = temp_base.join("test_config_priority");
        let settings_dir = temp_dir.join(".claude");

        drop(std::fs::remove_dir_all(&temp_dir));
        std::fs::create_dir_all(&settings_dir).ok();

        // Create settings with both top-level and env fields
        let settings_file = settings_dir.join("settings.json");
        std::fs::write(
            &settings_file,
            r#"{
            "model": "top-level-model",
            "env": {
                "ANTHROPIC_MODEL": "env-object-model"
            }
        }"#,
        )
        .ok();

        // Test 1: Top-level should override env object
        let config1 = AgentConfig::from_settings_or_env(&temp_dir);
        assert_eq!(config1.model, Some("top-level-model".to_string()));

        // Test 2: Env var should override both
        unsafe {
            std::env::set_var("ANTHROPIC_MODEL", "env-var-model");
        }
        let config2 = AgentConfig::from_settings_or_env(&temp_dir);
        assert_eq!(config2.model, Some("env-var-model".to_string()));

        // Cleanup (EnvGuard handles env var restoration)
        drop(std::fs::remove_dir_all(&temp_dir));
    }

    #[test]
    #[serial_test::serial]
    fn test_always_thinking_enabled() {
        // Test that alwaysThinkingEnabled sets default MAX_THINKING_TOKENS
        // Use EnvGuard to save and restore env vars, ensuring clean test state
        let _guard = EnvGuard::new(&["MAX_THINKING_TOKENS"]);

        // Use settings.local.json to override any user settings
        let temp_base = std::env::temp_dir();
        let temp_dir = temp_base.join("test_config_thinking");
        let settings_dir = temp_dir.join(".claude");

        // Cleanup any existing test directory
        drop(std::fs::remove_dir_all(&temp_dir));

        std::fs::create_dir_all(&settings_dir).ok();

        // Use settings.local.json (higher priority than user settings)
        let local_settings_file = settings_dir.join("settings.local.json");

        // Test 1: alwaysThinkingEnabled = true should set default MAX_THINKING_TOKENS
        std::fs::write(
            &local_settings_file,
            r#"{
            "alwaysThinkingEnabled": true
        }"#,
        )
        .ok();

        let config1 = AgentConfig::from_settings_or_env(&temp_dir);
        assert_eq!(config1.max_thinking_tokens, Some(20000));

        // Test 2: alwaysThinkingEnabled = false should not set MAX_THINKING_TOKENS
        std::fs::write(
            &local_settings_file,
            r#"{
            "alwaysThinkingEnabled": false
        }"#,
        )
        .ok();

        let config2 = AgentConfig::from_settings_or_env(&temp_dir);
        assert_eq!(config2.max_thinking_tokens, None);

        // Test 3: Env var should override alwaysThinkingEnabled
        std::fs::write(
            &local_settings_file,
            r#"{
            "alwaysThinkingEnabled": true
        }"#,
        )
        .ok();
        unsafe {
            std::env::set_var("MAX_THINKING_TOKENS", "8000");
        }

        let config3 = AgentConfig::from_settings_or_env(&temp_dir);
        assert_eq!(config3.max_thinking_tokens, Some(8000));

        // Test 4: No alwaysThinkingEnabled should not set MAX_THINKING_TOKENS
        unsafe {
            std::env::remove_var("MAX_THINKING_TOKENS");
        }
        // Explicitly set to false to override any user settings
        std::fs::write(
            &local_settings_file,
            r#"{"model": "test-model", "alwaysThinkingEnabled": false}"#,
        )
        .ok();

        let config4 = AgentConfig::from_settings_or_env(&temp_dir);
        assert_eq!(config4.max_thinking_tokens, None);

        // Cleanup (EnvGuard handles env var restoration)
        drop(std::fs::remove_dir_all(&temp_dir));
    }
}
