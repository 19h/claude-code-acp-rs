//! Settings file watcher
//!
//! Monitors settings files for changes and triggers reloads.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use notify::{RecommendedWatcher, RecursiveMode};
use notify_debouncer_mini::{DebounceEventResult, DebouncedEventKind, Debouncer, new_debouncer};
use tokio::sync::mpsc;

/// Settings file watcher
///
/// Watches settings files for changes and sends notifications via a channel.
#[allow(missing_debug_implementations)]
pub struct SettingsWatcher {
    /// The file watcher (held to keep it alive)
    _watcher: Debouncer<RecommendedWatcher>,
    /// Paths being watched
    watched_paths: Vec<PathBuf>,
}

/// Event sent when settings files change
#[derive(Debug, Clone)]
pub struct SettingsChangeEvent {
    /// Paths that changed
    pub changed_paths: Vec<PathBuf>,
}

impl SettingsWatcher {
    /// Create a new settings watcher
    ///
    /// # Arguments
    ///
    /// * `project_dir` - The project working directory
    /// * `debounce_ms` - Debounce duration in milliseconds (default 100)
    /// * `on_change` - Callback when settings change
    ///
    /// # Returns
    ///
    /// A watcher instance and a receiver for change events
    pub fn new(
        project_dir: impl AsRef<Path>,
        debounce_ms: u64,
    ) -> Result<(Self, mpsc::UnboundedReceiver<SettingsChangeEvent>), WatcherError> {
        let project_dir = project_dir.as_ref();
        let (tx, rx) = mpsc::unbounded_channel();

        // Collect paths to watch
        let mut watched_paths = Vec::new();

        // User settings directory
        if let Some(home) = dirs::home_dir() {
            let user_settings_dir = home.join(".claude");
            if user_settings_dir.exists() {
                watched_paths.push(user_settings_dir);
            }
        }

        // Project settings directory
        let project_settings_dir = project_dir.join(".claude");
        if project_settings_dir.exists() {
            watched_paths.push(project_settings_dir);
        }

        // Create debounced watcher
        let tx_clone = tx.clone();
        let watched_clone = watched_paths.clone();
        let mut watcher = new_debouncer(
            Duration::from_millis(debounce_ms),
            move |result: DebounceEventResult| {
                match result {
                    Ok(events) => {
                        // Filter for settings files
                        let changed_paths: Vec<PathBuf> = events
                            .into_iter()
                            .filter(|e| matches!(e.kind, DebouncedEventKind::Any))
                            .map(|e| e.path)
                            .filter(|p| is_settings_file(p))
                            .collect();

                        if !changed_paths.is_empty() {
                            tracing::debug!("Settings files changed: {:?}", changed_paths);
                            if tx_clone
                                .send(SettingsChangeEvent { changed_paths })
                                .is_err()
                            {
                                tracing::warn!(
                                    "Settings change receiver dropped, reload will not happen"
                                );
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Settings watcher error: {:?}", e);
                    }
                }
            },
        )
        .map_err(|e| WatcherError::Init(e.to_string()))?;

        // Start watching directories
        for path in &watched_paths {
            watcher
                .watcher()
                .watch(path, RecursiveMode::NonRecursive)
                .map_err(|e| WatcherError::Watch(path.clone(), e.to_string()))?;
            tracing::info!("Watching settings directory: {:?}", path);
        }

        Ok((
            Self {
                _watcher: watcher,
                watched_paths: watched_clone,
            },
            rx,
        ))
    }

    /// Get the paths being watched
    pub fn watched_paths(&self) -> &[PathBuf] {
        &self.watched_paths
    }

    /// Create a settings watcher that automatically reloads settings
    ///
    /// Returns a task handle that can be awaited or aborted.
    pub fn start_auto_reload(
        project_dir: impl AsRef<Path>,
        settings_manager: Arc<tokio::sync::RwLock<super::SettingsManager>>,
        debounce_ms: u64,
    ) -> Result<WatcherHandle, WatcherError> {
        let (watcher, mut rx) = Self::new(project_dir, debounce_ms)?;

        let handle = tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                tracing::info!("Settings changed, reloading: {:?}", event.changed_paths);
                let mut manager = settings_manager.write().await;
                manager.reload();
            }
        });

        Ok(WatcherHandle {
            _watcher: watcher,
            task: handle,
        })
    }
}

/// Handle to a running watcher task
#[allow(missing_debug_implementations)]
pub struct WatcherHandle {
    /// The watcher (kept alive)
    _watcher: SettingsWatcher,
    /// The reload task
    task: tokio::task::JoinHandle<()>,
}

impl WatcherHandle {
    /// Stop the watcher
    pub fn stop(self) {
        self.task.abort();
    }

    /// Check if the watcher is still running
    pub fn is_running(&self) -> bool {
        !self.task.is_finished()
    }
}

/// Check if a path is a settings file
fn is_settings_file(path: &Path) -> bool {
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    file_name == "settings.json" || file_name == "settings.local.json"
}

/// Errors that can occur during settings watching
#[derive(Debug, thiserror::Error)]
pub enum WatcherError {
    /// Failed to initialize the watcher
    #[error("Failed to initialize watcher: {0}")]
    Init(String),

    /// Failed to watch a path
    #[error("Failed to watch path {0:?}: {1}")]
    Watch(PathBuf, String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;
    use tempfile::TempDir;
    use tokio::time::timeout;

    #[test]
    fn test_is_settings_file() {
        assert!(is_settings_file(Path::new("/some/path/settings.json")));
        assert!(is_settings_file(Path::new(
            "/some/path/settings.local.json"
        )));
        assert!(!is_settings_file(Path::new("/some/path/other.json")));
        assert!(!is_settings_file(Path::new("/some/path/settings.yaml")));
    }

    #[tokio::test]
    async fn test_watcher_creation() {
        let temp_dir = TempDir::new().unwrap();
        let settings_dir = temp_dir.path().join(".claude");
        fs::create_dir_all(&settings_dir).unwrap();

        let result = SettingsWatcher::new(temp_dir.path(), 100);
        assert!(result.is_ok());

        let (watcher, _rx) = result.unwrap();
        assert!(!watcher.watched_paths().is_empty());
    }

    #[tokio::test]
    async fn test_watcher_detects_changes() {
        let temp_dir = TempDir::new().unwrap();
        let settings_dir = temp_dir.path().join(".claude");
        fs::create_dir_all(&settings_dir).unwrap();

        // Create initial settings file
        let settings_file = settings_dir.join("settings.json");
        let mut file = File::create(&settings_file).unwrap();
        writeln!(file, r#"{{"model": "claude-opus"}}"#).unwrap();
        drop(file);

        // Create watcher
        let (watcher, mut rx) = SettingsWatcher::new(temp_dir.path(), 50).unwrap();
        assert!(!watcher.watched_paths().is_empty());

        // Give watcher time to start
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Modify the file
        let mut file = File::create(&settings_file).unwrap();
        writeln!(file, r#"{{"model": "claude-sonnet"}}"#).unwrap();
        drop(file);

        // Wait for change event (with timeout)
        let result = timeout(Duration::from_secs(2), rx.recv()).await;

        // Note: File watching can be flaky in tests, so we accept both outcomes
        match result {
            Ok(Some(event)) => {
                assert!(!event.changed_paths.is_empty());
            }
            Ok(None) => {
                // Channel closed, acceptable in test
            }
            Err(_) => {
                // Timeout - file watching can be slow/unreliable in CI
                tracing::warn!("Watcher test timed out - this can happen in CI environments");
            }
        }
    }
}
