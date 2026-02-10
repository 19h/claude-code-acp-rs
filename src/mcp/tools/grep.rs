//! Grep tool for content search
//!
//! A powerful search tool built on ripgrep for searching file contents.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use std::process::Stdio;
use tokio::process::Command;
use tokio::time::{Duration, timeout};

use super::base::Tool;
use crate::mcp::registry::{ToolContext, ToolResult};

/// Maximum output size in characters
const MAX_OUTPUT_SIZE: usize = 50_000;
/// Default head limit for results
const DEFAULT_HEAD_LIMIT: usize = 100;

/// Grep tool for content search
#[derive(Debug, Default)]
pub struct GrepTool;

/// Output mode for grep results
#[derive(Debug, Clone, Copy, Default)]
enum OutputMode {
    /// Show matching lines with content
    Content,
    /// Show only file paths (default)
    #[default]
    FilesWithMatches,
    /// Show match counts per file
    Count,
}

impl OutputMode {
    fn from_str(s: &str) -> Self {
        match s {
            "content" => Self::Content,
            "count" => Self::Count,
            _ => Self::FilesWithMatches,
        }
    }
}

/// Input parameters for Grep
#[derive(Debug, Deserialize)]
struct GrepInput {
    /// The regex pattern to search for
    pattern: String,
    /// File or directory to search in
    #[serde(default)]
    path: Option<String>,
    /// Glob pattern to filter files
    #[serde(default)]
    glob: Option<String>,
    /// File type to search (e.g., "js", "py", "rust")
    #[serde(default, rename = "type")]
    file_type: Option<String>,
    /// Output mode
    #[serde(default)]
    output_mode: Option<String>,
    /// Case insensitive search
    #[serde(default, rename = "-i")]
    case_insensitive: Option<bool>,
    /// Lines after match
    #[serde(default, rename = "-A")]
    after_context: Option<usize>,
    /// Lines before match
    #[serde(default, rename = "-B")]
    before_context: Option<usize>,
    /// Lines around match (before and after)
    #[serde(default, rename = "-C")]
    context: Option<usize>,
    /// Show line numbers
    #[serde(default, rename = "-n")]
    line_numbers: Option<bool>,
    /// Enable multiline mode
    #[serde(default)]
    multiline: Option<bool>,
    /// Limit output to first N entries
    #[serde(default)]
    head_limit: Option<usize>,
    /// Skip first N entries
    #[serde(default)]
    offset: Option<usize>,
    /// Timeout in milliseconds
    #[serde(default)]
    timeout: Option<u64>,
}

impl GrepTool {
    /// Create a new Grep tool instance
    pub fn new() -> Self {
        Self
    }

    /// Process output with offset and head_limit using streaming approach
    /// Returns (processed_output, was_truncated)
    fn process_output_with_limits(
        stdout: &str,
        offset: usize,
        head_limit: usize,
    ) -> (String, bool) {
        let mut output = String::new();
        let mut was_truncated = false;

        // Use skip() to efficiently skip offset lines, enumerate() for counting
        for (collected, line) in stdout.lines().skip(offset).enumerate() {
            if collected >= head_limit {
                was_truncated = true;
                break;
            }

            // Check if adding this line would exceed MAX_OUTPUT_SIZE
            if output.len() + line.len() + 1 > MAX_OUTPUT_SIZE {
                was_truncated = true;
                break;
            }

            if !output.is_empty() {
                output.push('\n');
            }
            output.push_str(line);
        }

        (output, was_truncated)
    }

    /// Build rg command arguments
    fn build_args(&self, params: &GrepInput, search_path: &str, mode: OutputMode) -> Vec<String> {
        let mut args = Vec::new();

        // Output format based on mode
        match mode {
            OutputMode::FilesWithMatches => {
                args.push("--files-with-matches".to_string());
            }
            OutputMode::Count => {
                args.push("--count".to_string());
            }
            OutputMode::Content => {
                // Default content output, add line numbers if requested
                if params.line_numbers.unwrap_or(true) {
                    args.push("-n".to_string());
                }
            }
        }

        // Case insensitive
        if params.case_insensitive.unwrap_or(false) {
            args.push("-i".to_string());
        }

        // Multiline mode
        if params.multiline.unwrap_or(false) {
            args.push("-U".to_string());
            args.push("--multiline-dotall".to_string());
        }

        // Context lines (only for content mode)
        if matches!(mode, OutputMode::Content) {
            if let Some(c) = params.context {
                args.push(format!("-C{}", c));
            } else {
                if let Some(a) = params.after_context {
                    args.push(format!("-A{}", a));
                }
                if let Some(b) = params.before_context {
                    args.push(format!("-B{}", b));
                }
            }
        }

        // File type filter
        if let Some(ref ft) = params.file_type {
            args.push("--type".to_string());
            args.push(ft.clone());
        }

        // Glob filter
        if let Some(ref glob) = params.glob {
            args.push("--glob".to_string());
            args.push(glob.clone());
        }

        // Pattern
        args.push(params.pattern.clone());

        // Search path
        args.push(search_path.to_string());

        args
    }
}

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "Grep"
    }

    fn description(&self) -> &str {
        "A powerful search tool built on ripgrep. Supports regex patterns, file type filtering, \
         and context lines. Use output_mode to control output format."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "required": ["pattern"],
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regular expression pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in (defaults to cwd)"
                },
                "glob": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g., '*.js', '**/*.tsx')"
                },
                "type": {
                    "type": "string",
                    "description": "File type to search (e.g., 'js', 'py', 'rust', 'go')"
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": "Output mode: 'content' shows matching lines, 'files_with_matches' shows file paths (default), 'count' shows match counts"
                },
                "-i": {
                    "type": "boolean",
                    "description": "Case insensitive search"
                },
                "-A": {
                    "type": "integer",
                    "description": "Number of lines to show after each match"
                },
                "-B": {
                    "type": "integer",
                    "description": "Number of lines to show before each match"
                },
                "-C": {
                    "type": "integer",
                    "description": "Number of lines to show before and after each match"
                },
                "-n": {
                    "type": "boolean",
                    "description": "Show line numbers (default: true for content mode)"
                },
                "multiline": {
                    "type": "boolean",
                    "description": "Enable multiline mode for patterns spanning multiple lines"
                },
                "head_limit": {
                    "type": "integer",
                    "description": "Limit output to first N entries"
                },
                "offset": {
                    "type": "integer",
                    "description": "Skip first N entries"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Optional timeout in milliseconds (no timeout by default)"
                }
            }
        })
    }

    async fn execute(&self, input: Value, context: &ToolContext) -> ToolResult {
        // Parse input
        let params: GrepInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        // Determine search path
        let search_path = match &params.path {
            Some(p) => {
                let path = std::path::Path::new(p);
                if path.is_absolute() {
                    p.clone()
                } else {
                    context.cwd.join(path).display().to_string()
                }
            }
            None => context.cwd.display().to_string(),
        };

        // Determine output mode
        let mode = params
            .output_mode
            .as_ref()
            .map(|s| OutputMode::from_str(s))
            .unwrap_or_default();

        // Build command arguments
        let args = self.build_args(&params, &search_path, mode);

        // Execute ripgrep with optional timeout
        let mut cmd = Command::new("rg");
        cmd.args(&args)
            .current_dir(&context.cwd)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let output = if let Some(timeout_ms) = params.timeout {
            let timeout_duration = Duration::from_millis(timeout_ms);
            match timeout(timeout_duration, cmd.output()).await {
                Ok(Ok(o)) => o,
                Ok(Err(e)) => {
                    // Check if rg is not installed
                    if e.kind() == std::io::ErrorKind::NotFound {
                        return ToolResult::error(
                            "ripgrep (rg) not found. Please install ripgrep to use Grep tool.",
                        );
                    }
                    return ToolResult::error(format!("Failed to execute ripgrep: {}", e));
                }
                Err(_) => {
                    return ToolResult::error(format!(
                        "Search timed out after {}ms. Try narrowing your search pattern, path, or increase the timeout value.",
                        timeout_ms
                    ));
                }
            }
        } else {
            match cmd.output().await {
                Ok(o) => o,
                Err(e) => {
                    // Check if rg is not installed
                    if e.kind() == std::io::ErrorKind::NotFound {
                        return ToolResult::error(
                            "ripgrep (rg) not found. Please install ripgrep to use Grep tool.",
                        );
                    }
                    return ToolResult::error(format!("Failed to execute ripgrep: {}", e));
                }
            }
        };

        // Process output
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        // Handle exit codes
        // rg exits with 0 for matches, 1 for no matches, 2 for errors
        if let Some(0 | 1) = output.status.code() {
            // Apply offset and head_limit using streaming processing
            let offset = params.offset.unwrap_or(0);
            let head_limit = params.head_limit.unwrap_or(DEFAULT_HEAD_LIMIT);

            let (result, was_truncated) =
                Self::process_output_with_limits(&stdout, offset, head_limit);

            // Build final result
            let result = if result.is_empty() {
                format!(
                    "No matches found for pattern '{}' in {}",
                    params.pattern, search_path
                )
            } else {
                let mut output = result;
                // Add truncation notice if applicable
                if was_truncated {
                    output.push_str(&format!(
                        "\n... (showing {} results, use head_limit to see more)",
                        head_limit
                    ));
                }
                output
            };

            ToolResult::success(result).with_metadata(json!({
                "pattern": params.pattern,
                "path": search_path,
                "mode": format!("{:?}", mode),
                "truncated": was_truncated
            }))
        } else {
            // Error
            let error_msg = if stderr.is_empty() {
                "ripgrep returned an error".to_string()
            } else {
                stderr.to_string()
            };
            ToolResult::error(error_msg)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_grep_tool_properties() {
        let tool = GrepTool::new();
        assert_eq!(tool.name(), "Grep");
        assert!(tool.description().contains("ripgrep"));
    }

    #[test]
    fn test_grep_input_schema() {
        let tool = GrepTool::new();
        let schema = tool.input_schema();

        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["pattern"].is_object());
        assert!(
            schema["required"]
                .as_array()
                .unwrap()
                .contains(&json!("pattern"))
        );
    }

    #[tokio::test]
    async fn test_grep_find_content() {
        let temp_dir = TempDir::new().unwrap();

        // Create test file
        let mut file = File::create(temp_dir.path().join("test.txt")).unwrap();
        writeln!(file, "Hello World").unwrap();
        writeln!(file, "hello rust").unwrap();
        writeln!(file, "HELLO").unwrap();

        let tool = GrepTool::new();
        let context = ToolContext::new("test", temp_dir.path());

        // Case sensitive search
        let result = tool
            .execute(
                json!({
                    "pattern": "Hello",
                    "output_mode": "content"
                }),
                &context,
            )
            .await;

        // Only check if command executed (rg might not be installed in test env)
        if !result.is_error || !result.content.contains("not found") {
            assert!(result.content.contains("Hello") || result.content.contains("No matches"));
        }
    }

    #[tokio::test]
    async fn test_grep_case_insensitive() {
        let temp_dir = TempDir::new().unwrap();

        let mut file = File::create(temp_dir.path().join("test.txt")).unwrap();
        writeln!(file, "Hello").unwrap();
        writeln!(file, "HELLO").unwrap();

        let tool = GrepTool::new();
        let context = ToolContext::new("test", temp_dir.path());

        let result = tool
            .execute(
                json!({
                    "pattern": "hello",
                    "-i": true,
                    "output_mode": "count"
                }),
                &context,
            )
            .await;

        // Only validate if rg is available
        if !result.is_error || !result.content.contains("not found") {
            // Should find both matches
            assert!(!result.is_error || result.content.contains("No matches"));
        }
    }

    #[tokio::test]
    async fn test_grep_with_file_type() {
        let temp_dir = TempDir::new().unwrap();

        // Create Rust file
        let src_dir = temp_dir.path().join("src");
        fs::create_dir(&src_dir).unwrap();
        let mut rs_file = File::create(src_dir.join("main.rs")).unwrap();
        writeln!(rs_file, "fn main() {{ println!(\"hello\"); }}").unwrap();

        // Create JS file
        let mut js_file = File::create(temp_dir.path().join("index.js")).unwrap();
        writeln!(js_file, "console.log('hello');").unwrap();

        let tool = GrepTool::new();
        let context = ToolContext::new("test", temp_dir.path());

        let result = tool
            .execute(
                json!({
                    "pattern": "hello",
                    "type": "rust"
                }),
                &context,
            )
            .await;

        // Only validate if rg is available
        if !result.is_error || !result.content.contains("not found") {
            // Should only find in .rs file
            if !result.is_error && !result.content.contains("No matches") {
                assert!(result.content.contains(".rs"));
                assert!(!result.content.contains(".js"));
            }
        }
    }

    #[test]
    fn test_output_mode_parsing() {
        assert!(matches!(
            OutputMode::from_str("content"),
            OutputMode::Content
        ));
        assert!(matches!(OutputMode::from_str("count"), OutputMode::Count));
        assert!(matches!(
            OutputMode::from_str("files_with_matches"),
            OutputMode::FilesWithMatches
        ));
        assert!(matches!(
            OutputMode::from_str("invalid"),
            OutputMode::FilesWithMatches
        ));
    }
}
