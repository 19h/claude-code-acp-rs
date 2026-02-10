//! Glob tool for file pattern matching
//!
//! Fast file pattern matching using glob patterns like `**/*.rs`.

use async_trait::async_trait;
use globset::{Glob, GlobSetBuilder};
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::Path;
use walkdir::WalkDir;

use super::base::Tool;
use crate::mcp::registry::{ToolContext, ToolResult};

/// Maximum number of results to return
const MAX_RESULTS: usize = 1000;

/// Glob tool for file pattern matching
#[derive(Debug, Default)]
pub struct GlobTool;

/// Input parameters for Glob
#[derive(Debug, Deserialize)]
struct GlobInput {
    /// The glob pattern to match files against
    pattern: String,
    /// The directory to search in (defaults to cwd)
    #[serde(default)]
    path: Option<String>,
    /// Limit output to first N results
    #[serde(default)]
    head_limit: Option<usize>,
    /// Skip first N results
    #[serde(default)]
    offset: Option<usize>,
}

impl GlobTool {
    /// Create a new Glob tool instance
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &str {
        "Glob"
    }

    fn description(&self) -> &str {
        "Fast file pattern matching tool. Supports glob patterns like '**/*.js' or 'src/**/*.ts'. \
         Returns matching file paths sorted by modification time."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "required": ["pattern"],
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The glob pattern to match files against"
                },
                "path": {
                    "type": "string",
                    "description": "The directory to search in. If not specified, the current working directory will be used."
                },
                "head_limit": {
                    "type": "integer",
                    "description": "Limit output to first N results (default: 1000)"
                },
                "offset": {
                    "type": "integer",
                    "description": "Skip first N results (default: 0)"
                }
            }
        })
    }

    async fn execute(&self, input: Value, context: &ToolContext) -> ToolResult {
        // Parse input
        let params: GlobInput = match serde_json::from_value(input) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid input: {}", e)),
        };

        // Determine search directory
        let search_dir = match &params.path {
            Some(p) => {
                let path = Path::new(p);
                if path.is_absolute() {
                    path.to_path_buf()
                } else {
                    context.cwd.join(path)
                }
            }
            None => context.cwd.clone(),
        };

        // Validate directory exists
        if !search_dir.exists() {
            return ToolResult::error(format!("Directory not found: {}", search_dir.display()));
        }

        if !search_dir.is_dir() {
            return ToolResult::error(format!("Path is not a directory: {}", search_dir.display()));
        }

        // Build glob matcher
        let glob = match Glob::new(&params.pattern) {
            Ok(g) => g,
            Err(e) => return ToolResult::error(format!("Invalid glob pattern: {}", e)),
        };

        let mut builder = GlobSetBuilder::new();
        builder.add(glob);
        let glob_set = match builder.build() {
            Ok(gs) => gs,
            Err(e) => return ToolResult::error(format!("Failed to build glob set: {}", e)),
        };

        // Collect matching files with modification times
        let mut matches: Vec<(std::path::PathBuf, std::time::SystemTime)> = Vec::new();

        // Determine the head_limit for collection
        let collection_limit = params.head_limit.unwrap_or(MAX_RESULTS);
        let offset = params.offset.unwrap_or(0);
        // We need to collect more than offset + head_limit to account for filtering
        // Use a 3x multiplier to handle cases where many entries are filtered (directories, non-matching files)
        let total_to_collect = offset
            .saturating_add(collection_limit)
            .saturating_mul(3)
            .min(MAX_RESULTS * 3);

        for entry in WalkDir::new(&search_dir)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            let path = entry.path();

            // Skip directories
            if path.is_dir() {
                continue;
            }

            // Get relative path for matching
            let Ok(relative_path) = path.strip_prefix(&search_dir) else {
                continue;
            };

            // Check if matches pattern
            if glob_set.is_match(relative_path) {
                // Get modification time
                let mtime = entry
                    .metadata()
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH);

                matches.push((path.to_path_buf(), mtime));

                // Limit results for collection
                if matches.len() >= total_to_collect {
                    break;
                }
            }
        }

        // Sort by modification time (most recent first)
        matches.sort_by(|a, b| b.1.cmp(&a.1));

        // Apply offset and head_limit
        let total_found = matches.len();
        let was_truncated = total_found >= total_to_collect;

        let file_list: Vec<String> = matches
            .into_iter()
            .skip(offset)
            .take(collection_limit)
            .map(|(path, _)| path.display().to_string())
            .collect();

        let returned_count = file_list.len();

        let output = if file_list.is_empty() {
            format!("No files matching pattern '{}' found.", params.pattern)
        } else {
            let mut result = file_list.join("\n");
            if was_truncated {
                result.push_str(&format!(
                    "\n\n... (showing {} results, use head_limit to see more)",
                    returned_count
                ));
            }
            result
        };

        ToolResult::success(output).with_metadata(json!({
            "count": returned_count,
            "total_found": total_found,
            "truncated": was_truncated,
            "pattern": params.pattern,
            "offset": offset,
            "head_limit": collection_limit
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_glob_tool_properties() {
        let tool = GlobTool::new();
        assert_eq!(tool.name(), "Glob");
        assert!(tool.description().contains("pattern"));
    }

    #[test]
    fn test_glob_input_schema() {
        let tool = GlobTool::new();
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
    async fn test_glob_find_files() {
        let temp_dir = TempDir::new().unwrap();

        // Create test files
        let src_dir = temp_dir.path().join("src");
        fs::create_dir(&src_dir).unwrap();

        File::create(src_dir.join("main.rs"))
            .unwrap()
            .write_all(b"fn main() {}")
            .unwrap();
        File::create(src_dir.join("lib.rs"))
            .unwrap()
            .write_all(b"pub mod lib;")
            .unwrap();
        File::create(temp_dir.path().join("README.md"))
            .unwrap()
            .write_all(b"# README")
            .unwrap();

        let tool = GlobTool::new();
        let context = ToolContext::new("test", temp_dir.path());

        // Find all .rs files
        let result = tool.execute(json!({"pattern": "**/*.rs"}), &context).await;

        assert!(!result.is_error);
        assert!(result.content.contains("main.rs"));
        assert!(result.content.contains("lib.rs"));
        assert!(!result.content.contains("README.md"));
    }

    #[tokio::test]
    async fn test_glob_no_matches() {
        let temp_dir = TempDir::new().unwrap();

        let tool = GlobTool::new();
        let context = ToolContext::new("test", temp_dir.path());

        let result = tool.execute(json!({"pattern": "**/*.xyz"}), &context).await;

        assert!(!result.is_error);
        assert!(result.content.contains("No files matching"));
    }

    #[tokio::test]
    async fn test_glob_invalid_pattern() {
        let temp_dir = TempDir::new().unwrap();

        let tool = GlobTool::new();
        let context = ToolContext::new("test", temp_dir.path());

        let result = tool.execute(json!({"pattern": "[invalid"}), &context).await;

        assert!(result.is_error);
        assert!(result.content.contains("Invalid glob pattern"));
    }

    #[tokio::test]
    async fn test_glob_with_path() {
        let temp_dir = TempDir::new().unwrap();

        // Create nested structure
        let sub_dir = temp_dir.path().join("sub");
        fs::create_dir(&sub_dir).unwrap();
        File::create(sub_dir.join("test.txt"))
            .unwrap()
            .write_all(b"test")
            .unwrap();

        let tool = GlobTool::new();
        let context = ToolContext::new("test", temp_dir.path());

        let result = tool
            .execute(
                json!({
                    "pattern": "*.txt",
                    "path": "sub"
                }),
                &context,
            )
            .await;

        assert!(!result.is_error);
        assert!(result.content.contains("test.txt"));
    }
}
