fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(feature = "bundled-cli")]
    bundled_cli::copy_to_dist();
}

#[cfg(feature = "bundled-cli")]
mod bundled_cli {
    use std::fs;
    use std::path::{Path, PathBuf};

    pub fn copy_to_dist() {
        let manifest_dir = PathBuf::from(
            std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set"),
        );
        let output_dir = manifest_dir.join("bundled-cli");
        let cli_name = if cfg!(target_os = "windows") {
            "claude.exe"
        } else {
            "claude"
        };

        let Some(source_path) = find_bundled_cli(cli_name) else {
            eprintln!(
                "cargo:warning=Bundled CLI not found in ~/.claude/sdk/bundled/. \
                 Archive will not include Claude CLI. Runtime fallback will be used."
            );
            return;
        };

        if let Err(e) = fs::create_dir_all(&output_dir) {
            eprintln!("cargo:warning=Failed to create bundled-cli dir: {e}. Skipping.");
            return;
        }

        let dest_path = output_dir.join(cli_name);

        // Incremental build: skip copy if file size matches
        if dest_path.exists() {
            if let (Ok(src_meta), Ok(dst_meta)) =
                (fs::metadata(&source_path), fs::metadata(&dest_path))
            {
                if src_meta.len() == dst_meta.len() {
                    return;
                }
            }
        }

        match fs::copy(&source_path, &dest_path) {
            Ok(bytes) => {
                eprintln!(
                    "cargo:warning=Bundled CLI copied ({bytes} bytes): {}",
                    dest_path.display()
                );
                #[cfg(unix)]
                set_executable(&dest_path);
            }
            Err(e) => {
                eprintln!("cargo:warning=Failed to copy bundled CLI: {e}. Skipping.");
            }
        }
    }

    /// Scan ~/.claude/sdk/bundled/{version}/ to find CLI binary, preferring the latest version.
    fn find_bundled_cli(cli_name: &str) -> Option<PathBuf> {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .ok()?;
        let bundled_dir = PathBuf::from(home).join(".claude/sdk/bundled");
        if !bundled_dir.is_dir() {
            return None;
        }

        let mut dirs: Vec<_> = fs::read_dir(&bundled_dir)
            .ok()?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .collect();
        // Sort descending by name so newest version comes first
        dirs.sort_by_key(|e| std::cmp::Reverse(e.file_name()));

        dirs.into_iter()
            .map(|e| e.path().join(cli_name))
            .find(|p| p.is_file())
    }

    #[cfg(unix)]
    fn set_executable(path: &Path) {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(meta) = fs::metadata(path) {
            let mut perms = meta.permissions();
            perms.set_mode(0o755);
            drop(fs::set_permissions(path, perms));
        }
    }
}
