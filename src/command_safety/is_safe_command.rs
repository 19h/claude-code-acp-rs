//! Safe command detection
//!
//! Determines if a command is known to be safe and can be auto-approved.
//!
//! Reference: vendors/codex/codex-rs/core/src/command_safety/is_safe_command.rs

use super::extract_command_basename;

/// Check if a command is known to be safe (read-only, non-destructive)
///
/// Safe commands can be auto-approved without user confirmation in Default mode.
///
/// # Examples
/// ```ignore
/// assert!(is_known_safe_command("ls -la /tmp"));
/// assert!(is_known_safe_command("find . -name '*.rs'"));
/// assert!(!is_known_safe_command("find . -delete"));
/// assert!(!is_known_safe_command("rm -rf /"));
/// ```
pub fn is_known_safe_command(command: &str) -> bool {
    let parts: Vec<&str> = command.split_whitespace().collect();

    let Some(first) = parts.first() else {
        return false;
    };

    let cmd_name = extract_command_basename(first);

    match cmd_name {
        // Unconditionally safe: read-only file viewing
        "cat" | "head" | "tail" | "less" | "more" => true,

        // Unconditionally safe: system info queries
        "ls" | "pwd" | "whoami" | "id" | "uname" | "hostname" | "date" | "uptime" => true,

        // Unconditionally safe: text processing (read-only)
        "grep" | "egrep" | "fgrep" | "wc" | "cut" | "tr" | "sort" | "uniq" | "nl" | "paste"
        | "rev" | "seq" | "expr" => true,

        // Unconditionally safe: output commands
        "echo" | "printf" | "true" | "false" => true,

        // Unconditionally safe: path/file info
        "which" | "whereis" | "type" | "file" | "stat" | "realpath" | "basename" | "dirname" => {
            true
        }

        // Unconditionally safe: directory navigation
        "cd" => true,

        // Conditionally safe: find (without dangerous options)
        "find" => !has_unsafe_find_options(&parts),

        // Conditionally safe: git (only read-only subcommands)
        "git" => is_safe_git_subcommand(&parts),

        // Conditionally safe: cargo (only check)
        "cargo" => matches!(parts.get(1).copied(), Some("check")),

        // Conditionally safe: ripgrep (without unsafe options)
        "rg" => !has_unsafe_rg_options(&parts),

        // Conditionally safe: sed (only print mode)
        "sed" => is_safe_sed_command(&parts),

        // Conditionally safe: base64 (without output file)
        "base64" => !has_unsafe_base64_options(&parts),

        // Anything else is not known to be safe
        _ => false,
    }
}

/// Check if find command has unsafe options
///
/// Unsafe find options:
/// - `-exec`, `-execdir`, `-ok`, `-okdir`: Execute arbitrary commands
/// - `-delete`: Delete matching files
/// - `-fls`, `-fprint`, `-fprint0`, `-fprintf`: Write to files
fn has_unsafe_find_options(parts: &[&str]) -> bool {
    const UNSAFE_FIND_OPTIONS: &[&str] = &[
        "-exec", "-execdir", "-ok", "-okdir", "-delete", "-fls", "-fprint", "-fprint0", "-fprintf",
    ];

    parts.iter().any(|arg| UNSAFE_FIND_OPTIONS.contains(arg))
}

/// Check if git subcommand is safe (read-only)
fn is_safe_git_subcommand(parts: &[&str]) -> bool {
    matches!(
        parts.get(1).copied(),
        Some("status" | "log" | "diff" | "show" | "branch" | "remote" | "tag" | "describe")
    )
}

/// Check if ripgrep has unsafe options
///
/// Unsafe rg options:
/// - `--pre`: Execute arbitrary preprocessor command
/// - `--hostname-bin`: Execute command to get hostname
/// - `--search-zip`, `-z`: Calls out to decompression tools
fn has_unsafe_rg_options(parts: &[&str]) -> bool {
    parts.iter().any(|arg| {
        *arg == "--search-zip"
            || *arg == "-z"
            || *arg == "--pre"
            || arg.starts_with("--pre=")
            || *arg == "--hostname-bin"
            || arg.starts_with("--hostname-bin=")
    })
}

/// Check if sed command is safe (only print mode: sed -n Np)
fn is_safe_sed_command(parts: &[&str]) -> bool {
    // Only allow `sed -n {N|M,N}p [file]` pattern
    if parts.len() < 3 || parts.len() > 4 {
        return false;
    }

    if parts.get(1) != Some(&"-n") {
        return false;
    }

    // Check if the pattern is like "Np" or "M,Np"
    if let Some(pattern) = parts.get(2) {
        is_valid_sed_print_pattern(pattern)
    } else {
        false
    }
}

/// Check if sed pattern is valid print pattern (e.g., "10p", "1,5p")
fn is_valid_sed_print_pattern(pattern: &str) -> bool {
    let Some(core) = pattern.strip_suffix('p') else {
        return false;
    };

    let parts: Vec<&str> = core.split(',').collect();
    match parts.as_slice() {
        [num] => !num.is_empty() && num.chars().all(|c| c.is_ascii_digit()),
        [a, b] => {
            !a.is_empty()
                && !b.is_empty()
                && a.chars().all(|c| c.is_ascii_digit())
                && b.chars().all(|c| c.is_ascii_digit())
        }
        _ => false,
    }
}

/// Check if base64 has unsafe options (output to file)
fn has_unsafe_base64_options(parts: &[&str]) -> bool {
    parts.iter().any(|arg| {
        *arg == "-o" || *arg == "--output" || arg.starts_with("--output=") || arg.starts_with("-o") // covers -o and -ofilename
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unconditionally_safe_commands() {
        // File viewing
        assert!(is_known_safe_command("cat file.txt"));
        assert!(is_known_safe_command("head -n 10 file.txt"));
        assert!(is_known_safe_command("tail -f log.txt"));
        assert!(is_known_safe_command("less file.txt"));

        // System info
        assert!(is_known_safe_command("ls -la /tmp"));
        assert!(is_known_safe_command("pwd"));
        assert!(is_known_safe_command("whoami"));
        assert!(is_known_safe_command("id"));
        assert!(is_known_safe_command("uname -a"));

        // Text processing
        assert!(is_known_safe_command("grep pattern file.txt"));
        assert!(is_known_safe_command("wc -l file.txt"));
        assert!(is_known_safe_command("cut -d: -f1 /etc/passwd"));

        // Output
        assert!(is_known_safe_command("echo hello"));
        assert!(is_known_safe_command("printf '%s\\n' hello"));

        // Path info
        assert!(is_known_safe_command("which ls"));
        assert!(is_known_safe_command("file /bin/ls"));
    }

    #[test]
    fn test_safe_find_commands() {
        assert!(is_known_safe_command("find . -name '*.rs'"));
        assert!(is_known_safe_command("find /tmp -type f"));
        assert!(is_known_safe_command("find . -name '*.txt' -print"));
        assert!(is_known_safe_command("/usr/bin/find . -name '*.rs'"));
    }

    #[test]
    fn test_unsafe_find_commands() {
        assert!(!is_known_safe_command("find . -exec rm {} \\;"));
        assert!(!is_known_safe_command("find . -execdir python {} \\;"));
        assert!(!is_known_safe_command("find . -delete"));
        assert!(!is_known_safe_command("find . -ok rm {} \\;"));
        assert!(!is_known_safe_command("find . -fprint /tmp/out.txt"));
    }

    #[test]
    fn test_safe_git_commands() {
        assert!(is_known_safe_command("git status"));
        assert!(is_known_safe_command("git log --oneline"));
        assert!(is_known_safe_command("git diff HEAD~1"));
        assert!(is_known_safe_command("git show HEAD"));
        assert!(is_known_safe_command("git branch -a"));
        assert!(is_known_safe_command("/usr/bin/git status"));
    }

    #[test]
    fn test_unsafe_git_commands() {
        assert!(!is_known_safe_command("git reset --hard"));
        assert!(!is_known_safe_command("git rm file.txt"));
        assert!(!is_known_safe_command("git push"));
        assert!(!is_known_safe_command("git commit -m 'test'"));
        assert!(!is_known_safe_command("git checkout -b new-branch"));
    }

    #[test]
    fn test_safe_cargo_commands() {
        assert!(is_known_safe_command("cargo check"));
    }

    #[test]
    fn test_unsafe_cargo_commands() {
        assert!(!is_known_safe_command("cargo build"));
        assert!(!is_known_safe_command("cargo run"));
        assert!(!is_known_safe_command("cargo install foo"));
    }

    #[test]
    fn test_safe_rg_commands() {
        assert!(is_known_safe_command("rg pattern file.txt"));
        assert!(is_known_safe_command("rg -n pattern"));
    }

    #[test]
    fn test_unsafe_rg_commands() {
        assert!(!is_known_safe_command("rg --search-zip pattern"));
        assert!(!is_known_safe_command("rg -z pattern"));
        assert!(!is_known_safe_command("rg --pre=cat pattern"));
        assert!(!is_known_safe_command("rg --hostname-bin=hostname pattern"));
    }

    #[test]
    fn test_safe_sed_commands() {
        assert!(is_known_safe_command("sed -n 10p file.txt"));
        assert!(is_known_safe_command("sed -n 1,5p file.txt"));
    }

    #[test]
    fn test_unsafe_sed_commands() {
        assert!(!is_known_safe_command("sed -i 's/foo/bar/' file.txt"));
        assert!(!is_known_safe_command("sed 's/foo/bar/' file.txt"));
    }

    #[test]
    fn test_safe_base64_commands() {
        assert!(is_known_safe_command("base64 file.txt"));
        assert!(is_known_safe_command("base64 -d encoded.txt"));
    }

    #[test]
    fn test_unsafe_base64_commands() {
        assert!(!is_known_safe_command("base64 -o out.bin"));
        assert!(!is_known_safe_command("base64 --output=out.bin"));
    }

    #[test]
    fn test_unknown_commands() {
        assert!(!is_known_safe_command("rm file.txt"));
        assert!(!is_known_safe_command("mv a b"));
        assert!(!is_known_safe_command("cp a b"));
        assert!(!is_known_safe_command("chmod 755 file"));
        assert!(!is_known_safe_command("unknown_command"));
    }

    #[test]
    fn test_full_path_commands() {
        assert!(is_known_safe_command("/usr/bin/ls -la"));
        assert!(is_known_safe_command("/bin/cat file.txt"));
        assert!(is_known_safe_command("/usr/local/bin/rg pattern"));
    }

    #[test]
    fn test_empty_command() {
        assert!(!is_known_safe_command(""));
        assert!(!is_known_safe_command("   "));
    }
}
