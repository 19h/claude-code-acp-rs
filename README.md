# claude-code-acp-rs

[![Crates.io](https://img.shields.io/crates/v/claude-code-acp-rs.svg)](https://crates.io/crates/claude-code-acp-rs)
[![Docs.rs](https://docs.rs/claude-code-acp-rs/badge.svg)](https://docs.rs/claude-code-acp-rs)
[![CI](https://img.shields.io/github/actions/workflow/status/soddygo/claude-code-acp-rs/ci.yml?branch=main)](https://github.com/soddygo/claude-code-acp-rs/actions)
[![Coverage Status](https://coveralls.io/repos/github/soddygo/claude-code-acp-rs/badge.svg?branch=main)](https://coveralls.io/github/soddygo/claude-code-acp-rs?branch=main)

A Rust implementation of Claude Code ACP Agent. Use Claude Code from any ACP-compatible client such as Zed!

This is an alternative to the official [TypeScript implementation](https://github.com/zed-industries/claude-code-acp) (`@zed-industries/claude-code-acp`).

## Installation

### From Cargo

```bash
# Install the rust toolchain first: https://www.rust-lang.org/tools/install
cargo install claude-code-acp-rs
```

### From Source

```bash
git clone https://github.com/soddygo/claude-code-acp-rs.git
cd claude-code-acp-rs
cargo install --path .
```

### With OpenTelemetry Support

To enable distributed tracing with OpenTelemetry:

```bash
# Install with otel feature
cargo install claude-code-acp-rs --features otel

# Or from source
cargo install --path . --features otel
```

## Cross Compilation

Cross-compiling from macOS with plain `cargo build --target ...` will often fail for Linux/Windows targets because Cargo falls back to the host `cc` linker. On macOS that means Apple `ld`, which cannot link ELF/MinGW binaries.

The simplest local solution is `cargo-zigbuild`:

```bash
brew install zig
cargo install cargo-zigbuild

# Linux (verified on macOS)
cargo zigbuild --release \
  --target x86_64-unknown-linux-musl \
  --no-default-features \
  --features otel,sacp-flush

# Windows GNU (verified to Rust compile stage on macOS)
cargo zigbuild --release \
  --target x86_64-pc-windows-gnu \
  --no-default-features \
  --features otel,sacp-flush
```

Important: disable the default `bundled-cli` feature for cross builds. That feature downloads the Claude CLI for the build host, not the target platform, so a macOS cross-build would otherwise package a macOS `claude` binary into a Linux/Windows artifact.

If you prefer containerized builds, `cross build --release --target <target>` also works, but it requires a running Docker daemon.

## Usage

### Command Line

```bash
# Show help
claude-code-acp-rs --help

# Run with diagnostic mode (logs to file)
claude-code-acp-rs --diagnostic

# Run with verbose logging
claude-code-acp-rs -vv
```

### With Zed Editor

Configure Zed to use this agent by specifying `claude-code-acp-rs` as the agent command.

### Environment Variables

- `ANTHROPIC_BASE_URL`: Custom API base URL
- `ANTHROPIC_API_KEY`: API key (preferred)
- `ANTHROPIC_AUTH_TOKEN`: Authentication token (legacy, fallback)
- `ANTHROPIC_MODEL`: Model to use (default: claude-sonnet-4-20250514)
- `ANTHROPIC_SMALL_FAST_MODEL`: Model for fast operations
- `MAX_THINKING_TOKENS`: Maximum tokens for extended thinking mode

### Configuration Files

The agent can also load configuration from Claude Code settings files. When environment variables are not set, the agent will read from:

- `~/.claude/settings.json` (user settings)
- `.claude/settings.json` (project settings)
- `.claude/settings.local.json` (local settings)

**Configuration priority (highest to lowest):**
1. Environment variables
2. Settings files - Top-level fields
3. Settings files - `env` object
4. Defaults

**Supported settings formats:**

Using top-level fields:
```json
{
  "model": "claude-opus-4-20250514",
  "smallFastModel": "claude-haiku-4-20250514",
  "apiBaseUrl": "https://api.anthropic.com"
}
```

Using `env` object (compatible with Claude Code CLI):
```json
{
  "env": {
    "ANTHROPIC_MODEL": "claude-opus-4-20250514",
    "ANTHROPIC_SMALL_FAST_MODEL": "claude-haiku-4-20250514",
    "ANTHROPIC_BASE_URL": "https://api.anthropic.com"
  }
}
```

This allows you to use your existing Claude Code configuration without setting environment variables.

## OpenTelemetry Tracing

When compiled with the `otel` feature, you can enable distributed tracing to debug and monitor the agent:

```bash
# Send traces to Jaeger (default OTLP endpoint)
claude-code-acp-rs --otel-endpoint http://localhost:4317

# With custom service name
claude-code-acp-rs --otel-endpoint http://localhost:4317 --otel-service-name my-claude-agent

# Or use environment variable
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 claude-code-acp-rs
```

### Jaeger Quick Start

```bash
# Start Jaeger with Docker
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  jaegertracing/jaeger:latest

# Run the agent with tracing
claude-code-acp-rs --otel-endpoint http://localhost:4317

# View traces at http://localhost:16686
```

## Coexistence with npm Version

This Rust implementation uses the command name `claude-code-acp-rs` to avoid conflicts with the npm package `@zed-industries/claude-code-acp` (which uses `claude-code-acp`).

Both versions can be installed and used on the same system:
- `claude-code-acp` → npm version (TypeScript)
- `claude-code-acp-rs` → Rust version (this project)

## Extended Thinking Mode

This agent supports Claude's Extended Thinking mode for complex reasoning tasks. See the [Thinking Mode Guide](docs/THINKING_MODE_CN.md) for details.

**Quick Start:**

```bash
# Enable thinking mode with environment variable
export MAX_THINKING_TOKENS=4096
export ANTHROPIC_MODEL="claude-sonnet-4-20250514"

# Run the agent
claude-code-acp-rs
```

**Features:**
- 🧠 Deep reasoning for complex programming tasks
- 🔍 Better code analysis and generation quality
- 📊 Configurable thinking token budget (4096, 8000, 16000)
- 🎯 Support via environment variables or ACP `_meta` field

For more details, see:
- [English Documentation](docs/THINKING_MODE.md)
- [中文文档](docs/THINKING_MODE_CN.md)
- [Example Script](examples/thinking_mode.sh)

## License

* [MIT LICENSE](LICENSE)

## Contribution

[CONTRIBUTING.md](CONTRIBUTING.md)
