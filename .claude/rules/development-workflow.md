---
description: Code style, file naming, and project conventions. (project)
alwaysApply: true
---

# Development Workflow

This rule provides code style guidelines and project conventions for the StackOne AI Python SDK.

## Code Style

- Use [ruff](https://docs.astral.sh/ruff/) for linting and formatting
- Follow PEP 8 style guidelines
- Maximum line length: 88 characters (ruff default)
- Run `make lint` to check, `make format` to auto-fix

## Type Annotations

- Full type annotations required for all public APIs
- Use Python 3.10+ typing features
- Run `make ty` to verify type correctness
- Strict ty configuration is enforced

## Checks

There are no git hooks. Linting, type checking and tests run in CI on every push.
Run them locally before pushing with `make lint`, `make ty` and `make test`.

## Essential Commands

```bash
make install       # Install Python dependencies (EXTRAS="--all-extras" for optional groups)
make lint          # Run ruff lint + format check
make format        # Auto-fix lint issues and format
make ty            # Run type checking
make test          # Run all tests
make test-tools    # Run tool-specific tests
make test-examples # Run example tests
```

Integration tests need the MCP mock server: `git submodule update --init` and `pnpm install`.

## File Naming

- Use snake_case for Python files
- Use `.yaml` extension instead of `.yml` for YAML files
- Keep file names concise but meaningful

## Import Organization

- Standard library imports first
- Third-party imports second
- Local imports last
- Use absolute imports (see no-relative-imports rule)

## Working with Tools

- Use semantic tools for code exploration (avoid full file reads when possible)
- Leverage symbol indexing for fast navigation
- Use grep/ripgrep for pattern matching
- Read only necessary code sections
