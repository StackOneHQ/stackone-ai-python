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
- Run `just lint` to check, `just lint-fix` to auto-fix

## Type Annotations

- Full type annotations required for all public APIs
- Use Python 3.11+ typing features
- Run `just ty` to verify type correctness
- Strict ty configuration is enforced

## Pre-commit Hooks

Pre-commit hooks are configured for:

- ruff linting
- ty type checking

Run `just install` to set up hooks.

## Essential Commands

```bash
just install       # Install dependencies and pre-commit hooks
just lint          # Run ruff linting
just lint-fix      # Auto-fix linting issues
just ty            # Run type checking
just test          # Run all tests
just test-tools    # Run tool-specific tests
just test-examples # Run example tests
```

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
