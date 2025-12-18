# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules and Skills Structure

- **Rules** (`.claude/rules/`): Automatically loaded based on file paths. Source of truth for project conventions.
- **Skills** (`.claude/skills/`): Manually invoked for specific integrations.
- **Cursor rules** (`.cursor/rules/`): Symlinks to `.claude/rules/` for consistency.

## Available Rules

| Rule                         | Applies To          | Description                                        |
| ---------------------------- | ------------------- | -------------------------------------------------- |
| **git-workflow**             | All files           | Commit conventions, branch strategy, PR guidelines |
| **development-workflow**     | All files           | Code style, file naming, project conventions       |
| **release-please-standards** | All files           | Release versioning with release-please             |
| **no-relative-imports**      | `**/*.py`           | Enforce absolute imports in Python files           |
| **package-installation**     | `**/pyproject.toml` | UV package management standards                    |
| **uv-scripts**               | `scripts/**/*.py`   | Utility script standards with UV                   |
| **examples-standards**       | `examples/**/*`     | Example requirements and organization              |

## Project Overview

StackOne AI SDK is a Python library that provides a unified interface for accessing various SaaS tools through AI-friendly APIs. It acts as a bridge between AI applications and multiple SaaS platforms (HRIS, CRM, ATS, LMS, Marketing, etc.) with support for OpenAI, LangChain, CrewAI, and Model Context Protocol (MCP).

## Essential Development Commands

```bash
# Setup and installation
just install           # Install dependencies and pre-commit hooks

# Code quality
just lint             # Run ruff linting
just lint-fix         # Auto-fix linting issues
just ty               # Run type checking

# Testing
just test             # Run all tests
just test-tools       # Run tool-specific tests
just test-examples    # Run example tests
```

## Code Architecture

### Core Components

1. **StackOneToolSet** (`stackone_ai/toolset.py`): Main entry point
   - Handles authentication (API key + optional account ID)
   - Manages tool loading with glob pattern filtering
   - Provides format converters for OpenAI/LangChain

2. **Models** (`stackone_ai/models.py`): Data structures
   - `StackOneTool`: Base class with execution logic
   - `Tools`: Container for managing multiple tools
   - Format converters for different AI frameworks

3. **OpenAPI Parser** (`stackone_ai/specs/parser.py`): Spec conversion
   - Converts OpenAPI specs to tool definitions
   - Handles file upload detection (`format: binary` → `type: file`)
   - Resolves schema references

### OpenAPI Specifications

All tool definitions are generated from OpenAPI specs in `stackone_ai/oas/`:

- `core.json`, `ats.json`, `crm.json`, `documents.json`, `hris.json`, `iam.json`, `lms.json`, `marketing.json`

## Key Development Patterns

### Tool Filtering

```python
# Use glob patterns for tool selection
tools = StackOneToolSet(include_tools=["hris_*", "!hris_create_*"])
```

### Authentication

```python
# Uses environment variables or direct configuration
toolset = StackOneToolSet(
    api_key="your-api-key",  # or STACKONE_API_KEY env var
    account_id="optional-id"  # explicit account ID required
)
```

### Type Safety

- Full type annotations required (Python 3.11+)
- Strict ty configuration
- Use generics for better IDE support

### Testing

- Snapshot testing for tool parsing (`tests/snapshots/`)
- Async tests use `pytest-asyncio`
- See `examples-standards` rule for example validation

## Important Considerations

1. **Dependencies**: See `package-installation` rule for uv dependency management
2. **Pre-commit**: Hooks configured for ruff and ty - run on all commits
3. **Python Version**: Requires Python >=3.11
4. **Error Handling**: Custom exceptions (`StackOneError`, `StackOneAPIError`)
5. **File Uploads**: Binary parameters auto-detected from OpenAPI specs
6. **Context Window**: Tool loading warns when loading all tools (large context)

## Common Tasks

### Adding New SaaS Integration

1. Add OpenAPI spec to `stackone_ai/oas/`
2. Parser automatically converts to tool definitions
3. Test with `make test-tools`

### Modifying Tool Behavior

- Core execution logic in `StackOneTool.execute()` method
- HTTP configuration via `ExecuteConfig` class
- Response handling in `_process_response()`
