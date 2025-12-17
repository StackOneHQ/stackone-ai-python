# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Cursor Rules

- @./.cursor/rules/package-installation
- @./.cursor/rules/no-relative-imports
- @./.cursor/rules/uv-scripts
- @./.cursor/rules/release-please-standards
- @./.cursor/rules/examples-standards

## Project Overview

StackOne AI SDK is a Python library that provides a unified interface for accessing various SaaS tools through AI-friendly APIs. It acts as a bridge between AI applications and multiple SaaS platforms (HRIS, CRM, ATS, LMS, Marketing, etc.) with support for OpenAI, LangChain, CrewAI, and Model Context Protocol (MCP).

## Essential Development Commands

```bash
# Setup and installation
make install           # Install dependencies and pre-commit hooks

# Code quality
make lint             # Run ruff linting
make lint-fix         # Auto-fix linting issues  
make mypy             # Run type checking

# Testing
make test             # Run all tests
make test-tools       # Run tool-specific tests
make test-examples    # Run example tests

# Documentation
make docs-serve       # Build and serve docs locally (http://localhost:8000)
make docs-build       # Build docs for deployment

# MCP Development
make mcp-inspector    # Run MCP server inspector for debugging
```

## Code Architecture

### Core Components

1. **StackOneToolSet** (`stackone_ai/toolset.py`): Main entry point
   - Handles authentication (API key + optional account ID)
   - Fetches tools dynamically via MCP endpoint
   - Provides format converters for OpenAI/LangChain

2. **Models** (`stackone_ai/models.py`): Data structures
   - `StackOneTool`: Base class with execution logic
   - `Tools`: Container for managing multiple tools
   - Format converters for different AI frameworks

3. **MCP Server** (`stackone_ai/server.py`): Protocol implementation
   - Async tool execution
   - CLI interface via `stackmcp` command

## Key Development Patterns

### Tool Fetching
```python
# Fetch tools dynamically from MCP endpoint
toolset = StackOneToolSet(api_key="your-api-key")
tools = toolset.fetch_tools(
    account_ids=["account-1"],
    providers=["hibob"],
    actions=["*_list_*"]
)
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
- Strict mypy configuration
- Use generics for better IDE support

### Testing
- Async tests use `pytest-asyncio`
- Example validation: See @./.cursor/rules/examples-standards

## Important Considerations

1. **Dependencies**: See @./.cursor/rules/package-installation for uv dependency management
2. **Pre-commit**: Hooks configured for ruff and mypy - run on all commits
3. **Python Version**: Requires Python >=3.11
4. **Error Handling**: Custom exceptions (`StackOneError`, `StackOneAPIError`)

## Common Tasks

### Modifying Tool Behavior
- Core execution logic in `StackOneTool.execute()` method
- HTTP configuration via `ExecuteConfig` class
- RPC tool execution via `_StackOneRpcTool` class

### Updating Documentation
- Examples requirements: See @./.cursor/rules/examples-standards
- Run `make docs-serve` to preview changes
- MkDocs config in `mkdocs.yml`
