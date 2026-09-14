# CLAUDE.md

Guidance for coding agents working in this repository. `AGENTS.md` is a symlink to
this file, so Claude Code, Cursor and other agents read the same instructions.

## Project Overview

StackOne AI SDK is a Python library providing a unified interface for accessing SaaS
tools through AI-friendly APIs, with support for OpenAI, LangChain, CrewAI and the
Model Context Protocol (MCP).

Requires Python >= 3.10.

## Code Architecture

1. **StackOneToolSet** (`stackone_ai/toolset.py`): main entry point
   - Authentication (API key + optional account ID)
   - Tool loading with glob pattern filtering
   - Format converters for OpenAI/LangChain

2. **Models** (`stackone_ai/models.py`): data structures
   - `StackOneTool`: base class with execution logic
   - `Tools`: container for managing multiple tools
   - Format converters for different AI frameworks

## Commands

```bash
make install       # Install dependencies (EXTRAS="--all-extras" for optional groups)
make lint          # ruff lint + format check
make format        # Auto-fix lint issues and format
make ty            # Type checking
make test          # Run all tests
make test-tools    # Tool-specific tests
make test-examples # Example tests
make run-example FILE=search_tools.py
make build         # Build package
make publish       # Publish to PyPI
```

There are no git hooks. Lint, type checking and tests run in CI on every push; run
them locally before pushing.

## Code Style

- Use [ruff](https://docs.astral.sh/ruff/) for linting and formatting
- Line length: 110 (set in `pyproject.toml`)
- Target version: py310
- Use snake_case for Python files
- Use the `.yaml` extension, not `.yml`
- Prefer early returns and guard clauses over nested conditionals

## Type Annotations

- Full type annotations required for all public APIs
- Use Python 3.10+ typing features
- Strict `ty` configuration is enforced
- Use generics for better IDE support

## Imports

Always use absolute imports starting with the full package name. Never use relative
imports (`.` or `..`).

```python
from stackone_ai.tools import ToolDefinition   # good
from .tools import ToolDefinition              # bad
```

Order: standard library, then third-party, then local.

## Package Management

Use `uv` for all dependency management.

```bash
uv add --dev pytest     # dev dependency
uv add pydantic         # package dependency
uv run pytest           # run from root
```

Never use `uv pip install`, and never use editable installs (`-e`).

## Testing

- Snapshot testing for tool parsing (`tests/snapshots/`)
- Async tests use `pytest-asyncio`
- Examples are tested as part of CI and must work with the latest package version

Integration tests exercise an MCP mock server that runs under `tsx`. They need the
vendored submodule and its Node dependencies, and are skipped without them:

```bash
git submodule update --init --recursive
pnpm install
```

The mock's dependencies are pinned to the exact versions the vendored submodule
resolves. Do not loosen them to caret ranges: a newer MCP SDK rejects the raw
`inputSchema` objects the vendor mock passes.

## Examples

Live in `examples/`, organised into `basic_usage/` and `integrations/`.

- Every public function/class needs at least one example
- Examples are runnable scripts with type hints, following the main code style
- Start each file with a docstring explaining its purpose
- Load credentials from `.env` — never hardcode keys
- Include error handling; document prerequisites and expected output in comments

## Git Workflow

**Never push directly to main without permission.** Branch with
`git switch -c feature-name`, then open a PR.

Flow: branch → change → `make lint` → `make test` → `make format` → commit → PR.

### Commit Messages

Format: `type(scope): description`, where type is one of `feat`, `fix`, `docs`,
`refactor`, `test`, `chore`, `ci`, `perf`.

- Keep commits tiny but meaningful; use `git add -p` to stage selectively
- Explain the *why* in the body, not just the what
- Always write in English
- Reference issues and PRs where relevant
- When moving files, combine deletion and creation in one commit so git records a
  rename and preserves history

### Pull Requests

Use the same title format as commits. Include a summary (1-3 bullets) and a test
plan. Reference issues with `Closes #123`.

## Releases

Releases use release-please, configured in `.release-please-config.json` and
`.release-please-manifest.json`.

Version bumps follow commit type: `feat` bumps the minor version, `fix` the patch
version, and `feat!` signals a breaking change. `docs`, `chore` and `test` do not
trigger a release.

Merging the release PR updates `CHANGELOG.md`, creates the GitHub release, and
publishes to PyPI (requires the `PYPI_API_TOKEN` secret).

## Key Patterns

```python
# Tool filtering via glob patterns
tools = StackOneToolSet(include_tools=["bamboohr_*", "!bamboohr_create_*"])

# Authentication
toolset = StackOneToolSet(
    api_key="your-api-key",   # or STACKONE_API_KEY env var
    account_id="optional-id",
)
```

## Important Considerations

- **Error handling**: custom exceptions (`StackOneError`, `StackOneAPIError`)
- **File uploads**: binary parameters auto-detected from OpenAPI specs
- **Context window**: tool loading warns when loading all tools

### Adding a New SaaS Integration

1. Add the OpenAPI spec to `stackone_ai/oas/`
2. The parser converts it to tool definitions automatically
3. Test with `make test-tools`

### Modifying Tool Behaviour

- Core execution logic: `StackOneTool.execute()`
- HTTP configuration: `ExecuteConfig`
- Response handling: `_process_response()`
