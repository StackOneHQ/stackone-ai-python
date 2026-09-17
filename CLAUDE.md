# CLAUDE.md

Guidance for coding agents working in this repository. `AGENTS.md` is a symlink to
this file, so Claude Code, Cursor and other agents read the same instructions.

## Project Overview

StackOne AI SDK is a Python library providing a unified interface for accessing SaaS
tools through AI-friendly APIs, with support for OpenAI, LangChain, CrewAI and the
Model Context Protocol (MCP).

Requires Python >= 3.11.

## Code Architecture

The package is three modules. The guiding property is that **the toolset is the
served catalog** — the schema listed to a model is the schema the MCP server sent,
and the request sent to `/actions/rpc` matches it. Nothing invented, nothing lost.

1. **`types.py`** — `ToolParameters`, `ExecuteConfig`, `ParameterLocation`, the error
   hierarchy, shared aliases and `DEFAULT_BASE_URL`.
2. **`tools.py`** — `StackOneTool` (execution + framework converters), `Tools`
   (container), `StackOneRpcTool` (the RPC envelope), and the MCP listing client.
3. **`toolset.py`** — `StackOneToolSet`: fetches the catalog and exposes it.

Tools come from the MCP endpoint (`/mcp?param-style=flat_prefixed`) and execute
against `/actions/rpc`. There is no OpenAPI parsing and no client-side search.

## Commands

```bash
make               # List all targets (help is the default goal)
make install            # Core dependencies only
make install extras=1   # Adds adapters, examples and dev tooling
make format        # Fix lint, format, and type check — run this before committing
make test          # Run all tests
make validate      # Validate against conformance, ADK and Pydantic AI consumers
make build         # Build package
```

There are no git hooks. Lint, type checking and tests run in CI on every push; run
them locally before pushing.

## Code Style

- Use [ruff](https://docs.astral.sh/ruff/) for linting and formatting
- Line length: 110 (set in `pyproject.toml`)
- Target version: py311
- Use snake_case for Python files
- Use the `.yaml` extension, not `.yml`
- Prefer early returns and guard clauses over nested conditionals

## Type Annotations

- Full type annotations required for all public APIs
- Use Python 3.11+ typing features
- Use generics for better IDE support

## Imports

Always use absolute imports starting with the full package name. Never use relative
imports (`.` or `..`).

```python
from stackone_ai.tools import StackOneTool     # good
from .tools import StackOneTool                # bad
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

Two dependency tiers, nothing else. **Core** is what the SDK needs to function
at all — `pydantic`, `httpx`, `mcp`, since `fetch_tools()` is the only route to
a tool and it talks MCP. **Extras** are everything else: per-framework adapters
(`langchain`, `pydantic-ai`), the `examples` set, and `dev` tooling. Adapters are
imported lazily inside the method that needs them, so a bare install never drags
in a framework you do not use. A new adapter goes in an extra with a lazy
import, never in core.

## Testing

- Async tests use `pytest-asyncio`
- Examples are type-checked against the current SDK by `make validate` and in CI;
  each must fail loudly rather than succeed with an empty catalog

Integration tests exercise an MCP mock server (`tests/mocks/`) that runs under `tsx`:

```bash
pnpm install
```

These tests **fail** rather than skip if Node dependencies are missing — a silent
skip previously let them vanish while CI stayed green.

The mock's dependencies are pinned exactly (`@modelcontextprotocol/sdk`, `zod`,
`hono`, `@hono/mcp`). Do not loosen them to caret ranges: a newer MCP SDK rejects
the raw `inputSchema` objects the mock passes.

## Examples

Live in `examples/` (flat — no subdirectories).

- Every public function/class needs at least one example
- Examples are runnable scripts with type hints, following the main code style
- Start each file with a docstring explaining its purpose
- Load credentials from `.env` — never hardcode keys
- Include error handling; document prerequisites and expected output in comments

## Git Workflow

**Never push directly to main without permission.** Branch with
`git switch -c feature-name`, then open a PR.

Flow: branch → change → `make format` → `make test` → commit → PR.

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

Publishing only ever happens there. There is no `make publish`: releases go out
via release-please after a merge to main, never from a developer machine.

## Key Patterns

```python
# Tool filtering via glob patterns
tools = toolset.fetch_tools(actions=["bamboohr_*"], providers=["bamboohr"])

# Authentication
toolset = StackOneToolSet(
    api_key="your-api-key",   # or STACKONE_API_KEY env var
    account_id="optional-id",
)
```

## Important Considerations

- **Error handling**: custom exceptions (`StackOneError`, `StackOneAPIError`) in `types.py`
- **File downloads**: non-JSON responses return raw bytes plus metadata, not decoded text

### Modifying Tool Behaviour

- Core execution logic: `StackOneTool.execute()` in `tools.py`
- RPC envelope split: `StackOneRpcTool._split_envelope_params`
- HTTP configuration: `ExecuteConfig` in `types.py`

Schemas must reach the model intact. `to_openai_function` passes the served schema
through verbatim, stripping only the SDK's internal `nullable` marker (which becomes
the JSON Schema `required` list). Do not reintroduce an allowlist — the conformance
suite's `--strict-schema` gate checks exactly this.
