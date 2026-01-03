# Helper to run Python commands (uses uv run if not in Nix environment)
_run := if env("VIRTUAL_ENV", "") != "" { "" } else { "uv run " }

# Install dependencies and pre-commit hooks
install *extras:
	uv sync {{ extras }}

# Run linting and format check (ruff, typos, nixfmt, oxfmt)
lint:
	nix fmt -- --fail-on-change

# Format and auto-fix linting issues
format:
	nix fmt

# Alias for format
lint-fix: format

# Run all tests
test:
	{{ _run }}pytest

# Run tests with coverage
coverage:
	{{ _run }}pytest --cov --cov-report=term --cov-report=json --cov-report=html

# Run tool-specific tests
test-tools:
	{{ _run }}pytest tests

# Run example tests
test-examples:
	{{ _run }}pytest examples

# Run type checking
ty:
	{{ _run }}ty check stackone_ai

# Run gitleaks secret detection
gitleaks:
	gitleaks detect --source . --config .gitleaks.toml

# Update version in __init__.py
update-version:
	{{ _run }}python scripts/update_version.py

# Build package
build:
	uv build

# Publish package to PyPI
publish:
	uv publish
