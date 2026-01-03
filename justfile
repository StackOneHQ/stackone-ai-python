# Install dependencies and pre-commit hooks
install *extras:
	uv sync {{ extras }}

# Run linting and format check (ruff, typos, nixfmt, oxfmt)
lint:
	nix fmt -- --fail-on-change

# Format and auto-fix linting issues
format:
	nix fmt

# Run all tests
test:
	uv run pytest

# Run tests with coverage
coverage:
	uv run pytest --cov --cov-report=term --cov-report=json --cov-report=html

# Run tool-specific tests
test-tools:
	uv run pytest tests

# Run example tests
test-examples:
	uv run pytest examples

# Run type checking
ty:
	uv run ty check stackone_ai

# Run gitleaks secret detection
gitleaks:
	gitleaks detect --source . --config .gitleaks.toml

# Update version in __init__.py
update-version:
	uv run scripts/update_version.py

# Build package
build:
	uv build

# Publish package to PyPI
publish:
	uv publish
