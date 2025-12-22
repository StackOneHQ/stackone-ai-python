# Install dependencies and pre-commit hooks
install *extras:
	uv sync {{ extras }}

# Run ruff linting
lint:
	uv run ruff check .

# Auto-fix linting issues
lint-fix:
	uv run ruff check --fix .

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

# Run typos spell checker
typos:
	typos --config typos.toml .

# Run gitleaks secret detection
gitleaks:
	gitleaks detect --source . --config .gitleaks.toml

# Fix typos
typos-fix:
	typos --config typos.toml --write-changes .

# Update version in __init__.py
update-version:
	uv run scripts/update_version.py

# Build package
build:
	uv build

# Publish package to PyPI
publish:
	uv publish
