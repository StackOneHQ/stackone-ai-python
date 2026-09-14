.PHONY: install lint format test coverage test-tools test-examples run-example ty gitleaks build publish

# Install dependencies (EXTRAS="--all-extras" to include optional groups)
install:
	uv sync $(EXTRAS)

# Run linting and format check (ruff)
lint:
	uv run ruff check .
	uv run ruff format --check .

# Format and auto-fix linting issues
format:
	uv run ruff check --fix .
	uv run ruff format .

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

# Run a specific example (FILE=openai_integration.py)
run-example:
	uv run examples/$(FILE)

# Run type checking
ty:
	uv run ty check stackone_ai

# Run gitleaks secret detection (requires gitleaks on PATH)
gitleaks:
	gitleaks detect --source . --config .gitleaks.toml

# Build package
build:
	uv build

# Publish package to PyPI
publish:
	uv publish
