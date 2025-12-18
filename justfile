# Install dependencies and pre-commit hooks
install:
	uv sync --all-extras

# Run ruff linting
lint:
	uv run ruff check .

# Auto-fix linting issues
lint-fix:
	uv run ruff check --fix .

# Run all tests
test:
	uv run pytest

# Run tool-specific tests
test-tools:
	uv run pytest tests

# Run example tests
test-examples:
	uv run pytest examples

# Run type checking
mypy:
	uv run mypy stackone_ai

# Run typos spell checker
typos:
	typos .
