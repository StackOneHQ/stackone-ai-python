install:
	uv sync --all-extras
	uv run pre-commit install

lint:
	uv run ruff check .

test:
	uv run pytest

test-tools:
	uv run pytest stackone_ai

test-examples:
	uv run pytest examples

mypy:
	uv run mypy stackone_ai
