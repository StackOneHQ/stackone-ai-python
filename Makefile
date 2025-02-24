install:
	uv sync --all-extras
	uv run pre-commit install

lint:
	uv run ruff check .

test:
	uv run pytest

test-tools:
	uv run pytest packages/stackone-ai

test-examples:
	uv run pytest examples

mypy:
	uv run mypy packages/stackone-ai/stackone_ai
