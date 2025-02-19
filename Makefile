install:
	uv sync --all-extras
	uv run pre-commit install

lint:
	uv run ruff check .

test:
	uv run pytest

mypy:
	uv run mypy packages/stackone-ai/stackone_ai
