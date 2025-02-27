install:
	uv sync --all-extras
	uv run pre-commit install

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix .

test:
	uv run pytest

test-tools:
	uv run pytest tests

test-examples:
	uv run pytest examples

mypy:
	uv run mypy stackone_ai

docs-serve:
	uv run scripts/build_docs.py
	uv run mkdocs serve

docs-build:
	uv run scripts/build_docs.py
	uv run mkdocs build

mcp-inspector:
	uv sync --all-extras
	npx @modelcontextprotocol/inspector stackmcp
