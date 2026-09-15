.PHONY: help install lint format test coverage test-tools test-examples run-example ty gitleaks build publish validate

# `make` on its own lists the targets rather than running the first one.
.DEFAULT_GOAL := help

## Show this help
help:
	@echo "Usage: make <target>"
	@echo
	@grep -E '^##|^[a-z-]+:' $(MAKEFILE_LIST) \
		| sed -e 's/^## //' -e 's/:.*//' \
		| awk 'NR%2{desc=$$0; next} {printf "  \033[36m%-15s\033[0m %s\n", $$0, desc}'
	@echo
	@echo "Variables:"
	@echo "  EXTRAS         extra args for 'make install' (e.g. EXTRAS=\"--all-extras\")"
	@echo "  FILE           example filename for 'make run-example'"
	@echo
	@echo "Notes:"
	@echo "  publish        pushes to PyPI for real — CI runs this on release"
	@echo "  gitleaks       needs the gitleaks binary on PATH (brew install gitleaks)"
	@echo "  test-examples  only imports each example; the __main__ guard means no body runs"

## Install dependencies (EXTRAS="--all-extras" to include optional groups)
install:
	uv sync $(EXTRAS)

## Run linting and format check (ruff)
lint:
	uv run ruff check .
	uv run ruff format --check .

## Format and auto-fix linting issues
format:
	uv run ruff check --fix .
	uv run ruff format .

## Run all tests
test:
	uv run pytest

## Run tests with coverage
coverage:
	uv run pytest --cov --cov-report=term --cov-report=json --cov-report=html

## Run the tests/ directory
test-tools:
	uv run pytest tests

## Run example tests (import-only; see note above)
test-examples:
	uv run pytest examples

## Run a specific example (FILE=openai_integration.py)
run-example:
	uv run examples/$(FILE)

## Run type checking
ty:
	uv run ty check stackone_ai

## Run gitleaks secret detection (requires gitleaks on PATH)
gitleaks:
	gitleaks detect --source . --config .gitleaks.toml

## Validate against conformance, ADK and Pydantic AI consumers
validate:
	./scripts/validate.sh

## Build package
build:
	uv build

## Publish package to PyPI
publish:
	uv publish
