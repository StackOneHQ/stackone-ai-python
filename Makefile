.PHONY: help install format test coverage test-examples build validate

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
	@echo "  EXTRAS         args for 'make install' (default --all-extras;"
	@echo "                 EXTRAS=\"\" installs the minimal set and REMOVES extras)"
	@echo
	@echo "Notes:"
	@echo "  format         the one command to run before committing"
	@echo "  build          local artifact check only; publishing happens in the"
	@echo "                 release workflow after a merge to main, never by hand"
	@echo "  test-examples  only imports each example; the __main__ guard means no body runs"
	@echo
	@echo "CI runs the underlying ruff/ty/pytest commands directly rather than"
	@echo "these targets, so it can never mutate the tree to make itself pass."
	@echo "Secret scanning (gitleaks) runs in CI only."

# `uv sync` makes the environment match the requested set exactly, so a bare
# sync uninstalls every optional dependency. Default to the full set: the
# examples and the mcp-backed tests need it. Override with EXTRAS="" for a
# minimal environment.
EXTRAS ?= --all-extras

## Install dependencies (EXTRAS="" for the minimal set)
install:
	uv sync $(EXTRAS)

## Fix lint, format, and type check
format:
	uv run ruff check --fix .
	uv run ruff format .
	uv run ty check stackone_ai

## Run all tests
test:
	uv run pytest

## Run tests with coverage
coverage:
	uv run pytest --cov --cov-report=term --cov-report=json --cov-report=html

## Run example tests (import-only; see note above)
test-examples:
	uv run pytest examples

## Validate against conformance, ADK and Pydantic AI consumers
validate:
	./scripts/validate.sh

## Build package
build:
	uv build
