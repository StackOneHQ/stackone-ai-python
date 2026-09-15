.PHONY: help install format test test-examples build validate

# `make` on its own lists the targets rather than running the first one.
.DEFAULT_GOAL := help

## Show this help
help:
	@echo "Usage: make <target>"
	@echo
	@awk '/^## /{desc=substr($$0,4); next} \
		/^[a-z][a-z-]*:/{split($$0,t,":"); printf "  \033[36m%-15s\033[0m %s\n", t[1], desc}' \
		$(MAKEFILE_LIST)
	@echo
	@echo "Notes:"
	@echo "  install        core only; uv sync REMOVES anything outside that set."
	@echo "                 Use 'make install extras=1' for a dev environment."
	@echo "  format         the one command to run before committing"
	@echo "  build          local artifact check only; publishing happens in the"
	@echo "                 release workflow after a merge to main, never by hand"
	@echo "  test-examples  only imports each example; the __main__ guard means no body runs"
	@echo
	@echo "CI runs the underlying ruff/ty/pytest commands directly rather than"
	@echo "these targets, so it can never mutate the tree to make itself pass."
	@echo "Secret scanning (gitleaks) runs in CI only."

## Install dependencies (extras=1 adds adapters, examples and dev tooling)
install:
	uv sync $(if $(extras),--all-extras,)

## Fix lint, format, and type check
format:
	uv run ruff check --fix .
	uv run ruff format .
	uv run ty check stackone_ai

## Run all tests
test:
	uv run pytest

## Run example tests (import-only; see note above)
test-examples:
	uv run pytest examples

## Validate against the conformance mock (no live API, no credentials)
validate:
	./scripts/validate.sh

## Build package
build:
	uv build
