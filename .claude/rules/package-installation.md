---
description: Standards for installing packages with UV in StackOne. (project)
globs: "**/pyproject.toml"
paths: "**/pyproject.toml"
---

# Package Installation Standards

Standards for installing packages with UV in the StackOne repository.

## Root Level Dev Dependencies

```bash
# Install dev dependencies at root level
uv add --dev pytest
uv add --dev pytest-cov
uv add --dev black
```

## Package Level Dependencies

```bash
# Install package dependencies
uv add pydantic
uv add requests
```

## Never Use

```bash
# ❌ Don't use pip install
uv pip install package-name

# ❌ Don't use -e or editable installs
uv pip install -e .
```

## Running Tests

```bash
# Run from root directory
uv run pytest

# Run specific package tests
uv run pytest stackone_ai

# Run tests on examples
uv run pytest examples
```

## Package Dependencies in pyproject.toml

```toml
[project]
dependencies = [
    "pydantic>=2.10.6",
    "requests>=2.32.3",
]
```
