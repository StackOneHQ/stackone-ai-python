---
description: Standards for creating and maintaining examples for all functionality. (project)
globs: examples/**/*
paths: examples/**/*
---

# Examples Standards

Standards for creating and maintaining examples in the StackOne repository.

## Location Requirements

```
examples/
├── basic_usage/
│   ├── basic_tool_usage.py      # Basic usage examples
│   └── error_handling.py        # Error handling examples
├── integrations/                # Integration examples
│   ├── openai_integration.py
│   └── other_integration.py
└── README.md                    # Examples documentation
```

## Example Requirements

- Every public function/class needs at least one example
- Examples should be runnable Python scripts
- Include error handling cases
- Load credentials from .env
- Include type hints
- Follow the same code style as the main codebase

## Documentation

- Each example file should start with a docstring explaining its purpose
- Include expected output in comments
- Document any prerequisites (environment variables, etc)

## Testing

- Examples should be tested as part of CI
- Examples should work with the latest package version
- Include sample responses in comments

## Good Example Structure

```python
import os
from dotenv import load_dotenv
from stackone_ai import StackOneToolSet

def main():
    """Example showing basic usage of StackOneToolSet."""
    load_dotenv()

    api_key = os.getenv("STACKONE_API_KEY")
    if not api_key:
        raise ValueError("STACKONE_API_KEY not found")

    # Example code...

if __name__ == "__main__":
    main()
```

## Bad Example (avoid)

```python
# Missing error handling, docs, types
from stackone_ai import StackOneToolSet

toolset = StackOneToolSet("hardcoded_key")
tools = toolset.get_tools("crm")
result = tools["some_tool"].execute()
```
