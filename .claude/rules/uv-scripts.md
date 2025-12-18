---
description: Standards for creating and managing utility scripts with UV. (project)
globs: scripts/**/*.py
paths: scripts/**/*.py
---

# UV Scripts Standards

Standards for creating and managing utility scripts with UV.

## Location

- Place all utility scripts in the `scripts/` directory
- NOT for examples (use `examples/` directory instead)

## UV Script Dependencies Header

```python
# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "package1",
#   "package2>=1.0.0"
# ]
# ///
```

## Script Structure

- Type hints are required
- Use async/await when doing I/O operations
- Include main guard: `if __name__ == "__main__":`
- Add return types to functions

## Running Scripts

```bash
uv run scripts/your_script.py
```

## Error Handling

- Use try/except blocks for external calls
- Print meaningful error messages
- Exit with appropriate status codes

## Example Script

```python
# /// script
# requires-python = ">=3.8"
# dependencies = ["httpx"]
# ///

from typing import Dict
import asyncio
import httpx

async def fetch_data() -> Dict:
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com")
        return response.json()

if __name__ == "__main__":
    asyncio.run(fetch_data())
```
