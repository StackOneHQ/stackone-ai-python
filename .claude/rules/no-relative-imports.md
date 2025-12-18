---
description: Enforce the use of absolute imports instead of relative imports in Python files. (project)
globs: "**/*.py"
paths: "**/*.py"
---

# No Relative Imports

Standards for using absolute imports instead of relative imports in Python files.

## Guidelines

- Always use absolute imports starting with the full package name (`stackone_ai`)
- Never use relative imports (`.` or `..`)
- Keep imports organized and grouped

## Examples

```python
# Good - absolute imports
from stackone_ai.tools import ToolDefinition
from stackone_ai.constants import OAS_DIR

# Bad - relative imports (don't use)
from .tools import ToolDefinition
from ..constants import OAS_DIR
```
