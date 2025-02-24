import importlib.resources
from pathlib import Path

# Use bundled specs directly
OAS_DIR = Path(str(importlib.resources.files("stackone_ai") / "oas"))
