import importlib.util
import sys
from pathlib import Path

import pytest


def get_example_files() -> list[str]:
    """Get all example files from the directory"""
    examples_dir = Path(__file__).parent
    examples = []

    for file in examples_dir.glob("*.py"):
        # Skip __init__.py and test files
        if file.name.startswith("__") or file.name.startswith("test_"):
            continue
        examples.append(file.name)

    return examples


EXAMPLES = get_example_files()

# Map of example files to required optional packages
# Note: All examples now require MCP extra for fetch_tools()
OPTIONAL_DEPENDENCIES = {
    "openai_integration.py": ["openai", "mcp"],
    "langchain_integration.py": ["langchain_openai", "mcp"],
    "crewai_integration.py": ["crewai", "mcp"],
    "index.py": ["mcp"],
    "file_uploads.py": ["mcp"],
    "stackone_account_ids.py": ["mcp"],
    "meta_tools_example.py": ["mcp"],
    "mcp_server.py": ["mcp"],
}


def test_example_files_exist() -> None:
    """Verify that we found example files to test"""
    assert len(EXAMPLES) > 0, "No example files found"
    print(f"Found {len(EXAMPLES)} examples")


@pytest.mark.parametrize("example_file", EXAMPLES)
def test_run_example(example_file: str) -> None:
    """Run each example file directly using python"""
    # Skip if optional dependencies are not available
    if example_file in OPTIONAL_DEPENDENCIES:
        for module in OPTIONAL_DEPENDENCIES[example_file]:
            try:
                __import__(module)
            except ImportError:
                pytest.skip(f"Skipping {example_file}: {module} not installed")

    example_path = Path(__file__).parent / example_file

    # Import and run the example module directly
    spec = importlib.util.spec_from_file_location("example", example_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules["example"] = module
        spec.loader.exec_module(module)
