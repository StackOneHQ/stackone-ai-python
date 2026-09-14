import importlib.util
import os
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
    "langgraph_integration.py": ["langgraph", "langchain_openai", "mcp"],
    "pydantic_ai_integration.py": ["pydantic_ai", "mcp"],
    "search_tools.py": ["mcp"],
    "auth_management.py": ["mcp"],
}


# Every example talks to StackOne, so without these it cannot exercise the SDK.
STACKONE_CREDENTIALS = ("STACKONE_API_KEY", "STACKONE_ACCOUNT_ID")

# These additionally drive a live LLM, so pointing STACKONE_BASE_URL at a mock is
# not enough to make them runnable.
LLM_CREDENTIALS = ("OPENAI_API_KEY",)
LLM_EXAMPLES = {
    "crewai_integration.py",
    "langchain_integration.py",
    "langgraph_integration.py",
    "openai_integration.py",
    "pydantic_ai_integration.py",
}


def _missing(variables: tuple[str, ...]) -> list[str]:
    return [name for name in variables if not os.getenv(name)]


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

    # Credentials are checked here rather than left to the example's own guard:
    # a guard that returns early makes the test pass while exercising nothing.
    missing_stackone = _missing(STACKONE_CREDENTIALS)
    if missing_stackone:
        pytest.skip(f"Skipping {example_file}: {', '.join(missing_stackone)} not set")

    if example_file in LLM_EXAMPLES:
        missing_llm = _missing(LLM_CREDENTIALS)
        if missing_llm:
            pytest.skip(
                f"Skipping {example_file}: needs a live LLM ({', '.join(missing_llm)}), "
                "so it cannot run against a mock StackOne server"
            )

    example_path = Path(__file__).parent / example_file

    # Import and run the example module directly
    spec = importlib.util.spec_from_file_location("example", example_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules["example"] = module
        spec.loader.exec_module(module)
