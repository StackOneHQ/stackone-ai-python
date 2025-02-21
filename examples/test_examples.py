import subprocess
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


def test_example_files_exist():
    """Verify that we found example files to test"""
    assert len(EXAMPLES) > 0, "No example files found"
    print(f"Found {len(EXAMPLES)} examples")


@pytest.mark.parametrize("example_file", EXAMPLES)
def test_run_example(example_file):
    """Run each example file directly using python"""
    example_path = Path(__file__).parent / example_file
    result = subprocess.run([sys.executable, str(example_path)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        pytest.fail(f"Example {example_file} failed with return code {result.returncode}")
