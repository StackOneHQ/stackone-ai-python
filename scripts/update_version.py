#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "tomli",
# ]
# ///
"""
Script to ensure version consistency between pyproject.toml and __init__.py.
Run this script after release-please updates the version in pyproject.toml.
"""

import re
import subprocess
from pathlib import Path

import tomli


def main() -> None:
    """Update version in __init__.py to match pyproject.toml and refresh uv.lock."""
    # Read version from pyproject.toml
    pyproject_path = Path("pyproject.toml")
    init_path = Path("stackone_ai/__init__.py")

    with open(pyproject_path, "rb") as f:
        pyproject = tomli.load(f)
        version = pyproject["project"]["version"]

    # Update version in __init__.py
    init_content = init_path.read_text()
    new_init_content = re.sub(r'__version__ = "[^"]+"', f'__version__ = "{version}"', init_content)

    if init_content != new_init_content:
        init_path.write_text(new_init_content)
        print(f"Updated version in {init_path} to {version}")
    else:
        print(f"Version in {init_path} already matches {version}")

    # Update uv.lock to reflect version change in pyproject.toml
    print("Updating uv.lock...")
    subprocess.run(["uv", "lock"], check=True)
    print("uv.lock updated successfully")


if __name__ == "__main__":
    main()
