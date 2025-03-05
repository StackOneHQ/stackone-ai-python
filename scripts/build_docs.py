#!/usr/bin/env python
# /// script
# requires-python = ">=3.11"
# ///
import re
from pathlib import Path

DOCS_DIR = Path(".docs")
EXAMPLES_DIR = Path("examples")


def convert_file_to_markdown(py_file: Path) -> str:
    """Convert a Python file to markdown, preserving structure."""
    content = py_file.read_text()

    # Add title from filename
    if py_file.stem == "index":
        title = "StackOne AI SDK"
    else:
        title = py_file.stem.replace("_", " ").title()
    output = [f"# {title}\n"]

    # Find all docstrings and their positions
    # Match docstrings that start and end on their own lines
    docstring_pattern = r'(\n|\A)\s*"""(.*?)"""(\s*\n|\Z)'
    current_pos = 0

    for match in re.finditer(docstring_pattern, content, re.DOTALL):
        start, end = match.span()
        docstring_content = match.group(2).strip()  # The actual docstring content is in group 2

        # If there's code before this docstring, wrap it
        if current_pos < start:
            code = content[current_pos:start].strip()
            if code:
                output.append("\n```python")
                output.append(code)
                output.append("```\n")

        # Add the docstring content as markdown
        output.append(docstring_content)
        current_pos = end

    # Add any remaining code
    if current_pos < len(content):
        remaining_code = content[current_pos:].strip()
        if remaining_code:
            output.append("\n```python")
            output.append(remaining_code)
            output.append("```\n")

    return "\n".join(output)


def create_markdown_file(py_file: Path) -> None:
    """Convert a Python file to markdown documentation."""
    markdown_content = convert_file_to_markdown(py_file)

    # Output to .docs directory
    output_path = DOCS_DIR / f"{py_file.stem.replace('_', '-')}.md"

    # Create directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the file
    output_path.write_text(markdown_content)
    print(f"Created {output_path}")


def main() -> None:
    """Main function to build documentation."""
    DOCS_DIR.mkdir(exist_ok=True)

    # Process all Python files in examples directory
    for py_file in EXAMPLES_DIR.glob("*.py"):
        if py_file.stem.startswith("test_"):
            continue
        create_markdown_file(py_file)


if __name__ == "__main__":
    main()
