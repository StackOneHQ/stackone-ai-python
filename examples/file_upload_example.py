"""
Example demonstrating file upload functionality with StackOne.
Shows how to upload an employee document using the HRIS integration.

This example is runnable with the following command:
```bash
uv run examples/file_upload_example.py
```
"""

import base64
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from stackone_ai import StackOneToolSet

load_dotenv()

account_id = "45072196112816593343"
employee_id = "c28xIQaWQ6MzM5MzczMDA2NzMzMzkwNzIwNA"


def upload_employee_document() -> None:
    """Demonstrate uploading an employee document using StackOne."""
    with tempfile.TemporaryDirectory() as temp_dir:
        resume_content = """
        JOHN DOE
        Software Engineer

        EXPERIENCE
        Senior Developer - Tech Corp
        2020-Present
        - Led development of core features
        - Managed team of 5 engineers

        EDUCATION
        BS Computer Science
        University of Technology
        2016-2020
        """

        resume_file = Path(temp_dir) / "resume.pdf"
        resume_file.write_text(resume_content)

        # Initialize StackOne
        toolset = StackOneToolSet()
        tools = toolset.get_tools(vertical="hris", account_id=account_id)

        # Get the upload document tool
        upload_tool = tools.get_tool("hris_upload_employee_document")
        if not upload_tool:
            print("Upload tool not available")
            return

        with open(resume_file, "rb") as f:
            file_content = base64.b64encode(f.read()).decode()

        upload_params = {
            "x-account-id": account_id,
            "id": employee_id,
            "name": "resume",
            "content": file_content,
            "category": {"value": "shared"},
            "file_format": {"value": "txt"},
        }

        try:
            upload_tool.execute(upload_params)
        except Exception as e:
            print(f"Error uploading document: {e}")


if __name__ == "__main__":
    upload_employee_document()
