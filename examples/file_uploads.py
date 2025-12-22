"""
Example demonstrating file upload functionality using StackOne.
Shows how to upload an employee document using a BambooHR integration.

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

"""
# Resume content

This is a sample resume content that will be uploaded using the `bamboohr_upload_employee_document` tool.
"""

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
        2016-2020 """


"""
# Upload employee document

This function uploads a resume using the `bamboohr_upload_employee_document` tool.

"""


def upload_employee_document() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        resume_file = Path(temp_dir) / "resume.pdf"
        resume_file.write_text(resume_content)

        toolset = StackOneToolSet()
        tools = toolset.fetch_tools(actions=["bamboohr_*"], account_ids=[account_id])

        upload_tool = tools.get_tool("bamboohr_upload_employee_document")
        assert upload_tool is not None

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

        result = upload_tool.execute(upload_params)
        assert result is not None
        assert result.get("message") == "File uploaded successfully"


if __name__ == "__main__":
    upload_employee_document()
