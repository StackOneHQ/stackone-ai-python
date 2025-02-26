import json
from pathlib import Path
from typing import Any

import pytest

from stackone_ai.specs.parser import OpenAPIParser


@pytest.fixture
def sample_openapi_spec() -> dict[str, Any]:
    return {
        "openapi": "3.0.0",
        "servers": [{"url": "https://api.test.com"}],
        "paths": {
            "/employees/{id}": {
                "get": {
                    "operationId": "get_employee",
                    "summary": "Get employee details",
                    "parameters": [
                        {
                            "name": "id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "x-api-version",
                            "in": "header",
                            "schema": {"type": "string"},
                            "description": "API Version",
                        },
                        {
                            "name": "include",
                            "in": "query",
                            "schema": {"type": "string"},
                            "description": "Fields to include",
                        },
                    ],
                }
            },
            "/employees": {
                "post": {
                    "operationId": "create_employee",
                    "summary": "Create new employee",
                    "parameters": [
                        {
                            "name": "x-idempotency-key",
                            "in": "header",
                            "schema": {"type": "string"},
                            "description": "Idempotency Key",
                        }
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "Employee name",
                                        },
                                        "email": {
                                            "type": "string",
                                            "description": "Employee email",
                                        },
                                    },
                                }
                            }
                        },
                    },
                }
            },
            "/employees/{id}/documents": {
                "post": {
                    "operationId": "upload_document",
                    "description": "Upload employee document",
                    "parameters": [
                        {
                            "name": "id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/x-www-form-urlencoded": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "document_type": {"type": "string"},
                                        "file": {"type": "string", "format": "binary"},
                                    },
                                }
                            }
                        },
                    },
                }
            },
        },
        "components": {
            "schemas": {
                "Employee": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                    },
                }
            }
        },
    }


@pytest.fixture
def parser(tmp_path: Path, sample_openapi_spec: dict[str, Any]) -> OpenAPIParser:
    """Parser for test OpenAPI spec"""
    spec_file = tmp_path / "test_spec.json"
    spec_file.write_text(json.dumps(sample_openapi_spec))
    return OpenAPIParser(spec_file)


def test_parser_initialization(parser: OpenAPIParser) -> None:
    """Test parser initialization and base URL handling"""
    assert parser.base_url == "https://api.test.com"


def test_resolve_schema_ref(parser: OpenAPIParser) -> None:
    """Test schema reference resolution"""
    ref = "#/components/schemas/Employee"
    resolved = parser._resolve_schema_ref(ref)

    assert resolved == {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "email": {"type": "string"},
        },
    }


def test_resolve_schema_ref_invalid(parser: OpenAPIParser) -> None:
    """Test invalid schema reference handling"""
    with pytest.raises(ValueError, match="Only local references are supported"):
        parser._resolve_schema_ref("https://external.com/schema.json#/definitions/Type")


def test_parse_request_body_json(parser: OpenAPIParser) -> None:
    """Test JSON request body parsing"""
    operation = parser.spec["paths"]["/employees"]["post"]
    schema, body_type = parser._parse_request_body(operation)

    assert body_type == "json"
    assert schema == {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Employee name"},
            "email": {"type": "string", "description": "Employee email"},
        },
    }


def test_parse_request_body_form(parser: OpenAPIParser) -> None:
    """Test form request body parsing"""
    operation = parser.spec["paths"]["/employees/{id}/documents"]["post"]
    schema, body_type = parser._parse_request_body(operation)

    assert body_type == "form"
    assert schema == {
        "type": "object",
        "properties": {
            "document_type": {"type": "string"},
            "file": {"type": "string", "format": "binary"},
        },
    }


def test_parse_request_body_none(parser: OpenAPIParser) -> None:
    """Test parsing operation without request body"""
    operation = parser.spec["paths"]["/employees/{id}"]["get"]
    schema, body_type = parser._parse_request_body(operation)

    assert schema is None
    assert body_type is None


def test_parse_tools(parser: OpenAPIParser) -> None:
    """Test parsing complete tools from OpenAPI spec"""
    tools = parser.parse_tools()

    # Check get_employee tool
    get_employee = tools["get_employee"]
    assert get_employee.description == "Get employee details"
    assert get_employee.execute.method == "GET"
    assert get_employee.execute.url == "https://api.test.com/employees/{id}"

    # Check parameter locations
    assert get_employee.execute.parameter_locations == {
        "id": "path",
        "x-api-version": "header",
        "include": "query",
    }

    # Check create_employee tool
    create_employee = tools["create_employee"]
    assert create_employee.execute.body_type == "json"
    assert "name" in create_employee.parameters.properties
    assert "email" in create_employee.parameters.properties

    # Check upload_document tool
    upload_document = tools["upload_document"]
    assert upload_document.execute.body_type == "form"
    assert "document_type" in upload_document.parameters.properties
    assert "file" in upload_document.parameters.properties


def test_parse_tools_empty_spec(tmp_path: Path) -> None:
    """Test parsing empty OpenAPI spec"""
    empty_spec = {"openapi": "3.0.0", "paths": {}}
    spec_file = tmp_path / "empty_spec.json"
    spec_file.write_text(json.dumps(empty_spec))

    parser = OpenAPIParser(spec_file)
    tools = parser.parse_tools()

    assert len(tools) == 0


def test_operation_id_required(parser: OpenAPIParser) -> None:
    """Test that operationId is required for all operations"""
    # Remove operationId from a path
    del parser.spec["paths"]["/employees/{id}"]["get"]["operationId"]

    # Attempt to parse tools should raise ValueError
    with pytest.raises(ValueError, match="Operation ID is required for tool parsing"):
        parser.parse_tools()

    # Verify error contains useful operation details
    try:
        parser.parse_tools()
    except ValueError as e:
        assert "Get employee details" in str(e), "Error should contain operation description"
        assert "parameters" in str(e), "Error should contain operation details"


@pytest.fixture
def nested_components_spec() -> dict[str, Any]:
    return {
        "openapi": "3.0.0",
        "servers": [{"url": "https://api.test.com"}],
        "paths": {
            "/employees": {
                "post": {
                    "operationId": "create_employee",
                    "summary": "Create new employee",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/CreateEmployeeRequest"}
                            }
                        },
                    },
                }
            }
        },
        "components": {
            "schemas": {
                "CreateEmployeeRequest": {
                    "type": "object",
                    "properties": {
                        "personal_info": {"$ref": "#/components/schemas/PersonalInfo"},
                        "employment": {"$ref": "#/components/schemas/Employment"},
                    },
                },
                "PersonalInfo": {
                    "type": "object",
                    "properties": {
                        "first_name": {"type": "string", "description": "First name of employee"},
                        "last_name": {"type": "string", "description": "Last name of employee"},
                        "contact": {"$ref": "#/components/schemas/Contact"},
                    },
                },
                "Contact": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string", "description": "Email address"},
                        "phone": {"type": "string", "description": "Phone number"},
                    },
                },
                "Employment": {
                    "type": "object",
                    "properties": {
                        "department": {"type": "string", "description": "Department name"},
                        "position": {"type": "string", "description": "Job position"},
                    },
                },
            }
        },
    }


@pytest.fixture
def nested_parser(tmp_path: Path, nested_components_spec: dict[str, Any]) -> OpenAPIParser:
    spec_file = tmp_path / "nested_spec.json"
    spec_file.write_text(json.dumps(nested_components_spec))
    return OpenAPIParser(spec_file)


def test_nested_schema_resolution(nested_parser: OpenAPIParser) -> None:
    """Test resolution of nested schema references"""
    tools = nested_parser.parse_tools()

    create_employee = tools["create_employee"]
    properties = create_employee.parameters.properties

    # Check top level properties are resolved
    assert "personal_info" in properties
    assert "employment" in properties

    # Check nested properties structure
    personal_info = properties["personal_info"]
    assert personal_info["type"] == "object"
    assert "first_name" in personal_info["properties"]
    assert "last_name" in personal_info["properties"]
    assert "contact" in personal_info["properties"]

    # Check deeply nested properties
    contact = personal_info["properties"]["contact"]
    assert contact["type"] == "object"
    assert "email" in contact["properties"]
    assert "phone" in contact["properties"]

    # Check employment properties
    employment = properties["employment"]
    assert employment["type"] == "object"
    assert "department" in employment["properties"]
    assert "position" in employment["properties"]

    # Verify property types and descriptions are preserved
    assert personal_info["properties"]["first_name"]["type"] == "string"
    assert "First name of employee" in personal_info["properties"]["first_name"]["description"]


def test_circular_reference_detection(nested_parser: OpenAPIParser) -> None:
    """Test detection of circular references in schemas"""
    # Add a circular reference
    nested_parser.spec["components"]["schemas"]["Contact"]["properties"]["employee"] = {
        "$ref": "#/components/schemas/CreateEmployeeRequest"
    }

    with pytest.raises(ValueError, match="Circular reference detected"):
        nested_parser.parse_tools()


@pytest.fixture
def oas_specs() -> list[tuple[str, dict[str, Any]]]:
    """Load all OpenAPI specs from the oas directory"""
    oas_dir = Path("stackone_ai/oas")
    specs = []

    for spec_file in oas_dir.glob("*.json"):
        if spec_file.name == ".gitignore":
            continue
        with open(spec_file) as f:
            specs.append((spec_file.stem, json.load(f)))

    return specs


def test_parse_all_oas_specs(tmp_path: Path, oas_specs: list[tuple[str, dict[str, Any]]], snapshot) -> None:
    """Test parsing all OpenAPI specs with separate snapshots for each spec"""

    for name, spec in oas_specs:
        # Create temporary file for each spec
        spec_file = tmp_path / f"{name}_spec.json"
        spec_file.write_text(json.dumps(spec))

        parser = OpenAPIParser(spec_file)
        tools = parser.parse_tools()

        # Basic validation of parsed tools
        assert tools, f"No tools parsed from {name} spec"
        for tool_name, tool in tools.items():
            assert tool.description, f"Tool {tool_name} in {name} spec has no description"
            assert tool.execute.method, f"Tool {tool_name} in {name} spec has no HTTP method"
            assert tool.execute.url, f"Tool {tool_name} in {name} spec has no URL"
            assert tool.parameters.properties, f"Tool {tool_name} in {name} spec has no parameters"

        # Convert tools to serializable format for snapshot
        serialized_tools = {
            tool_name: {
                "description": tool.description,
                "parameters": tool.parameters.model_dump(),
                "execute": tool.execute.model_dump(),
            }
            for tool_name, tool in tools.items()
        }

        # Create separate snapshot for each spec
        snapshot_json = json.dumps(serialized_tools, indent=2, sort_keys=True)
        snapshot.assert_match(snapshot_json, f"{name}_tools.json")


def test_resolve_schema_with_allof(tmp_path: Path) -> None:
    """Test resolving schema with allOf references"""

    # Create a minimal OpenAPI spec with nested component references
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "components": {
            "schemas": {
                "ApplicationStatusEnum": {"type": "string", "enum": ["pending", "accepted", "rejected"]},
                "BaseCandidate": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
                },
                "CreateCandidate": {
                    "allOf": [
                        {"$ref": "#/components/schemas/BaseCandidate"},
                        {
                            "type": "object",
                            "properties": {"phone": {"type": "string"}},
                        },
                    ],
                    "description": "Extended candidate model",
                },
            }
        },
        "paths": {
            "/test": {
                "post": {
                    "operationId": "test_operation",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "application_status": {
                                            "allOf": [{"$ref": "#/components/schemas/ApplicationStatusEnum"}],
                                            "nullable": True,
                                        },
                                        "candidate": {
                                            "allOf": [{"$ref": "#/components/schemas/CreateCandidate"}],
                                            "description": "Candidate Properties",
                                            "nullable": True,
                                        },
                                    },
                                }
                            }
                        }
                    },
                }
            }
        },
    }

    # Write spec to temporary file
    spec_file = tmp_path / "test_spec.json"
    spec_file.write_text(json.dumps(spec))

    # Parse the spec
    parser = OpenAPIParser(spec_file)
    tools = parser.parse_tools()

    # Get the resolved schema for our test operation
    tool = tools["test_operation"]

    # Verify application_status schema
    status_schema = tool.parameters.properties["application_status"]
    assert status_schema["type"] == "string"
    assert status_schema["enum"] == ["pending", "accepted", "rejected"]
    assert status_schema["nullable"] is True

    # Verify candidate schema with nested references
    candidate_schema = tool.parameters.properties["candidate"]
    assert candidate_schema["type"] == "object"
    assert "name" in candidate_schema["properties"]
    assert "email" in candidate_schema["properties"]
    assert "phone" in candidate_schema["properties"]  # From CreateCandidate extension
    assert candidate_schema["description"] == "Candidate Properties"
    assert candidate_schema["nullable"] is True


@pytest.fixture
def temp_spec_file(tmp_path: Path) -> Path:
    """Create a temporary OpenAPI spec file for testing."""

    def write_spec(spec: dict[str, Any]) -> Path:
        spec_file = tmp_path / "test_spec.json"
        with open(spec_file, "w") as f:
            json.dump(spec, f)
        return spec_file

    return write_spec


def test_parse_file_upload(temp_spec_file: Path) -> None:
    """Test parsing an OpenAPI spec with file upload endpoints."""
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "paths": {
            "/upload": {
                "post": {
                    "operationId": "uploadFile",
                    "summary": "Upload a file",
                    "requestBody": {
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "file": {
                                            "type": "string",
                                            "format": "binary",
                                            "description": "The file to upload",
                                        },
                                        "description": {"type": "string", "description": "File description"},
                                    },
                                    "required": ["file"],
                                }
                            }
                        }
                    },
                }
            }
        },
    }

    parser = OpenAPIParser(temp_spec_file(spec))
    tools = parser.parse_tools()

    assert "uploadFile" in tools
    tool = tools["uploadFile"]

    # Check file parameter is correctly marked
    assert "file" in tool.parameters.properties
    assert tool.parameters.properties["file"]["type"] == "file"
    assert tool.execute.parameter_locations["file"] == "file"

    # Check non-file parameter
    assert "description" in tool.parameters.properties
    assert tool.parameters.properties["description"]["type"] == "string"
    assert tool.execute.parameter_locations["description"] == "body"

    # Check body type
    assert tool.execute.body_type == "multipart"


def test_parse_multiple_files(temp_spec_file: Path) -> None:
    """Test parsing an endpoint that accepts multiple files."""
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "paths": {
            "/upload-multiple": {
                "post": {
                    "operationId": "uploadMultipleFiles",
                    "summary": "Upload multiple files",
                    "requestBody": {
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "files": {
                                            "type": "array",
                                            "items": {"type": "string", "format": "binary"},
                                            "description": "Multiple files to upload",
                                        },
                                        "metadata": {
                                            "type": "object",
                                            "properties": {"category": {"type": "string"}},
                                        },
                                    },
                                }
                            }
                        }
                    },
                }
            }
        },
    }

    parser = OpenAPIParser(temp_spec_file(spec))
    tools = parser.parse_tools()

    assert "uploadMultipleFiles" in tools
    tool = tools["uploadMultipleFiles"]

    # Check array of files
    assert "files" in tool.parameters.properties
    assert tool.parameters.properties["files"]["type"] == "array"
    assert tool.parameters.properties["files"]["items"]["type"] == "file"
    assert tool.execute.parameter_locations["files"] == "file"

    # Check nested object parameter
    assert "metadata" in tool.parameters.properties
    assert tool.parameters.properties["metadata"]["type"] == "object"
    assert tool.execute.parameter_locations["metadata"] == "body"


def test_mixed_parameter_types(temp_spec_file: Path) -> None:
    """Test parsing an endpoint with mixed parameter types (path, query, file)."""
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "paths": {
            "/users/{userId}/files": {
                "post": {
                    "operationId": "uploadUserFile",
                    "summary": "Upload a user file",
                    "parameters": [
                        {"name": "userId", "in": "path", "required": True, "schema": {"type": "string"}},
                        {"name": "overwrite", "in": "query", "schema": {"type": "boolean"}},
                    ],
                    "requestBody": {
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"file": {"type": "string", "format": "binary"}},
                                }
                            }
                        }
                    },
                }
            }
        },
    }

    parser = OpenAPIParser(temp_spec_file(spec))
    tools = parser.parse_tools()

    assert "uploadUserFile" in tools
    tool = tools["uploadUserFile"]

    # Check path parameter
    assert tool.execute.parameter_locations["userId"] == "path"
    assert tool.parameters.properties["userId"]["type"] == "string"

    # Check query parameter
    assert tool.execute.parameter_locations["overwrite"] == "query"
    assert tool.parameters.properties["overwrite"]["type"] == "boolean"

    # Check file parameter
    assert tool.execute.parameter_locations["file"] == "file"
    assert tool.parameters.properties["file"]["type"] == "file"

    # Check body type
    assert tool.execute.body_type == "multipart"


def test_form_data_without_files(temp_spec_file: Path) -> None:
    """Test parsing form data without file uploads."""
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "paths": {
            "/submit-form": {
                "post": {
                    "operationId": "submitForm",
                    "summary": "Submit a form",
                    "requestBody": {
                        "content": {
                            "application/x-www-form-urlencoded": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                                }
                            }
                        }
                    },
                }
            }
        },
    }

    parser = OpenAPIParser(temp_spec_file(spec))
    tools = parser.parse_tools()

    assert "submitForm" in tools
    tool = tools["submitForm"]

    # Check form parameters
    assert tool.execute.parameter_locations["name"] == "body"
    assert tool.execute.parameter_locations["age"] == "body"
    assert tool.parameters.properties["name"]["type"] == "string"
    assert tool.parameters.properties["age"]["type"] == "integer"

    # Check body type
    assert tool.execute.body_type == "form"


def test_parser_with_base_url_override(tmp_path: Path, sample_openapi_spec: dict[str, Any]) -> None:
    """Test that the parser uses the provided base_url instead of the one from the spec."""
    # Write the spec to a temporary file
    spec_file = tmp_path / "test_spec.json"
    with open(spec_file, "w") as f:
        json.dump(sample_openapi_spec, f)

    # Create parser with default base_url
    default_parser = OpenAPIParser(spec_file)
    assert default_parser.base_url == "https://api.test.com"

    # Create parser with development base_url
    dev_parser = OpenAPIParser(spec_file, base_url="https://api.example-dev.com")
    assert dev_parser.base_url == "https://api.example-dev.com"

    # Create parser with experimental base_url
    exp_parser = OpenAPIParser(spec_file, base_url="https://api.example-exp.com")
    assert exp_parser.base_url == "https://api.example-exp.com"

    # Verify the base_url is used in the tool definitions for development environment
    dev_tools = dev_parser.parse_tools()
    assert dev_tools["get_employee"].execute.url.startswith("https://api.example-dev.com")
    assert not dev_tools["get_employee"].execute.url.startswith("https://api.test.com")

    # Verify the base_url is used in the tool definitions for experimental environment
    exp_tools = exp_parser.parse_tools()
    assert exp_tools["get_employee"].execute.url.startswith("https://api.example-exp.com")
    assert not exp_tools["get_employee"].execute.url.startswith("https://api.test.com")
