import argparse
import asyncio
import logging
import os
import sys
from typing import Any, TypeVar

import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError
from mcp.types import EmbeddedResource, ErrorData, ImageContent, TextContent, Tool
from pydantic import ValidationError

from stackone_ai import StackOneToolSet
from stackone_ai.models import StackOneAPIError, StackOneError

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("stackone.mcp")

app: Server = Server("stackone-ai")
toolset: StackOneToolSet | None = None

NO_ACCOUNT_ID_PREFIXES = [
    "stackone_",
]

# Type variables for function annotations
T = TypeVar("T")
R = TypeVar("R")


def tool_needs_account_id(tool_name: str) -> bool:
    for prefix in NO_ACCOUNT_ID_PREFIXES:
        if tool_name.startswith(prefix):
            return False

    # By default, assume all other tools need account_id
    return True


@app.list_tools()  # type: ignore[misc]
async def list_tools() -> list[Tool]:
    """List all available StackOne tools as MCP tools."""
    if not toolset:
        logger.error("Toolset not initialized")
        raise McpError(
            ErrorData(
                code=types.INTERNAL_ERROR,
                message="Toolset not initialized, please check your STACKONE_API_KEY.",
            )
        )

    try:
        mcp_tools: list[Tool] = []
        tools = toolset.get_tools()
        # Convert to a list if it's not already iterable in the expected way
        tool_list = list(tools.tools) if hasattr(tools, "tools") else []

        for tool in tool_list:
            # Convert StackOne tool parameters to MCP schema
            properties = {}
            required = []

            # Add account_id parameter only for tools that need it
            if tool_needs_account_id(tool.name):
                properties["account_id"] = {
                    "type": "string",
                    "description": "The StackOne account ID to use for this tool call",
                }

            for name, details in tool.parameters.properties.items():
                if isinstance(details, dict):
                    prop = {
                        "type": details.get("type", "string"),
                        "description": details.get("description", ""),
                    }
                    if not details.get("nullable", False):
                        required.append(name)
                    properties[name] = prop

            schema = {"type": "object", "properties": properties}
            if required:
                schema["required"] = required

            mcp_tools.append(Tool(name=tool.name, description=tool.description, inputSchema=schema))

        logger.info(f"Listed {len(mcp_tools)} tools")
        return mcp_tools
    except Exception as e:
        logger.error(f"Error listing tools: {str(e)}", exc_info=True)
        raise McpError(
            ErrorData(
                code=types.INTERNAL_ERROR,
                message=f"Error listing tools: {str(e)}",
            )
        ) from e


@app.call_tool()  # type: ignore[misc]
async def call_tool(
    name: str, arguments: dict[str, Any]
) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Execute a StackOne tool and return its result."""
    if not toolset:
        logger.error("Toolset not initialized")
        raise McpError(
            ErrorData(
                code=types.INTERNAL_ERROR,
                message="Server configuration error: Toolset not initialized",
            )
        )

    try:
        tool = toolset.get_tool(name)
        if not tool:
            logger.warning(f"Tool not found: {name}")
            raise McpError(
                ErrorData(
                    code=types.INVALID_PARAMS,
                    message=f"Tool not found: {name}",
                )
            )

        if "account_id" in arguments:
            tool.set_account_id(arguments.pop("account_id"))

        if tool_needs_account_id(name) and tool.get_account_id() is None:
            logger.warning(f"Tool {name} needs account_id but none provided")
            raise McpError(
                ErrorData(
                    code=types.INVALID_PARAMS,
                    message=f"Tool {name} needs account_id but none provided",
                )
            )

        result = tool.execute(arguments)
        return [TextContent(type="text", text=str(result))]

    except ValidationError as e:
        logger.warning(f"Invalid parameters for tool {name}: {str(e)}")
        raise McpError(
            ErrorData(
                code=types.INVALID_PARAMS,
                message=f"Invalid parameters for tool {name}: {str(e)}",
            )
        ) from e
    except StackOneAPIError as e:
        logger.error(f"API error: {str(e)}")
        raise McpError(
            ErrorData(
                code=types.INTERNAL_ERROR,
                message=f"API error: {str(e)}",
            )
        ) from e
    except StackOneError as e:
        logger.error(f"Error: {str(e)}")
        raise McpError(
            ErrorData(
                code=types.INTERNAL_ERROR,
                message=f"Error: {str(e)}",
            )
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise McpError(
            ErrorData(
                code=types.INTERNAL_ERROR,
                message="An unexpected error occurred. Please try again later.",
            )
        ) from e


async def main(api_key: str | None = None) -> None:
    """Run the MCP server."""

    if not api_key:
        api_key = os.getenv("STACKONE_API_KEY")
        if not api_key:
            raise ValueError("STACKONE_API_KEY not found in environment variables")

    global toolset
    toolset = StackOneToolSet(api_key=api_key)
    logger.info("StackOne toolset initialized successfully")

    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="stackone-ai",
                server_version="0.1.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def cli() -> None:
    """CLI entry point for the MCP server."""
    parser = argparse.ArgumentParser(description="StackOne AI MCP Server")
    parser.add_argument("--api-key", help="StackOne API key (can also be set via STACKONE_API_KEY env var)")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    args = parser.parse_args()

    logger.setLevel(args.log_level)

    try:
        asyncio.run(main(args.api_key))
    except Exception as e:
        logger.critical(f"Failed to start server: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
