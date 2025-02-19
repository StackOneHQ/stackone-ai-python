from stackone_ai.models import BaseTool, Tools
from stackone_ai.specs.loader import load_specs


class StackOneToolSet:
    def __init__(self, api_key: str):
        self._api_key = api_key
        self._tools = load_specs()

    def get_tools(self, vertical: str, account_id: str | None = None) -> Tools:
        """
        Get tools filtered by vertical and optionally bound to an account_id

        Args:
            vertical: The vertical to filter tools by (e.g. "crm")
            account_id: Optional account_id to bind the tools to

        Returns:
            Tools instance containing the filtered tools
        """
        if vertical not in self._tools:
            return Tools(tools=[])

        tools = [
            BaseTool(
                description=tool_def.description,
                parameters=tool_def.parameters,
                _execute_config=tool_def.execute,
                _api_key=self._api_key,
                _account_id=account_id,
            )
            for name, tool_def in self._tools[vertical].items()
        ]
        return Tools(tools=tools)
