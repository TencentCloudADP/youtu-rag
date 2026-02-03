import re
from collections.abc import Callable

import mcp.types as types
from agents import Agent, FunctionTool, RunContextWrapper
from agents.function_schema import FuncSchema, function_schema
from agents.mcp import MCPServerSse, MCPServerStdio, MCPServerStreamableHttp, ToolFilterStatic
from mcp import Tool as MCPTool

from ..config import ToolkitConfig

# ------------------------------------------------------------------------------
# MCP
MCP_SERVER_MAP = {
    "sse": MCPServerSse,
    "stdio": MCPServerStdio,
    "streamable_http": MCPServerStreamableHttp,
}


def get_mcp_server(config: ToolkitConfig) -> MCPServerSse | MCPServerStdio | MCPServerStreamableHttp:
    """Get mcp server from config, with tool_filter if activated_tools is set.
    NOTE: you should manage the lifecycle of the returned server (.connect & .cleanup), e.g. using `async with`."""
    assert config.mode == "mcp", f"config mode must be 'mcp', got {config.mode}"
    assert config.mcp_transport in MCP_SERVER_MAP, f"Unsupported mcp transport: {config.mcp_transport}"
    tool_filter = ToolFilterStatic(allowed_tool_names=config.activated_tools) if config.activated_tools else None
    return MCP_SERVER_MAP[config.mcp_transport](
        params=config.config,
        name=config.name,
        client_session_timeout_seconds=config.mcp_client_session_timeout_seconds,
        tool_filter=tool_filter,
    )


async def get_mcp_tools(config: ToolkitConfig) -> list[MCPTool]:
    async with get_mcp_server(config) as mcp_server:
        # It is required to pass agent and run_context when using `tool_filter`, we pass a dummy agent here
        tools = await mcp_server.list_tools(agent=Agent(name="dummy"), run_context=RunContextWrapper(context=None))
        return tools


async def get_mcp_tools_schema(config: ToolkitConfig) -> dict[str, FuncSchema]:
    tools = await get_mcp_tools(config)
    tools_map = {}
    for tool in tools:
        tools_map[tool.name] = FuncSchema(
            name=tool.name,
            description=tool.description,
            params_pydantic_model=None,
            params_json_schema=tool.inputSchema,
            signature=None,
        )
    return tools_map


class MCPConverter:
    @classmethod
    def function_tool_to_mcp(cls, tool: FunctionTool) -> types.Tool:
        return types.Tool(
            name=tool.name,
            description=tool.description,
            inputSchema=tool.params_json_schema,
        )


# ------------------------------------------------------------------------------
# AsyncBaseToolkit utils
def register_tool(name: str = None):
    """Decorator to register a method as a tool.

    Usage:
        @register_tool  # uses method name
        @register_tool()  # uses method name
        @register_tool("custom_name")  # uses custom name

    Args:
        name (str, optional): The name of the tool. (Also support passing the function)
    """

    def decorator(func: Callable):
        if isinstance(name, str):
            tool_name = name
        else:
            tool_name = func.__name__
        func._is_tool = True
        func._tool_name = tool_name
        return func

    if callable(name):
        return decorator(name)
    return decorator


def get_tools_map(cls: type) -> dict[str, Callable]:
    """Get tools map from a class, without instance the class."""
    tools_map = {}
    # Iterate through all methods of the class and register @tool
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if callable(attr) and getattr(attr, "_is_tool", False):
            tools_map[attr._tool_name] = attr
    return tools_map


def get_tools_schema(cls: type) -> dict[str, FuncSchema]:
    """Get tools schema from a class, without instance the class."""
    tools_map = {}
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if callable(attr) and getattr(attr, "_is_tool", False):
            tools_map[attr._tool_name] = function_schema(attr)
    return tools_map


# ------------------------------------------------------------------------------
# misc
class ContentFilter:
    def __init__(self, banned_sites: list[str] = None):
        if banned_sites:
            self.RE_MATCHED_SITES = re.compile(r"^(" + "|".join(banned_sites) + r")")
        else:
            self.RE_MATCHED_SITES = None

    def filter_results(self, results: list[dict], limit: int, key: str = "link") -> list[dict]:
        # can also use search operator `-site:huggingface.co`
        # ret: {title, link, snippet, position, | sitelinks}
        res = []
        for result in results:
            if self.RE_MATCHED_SITES is None or not self.RE_MATCHED_SITES.match(result[key]):
                res.append(result)
            if len(res) >= limit:
                break
        return res
