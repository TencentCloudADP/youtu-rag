import logging
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import timedelta
from typing import Literal

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)


class MCPClient:
    """MCP client with async context manager support for automatic start/stop.
    ref: [doc](https://modelcontextprotocol.io/docs/concepts/transports#python)

    - RuntimeError: Attempted to exit cancel scope in a different task than it was entered in with AsyncExitStack
    [github](https://github.com/modelcontextprotocol/python-sdk/issues/79)

    Usage:
        async with MCPClient.get_mcp_client(mcp_url) as session:
            result = await session.call_tool("tool_name", {"arg": "value"})
    """

    # session: ClientSession | None = None
    # session_id: str | None = None
    # _exit_stack: AsyncExitStack | None = None

    @asynccontextmanager
    async def start_http_session(self, url: str) -> AsyncGenerator[ClientSession, None]:
        async with AsyncExitStack() as stack:
            logger.info("Starting MCP with HTTP...")
            read_stream, write_stream, get_session_id = await stack.enter_async_context(
                streamablehttp_client(
                    url=url,
                    timeout=timedelta(seconds=60 * 10),
                    sse_read_timeout=timedelta(seconds=60 * 10),
                    terminate_on_close=True,
                )
            )
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            logger.info(f"HTTP session started, session_id: {get_session_id()}")
            yield session

    @asynccontextmanager
    async def start_sse_session(self, url: str) -> AsyncGenerator[ClientSession, None]:
        async with AsyncExitStack() as stack:
            logger.info("Starting MCP with SSE...")
            read_stream, write_stream = await stack.enter_async_context(
                sse_client(
                    url=url,
                    headers={},
                    timeout=timedelta(seconds=60 * 10),  # Increased timeout
                    sse_read_timeout=timedelta(seconds=60 * 10),  # Increased read timeout
                )
            )
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            logger.info("SSE session started")
            yield session

    @classmethod
    @asynccontextmanager
    async def get_mcp_client(cls, url: str) -> AsyncGenerator[ClientSession, None]:
        match cls.get_url_type(url):
            case "http":
                async with cls().start_http_session(url) as session:
                    yield session
            case "sse":
                async with cls().start_sse_session(url) as session:
                    yield session
            case _:
                raise ValueError(f"Unknown url type: {url}")

    @classmethod
    def get_url_type(cls, url: str) -> Literal["http", "sse"]:
        if url.strip("/").endswith("sse"):
            return "sse"
        elif url.strip("/").endswith("mcp"):
            return "http"
        else:
            raise ValueError(f"Unknown url: {url}")

    # async def __aexit__(self, exc_type, exc_val, exc_tb):
    #     try:
    #         await self.cleanup()
    #     except BaseException as e:
    #         logger.error(f"Error in __aexit__: {e}")

    # async def cleanup(self) -> None:
    #     if self._exit_stack:
    #         try:
    #             await self._exit_stack.aclose()
    #         except BaseException as e:
    #             logger.error(f"Error closing exit stack: {e}")
    #         self._exit_stack = None
    #         self.session = None
    #         logger.info("MCPClient stopped")

    # async def list_tools(self) -> types.ListToolsResult:
    #     return await self.session.list_tools()
