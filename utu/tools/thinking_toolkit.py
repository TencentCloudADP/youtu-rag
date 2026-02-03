"""
https://github.com/sierra-research/tau-bench/
https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking
"""

from .base import AsyncBaseToolkit


class ThinkingToolkit(AsyncBaseToolkit):
    async def think(self, thought: str) -> str:
        """Use the tool to think about something.
        It will not obtain new information or change the database, but just append the thought to the log.
        Use it when complex reasoning or some cache memory is needed.

        Args:
            thought (str): Your thoughts.
        """
        raise NotImplementedError
