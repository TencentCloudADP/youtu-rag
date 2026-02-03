import logging

import pytest

from utu.agents import SimpleAgent
from utu.config import ConfigLoader


@pytest.fixture
async def agent():
    agent = SimpleAgent(config=ConfigLoader.load_agent_config("simple/search_agent"))
    await agent.build()
    yield agent
    await agent.cleanup()


async def test_chat_streamed(agent: SimpleAgent):
    tools = await agent.get_tools()
    logging.info(f"Loaded {len(tools)} tools: {tools}")
    await agent.chat_streamed("That's the weather in Beijing today?")
