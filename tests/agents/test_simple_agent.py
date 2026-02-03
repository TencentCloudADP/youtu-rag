import json

import pytest

from utu.agents import SimpleAgent
from utu.config import ConfigLoader


@pytest.fixture
async def agent():
    agent = SimpleAgent(config=ConfigLoader.load_agent_config("simple/base_search"))
    await agent.build()
    yield agent
    await agent.cleanup()


async def test_chat_streamed(agent: SimpleAgent):
    run_result_streaming = await agent.chat_streamed("That's the weather in Beijing today?")
    print(run_result_streaming)


async def test_chat(agent: SimpleAgent):
    run_result = await agent.chat("That's the weather in Beijing today?")
    print(json.dumps(run_result.to_input_list(), ensure_ascii=False, indent=2))


async def test_run(agent: SimpleAgent):
    res = await agent.run("Hello! I'm Eason", save=True)
    print(f"res.final_output: {res.final_output}")
    res = await agent.run("Do you know my name? BTW, my last name is Shi.")
    print(f"res.final_output: {res.final_output}")
    res = await agent.run("Please tell me my full name.")
    print(f"res.final_output: {res.final_output}")


async def test_init_toolkit():
    """test init SimpleAgent with toolkits config"""
    agent = SimpleAgent(toolkits=["user_interaction"])
    async with agent:
        tools = await agent.get_tools()
        print(f"Loaded {len(tools)} tools: {tools}")
        await agent.chat_streamed("That's the weather in Beijing today?")
