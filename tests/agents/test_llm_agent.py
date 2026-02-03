import json

import pytest

from utu.agents import LLMAgent
from utu.config import ConfigLoader
from utu.utils import AgentsUtils


@pytest.fixture
async def agent():
    agent = LLMAgent(model_config=ConfigLoader.load_model_config("base"))
    return agent


async def test_run_streamed(agent: LLMAgent):
    run_result_streaming = agent.run_streamed("Who are you?")
    await AgentsUtils.print_stream_events(run_result_streaming.stream_events())


async def test_run(agent: LLMAgent):
    run_result = await agent.run("Who are you?")
    print(json.dumps(run_result.trajectories, ensure_ascii=False, indent=2))
