# ruff: noqa

import pytest

from utu.agents import WorkforceAgent
from utu.config import ConfigLoader

overall_task = "It's May 2023, and I'm about to drive across the U.S. from California to Maine. I always recycle my water bottles at the end of a trip, and I drink 5 12-ounce water bottles for every 100 miles I travel, rounded to the nearest 100. Assuming I follow I-40 from Los Angeles to Cincinnati, then take I-90 from Cincinnati to Augusta, how many dollars will I get back according to Wikipedia?"


@pytest.fixture
async def agent():
    agent = WorkforceAgent(config=ConfigLoader.load_agent_config("workforce"))
    return agent


async def test_run(agent: WorkforceAgent):
    recorder = await agent.run(overall_task)
    print(recorder)
