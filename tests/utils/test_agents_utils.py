import logging
import os
from datetime import datetime

from agents import Agent, Runner, function_tool, trace
from openai import AsyncOpenAI

from utu.config import ConfigLoader
from utu.utils.agents_utils import AgentsUtils, SimplifiedOpenAIChatCompletionsModel


@function_tool
def get_weather(city: str, date: str) -> str:
    """Get the weather in a city.

    Args:
        city (str): The city to get the weather for.
        date (str): The date to get the weather for. In the format of "YYYY-MM-DD".
    """
    logging.info(f"> calling tool: get_weather({city}, {date})")
    return f"The weather in {city} at {date} is sunny."


config = ConfigLoader.load_model_config("base")
model = AgentsUtils.get_agents_model(**config.model_provider.model_dump())
sp = f"You are a helpful assistant. The date of today is {datetime.now().strftime('%Y-%m-%d (%A)')}."
agent = Agent(
    name="General Assistant",
    instructions=sp,
    model=model,
    tools=[get_weather],
)


async def test_print_stream_events():
    with trace(workflow_name="test_print_stream_events"):
        stream = Runner.run_streamed(agent, "tell me a joke. And what is the weather like in Shanghai?")
        await AgentsUtils.print_stream_events(stream.stream_events())


async def test_print_items():
    result = await Runner.run(agent, "tell me a joke. And what is the weather like in Shanghai?")
    AgentsUtils.print_new_items(result.new_items)


messages = [{"role": "user", "content": "给我讲两个笑话, 然后帮我查一下北京天津的天气"}]
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for a given location.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City and country e.g. Bogotá, Colombia"}},
                "required": ["location"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]


async def test_simplified_openai_chat_completions_model():
    model = os.getenv("UTU_LLM_MODEL")
    api_key = os.getenv("UTU_LLM_API_KEY")
    base_url = os.getenv("UTU_LLM_BASE_URL")
    openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    simplified_openai_model = SimplifiedOpenAIChatCompletionsModel(model=model, openai_client=openai_client)
    with trace(workflow_name="test_agent"):
        res = await simplified_openai_model.query_one(messages=messages, tools=tools, model=model)
        print(res)
