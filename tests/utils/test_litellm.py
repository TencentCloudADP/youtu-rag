from agents import ModelSettings, ModelTracing
from agents.extensions.models.litellm_model import LitellmModel

from utu.utils import AgentsUtils, EnvUtils

params = {
    "system_instructions": None,
    "input": "hello",
    "model_settings": ModelSettings(),
    "tools": [],
    "output_schema": None,
    "handoffs": [],
    "tracing": ModelTracing.DISABLED,
    "previous_response_id": None,
    # conversation_id=None,
    "prompt": None,
}


async def test_litellm():
    model = LitellmModel(
        "azure/gpt-5-nano",
        # these two envs can be omitted if using the default name
        # https://docs.litellm.ai/docs/providers/azure/
        base_url=EnvUtils.get_env("AZURE_API_BASE"),
        api_key=EnvUtils.get_env("AZURE_API_KEY"),
    )
    async for event in model.stream_response(**params):
        print(event)


async def test_get_agents_model():
    model = AgentsUtils.get_agents_model(type="litellm", model="azure/gpt-5-nano")
    async for event in model.stream_response(**params):
        print(event)
