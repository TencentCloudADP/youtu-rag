import json

from agents._run_impl import TraceCtxManager
from agents.tracing import function_span, response_span

from utu.tracing import setup_db_tracing
from utu.utils import SimplifiedAsyncOpenAI

setup_db_tracing()

client = SimplifiedAsyncOpenAI()
input = [{"role": "user", "content": "给我讲两个笑话, 然后帮我查一下北京天津的天气"}]
tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string", "description": "City and country e.g. Bogotá, Colombia"}},
            "required": ["location"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]


async def mock_function():
    with function_span(
        name="mock_function",
        input=json.dumps({"input": "mock_function_input"}),  # str
    ) as span:
        span.span_data.output = {"output": "mock_function_output"}


async def test_function_span():
    with TraceCtxManager(
        workflow_name="test_db_tracer",
        trace_id="trace_db_tracer",
        group_id="trace_db_tracer",
        metadata={"test": "test"},
        disabled=False,
    ):
        await mock_function()


async def mock_response():
    with response_span() as span:
        response = await client.responses_create(input=input, tools=tools)
        span.span_data.input = input
        span.span_data.response = response
        print(response)


async def test_response_span():
    with TraceCtxManager(
        workflow_name="test_db_tracer",
        trace_id="trace_db_tracer",
        group_id="trace_db_tracer",
        metadata={"test": "test"},
        disabled=False,
    ):
        await mock_response()
