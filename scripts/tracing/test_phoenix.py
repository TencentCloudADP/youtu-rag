"""Test the performance of Phoenix
- test_len:
- concurrency:
"""

import asyncio
import random

import tqdm
from agents._run_impl import TraceCtxManager
from agents.tracing import agent_span, function_span, generation_span

from utu.tracing import setup_otel_tracing

setup_otel_tracing(project_name="test_phoenix")

TEST_LEN = 100_000


async def mock_function(output_len: int):
    with function_span(
        name="mock_function",
        input="mock_function_input",
    ) as span:
        output = "mock_function_output " * output_len
        span.span_data.output = output


async def mock_llm(input_len: int):
    mocked_input = [{"role": "user", "content": "mock_llm_input " * input_len}]
    with generation_span(
        input=mocked_input,
        model="mock_llm_model",
        model_config={"mock_llm_model_config": "mock_llm_model_config"},
    ) as span:
        await asyncio.sleep(random.random() * 2)
        output = [{"role": "assistant", "content": "mock_llm_output"}]
        span.span_data.output = output
        # span.span_data.usage = {"mock_llm_usage": "mock_llm_usage"}


async def exp(idx: int = 0):
    with TraceCtxManager(
        workflow_name="test_phoenix",
        trace_id=f"trace_phoenix_{idx}",
        group_id="trace_phoenix",
        metadata={"idx": str(idx)},
        disabled=False,
    ):
        with agent_span("agent"):
            # span_agent.start(mark_as_current=True)  # mark start
            await mock_llm(input_len=TEST_LEN)
            await mock_function(output_len=TEST_LEN)
            # span_agent.finish(reset_current=True)
        return f"Finished {idx}"


async def test_concurrency(concurrency: int = 10, total: int = 100):
    semaphore = asyncio.Semaphore(concurrency)

    async def exp_with_semaphore(idx: int):
        async with semaphore:
            return await exp(idx)

    tasks = [exp_with_semaphore(i) for i in range(total)]
    for task in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Testing concurrency"):
        _ = await task
        # print(res)


if __name__ == "__main__":
    asyncio.run(test_concurrency(concurrency=50, total=100))
