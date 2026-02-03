import asyncio
import threading


# 限制并发数量的装饰器
def limit_concurrency(max_concurrent: int):
    semaphore = asyncio.Semaphore(max_concurrent)

    def decorator(func):
        async def wrapper(*args, **kwargs):
            async with semaphore:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


# 限制线程级并发数量的装饰器
def limit_concurrency_thread(max_concurrent: int):
    semaphore = threading.Semaphore(max_concurrent)

    def decorator(func):
        def wrapper(*args, **kwargs):
            with semaphore:
                return func(*args, **kwargs)

        return wrapper

    return decorator


# 将线程中的异步函数转换为同步函数的转换器
def async_to_sync(func):
    def wrapper(index, *args, **kwargs):
        result = asyncio.run(func(*args, **kwargs))
        return index, result

    return wrapper


# def get_trajectory_from_agent_result(agent_result: RunResult) -> list[dict]:
#     messages = ChatCompletionConverter.items_to_messages(agent_result.to_input_list())
#     trajectory = [dict(message) for message in messages]
#     model_responses = agent_result.raw_responses
#     usage_list = []
#     for model_response in model_responses:
#         usage_list.append(asdict(model_response.usage))
#     # 添加模型响应的使用情况
#     usage_idx = 0
#     for i, message in enumerate(trajectory):
#         if message.get("role") == "assistant":
#             if usage_idx < len(usage_list):
#                 curr_usage = usage_list[usage_idx]
#                 message["usage"] = {"input_tokens": curr_usage["input_tokens"],
#                                     "output_tokens": curr_usage["output_tokens"],
#                                     "total_tokens": curr_usage["total_tokens"],
#                                     "requests": curr_usage["requests"]}
#                 usage_idx += 1
#             else:
#                 message["usage"] = {}
#     return trajectory
