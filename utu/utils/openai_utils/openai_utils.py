import logging
from typing import Any

from openai._streaming import AsyncStream
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
)
from openai.types.responses import (
    FunctionToolParam,
    Response,
    ResponseStreamEvent,
)
from pydantic import BaseModel

from ..print_utils import PrintUtils

logger = logging.getLogger(__name__)


class OpenAIUtils:
    # --------------------------------------------------------
    # chat completions
    # --------------------------------------------------------
    @staticmethod
    def print_message(message: ChatCompletionMessage) -> None:
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            PrintUtils.print_info(f"{message.reasoning_content}")
        if message.content:
            PrintUtils.print_bot(f"{message.content}", add_prefix=True)
        if message.tool_calls:
            for tool_call in message.tool_calls:
                PrintUtils.print_bot(f"<{tool_call.function.name}>{tool_call.function.arguments}", add_prefix=True)

    @staticmethod
    async def print_stream(stream: AsyncStream[ChatCompletionChunk]) -> ChatCompletionMessage:
        final_tool_calls: dict[int, ChatCompletionMessageToolCall] = {}
        content = ""
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                PrintUtils.print_info(f"{delta.reasoning_content}", end="", color="green")
            if delta.content:
                content += delta.content
                PrintUtils.print_info(f"{delta.content}", end="", color="gray")
            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    index = tool_call.index
                    if index not in final_tool_calls:
                        final_tool_calls[index] = tool_call
                        PrintUtils.print_info(
                            f"<{tool_call.function.name}>{tool_call.function.arguments}", end="", color="blue"
                        )
                    else:
                        if final_tool_calls[index].function.arguments:
                            final_tool_calls[index].function.arguments += tool_call.function.arguments
                        else:
                            final_tool_calls[index].function.arguments = tool_call.function.arguments
                        PrintUtils.print_info(f"{tool_call.function.arguments}", end="", color="blue")
        PrintUtils.print_info("")  # print a newline
        tool_calls = [
            ChatCompletionMessageFunctionToolCall(
                id=tool_call.id,
                function=tool_call.function.model_dump(),
                type=tool_call.type,  # type is always "function"
            )
            for tool_call in final_tool_calls.values()
        ]
        message = ChatCompletionMessage(role="assistant", content=content, tool_calls=tool_calls)
        OpenAIUtils.print_message(message)
        return message

    # --------------------------------------------------------
    # responses
    # --------------------------------------------------------
    @staticmethod
    def print_response(response: Response) -> None:
        for item in response.output:
            # print(f"> responses item: {item}")
            match item.type:
                case "reasoning":
                    content = getattr(item, "content", item.summary)
                    PrintUtils.print_bot(f"<reasoning>{content}</reasoning>", add_prefix=True, color="gray")
                case "message":
                    PrintUtils.print_bot(f"{item.content}", add_prefix=True)
                case "function_call":
                    PrintUtils.print_info(f"<{item.name}>({item.arguments})")
                case "file_search_call":
                    PrintUtils.print_info(f"<{item.type}>({item.queries})")
                case "web_search_call":
                    PrintUtils.print_info(f"<{item.type}>({item.action})")
                case "computer_call":
                    PrintUtils.print_info(f"<{item.type}>({item.action})")
                case "image_generation_call":
                    PrintUtils.print_info(f"<{item.type}> -> {item.result[:4]}")
                case "code_interpreter_call":
                    PrintUtils.print_info(
                        f"<{item.type}>(container_id={item.container_id}, code={item.code}) -> {item.outputs}"
                    )
                case "local_shell_call":
                    PrintUtils.print_info(f"<{item.type}>(action={item.action})")
                case "mcp_list_tools":
                    PrintUtils.print_info(f"<{item.type}>(server={item.server_label}) -> {item.tools}")
                case "mcp_call":
                    PrintUtils.print_info(
                        f"<{item.type}>(server={item.server_label}) {item.name}({item.arguments}) -> {item.output}"
                    )
                case "mcp_approval_request":
                    PrintUtils.print_info(f"<{item.type}>(server={item.server_label}) {item.name}({item.arguments})")
                case _:
                    PrintUtils.print_error(f"Unknown item type: {item.type}\n{item}")

    @staticmethod
    def print_response_stream(stream: AsyncStream[ResponseStreamEvent]) -> Response:
        raise NotImplementedError

    @staticmethod
    def get_response_configs(response: Response, include_output: bool = False) -> dict:
        """Get response configs from response"""
        data = response.model_dump()
        if not include_output:
            del data["output"]
        return data

    @staticmethod
    def get_response_output(response: Response) -> list[dict]:
        """Get response output from response"""
        return response.model_dump()["output"]

    @classmethod
    def tool_chatcompletion_to_responses(cls, tool: ChatCompletionToolParam) -> FunctionToolParam:
        assert tool["type"] == "function"
        return FunctionToolParam(
            name=tool["function"]["name"],
            description=tool["function"].get("description", ""),
            parameters=tool["function"].get("parameters", None),
            type="function",
        )

    @staticmethod
    def maybe_basemodel_to_dict(obj: Any) -> dict | None:
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        return obj
