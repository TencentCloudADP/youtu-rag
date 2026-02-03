from collections.abc import AsyncIterator

from agents import (
    AgentOutputSchema,
    AgentOutputSchemaBase,
    Handoff,
    ModelResponse,
    ModelSettings,
    ModelTracing,
    OpenAIChatCompletionsModel,
    Tool,
    TResponseInputItem,
)
from agents.items import TResponseStreamEvent
from openai import AsyncOpenAI
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    # Response,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
)
from openai.types.responses.response_prompt_param import ResponsePromptParam

from ..utils import get_logger
from .react_converter import ConverterPreprocessInput, ReactConverter

logger = get_logger(__name__)


converter = ReactConverter()


class ReactModel(OpenAIChatCompletionsModel):
    # def __init__(
    #     self,
    #     model: str | ChatModel,
    #     openai_client: AsyncOpenAI,
    #     context: AgentContext,
    #     preprocessors: list[Preprocessor] = [],
    # ) -> None:
    #     super().__init__(
    #         model=model,
    #         openai_client=openai_client
    #     )
    #     self._context = context
    #     self._preprocessors = preprocessors

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        previous_response_id: str | None,
        prompt: ResponsePromptParam | None = None,
    ) -> ModelResponse:
        preprocess_input = ConverterPreprocessInput(
            system_instructions, input, tools, output_schema, handoffs, model_settings
        )
        preprocess_input = converter.preprocess(preprocess_input)
        model_response = await super().get_response(
            tracing=tracing,
            previous_response_id=previous_response_id,
            prompt=prompt,
            system_instructions=preprocess_input.system_instructions,
            input=preprocess_input.input,
            tools=preprocess_input.tools,
            output_schema=preprocess_input.output_schema,
            handoffs=preprocess_input.handoffs,
            model_settings=preprocess_input.model_settings,
        )
        model_response.output = converter.postprocess(model_response.output)
        return model_response

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchema | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        previous_response_id: str | None,
        prompt: ResponsePromptParam | None = None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        """
        yield:
            1. All events will be wrapped in RawResponsesStreamEvent(data=event) and sent to `_event_queue`
            2. ResponseCompletedEvent: used in AgentRunner._run_single_turn_streamed()
        Here, we only yield the final event
        """
        preprocess_input = ConverterPreprocessInput(
            system_instructions, input, tools, output_schema, handoffs, model_settings
        )
        preprocess_input = converter.preprocess(preprocess_input)
        content = ""
        final_event: ResponseCompletedEvent | None = None
        async for event in super().stream_response(
            tracing=tracing,
            previous_response_id=previous_response_id,
            prompt=prompt,
            system_instructions=preprocess_input.system_instructions,
            input=preprocess_input.input,
            tools=preprocess_input.tools,
            output_schema=preprocess_input.output_schema,
            handoffs=preprocess_input.handoffs,
            model_settings=preprocess_input.model_settings,
        ):
            if isinstance(event, ResponseOutputItemDoneEvent):
                item = event.item
                if isinstance(item, ResponseOutputMessage):
                    if content:
                        logger.warning("ReAct mode should not have multiple messages!")
                    content = item.content
                elif isinstance(item, ResponseFunctionToolCall):
                    logger.error("ReAct mode should not have tool call!")
                else:
                    logger.warning(f">> Unknown item type: {item.__class__.__name__}")
            elif isinstance(event, ResponseCompletedEvent):
                final_event = event  # TODO: cross-check with final output?
            else:
                pass
        assert content, "Model does not return any content!"
        assert final_event, "Model does not return any final event!"
        final_event.response.output = converter.postprocess(final_event.response.output)
        yield final_event


def get_react_model(model: str, api_key: str, base_url: str) -> ReactModel:
    openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return ReactModel(model=model, openai_client=openai_client)
