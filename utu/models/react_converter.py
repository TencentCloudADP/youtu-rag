# pylint: disable=line-too-long
# ruff: noqa: E501

import ast
import json
from copy import deepcopy
from dataclasses import dataclass, field

import jinja2
from agents import AgentOutputSchema, Handoff, ModelSettings, Tool, TResponseInputItem

# FIXME: change to ChatCompletionConverter
from agents.models.chatcmpl_converter import Converter
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseOutputItem,
    ResponseOutputMessage,
)

# TODO: handle handoffs: list[Handoff]
TEMPLATE_SP = r"""
You are an expert assistant who can solve any task using tool calls. You will be given a task to solve as best you can.
To do so, you have been given access to some tools.

The tool call you write is an action: after the tool is executed, you will get the result of the tool call as an "observation".
This Action/Observation can repeat N times, you should take several steps when needed.

Here are a few examples using notional tools:
---
Task: "Generate an image of the oldest person in this document."

Action:
{
    "name": "document_qa",
    "arguments": {"document": "document.pdf", "question": "Who is the oldest person mentioned?"}
}
Observation: "The oldest person in the document is John Doe, a 55 year old lumberjack living in Newfoundland."

Action:
{
    "name": "image_generator",
    "arguments": {"prompt": "A portrait of John Doe, a 55-year-old man living in Canada."}
}
Observation: "image.png"

Action:
{
    "name": "final_answer",
    "arguments": "image.png"
}

---
Task: "What is the result of the following operation: 5 + 3 + 1294.678?"

Action:
{
    "name": "python_interpreter",
    "arguments": {"code": "5 + 3 + 1294.678"}
}
Observation: 1302.678

Action:
{
    "name": "final_answer",
    "arguments": "1302.678"
}

Above example were using notional tools that might not exist for you. You only have access to these tools:
{%- for tool in tools %}
- {{ tool.name }}: {{ tool.description }}
    Takes inputs: {{tool.params_json_schema}}
{%- endfor %}

{%- if handoffs %}
You can also give tasks to team members.
Calling a team member works the same as for calling a tool: simply, the only argument you can give in the call is 'task', a long string explaining your task.
Given that this team member is a real human, you should be very verbose in your task.
Here is a list of the team members that you can call:
{%- for handoff in handoffs %}
- {{ handoff.name }}: {{ handoff.description }}
{%- endfor %}
{%- endif %}
""".strip()

TEMPLATE_ACTION = r"""Action:
{
  "name": "{{action_name}}",
  "arguments": {{action_arguments}}
}"""


@dataclass
class ConverterPreprocessInput:
    system_instructions: str | None
    input: str | list[TResponseInputItem]
    tools: list[Tool] = field(default_factory=list)
    output_schema: AgentOutputSchema | None = None
    handoffs: list[Handoff] = field(default_factory=list)
    model_settings: ModelSettings | None = None


class ReactConverter:
    """
    Format: ReAct style function_call, see https://github.com/huggingface/smolagents/blob/main/src/smolagents/prompts/toolcalling_agent.yaml
    Agent output: Purely function_call! (cannot only return content)
        ONLY one function_call is allowed!
    """

    def __init__(self) -> None:
        self.jinja_env = jinja2.Environment()
        self.template_sp = self.jinja_env.from_string(TEMPLATE_SP)
        self.template_action = self.jinja_env.from_string(TEMPLATE_ACTION)
        self.observation_str = "Observation:"
        self.action_str = "Action:"

    def preprocess(self, input: ConverterPreprocessInput) -> ConverterPreprocessInput:
        """Preprocess input for ReAct mode
        - convert SP+tools+handoffs -> new SP
        - process input
        - output_schema: BACKLOG:
        """
        converted_sp = self._handle_sp(input.system_instructions, input.tools, input.handoffs)
        converted_input = self._handle_input(input.system_instructions, input.input)
        converted_model_settings = self._handle_model_settings(input.model_settings)
        return ConverterPreprocessInput(
            system_instructions=converted_sp, input=converted_input, model_settings=converted_model_settings
        )

    def _handle_sp(self, system_instructions: str | None, tools: list[Tool], handoffs: list[Handoff]) -> str | None:
        sp = self.template_sp.render(tools=tools, handoffs=handoffs)
        if system_instructions:  # Handle of TWO system_instructions?
            sp = f"{system_instructions}\n\n{sp}"
        return sp

    def _handle_input(
        self, system_instructions: str | None, input: str | list[TResponseInputItem]
    ) -> str | list[TResponseInputItem]:
        """Basic conversion, see logic in Converter.items_to_messages()

        Rules:
        - tool outputs => InputMessage (role=user)
        - InputMessage (role=assistant) => remove tool_calls
        """
        results = []
        for item in input:
            # type == "message" & role == "user|system|developer"
            if Converter.maybe_easy_input_message(item) or Converter.maybe_input_message(item):
                # content: str | List[ResponseInputContentParam] -- do not convert now!
                results.append(deepcopy(item))
            # type == "message" & role == "assistant"
            elif _ := Converter.maybe_response_output_message(item):
                print(f">> [WARNING] got response_output_message: {item}")
                results.append(deepcopy(item))
            # type == "function_call"
            elif func_call := Converter.maybe_function_tool_call(item):
                message = EasyInputMessageParam(
                    role="assistant",
                    content=self.template_action.render(
                        action_name=func_call["name"], action_arguments=func_call["arguments"]
                    ),
                )
                # print(f">> converted function_call to {message}")
                results.append(message)
            # type == "function_call_output"
            elif func_output := Converter.maybe_function_tool_call_output(item):
                message = EasyInputMessageParam(role="user", content=f"{self.observation_str}: {func_output['output']}")
                # print(f">> converted function_call_output to {message}")
                results.append(message)
            else:
                print(f">> [WARNING] Item with unknown type: {item}")
                results.append(deepcopy(item))
        return results

    def _handle_model_settings(self, model_settings: ModelSettings) -> ModelSettings:
        if not model_settings.extra_args:
            model_settings.extra_args = {}
        model_settings.extra_args["stop"] = [self.observation_str]  # add stop tokens
        return model_settings

    def postprocess(self, items: list[ResponseOutputItem]) -> list[ResponseOutputItem]:
        """Parse FCs from text output"""
        text_output = ""
        for item in items:
            if not isinstance(item, ResponseOutputMessage):
                print(f">> Unknown item type: {item.__class__.__name__}")
            else:
                assert len(item.content) == 1 and item.content[0].type == "output_text"
                text_output += item.content[0].text
        return self._parse_react_output(text_output)

    def _parse_react_output(self, text_output: str) -> list[ResponseOutputItem]:
        """Parse output text into list of ResponseOutputMessage|ResponseFunctionToolCall

        Sample input:
            Action:\n{\n    "name": "search_google_api",\n    "arguments": {\'query\': \'smolagents package\'}\n}
        """
        assert self.observation_str not in text_output
        if self.action_str in text_output:
            # only one action is allowed for now!
            assert text_output.count(self.action_str) == 1
            text_output = text_output.split(self.action_str)[1].strip()
            try:
                action = json.loads(text_output)
            except json.JSONDecodeError:
                try:
                    action = ast.literal_eval(text_output)
                except Exception as e:  # pylint: disable=broad-except
                    raise ValueError(f"Invalid action: {text_output}") from e
            assert "name" in action and "arguments" in action
            function = Function(name=action["name"], arguments=json.dumps(action["arguments"], ensure_ascii=False))
            message = ChatCompletionMessage(
                role="assistant",
                # TODO: also parse "Think" into content
                tool_calls=[ChatCompletionMessageToolCall(function=function, id="FAKE_ID", type="function")],
            )
            return Converter.message_to_output_items(message)
        else:
            message = ChatCompletionMessage(role="assistant", content=text_output)
            return Converter.message_to_output_items(message)
