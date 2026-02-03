import asyncio
import logging
import sys
import warnings

import agents as ag
import gradio as gr

from utu.agents import OrchestraAgent, SimpleAgent
from utu.agents.orchestra import OrchestraStreamEvent

# Add a visible deprecation warning that will be shown when the module is imported
warnings.simplefilter("always", DeprecationWarning)  # Ensure deprecation warnings are shown
warnings.warn(
    "The gradio_chatbot module is deprecated and will be removed in a future release. "
    "Please migrate to the new webui implementation.",
    DeprecationWarning,
    stacklevel=2,
)

# Print a warning to stderr for better visibility
print(
    "WARNING: The gradio_chatbot module is deprecated and will be removed in a future release. "
    "Please migrate to the new webui implementation.",
    file=sys.stderr,
)


warnings.warn(
    "The gradio_chatbot module is deprecated and will be removed in a future release. "
    "Please update your code to use the recommended alternatives.",
    DeprecationWarning,
    stacklevel=2,
)


class GradioChatbot:
    """
    DEPRECATED: This module is deprecated and will be removed in a future release.

    This module has been superseded by newer implementations in the webui package.
    Please migrate to the new implementation as soon as possible.

    Migration Guide:
    - Replace usage of GradioChatbot with the new web interface components
    - Update your code to use the new API endpoints
    - Refer to the project documentation for more details
    """

    def __init__(self, agent: SimpleAgent | OrchestraAgent, example_query=""):
        self.agent = agent
        self.user_interrupted = False
        self.user_interrupted_lock = asyncio.Lock()
        self.example_query = example_query
        self.ui = self._setup_ui()

    def _handle_output_text_delta(self, event, history):
        """Handle response.output_text.delta event."""
        if history and history[-1]["role"] == "assistant" and "metadata" not in history[-1]:
            history[-1]["content"] += event.data.delta
        else:
            history.append({"role": "assistant", "content": event.data.delta})

    def _handle_reasoning_delta(self, event, history):
        """Handle response.reasoning_summary_text.delta event."""
        if (
            history
            and history[-1]["role"] == "assistant"
            and ("metadata" in history[-1])
            and history[-1]["metadata"]["type"] == "reasoning"
        ):
            history[-1]["content"] += event.data.delta
        else:
            history.append(
                {
                    "role": "assistant",
                    "content": f"[Reasoning]: {event.data.delta}",
                    "metadata": {
                        "title": "🤔 reasoning",
                        "type": "reasoning",
                    },
                }
            )

    def _handle_reasoning_done(self, history):
        """Handle response.reasoning_summary_text.done event."""
        if (
            history
            and history[-1]["role"] == "assistant"
            and ("metadata" in history[-1])
            and history[-1]["metadata"]["type"] == "reasoning"
        ):
            history[-1]["content"] += "... [Reasoning Completed] ..."

    def _handle_output_item_added(self, event, history):
        """Handle response.output_item.added event."""
        item = event.data.item
        if item.type == "function_call":
            history.append(
                {
                    "role": "assistant",
                    "content": "",
                    "metadata": {
                        "title": f"🛠️ tool_call {item.name}",
                        "type": "tool_call",
                    },
                }
            )
        elif item.type == "reasoning":
            history.append(
                {
                    "role": "assistant",
                    "content": "",
                    "metadata": {
                        "title": "🤔 reasoning",
                        "type": "reasoning",
                    },
                }
            )

    def _handle_function_call_arguments_delta(self, event, history):
        """Handle response.function_call_arguments.delta event."""
        if (
            history
            and history[-1]["role"] == "assistant"
            and ("metadata" in history[-1])
            and history[-1]["metadata"]["type"] == "tool_call"
        ):
            history[-1]["content"] += event.data.delta
        else:
            history.append(
                {
                    "role": "assistant",
                    "content": event.data.delta,
                    "metadata": {
                        "title": "🛠️ tool_call",
                        "type": "tool_call",
                    },
                }
            )

    def _handle_function_call_arguments_done(self, history):
        """Handle response.function_call_arguments.done event."""
        if (
            history
            and history[-1]["role"] == "assistant"
            and ("metadata" in history[-1])
            and history[-1]["metadata"]["type"] == "tool_call"
        ):
            history[-1]["content"] += "... [Tool Call Arguments Completed] ..."

    def _handle_raw_response_event(self, event, history):
        """Handle RawResponsesStreamEvent."""
        event_type = event.data.type

        if event_type == "response.output_text.delta":
            self._handle_output_text_delta(event, history)
        elif event_type == "response.reasoning_summary_text.delta":
            self._handle_reasoning_delta(event, history)
        elif event_type == "response.reasoning_summary_text.done":
            self._handle_reasoning_done(history)
        elif event_type == "response.output_item.added":
            self._handle_output_item_added(event, history)
        elif event_type == "response.function_call_arguments.delta":
            self._handle_function_call_arguments_delta(event, history)
        elif event_type == "response.function_call_arguments.done":
            self._handle_function_call_arguments_done(history)

    def _handle_run_item_event(self, event, history):
        """Handle RunItemStreamEvent."""
        print(f"-------- {event.item.type} --------")
        item = event.item
        if item.type == "tool_call_output_item":
            history.append(
                {
                    "role": "assistant",
                    "content": f"{item.output}",
                    "metadata": {"title": "tool_output"},
                }
            )

    def _handle_orchestra_plan_event(self, item, history):
        """Handle Orchestra plan event."""
        analysis = item.analysis
        todo_str = []
        for i, subtask in enumerate(item.todo, 1):
            todo_str.append(f"{i}. {subtask.task} ({subtask.agent_name})")
        todo_str = "\n".join(todo_str)
        history.append(
            {
                "role": "assistant",
                "content": f"{analysis}",
                "metadata": {"title": "💭 Plan Analysis"},
            },
        )
        history.append(
            {
                "role": "assistant",
                "content": f"{todo_str}",
                "metadata": {"title": "📋 Todo"},
            }
        )

    def _handle_orchestra_worker_event(self, item, history):
        """Handle Orchestra worker event."""
        task = item.task
        output = item.output
        history.append(
            {"role": "assistant", "content": f"{task}", "metadata": {"title": "Worker Task"}}
        )
        history.append(
            {"role": "assistant", "content": f"{output}", "metadata": {"title": "Worker Output"}}
        )

    def _handle_orchestra_report_event(self, item, history):
        """Handle Orchestra report event."""
        output = item.output
        history.append(
            {"role": "assistant", "content": f"{output}", "metadata": {"title": "Report Output"}}
        )

    def _handle_orchestra_event(self, event, history):
        """Handle OrchestraStreamEvent."""
        item = event.item
        if event.name == "plan":
            self._handle_orchestra_plan_event(item, history)
        elif event.name == "worker":
            self._handle_orchestra_worker_event(item, history)
        elif event.name == "report":
            self._handle_orchestra_report_event(item, history)

    def _process_event(self, event, history):
        """Process a single stream event and update history."""
        if isinstance(event, ag.RawResponsesStreamEvent):
            self._handle_raw_response_event(event, history)
        elif isinstance(event, ag.RunItemStreamEvent):
            self._handle_run_item_event(event, history)
        elif isinstance(event, OrchestraStreamEvent):
            self._handle_orchestra_event(event, history)

    def _setup_ui(self):
        """Setup the Gradio UI."""
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            gr.Markdown("# uTu Agent Gradio Demo")
            with gr.Row():
                with gr.Column(scale=1):
                    chatbot = gr.Chatbot(label="Chatbot", type="messages")
                    user_input = gr.Textbox(
                        label="Your Input",
                        placeholder="Type your message here...",
                        value=self.example_query,
                    )
                    with gr.Row():
                        with gr.Column():
                            submit_button = gr.Button("Submit")
                        with gr.Column():
                            cancel_button = gr.Button("Cancel")

            gr.Markdown("This is a simple agent demo using uTu framework.")

            async def check_and_reset_user_interrupt():
                async with self.user_interrupted_lock:
                    if self.user_interrupted:
                        self.user_interrupted = False
                        return True
                return False

            async def set_user_interrupt(user_interrupted_: bool = True):
                async with self.user_interrupted_lock:
                    self.user_interrupted = user_interrupted_

            async def respond(message, history):
                if message.strip() == "":
                    return
                global built

                await set_user_interrupt(False)

                chat_message = {"role": "user", "content": message}
                history.append(chat_message)
                yield history

                if isinstance(self.agent, OrchestraAgent):
                    stream = self.agent.run_streamed(message)
                elif isinstance(self.agent, SimpleAgent):
                    self.agent.input_items.append(chat_message)
                    stream = self.agent.run_streamed(self.agent.input_items)

                async for event in stream.stream_events():
                    logging.info(f"Event: {event}")
                    if await check_and_reset_user_interrupt():
                        stream.cancel()
                        history.append(
                            {
                                "role": "assistant",
                                "content": "User interrupted the response.",
                                "metadata": {"title": "🛑 interrupted"},
                            }
                        )
                        yield history
                        break

                    self._process_event(event, history)
                    yield history

                self.agent.input_items = stream.to_input_list()
                self.agent.current_agent = stream.last_agent

            async def cancel_response():
                await set_user_interrupt(True)

            submit_button.click(respond, inputs=[user_input, chatbot], outputs=[chatbot])
            cancel_button.click(cancel_response, inputs=[], outputs=[])
            user_input.submit(respond, inputs=[user_input, chatbot], outputs=[chatbot])

        return demo

    def launch(self, port=8848):
        asyncio.run(self.agent.build())
        self.ui.launch(share=False, server_port=port)
