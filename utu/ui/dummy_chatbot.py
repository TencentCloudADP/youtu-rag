import asyncio
import pickle
import time

import agents as ag

from utu.agents.orchestra import OrchestraStreamEvent
from utu.agents.orchestra_agent import OrchestraAgent
from utu.agents.simple_agent import SimpleAgent

from .common import (
    Event,
    NewAgentContent,
    OrchestraContent,
    PlanItem,
    ReportItem,
    TextDeltaContent,
    WorkerItem,
    handle_orchestra_events,
)

event_list = []
start_time = time.time()


async def send_event(event: Event):
    event_list.append({"timestamp": time.time() - start_time, "event": event})


def save_event_list():
    with open(f"event_list_{time.time()}.pkl", "wb") as f:
        pickle.dump(event_list, f)


async def run(agent: SimpleAgent | OrchestraAgent, query: str):
    if isinstance(agent, OrchestraAgent):
        stream = agent.run_streamed(query)
    elif isinstance(agent, SimpleAgent):
        agent.input_items.append({"role": "user", "content": query})
        # print in red color
        print(f"\033[91mInput items: {agent.input_items}\033[0m")
        stream = agent.run_streamed(agent.input_items)
    else:
        raise ValueError(f"Unsupported agent type: {type(agent).__name__}")

    async for event in stream.stream_events():
        event_to_send = None
        print(f"--------------------\n{event}")
        if isinstance(event, ag.RawResponsesStreamEvent):
            if event.data.type == "response.output_text.delta":
                if event.data.delta != "":
                    event_to_send = Event(
                        type="raw",
                        data=TextDeltaContent(
                            type="text",
                            delta=event.data.delta,
                            inprogress=True,
                        ),
                    )
            elif event.data.type == "response.output_text.done":
                event_to_send = Event(
                    type="raw",
                    data=TextDeltaContent(
                        type="text",
                        delta="",
                        inprogress=False,
                    ),
                )
            elif (
                event.data.type == "response.reasoning_summary_text.delta"
                or event.data.type == "response.reasoning_text.delta"
            ):
                if event.data.delta != "":
                    event_to_send = Event(
                        type="raw",
                        data=TextDeltaContent(
                            type="reason",
                            delta=event.data.delta,
                            inprogress=True,
                        ),
                    )
            elif (
                event.data.type == "response.reasoning_summary_text.done"
                or event.data.type == "response.reasoning_text.done"
            ):
                event_to_send = Event(
                    type="raw",
                    data=TextDeltaContent(
                        type="reason",
                        delta="",
                        inprogress=True,
                    ),
                )
            elif event.data.type == "response.function_call_arguments.delta":
                # if event.data.delta != '':
                #     event_to_send = Event(type="raw", data=TextDeltaContent(
                #         type="tool_call_argument",
                #         delta=event.data.delta,
                #         inprogress=True,
                #     ))
                pass
            elif event.data.type == "response.output_item.done":
                item = event.data.item
                if item.type == "function_call":
                    event_to_send = Event(
                        type="raw",
                        data=TextDeltaContent(
                            type="tool_call",
                            delta=item.name,
                            argument=item.arguments,
                            callid=item.call_id,
                            inprogress=True,
                        ),
                    )
                elif item.type == "reasoning":
                    event_to_send = Event(
                        type="raw",
                        data=TextDeltaContent(
                            type="reason",
                            delta="",
                            inprogress=False,
                        ),
                    )
                elif item.type == "message":
                    pass
            elif event.data.type == "response.function_call_arguments.done":
                # event_to_send = Event(type="raw", data=TextDeltaContent(
                #     type="tool_call_argument",
                #     delta=event.data.delta,
                #     inprogress=False,
                # ))
                pass
            elif event.data.type == "response.output_item.added":
                item = event.data.item
                if item.type == "function_call":
                    # event_to_send = Event(type="raw", data=TextDeltaContent(
                    #     type="tool_call",
                    #     delta=item.name,
                    #     inprogress=True,
                    #     tool_name=item.name,
                    # ))
                    pass
                elif item.type == "reasoning":
                    event_to_send = Event(
                        type="raw",
                        data=TextDeltaContent(
                            type="reason",
                            delta=item.summary,
                            inprogress=True,
                        ),
                    )
                elif item.type == "message":
                    event_to_send = Event(
                        type="raw",
                        data=TextDeltaContent(
                            type="text",
                            delta="",
                            inprogress=True,
                        ),
                    )
                else:
                    event_to_send = None
            else:
                event_to_send = None
        elif isinstance(event, ag.RunItemStreamEvent):
            item = event.item
            if item.type == "tool_call_output_item":
                event_to_send = Event(
                    type="raw",
                    data=TextDeltaContent(
                        type="tool_call_output",
                        delta=item.output,
                        callid=item.raw_item["call_id"],
                        inprogress=False,
                    ),
                )
            else:
                pass
        elif isinstance(event, ag.AgentUpdatedStreamEvent):
            if event.new_agent:
                if hasattr(event.new_agent, "name"):
                    new_agent_name = f"{event.new_agent.name} ({event.new_agent.__class__.__name__})"
                else:
                    new_agent_name = event.new_agent.__class__.__name__

                event_to_send = Event(
                    type="new",
                    data=NewAgentContent(type="new", name=new_agent_name),
                )
        elif isinstance(event, OrchestraStreamEvent):
            event_to_send = await handle_orchestra_events(event)
        else:
            pass
        if event_to_send:
            # print(f"Sending event: {asdict(event_to_send)}")
            await send_event(event_to_send)
    else:
        pass
    event_to_send = Event(type="finish")
    # self.write_message(asdict(event_to_send))
    await send_event(event_to_send)


class DummyChatbot:
    def __init__(self, agent: SimpleAgent | OrchestraAgent, example_query: str = ""):
        self.agent = agent
        self.example_query = example_query

    async def __launch(self, port: int = 8848):
        print("this is dummy chatbot used for recording events")
        await self.agent.build()
        await run(self.agent, self.example_query)
        save_event_list()

    async def launch_async(self, port: int = 8848):
        await self.__launch(port=port)

    def launch(self, port: int = 8848):
        asyncio.run(self.__launch(port=port))
