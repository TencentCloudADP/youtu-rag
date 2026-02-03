import argparse
import asyncio
import json
import pickle
import traceback
from dataclasses import asdict
from importlib import resources

import tornado.web
import tornado.websocket

from utu.ui.common import Event, ExampleContent, UserRequest

REPLAY_INTERVAL = 0.5


class ReplayWebSocketHandler(tornado.websocket.WebSocketHandler):
    def initialize(self, example_query: str = "", events: list[Event] | None = None):
        self.example_query = example_query
        if events is None:
            events = []
        self.events = events

    def check_origin(self, origin):
        # Allow all origins to connect
        return True

    def open(self):
        # send example query
        self.write_message(asdict(Event("example", ExampleContent(type="example", query=self.example_query))))

    async def send_event(self, event: Event):
        # print in green color
        print(f"\033[92mSending event: {asdict(event)}\033[0m")
        self.write_message(asdict(event))

    async def on_message(self, message: str):
        try:
            data = json.loads(message)
            user_request = UserRequest(**data)
            if user_request.type == "query":
                try:
                    content = user_request.content
                    # check query not empty
                    if content.query.strip() == "":
                        raise ValueError("Query cannot be empty")

                    # last_time = 0
                    for event in self.events:
                        # timestamp = event["timestamp"]
                        real_event = event["event"]
                        await asyncio.sleep(REPLAY_INTERVAL)
                        await self.send_event(real_event)
                        # last_time = timestamp

                except TypeError as e:
                    print(f"Invalid query format: {e}")
                    # stack trace
                    print(traceback.format_exc())
                    self.close(1002, "Invalid query format")
            else:
                pass
        except json.JSONDecodeError:
            self.close(1002, "Invalid JSON received")
        except Exception as e:
            print(f"Error processing message: {str(e)}")
            print(traceback.format_exc())
            self.close(1002, "Error processing message")

    def on_close(self):
        pass


class ReplayWebUIChatbot:
    def __init__(self, example_query: str = "", events_file: str = ""):
        self.example_query = example_query
        self.events_file = events_file
        with open(events_file, "rb") as f:
            self.events = pickle.load(f)
        # hack
        with resources.as_file(resources.files("utu_agent_ui.static").joinpath("index.html")) as static_dir:
            self.static_path = str(static_dir).replace("index.html", "")

    def make_app(self) -> tornado.web.Application:
        return tornado.web.Application(
            [
                (r"/ws", ReplayWebSocketHandler, {"example_query": self.example_query, "events": self.events}),
                (
                    r"/",
                    tornado.web.RedirectHandler,
                    {"url": "/index.html"},
                ),
                (
                    r"/(.*)",
                    tornado.web.StaticFileHandler,
                    {"path": self.static_path, "default_filename": "index.html"},
                ),
            ],
            debug=True,
        )

    async def __launch(self, port: int = 8848):
        app = self.make_app()
        app.listen(port)
        print(f"Server started at http://localhost:{port}/")
        await asyncio.Event().wait()

    async def launch_async(self, port: int = 8848):
        await self.__launch(port=port)

    def launch(self, port: int = 8848):
        asyncio.run(self.__launch(port=port))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()
    webui = ReplayWebUIChatbot(example_query=args.query, events_file=args.events)
    webui.launch()
