import argparse

from utu.utils import EnvUtils


class ExampleConfig:
    def __init__(self):
        self.example_query = ""
        self.ip = EnvUtils.get_env("UTU_WEBUI_IP", "127.0.0.1")
        self.port = EnvUtils.get_env("UTU_WEBUI_PORT", "8848")
        self.autoload = EnvUtils.get_env("UTU_WEBUI_AUTOLOAD", "false") == "true"

        parser = argparse.ArgumentParser()
        parser.add_argument("--example_query", type=str, default=self.example_query)
        parser.add_argument("--ip", type=str, default=self.ip)
        parser.add_argument("--port", type=int, default=self.port)
        parser.add_argument("--autoload", type=bool, default=self.autoload)
        args = parser.parse_args()

        self.example_query = args.example_query
        self.ip = args.ip
        self.port = args.port
        self.autoload = args.autoload
