"""
https://github.com/bytedance/SandboxFusion
https://bytedance.github.io/SandboxFusion/docs/docs/get-started
"""

import requests

from ..config import ToolkitConfig
from ..utils import get_logger, oneline_object
from .base import AsyncBaseToolkit, register_tool

logger = get_logger(__name__)

SUPPORTED_LANGUAGES = [
    "python",
    "cpp",
    "nodejs",
    "go",
    "go_test",
    "java",
    "php",
    "csharp",
    "bash",
    "typescript",
    "sql",
    "rust",
    "cuda",
    "lua",
    "R",
    "perl",
    "D_ut",
    "ruby",
    "scala",
    "julia",
    "pttest",
    "junit",
    "kotlin_script",
    "jest",
    "verilog",
    "python_gpu",
    "lean",
    "swift",
    "racket",
]


class CodesnipToolkit(AsyncBaseToolkit):
    def __init__(self, config: ToolkitConfig = None) -> None:
        super().__init__(config)
        self.server_url = self.config.config.get("server_url")

    @register_tool
    async def run_code(self, code: str, language: str = "python") -> str:
        """Run code in sandbox and return the result.
        Supported languages: python, cpp, nodejs, go, go_test, java, php, csharp, bash, typescript, sql, rust, cuda,
         lua, R, perl, D_ut, ruby, scala, julia, pttest, junit, kotlin_script, jest, verilog, python_gpu, lean, swift,
         racket

        Args:
            code (str): The code to run.
            language (str, optional): The language of the code. Defaults to "python".
        Returns:
            str: The result of the code.
        """
        payload = {
            "code": code,
            "language": language,
        }
        response = requests.post(f"{self.server_url}/run_code", json=payload)
        result = response.json()
        logger.info(f"[tool] run_code ```{oneline_object(payload)}``` got result: {oneline_object(result)}")
        return str(result)
