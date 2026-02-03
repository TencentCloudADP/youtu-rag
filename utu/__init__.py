# ruff: noqa
from agents.run import set_default_agent_runner

from .utils import EnvUtils, setup_logging
from .patch.runner import UTUAgentRunner
from .tracing import setup_tracing

EnvUtils.assert_env(["UTU_LLM_TYPE", "UTU_LLM_MODEL"])
setup_logging(EnvUtils.get_env("UTU_LOG_LEVEL", "WARNING"))
setup_tracing()
# patched runner
set_default_agent_runner(UTUAgentRunner())
