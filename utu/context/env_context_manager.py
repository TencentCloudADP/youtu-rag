import copy
import logging

from agents import RunContextWrapper, TContext, TResponseInputItem
from openai.types.responses import EasyInputMessageParam

from ..env import BaseEnv
from .base_context_manager import BaseContextManager

logger = logging.getLogger(__name__)


class EnvContextManager(BaseContextManager):
    def preprocess(
        self, input: str | list[TResponseInputItem], run_context: RunContextWrapper[TContext] = None
    ) -> str | list[TResponseInputItem]:
        if run_context is None or run_context.context.get("env", None) is None:
            logger.warning(f"run_context {run_context} or env is None")
            return input
        env: BaseEnv = run_context.context["env"]
        env_state = env.get_state()
        if env_state:
            input = copy.deepcopy(input)
            input.append(EasyInputMessageParam(content=env_state, role="user"))
        return input
