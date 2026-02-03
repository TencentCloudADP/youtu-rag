from agents import RunContextWrapper, TContext, TResponseInputItem

from ..utils import get_logger

logger = get_logger(__name__)


class BaseContextManager:
    def preprocess(
        self, input: str | list[TResponseInputItem], run_context: RunContextWrapper[TContext] = None
    ) -> str | list[TResponseInputItem]:
        return input


class DummyContextManager(BaseContextManager):
    def preprocess(
        self, input: str | list[TResponseInputItem], run_context: RunContextWrapper[TContext] = None
    ) -> str | list[TResponseInputItem]:
        # NOTE: filter type="reasoning" items for vllm cannot process it for now!
        # return ChatCompletionConverter.filter_items(input)

        # TODO: add action limit
        # if run_context is None or run_context.context.get("env", None) is None:
        #     logger.warning(f"run_context {run_context} or env is None")
        #     return input
        return input
