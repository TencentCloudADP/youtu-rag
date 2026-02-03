import re

from ...config import AgentConfig
from ...utils import FileUtils, get_logger
from ..llm_agent import LLMAgent
from .data import WorkspaceTaskRecorder

logger = get_logger(__name__)
PROMPTS: dict[str, str] = FileUtils.load_prompts("agents/workforce/answerer.yaml")


class AnswererAgent:
    """Answer extractor that handles final answer generation from task execution results."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = LLMAgent(model_config=config.workforce_answerer_model)

    async def extract_final_answer(self, recorder: WorkspaceTaskRecorder) -> str:
        """Extract the final answer from formatted task execution results."""
        # Generate final answer prompt
        final_prompt = (
            PROMPTS["FINAL_ANSWER_PROMPT"]
            .strip()
            .format(
                question=recorder.overall_task,
                task_results="\n".join(recorder.formatted_task_plan_list_with_task_results),
            )
        )
        final_recorder = await self.llm.run(final_prompt)
        recorder.add_run_result(final_recorder.get_run_result(), "answerer")  # add answerer trajectory
        final_answer = self._parse_final_response(final_recorder.final_output)
        return final_answer

    def _parse_final_response(self, response: str) -> str:
        answer_pattern = r"<answer>(.*?)</answer>"
        answer_match = re.search(answer_pattern, response, re.DOTALL)
        final_answer = answer_match.group(1).strip()
        return final_answer

    async def answer_check(self, question: str, model_answer: str, ground_truth: str) -> bool:
        """Check if model answer and ground truth are semantically equivalent using LLM."""
