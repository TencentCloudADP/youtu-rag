"""
- [ ] standardize parser?
- [ ] only LLM config is needed!
"""

import re

from ...config import AgentConfig
from ...utils import FileUtils, get_logger
from ..llm_agent import LLMAgent
from .data import Subtask, WorkspaceTaskRecorder

logger = get_logger(__name__)
PROMPTS = FileUtils.load_prompts("agents/workforce/planner.yaml")


class PlannerAgent:
    """Task planner that handles task decomposition."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.llm = LLMAgent(model_config=config.workforce_planner_model)

    async def plan_task(self, recorder: WorkspaceTaskRecorder) -> None:
        """Plan tasks based on the overall task and available agents."""
        # TODO: replan with `failure_info`
        plan_prompt = PROMPTS["TASK_PLAN_PROMPT"].format(
            overall_task=recorder.overall_task,
            executor_agents_info=recorder.executor_agents_info,
        )
        plan_recorder = await self.llm.run(plan_prompt)
        recorder.add_run_result(plan_recorder.get_run_result(), "planner")  # add planner trajectory

        # parse tasks
        pattern = "<task>(.*?)</task>"
        tasks_content: list[str] = re.findall(pattern, plan_recorder.final_output, re.DOTALL)
        tasks_content = [task.strip() for task in tasks_content if task.strip()]
        tasks = [Subtask(task_id=i + 1, task_name=task) for i, task in enumerate(tasks_content)]
        recorder.plan_init(tasks)

    async def plan_update(self, recorder: WorkspaceTaskRecorder, task: Subtask) -> str:
        """Update the task plan based on completed tasks."""
        task_plan_list = recorder.formatted_task_plan_list_with_task_results
        last_task_id = task.task_id
        previous_task_plan = "\n".join(f"{task}" for task in task_plan_list[: last_task_id + 1])
        unfinished_task_plan = "\n".join(f"{task}" for task in task_plan_list[last_task_id + 1 :])

        task_update_plan_prompt = (
            PROMPTS["TASK_UPDATE_PLAN_PROMPT"]
            .strip()
            .format(
                overall_task=recorder.overall_task,
                previous_task_plan=previous_task_plan,
                unfinished_task_plan=unfinished_task_plan,
            )
        )
        plan_update_recorder = await self.llm.run(task_update_plan_prompt)
        recorder.add_run_result(plan_update_recorder.get_run_result(), "planner")  # add planner trajectory
        choice, updated_plan = self._parse_update_response(plan_update_recorder.final_output)
        # choice: continue, update, stop
        if choice == "update":
            recorder.plan_update(task, updated_plan)
        return choice

    def _parse_update_response(self, response: str) -> tuple[str, list[str] | None]:
        # TODO: split "stop" into "early_completion" and "task_collapse"
        # Parse choice
        pattern_choice = r"<choice>(.*?)</choice>"
        match_choice = re.search(pattern_choice, response, re.DOTALL)
        if match_choice:
            choice = match_choice.group(1).strip().lower()
            if choice not in ["continue", "update", "stop"]:
                logger.warning(f"Unexpected choice value: {choice}. Defaulting to 'continue'.")
                choice = "continue"
        else:
            logger.warning("No choice found in response. Defaulting to 'continue'.")
            choice = "continue"

        # Parse updated plan if choice is "update"
        updated_tasks = None
        if choice == "update":
            pattern_updated_plan = r"<updated_unfinished_task_plan>(.*?)</updated_unfinished_task_plan>"
            match_updated_plan = re.search(pattern_updated_plan, response, re.DOTALL)
            if match_updated_plan:
                # Try both task formats: <task> and <task_id:X>
                updated_plan_content = match_updated_plan.group(1).strip()
                task_pattern = r"<task>(.*?)</task>"
                task_matches = re.findall(task_pattern, updated_plan_content, re.DOTALL)
                # If no standard task tags found, try task_id format
                if not task_matches:
                    task_id_pattern = r"<task_id:\d+>(.*?)</task_id:\d+>"
                    task_matches = re.findall(task_id_pattern, updated_plan_content, re.DOTALL)

                updated_tasks = [task.strip() for task in task_matches if task.strip()]
                if not updated_tasks:
                    logger.warning("No tasks found in updated plan. Defaulting to None.")
                    updated_tasks = None
            else:
                logger.warning("No updated plan found in response. Defaulting to None.")
                updated_tasks = None

        return choice, updated_tasks

    async def plan_check(self, recorder: WorkspaceTaskRecorder, task: Subtask) -> None:
        task_check_prompt = (
            PROMPTS["TASK_CHECK_PROMPT"]
            .strip()
            .format(
                overall_task=recorder.overall_task,
                task_plan=recorder.formatted_task_plan,
                last_completed_task=task.task_name,
                last_completed_task_id=task.task_id,
                last_completed_task_description=task.task_description,
                last_completed_task_result=task.task_result,
            )
        )
        res = await self.llm.run(task_check_prompt)
        recorder.add_run_result(res.get_run_result(), "planner")  # add planner trajectory
        # parse and update task status
        task_check_result = self._parse_check_response(res.final_output)
        task.task_status = task_check_result

    def _parse_check_response(self, response: str) -> str:
        pattern = r"<task_status>(.*?)</task_status>"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            task_status = match.group(1).strip().lower()
            if "partial" in task_status:  # in case that models output "partial_success"
                return "partial success"
            if task_status in ["success", "failed", "partial success"]:
                return task_status
            else:
                logger.warning(f"Unexpected task status value: {task_status}. Defaulting to 'partial success'.")
                return "partial success"
        else:
            logger.warning("No task status found in response. Defaulting to 'partial success'.")
            return "partial success"
