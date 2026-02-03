"""
- [ ] error tracing
"""

import re

from ...config import AgentConfig
from ...utils import FileUtils, get_logger
from ..simple_agent import SimpleAgent
from .data import Subtask, WorkspaceTaskRecorder

logger = get_logger(__name__)

PROMPTS = FileUtils.load_prompts("agents/workforce/executor.yaml")


class ExecutorAgent:
    """Executor agent that executes tasks assigned by the planner.

    - TODO: self-reflection
    """

    def __init__(self, config: AgentConfig, workforce_config: AgentConfig):
        self.config = config
        self.executor_agent = SimpleAgent(config=config)

        executor_config = workforce_config.workforce_executor_config
        self.max_tries = executor_config.get("max_tries", 1)
        self.return_summary = executor_config.get("return_summary", False)

        self.reflection_history = []

    async def execute_task(
        self,
        recorder: WorkspaceTaskRecorder,
        task: Subtask,
    ) -> None:
        """Execute the task and check the result."""
        task.task_status = "in progress"

        tries = 1
        final_result = None
        executor_res = None
        while tries <= self.max_tries:
            try:
                self.executor_agent.clear_input_items()  # clear chat history!

                # * 1. Task execution
                if tries == 1:
                    user_prompt = PROMPTS["TASK_EXECUTE_USER_PROMPT"].format(
                        overall_task=recorder.overall_task,
                        overall_plan=recorder.formatted_task_plan,
                        task_name=task.task_name,
                        task_description=task.task_description,
                    )
                else:
                    user_prompt = PROMPTS["TASK_EXECUTE_WITH_REFLECTION_USER_PROMPT"].format(
                        overall_task=recorder.overall_task,
                        overall_plan=recorder.formatted_task_plan,
                        task_name=task.task_name,
                        task_description=task.task_description,
                        previous_attempts=self.reflection_history[-1] if self.reflection_history else "",
                    )
                executor_res = await self.executor_agent.run(user_prompt, save=True)  # save chat history!
                final_result = executor_res.final_output

                # * 2. Task check
                task_check_prompt = PROMPTS["TASK_CHECK_PROMPT"].format(
                    task_name=task.task_name,
                    task_description=task.task_description,
                )
                response_content = await self.executor_agent.run(task_check_prompt)  # do not save chat history!
                if self._parse_task_check_result(response_content.final_output):
                    logger.info(f"Task '{task.task_name}' completed successfully.")
                    break

                # * 3. Task reflection (when failed)
                reflection_prompt = PROMPTS["TASK_REFLECTION_PROMPT"].format(
                    task_name=task.task_name,
                    task_description=task.task_description,
                )
                reflection_res = await self.executor_agent.run(reflection_prompt)  # do not save chat history!
                self.reflection_history.append(reflection_res.final_output)
                logger.info(f"Task '{task.task_name}' reflection: {reflection_res.final_output}")

                logger.warning(f"Task '{task.task_name}' not completed. Retrying... (Attempt {tries}/{self.max_tries})")
                tries += 1

            except Exception as e:
                logger.error(f"Error executing task `{task.task_name}` on attempt {tries}: {str(e)}")
                tries += 1
                if tries > self.max_tries:
                    final_result = f"Task execution failed: {str(e)}"
                    break

        if executor_res is None:
            logger.error(f"Task `{task.task_name}` execution failed after {tries} attempts!")
            task.task_result = final_result
            task.task_status = "failed"
            return

        recorder.add_run_result(executor_res.get_run_result(), "executor")  # add executor trajectory
        task.task_result = final_result
        task.task_status = "completed"

        if self.return_summary:
            # WARNING: reset instructions is dangerous! DONOT use here!
            # self.executor_agent.set_instructions(PROMPTS["TASK_SUMMARY_SYSTEM_PROMPT"])
            summary_prompt = PROMPTS["TASK_SUMMARY_USER_PROMPT"].format(
                task_name=task.task_name,
                task_description=task.task_description,
            )
            summary_response = await self.executor_agent.run(summary_prompt)
            recorder.add_run_result(summary_response.get_run_result(), "executor_summary")  # add executor trajectory
            task.task_result_detailed, task.task_result = summary_response.final_output, summary_response.final_output
            logger.info(f"Task result summarized: {task.task_result_detailed} -> {task.task_result}")

    def _parse_task_check_result(self, response) -> bool:
        task_check_result = re.search(r"<task_check>(.*?)</task_check>", response, re.DOTALL)
        if task_check_result and task_check_result.group(1).strip().lower() == "yes":
            return True
        return False
