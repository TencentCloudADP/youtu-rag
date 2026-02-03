"""
Parallel Orchestrator - 并行任务规划和结果融合

包含:
- ParallelPlanner: 并行任务规划器
- ResultMerger: 结果融合器
"""

import json
import re
from typing import List

from ...config import AgentConfig
from ...utils import FileUtils, get_logger
from ..llm_agent import LLMAgent
from .parallel_common import ParallelTask, ParallelGroup, ParallelPlan, TaskStatus

logger = get_logger(__name__)


class ParallelPlanner:
    """
    并行任务规划器

    负责分析用户输入，决定:
    1. 需要调用哪些 agents
    2. 哪些任务可以并行
    3. 哪些任务需要顺序执行
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.prompts = FileUtils.load_prompts("agents/orchestrator/parallel.yaml")

    async def create_parallel_plan(self, recorder) -> ParallelPlan:
        """
        创建并行执行计划

        Args:
            recorder: 记录器对象（包含用户输入和历史）

        Returns:
            ParallelPlan: 并行执行计划
        """
        logger.info("Creating parallel execution plan...")

        # 1. 构建 system prompt
        sp = self._build_system_prompt()

        # 2. 构建 user prompt
        up = self._build_user_prompt(recorder)

        # 3. 调用 LLM 生成计划
        llm = LLMAgent(
            name="parallel_planner",
            instructions=sp,
            model_config=self.config.orchestrator_model,
        )

        logger.debug(f"Planning prompt: {up[:200]}...")
        result = await llm.run(up)
        plan_text = result.final_output

        # 4. 解析计划
        plan = self._parse_parallel_plan(plan_text, recorder)

        logger.info(
            f"Plan created: {len(plan.parallel_groups)} groups, "
            f"{plan.get_total_task_count()} total tasks"
        )
        logger.debug(f"Plan: {plan}")

        return plan

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        # 使用 Jinja2 模板渲染
        template = FileUtils.get_jinja_template_str(self.prompts["planner_sp"])
        return template.render(
            available_agents=self.config.orchestrator_workers_info
        )

    def _build_user_prompt(self, recorder) -> str:
        """构建用户提示词"""
        additional_instructions = self.config.orchestrator_config.get(
            "additional_instructions", ""
        )

        # 获取历史消息（如果有）
        history_messages = getattr(recorder, 'history_messages', None)

        template = FileUtils.get_jinja_template_str(self.prompts["planner_up"])
        return template.render(
            additional_instructions=additional_instructions,
            question=recorder.input,
            history_messages=history_messages
        )

    def _parse_parallel_plan(self, text: str, recorder) -> ParallelPlan:
        """
        解析 LLM 输出为结构化的 ParallelPlan

        期望格式:
        <analysis>...</analysis>
        <plan>
        {
            "parallel_groups": [
                [
                    {"name": "ExcelQA", "task": "...", "priority": 1}
                ]
            ]
        }
        </plan>

        Args:
            text: LLM 输出文本
            recorder: 记录器对象

        Returns:
            ParallelPlan: 解析后的计划

        Raises:
            ValueError: 如果解析失败
        """
        # 提取 analysis
        analysis_match = re.search(r"<analysis>(.*?)</analysis>", text, re.DOTALL)
        analysis = analysis_match.group(1).strip() if analysis_match else ""

        # 提取 plan
        plan_match = re.search(r"<plan>(.*?)</plan>", text, re.DOTALL)
        if not plan_match:
            logger.error(f"Failed to parse plan from LLM output: {text}")
            raise ValueError("Failed to parse plan from LLM output")

        # 解析 JSON
        plan_json_str = plan_match.group(1).strip()
        try:
            plan_json = json.loads(plan_json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in plan: {plan_json_str}")
            raise ValueError(f"Invalid JSON in plan: {e}")

        # 构建 ParallelPlan
        parallel_groups = []
        for group_idx, group_data in enumerate(plan_json["parallel_groups"]):
            tasks = []
            for task_def in group_data:
                task = ParallelTask(
                    agent_name=task_def["name"],
                    task=task_def["task"],
                    priority=task_def.get("priority", 0)
                )
                tasks.append(task)

            parallel_groups.append(ParallelGroup(
                group_id=group_idx,
                tasks=tasks
            ))

        return ParallelPlan(
            input=recorder.input,
            analysis=analysis,
            parallel_groups=parallel_groups
        )


class ResultMerger:
    """
    结果融合器

    负责将多个 agent 的结果合并为最终答案
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.prompts = FileUtils.load_prompts("agents/orchestrator/parallel.yaml")

    async def merge_results(
        self,
        recorder,
        tasks: List[ParallelTask]
    ) -> str:
        """
        融合多个任务的结果

        Args:
            recorder: 记录器对象
            tasks: 已完成的任务列表

        Returns:
            str: 融合后的最终结果
        """
        logger.info(f"Merging results from {len(tasks)} tasks...")

        # 1. 过滤成功完成的任务
        successful_tasks = [
            t for t in tasks
            if t.status == TaskStatus.COMPLETED and t.result
        ]

        if not successful_tasks:
            logger.warning("No successful tasks to merge")
            return "所有任务都失败了，无法生成结果。"

        logger.info(f"Merging {len(successful_tasks)} successful results")

        # 2. 构建融合 prompt
        merge_prompt = self._build_merge_prompt(recorder, successful_tasks)

        # 3. 调用 LLM 融合
        merge_instructions = self.prompts["merge_instructions"]
        merger = LLMAgent(
            name="result_merger",
            instructions=merge_instructions,
            model_config=self.config.orchestrator_model
        )

        logger.debug(f"Merge prompt: {merge_prompt[:200]}...")
        result = await merger.run(merge_prompt)
        merged_output = result.final_output

        logger.info("Results merged successfully")
        logger.debug(f"Merged result: {merged_output[:200]}...")

        return merged_output

    def _build_merge_prompt(
        self,
        recorder,
        tasks: List[ParallelTask]
    ) -> str:
        """构建融合提示词"""
        # 格式化每个任务的结果
        results_text = []
        for task in tasks:
            results_text.append(
                f"### {task.agent_name}\n"
                f"任务: {task.task}\n"
                f"结果:\n{task.result}\n"
            )

        results_combined = "\n".join(results_text)

        return f"""## Original Question
{recorder.input}

## Agent Results
{results_combined}

## Your Task
Please synthesize these results into a coherent, comprehensive answer following the guidelines in your instructions.
"""
