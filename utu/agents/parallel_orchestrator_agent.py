"""
Parallel Orchestrator Agent - 并行编排器

支持并行执行多个不同类型的 Agents，并智能融合结果。
"""

import asyncio
import time
from typing import List, Dict, Union

from agents import trace

from ..config import AgentConfig, ConfigLoader
from ..db import DBService, TrajectoryModel
from ..utils import AgentsUtils, get_logger
from .common import QueueCompleteSentinel
from .orchestrator import Recorder
from .orchestrator.parallel import ParallelPlanner, ResultMerger
from .orchestrator.parallel_common import (
    ParallelTask,
    ParallelPlan,
    ParallelOrchestratorEvent,
    TaskStatus
)
from .simple_agent import SimpleAgent
from .orchestra_agent import OrchestraAgent
from .orchestrator_agent import OrchestratorAgent

logger = get_logger(__name__)


class ParallelOrchestratorAgent:
    """
    并行编排器 Agent

    功能:
    1. 根据用户输入创建并行执行计划
    2. 并行执行多个 sub-agent
    3. 融合结果生成最终答案
    """

    def __init__(self, config: AgentConfig):
        self._handle_config(config)

        # 初始化组件
        self.planner = ParallelPlanner(self.config)
        self.merger = ResultMerger(self.config)
        self.workers = self._setup_workers()

        # 配置参数
        self.max_parallel = config.orchestrator_config.get("max_parallel", 4)
        self.task_timeout = config.orchestrator_config.get("task_timeout", 300)  # 5分钟

        logger.info(
            f"ParallelOrchestratorAgent initialized: "
            f"max_parallel={self.max_parallel}, "
            f"workers={list(self.workers.keys())}"
        )

    @property
    def name(self) -> str:
        return self.config.orchestrator_config.get("name", "ParallelOrchestratorAgent")

    def _handle_config(self, config: AgentConfig) -> None:
        """处理配置（例如添加默认 agents）"""
        # 可选：添加默认的 chitchat agent
        add_chitchat = config.orchestrator_config.get("add_chitchat_subagent", False)
        if add_chitchat:
            config.orchestrator_workers["ChitchatAgent"] = ConfigLoader.load_agent_config("simple/chitchat")
            config.orchestrator_workers_info.append({
                "name": "ChitchatAgent",
                "description": "Engages in light, informal conversations and handles straightforward queries."
            })
        self.config = config

    def _setup_workers(self) -> Dict[str, Union[SimpleAgent, OrchestraAgent, OrchestratorAgent]]:
        """设置工作 agents"""
        workers = {}
        for name, config in self.config.orchestrator_workers.items():
            # 根据配置类型创建对应的 agent
            if config.type == "simple":
                workers[name] = SimpleAgent(config=config)
            elif config.type == "orchestra":
                workers[name] = OrchestraAgent(config=config)
            elif config.type == "orchestra_react_sql":
                # Lazy import to avoid circular dependency
                from ..rag.rag_agents.orchestra_react_text2sql import OrchestraReactSqlAgent
                workers[name] = OrchestraReactSqlAgent(config=config)
            elif config.type == "orchestrator":
                workers[name] = OrchestratorAgent(config=config)
            else:
                raise ValueError(f"Unsupported worker agent type: {config.type} for {name}")
            logger.debug(f"Worker '{name}' ({config.type}) initialized")
        return workers

    async def run(self, input: str, history: Recorder = None, trace_id: str = None) -> Recorder:
        """
        运行（非流式）

        Args:
            input: 用户输入
            history: 历史记录器（可选）
            trace_id: 追踪ID（可选）

        Returns:
            Recorder: 完成的记录器
        """
        recorder = self.run_streamed(input, history, trace_id)
        async for _ in recorder.stream_events():
            pass
        return recorder

    def run_streamed(self, input: str, history: Recorder = None, trace_id: str = None) -> Recorder:
        """
        流式运行

        Args:
            input: 用户输入
            history: 历史记录器（可选）
            trace_id: 追踪ID（可选）

        Returns:
            Recorder: 记录器（可以流式读取事件）
        """
        trace_id = trace_id or AgentsUtils.gen_trace_id()

        if history:
            recorder = history.new(input=input, trace_id=trace_id)
        else:
            recorder = Recorder(input=input, trace_id=trace_id)

        recorder._run_impl_task = asyncio.create_task(self._start_streaming(recorder))
        return recorder

    async def _start_streaming(self, recorder: Recorder):
        """
        主执行流程

        流程:
        1. 创建并行计划
        2. 按组执行任务（组内并行，组间顺序）
        3. 融合结果
        """
        with trace(workflow_name=self.name, trace_id=recorder.trace_id):
            try:
                logger.info(f"Starting parallel orchestration for: {recorder.input}")

                # 1. 创建并行计划
                recorder._event_queue.put_nowait(
                    ParallelOrchestratorEvent(name="plan.start")
                )

                plan = await self.planner.create_parallel_plan(recorder)
                recorder.add_plan(plan)

                recorder._event_queue.put_nowait(
                    ParallelOrchestratorEvent(
                        name="plan.done",
                        data={"plan": plan.to_dict()}
                    )
                )

                # 2. 按组执行任务
                all_results = []
                for group in plan.parallel_groups:
                    logger.info(
                        f"Executing group {group.group_id} with {len(group.tasks)} tasks"
                    )

                    # 发送任务组开始事件
                    recorder._event_queue.put_nowait(
                        ParallelOrchestratorEvent(
                            name="parallel_group.start",
                            group_idx=group.group_id,
                            tasks=group.tasks
                        )
                    )

                    # 并行执行该组的任务
                    results = await self._run_parallel_tasks(recorder, group.tasks)
                    all_results.extend(results)

                    # 发送任务组完成事件
                    recorder._event_queue.put_nowait(
                        ParallelOrchestratorEvent(
                            name="parallel_group.done",
                            group_idx=group.group_id
                        )
                    )

                # 3. 融合结果
                logger.info("Merging results from all tasks")
                recorder._event_queue.put_nowait(
                    ParallelOrchestratorEvent(name="merge.start")
                )

                final_output = await self.merger.merge_results(recorder, all_results)
                recorder.add_final_output(final_output)

                recorder._event_queue.put_nowait(
                    ParallelOrchestratorEvent(
                        name="merge.done",
                        result=final_output
                    )
                )

                # 记录到数据库
                DBService.add(TrajectoryModel.from_task_recorder(recorder))

                logger.info("Parallel orchestration completed successfully")

            except Exception as e:
                logger.error(f"Error in parallel orchestration: {str(e)}", exc_info=True)
                recorder._event_queue.put_nowait(
                    ParallelOrchestratorEvent(
                        name="error",
                        data={"error": str(e)}
                    )
                )
                raise
            finally:
                recorder._event_queue.put_nowait(QueueCompleteSentinel())
                recorder._is_complete = True

    async def _run_parallel_tasks(
        self,
        recorder: Recorder,
        tasks: List[ParallelTask]
    ) -> List[ParallelTask]:
        """
        并行执行一组任务

        使用 semaphore 控制最大并发数

        Args:
            recorder: 记录器
            tasks: 任务列表

        Returns:
            List[ParallelTask]: 执行完成的任务列表
        """
        # 创建信号量限制并发
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def _run_single_task(task: ParallelTask) -> ParallelTask:
            """执行单个任务"""
            async with semaphore:
                try:
                    logger.info(f"Starting task: {task.agent_name} - {task.task[:50]}")

                    # 1. 获取 worker
                    worker = self.workers[task.agent_name]

                    # 2. 初始化 worker（仅 SimpleAgent 需要调用 build）
                    # OrchestraAgent 的 workers 会在 run_streamed() 内部自动初始化
                    if isinstance(worker, SimpleAgent):
                        await worker.build()

                    # 3. 更新任务状态
                    task.status = TaskStatus.RUNNING
                    task.start_time = time.time()

                    # 发送任务开始事件
                    recorder._event_queue.put_nowait(
                        ParallelOrchestratorEvent(
                            name="parallel_task.start",
                            agent_name=task.agent_name,
                            task=task
                        )
                    )

                    # 4. 构建任务上下文
                    task_input = self._build_task_context(recorder, task)

                    # 5. 运行任务（流式输出）
                    result = worker.run_streamed(task_input)

                    # 流式转发事件（添加 agent_name 标签）
                    async for event in result.stream_events():
                        # 关键：为事件添加 agent_name 标签，前端用于路由到对应窗口
                        if hasattr(event, '__dict__'):
                            event.agent_name = task.agent_name
                        recorder._event_queue.put_nowait(event)

                    # 6. 保存结果
                    task.result = result.final_output
                    task.status = TaskStatus.COMPLETED
                    task.end_time = time.time()

                    duration = task.end_time - task.start_time
                    logger.info(
                        f"Task completed: {task.agent_name} "
                        f"(duration: {duration:.2f}s)"
                    )

                    # 7. 发送任务完成事件
                    recorder._event_queue.put_nowait(
                        ParallelOrchestratorEvent(
                            name="parallel_task.done",
                            agent_name=task.agent_name,
                            task=task,
                            result=task.result
                        )
                    )

                except asyncio.TimeoutError:
                    logger.error(f"Task timeout: {task.agent_name}")
                    task.status = TaskStatus.FAILED
                    task.error = f"Task timed out after {self.task_timeout}s"

                    recorder._event_queue.put_nowait(
                        ParallelOrchestratorEvent(
                            name="parallel_task.error",
                            agent_name=task.agent_name,
                            task=task,
                            data={"error": task.error}
                        )
                    )

                except Exception as e:
                    logger.error(
                        f"Task failed: {task.agent_name} - {str(e)}",
                        exc_info=True
                    )
                    task.status = TaskStatus.FAILED
                    task.error = str(e)

                    recorder._event_queue.put_nowait(
                        ParallelOrchestratorEvent(
                            name="parallel_task.error",
                            agent_name=task.agent_name,
                            task=task,
                            data={"error": str(e)}
                        )
                    )

                return task

        # 并行执行所有任务（带超时）
        try:
            results = await asyncio.gather(
                *[
                    asyncio.wait_for(
                        _run_single_task(task),
                        timeout=self.task_timeout
                    )
                    for task in tasks
                ],
                return_exceptions=True
            )
        except Exception as e:
            logger.error(f"Error in parallel execution: {str(e)}")
            results = tasks  # 返回原始任务列表

        # 过滤掉异常
        valid_results = [r for r in results if isinstance(r, ParallelTask)]

        # 统计结果
        completed = sum(1 for t in valid_results if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in valid_results if t.status == TaskStatus.FAILED)
        logger.info(
            f"Parallel tasks finished: {completed} completed, {failed} failed"
        )

        return valid_results

    def _build_task_context(self, recorder: Recorder, task: ParallelTask) -> str:
        """
        为任务构建上下文

        包括:
        - 原始问题（保留KB/文件上下文信息）
        - 执行计划
        - 当前任务的具体描述

        Args:
            recorder: 记录器
            task: 任务对象

        Returns:
            str: 任务上下文字符串
        """
        # 提取原始输入中的KB/文件上下文信息
        # 格式: [Knowledge Base ID: xxx]\n[Selected Files: xxx]\n...User Question: xxx
        kb_context = ""
        original_input = recorder.input

        # 检查是否包含 KB/文件上下文信息
        if "[Knowledge Base ID:" in original_input or "[Selected Files:" in original_input:
            # 提取上下文信息（到 "User Question:" 之前）
            if "User Question:" in original_input:
                parts = original_input.split("User Question:", 1)
                kb_context = parts[0].strip()
                # 保留原始问题（去掉KB上下文）
                original_question = parts[1].strip()
            else:
                original_question = original_input
        else:
            original_question = original_input

        # 构建任务输入
        context_parts = []

        # 添加KB/文件上下文（如果存在）
        if kb_context:
            context_parts.append(kb_context)

        # 添加整体问题和任务描述
        context_parts.append(f"""# Overall Question
{original_question}

# Your Task
As {task.agent_name}, you need to: {task.task}

Please complete your task and provide a detailed result.""")

        return "\n\n".join(context_parts)
