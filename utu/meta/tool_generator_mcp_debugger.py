"""
A naive implementation of a tool generator debugger.

- [ ] optimize the bash environment for coding
"""

import asyncio
import pathlib
from dataclasses import dataclass

from agents import RunResultStreaming, trace

from ..agents import SimpleAgent
from ..agents.common import DataClassWithStreamEvents, QueueCompleteSentinel
from ..utils import DIR_ROOT, get_logger

logger = get_logger(__name__)


@dataclass
class TaskRecorder(DataClassWithStreamEvents):
    pass


class ToolGeneratorDebugger:
    def __init__(self):
        self.llm = SimpleAgent(config="meta/tool_generator_mcp_debugger")

    # async def run(self):
    #     self.llm.clear_input_items()
    #     with trace("tool_debugger"):
    #         task_recorder = TaskRecorder()
    #         await self.test(task_recorder)

    def run_streamed(self, workspace_dir: str) -> TaskRecorder:
        task_recorder = TaskRecorder()
        task_recorder._run_impl_task = asyncio.create_task(self._start_streaming(task_recorder, workspace_dir))
        return task_recorder

    async def _start_streaming(self, task_recorder: TaskRecorder, workspace_dir: str):
        with trace("tool_debugger"):
            try:
                await self.test(task_recorder=task_recorder, workspace_dir=workspace_dir)
            except Exception as e:
                task_recorder._is_complete = True
                task_recorder._event_queue.put_nowait(QueueCompleteSentinel())
                raise e

        task_recorder._event_queue.put_nowait(QueueCompleteSentinel())
        task_recorder._is_complete = True

    async def test(self, task_recorder: TaskRecorder = None, workspace_dir: str | pathlib.Path = None):
        if isinstance(workspace_dir, str):
            workspace_dir = pathlib.Path(DIR_ROOT / "configs/tools/generated" / workspace_dir)
        assert workspace_dir.exists()
        async with self.llm as agent:
            agent._toolkits["bash"].setup_workspace(workspace_dir)
            agent._toolkits["file_edit"].setup_workspace(workspace_dir)
            res = agent.run_streamed(f"当前目录: {workspace_dir}")
            await self._process_streamed(res, task_recorder)

    async def _process_streamed(self, run_result_streaming: RunResultStreaming, task_recorder: TaskRecorder):
        async for event in run_result_streaming.stream_events():
            # logger.info(event)
            task_recorder._event_queue.put_nowait(event)
        self.llm.input_items = run_result_streaming.to_input_list()
