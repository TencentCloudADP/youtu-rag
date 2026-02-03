"""
ADG Benchmark Runner V4 - 分步执行 + 丰富上下文
"""

import os
import sys
import json
import datetime
import argparse
import yaml
import asyncio
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
from threading import Lock
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.model.adg_benchmark.adg_workflow_v4 import build_adg_workflow_v4, ADGWorkflowStateV4
from utils.logger import logger

from utu.agents.common import TaskRecorder, QueueCompleteSentinel
from utu.tools.memory_toolkit import VectorMemoryToolkit


@dataclass
class ExcelAgentStreamEvent:
    """ExcelAgent 流式事件"""
    name: Literal[
        "excel_agent.plan.start",
        "excel_agent.plan.delta",
        "excel_agent.plan.done",
        "excel_agent.task.start",
        "excel_agent.task.delta",
        "excel_agent.task.done",
        "excel_agent.answer.start",
        "excel_agent.answer.delta",
        "excel_agent.answer.done",
    ]
    item: dict | None = None
    type: Literal["excel_agent_stream_event"] = "excel_agent_stream_event"


@dataclass
class ExcelAgentRecorder(TaskRecorder):
    """用于记录和流式传输 ExcelAgent 的执行结果
    
    继承自 TaskRecorder，保持与其他 Agent 接口一致
    """
    # ExcelAgent 特有字段
    question_type: str = ""
    execution_trace: list = field(default_factory=list)


class ExcelAgent:

    def __init__(self, config):
        self.config = self._load_config(config)
        # workflow 和 instance 将在运行时根据 event_callback 创建
        self.workflow = None
        self.instance = None
        self._memory_toolkit = None

    def set_memory_toolkit(self, memory_toolkit: "VectorMemoryToolkit") -> None:
        """Set the memory toolkit for this agent.

        Args:
            memory_toolkit: VectorMemoryToolkit instance for memory operations.
        """
        self._memory_toolkit = memory_toolkit

    @property
    def memory_toolkit(self) -> "VectorMemoryToolkit | None":
        """Get the memory toolkit if set."""
        return self._memory_toolkit

    def _load_config(self, config):
        with open(config, 'r') as f:
            return yaml.safe_load(f)
    
    def _build_workflow(self, event_callback=None):
        """构建 workflow，支持传入事件回调"""
        self.workflow, self.instance = build_adg_workflow_v4(
            data_sample_rows=self.config['config']['data_sample_rows'],
            event_callback=event_callback
        )

    async def run(self, input, question_type=None):
        """运行查询（异步版本）"""
        recorder = self.run_streamed(input, question_type)
        async for _ in recorder.stream_events():
            pass
        return {
            'question': recorder.task,
            'FileName': self._get_file_name(),
            'model_answer': recorder.final_output,
        }

    def run_streamed(self, input, question_type=None, use_memory: bool = True) -> ExcelAgentRecorder:
        """流式运行查询"""
        recorder = ExcelAgentRecorder(task=input, question_type=question_type, trace_id="")
        recorder._run_impl_task = asyncio.create_task(self._start_streaming(recorder, use_memory=use_memory))
        return recorder
    
    def _get_file_name(self):
        """获取文件名"""
        file_path = os.environ.get("FILE_PATH", None)
        if file_path:
            return Path(file_path.split(",")[0]).stem
        return "unknown"
    
    async def _start_streaming(self, recorder: ExcelAgentRecorder, use_memory: bool = True):
        """异步执行流程"""
        try:
            # 从环境变量读取全局配置，覆盖传入参数
            env_memory_setting = os.environ.get("memoryEnabled", "false").lower() == "true"
            use_memory = env_memory_setting
            logger.info(f"[ExcelAgent] use_memory from env: {use_memory}")
            logger.info(f"[ExcelAgent] self._memory_toolkit: {self._memory_toolkit}")
            question = recorder.task
            question_type = recorder.question_type
            original_question = question  # 保存原始问题用于 episodic memory

            if use_memory:
                logger.info(f"[ExcelAgent] use_memory: {use_memory}")

            if use_memory and self._memory_toolkit:
                await self._memory_toolkit.store_working_memory(question, role="user")
                logger.debug("Stored user question to working memory")

                # 使用统一的 memory 检索方法
                memory_contexts = await self._memory_toolkit.retrieve_all_context(
                    query=question,
                    include_skills=False,
                )
                memory_context = memory_contexts["memory_context"]

                if memory_context:
                    logger.info(f"Retrieved memory context: {len(memory_context)} chars")

                enhanced_input = f"# 相关历史上下文\n{memory_context}\n\n---\n# 当前问题\n{question}"
                recorder.task = enhanced_input
                question = enhanced_input

            file_path = os.environ.get("FILE_PATH", None)
            file_path = Path(file_path.split(",")[0])
            file_name = file_path.stem
            
            # 获取当前事件循环，供回调函数使用
            current_loop = asyncio.get_running_loop()
            
            # 定义线程安全的事件回调函数
            def event_callback(name: str, event_data: dict):
                """接收来自 workflow 的事件并转发（线程安全）"""
                try:
                    event = ExcelAgentStreamEvent(
                        name=name,
                        item=event_data
                    )
                    # 使用 call_soon_threadsafe 确保跨线程安全
                    current_loop.call_soon_threadsafe(
                        recorder._event_queue.put_nowait,
                        event
                    )
                    logger.debug(f"Event sent from thread: {event_data.get('step', 'unknown')}")
                except Exception as e:
                    logger.warning(f"Failed to send event from callback: {e}")
            
            # 构建 workflow（带事件回调）
            self._build_workflow(event_callback=event_callback)

            initial_state = ADGWorkflowStateV4(
                user_question=question,
                question_type=question_type,
                current_table={
                    "file_name": f"{file_path.name}",
                    "file_path": str(file_path)
                },
            )
            config = {"recursion_limit": 60, "configurable": {"thread_id": "1"}}

            # 发送开始事件
            recorder._event_queue.put_nowait(
                ExcelAgentStreamEvent(
                    name="excel_agent.plan.start",
                    item={
                        "question": question,
                        "file_path": str(file_path)
                    }
                )
            )

            try:
                # 在线程池中执行同步的 workflow.invoke
                final_state = await current_loop.run_in_executor(
                    None, 
                    lambda: self.workflow.invoke(initial_state, config=config)
                )
                final_answer = final_state.get("final_answer", "")
                execution_trace = final_state.get("execution_trace", [])
            except Exception as invoke_e:
                if "Recursion limit" in str(invoke_e):
                    logger.warning(f"Query hit recursion limit.")
                    final_answer = "[Final Answer]: Processing timeout"
                    execution_trace = []
                else:
                    raise invoke_e
            
            # Visualization 处理
            if question_type == 'Visualization':
                all_code = "\n".join([t.get("code", "") for t in execution_trace if t.get("code")])
                if "plt." not in final_answer and "plt." in all_code:
                    final_answer = all_code
                if "plt.show()" not in final_answer:
                    final_answer += "\nplt.show()"
            
            # 更新 recorder
            recorder.final_output = final_answer
            recorder.execution_trace = execution_trace

            final_output = str(recorder.final_output or "")
            logger.debug(f"Final output: {final_output}")

            # if use_memory and self._memory_toolkit:
            #     # 存储 working memory
            #     await self._memory_toolkit.store_working_memory(final_output, role="assistant")
            #     logger.debug("Saved model output to memory")

            # 存储到 Memory（包括 episodic memory）
            if use_memory and self._memory_toolkit:
                try:
                    # 存储 working memory
                    await self._memory_toolkit.store_working_memory(final_output, role="assistant")
                    
                    # 存储到 episodic memory（持久化）
                    # 恢复原始问题（去除上下文注入部分）
                    clean_question = original_question
                    if "\n# 当前问题\n" in str(recorder.task):
                        clean_question = str(recorder.task).split("\n# 当前问题\n")[-1]
                    
                    await self._memory_toolkit.save_conversation_to_episodic(
                        question=clean_question,
                        answer=final_output,
                        importance_score=0.6,  # Excel 分析通常比较重要
                    )
                    logger.debug("Saved conversation to episodic memory")
                except Exception as e:
                    logger.warning(f"Memory storage error: {e}")
            
            # 发送完成事件
            recorder._event_queue.put_nowait(
                ExcelAgentStreamEvent(
                    name="excel_agent.answer.delta",
                    item={
                        "type": "answer_generation",
                        "content": recorder.final_output
                    }
                )
            )
            
        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            recorder._event_queue.put_nowait(QueueCompleteSentinel())
            recorder._is_complete = True
            raise e
        finally:
            recorder._event_queue.put_nowait(QueueCompleteSentinel())
            recorder._is_complete = True

if __name__ == "__main__":

    async def main():
        os.environ["FILE_PATH"] = "ADG/benchmarks/realhitbench/tables/employment-table01.xlsx"
        query = "What was the employment in non-agricultural industries in the year with the highest civilian labor force total?"
        agent = ExcelAgent(config="configs/agents/ragref/excel/excel.yaml")
        
        # 使用异步流式调用
        rec = agent.run_streamed(query)
        async for event in rec.stream_events():
            print(f"Event: {event}")
        
        print(f"\nFinal Answer:\n{rec.final_output}")
    
    asyncio.run(main())



# """
# ADG Benchmark Runner V4 - 分步执行 + 丰富上下文
# """

# import os
# import sys
# import json
# import datetime
# import argparse
# import yaml
# import asyncio
# from tqdm import tqdm
# from pathlib import Path
# import concurrent.futures
# from threading import Lock
# from dataclasses import dataclass, field
# from typing import Any, AsyncIterator, Literal

# project_root = Path(__file__).parent.parent.parent.parent
# sys.path.append(str(project_root))

# from src.model.adg_benchmark.adg_workflow_v4 import build_adg_workflow_v4, ADGWorkflowStateV4
# from utils.logger import logger

# from utu.agents.common import TaskRecorder


# class QueueCompleteSentinel:
#     pass


# @dataclass
# class ExcelAgentStreamEvent:
#     """ExcelAgent 流式事件"""
#     name: Literal[
#         "excel_agent.plan.start",
#         "excel_agent.plan.delta",
#         "excel_agent.plan.done",
#         "excel_agent.task.start",
#         "excel_agent.task.delta",
#         "excel_agent.task.done",
#         "excel_agent.answer.start",
#         "excel_agent.answer.delta",
#         "excel_agent.answer.done",
#     ]
#     item: dict | None = None
#     type: Literal["excel_agent_stream_event"] = "excel_agent_stream_event"


# @dataclass
# class ExcelAgentRecorder(TaskRecorder):
#     """用于记录和流式传输 ExcelAgent 的执行结果
    
#     继承自 TaskRecorder，保持与其他 Agent 接口一致
#     """
#     # ExcelAgent 特有字段
#     question_type: str = ""
#     execution_trace: list = field(default_factory=list)
    
#     async def stream_events(self) -> AsyncIterator[ExcelAgentStreamEvent]:
#         """流式输出事件"""
#         while True:
#             self._check_errors()
#             if self._stored_exception:
#                 logger.debug("Breaking due to stored exception")
#                 self._is_complete = True
#                 break

#             if self._is_complete and self._event_queue.empty():
#                 break

#             try:
#                 item = await self._event_queue.get()
#             except asyncio.CancelledError:
#                 logger.debug("Breaking due to asyncio.CancelledError")
#                 break

#             if isinstance(item, QueueCompleteSentinel):
#                 self._event_queue.task_done()
#                 self._check_errors()
#                 break

#             yield item
#             self._event_queue.task_done()

#         self._cleanup_tasks()

#         if self._stored_exception:
#             raise self._stored_exception


# class ExcelAgent:

#     def __init__(self, config):
#         self.config = self._load_config(config)
#         # workflow 和 instance 将在运行时根据 event_callback 创建
#         self.workflow = None
#         self.instance = None

#     def _load_config(self, config):
#         with open(config, 'r') as f:
#             return yaml.safe_load(f)
    
#     def _build_workflow(self, event_callback=None):
#         """构建 workflow，支持传入事件回调"""
#         self.workflow, self.instance = build_adg_workflow_v4(
#             data_sample_rows=self.config['config']['data_sample_rows'],
#             event_callback=event_callback
#         )

#     async def run(self, input, question_type=None):
#         """运行查询（异步版本）"""
#         recorder = self.run_streamed(input, question_type)
#         async for _ in recorder.stream_events():
#             pass
#         return {
#             'question': recorder.task,
#             'FileName': self._get_file_name(),
#             'model_answer': recorder.final_output,
#         }

#     def run_streamed(self, input, question_type=None) -> ExcelAgentRecorder:
#         """流式运行查询"""
#         recorder = ExcelAgentRecorder(task=input, question_type=question_type, trace_id="")
#         recorder._run_impl_task = asyncio.create_task(self._start_streaming(recorder))
#         return recorder
    
#     def _get_file_name(self):
#         """获取文件名"""
#         file_path = os.environ.get("FILE_PATH", None)
#         if file_path:
#             return Path(file_path.split(",")[0]).stem
#         return "unknown"
    
#     async def _start_streaming(self, recorder: ExcelAgentRecorder):
#         """异步执行流程"""
#         try:
#             question = recorder.task
#             question_type = recorder.question_type

#             file_path = os.environ.get("FILE_PATH", None)
#             file_path = Path(file_path.split(",")[0])
#             file_name = file_path.stem
            
#             # 获取当前事件循环，供回调函数使用
#             current_loop = asyncio.get_running_loop()
            
#             # 定义线程安全的事件回调函数
#             def event_callback(name: str, event_data: dict):
#                 """接收来自 workflow 的事件并转发（线程安全）"""
#                 try:
#                     event = ExcelAgentStreamEvent(
#                         name=name,
#                         item=event_data
#                     )
#                     # 使用 call_soon_threadsafe 确保跨线程安全
#                     current_loop.call_soon_threadsafe(
#                         recorder._event_queue.put_nowait,
#                         event
#                     )
#                     logger.debug(f"Event sent from thread: {event_data.get('step', 'unknown')}")
#                 except Exception as e:
#                     logger.warning(f"Failed to send event from callback: {e}")
            
#             # 构建 workflow（带事件回调）
#             self._build_workflow(event_callback=event_callback)

#             initial_state = ADGWorkflowStateV4(
#                 user_question=question,
#                 question_type=question_type,
#                 current_table={
#                     "file_name": f"{file_path.name}",
#                     "file_path": str(file_path)
#                 },
#             )
#             config = {"recursion_limit": 60, "configurable": {"thread_id": "1"}}

#             # 发送开始事件
#             recorder._event_queue.put_nowait(
#                 ExcelAgentStreamEvent(
#                     name="excel_agent.plan.start",
#                     item={
#                         "question": question,
#                         "file_path": str(file_path)
#                     }
#                 )
#             )

#             try:
#                 # 在线程池中执行同步的 workflow.invoke
#                 final_state = await current_loop.run_in_executor(
#                     None, 
#                     lambda: self.workflow.invoke(initial_state, config=config)
#                 )
#                 final_answer = final_state.get("final_answer", "")
#                 execution_trace = final_state.get("execution_trace", [])
#             except Exception as invoke_e:
#                 if "Recursion limit" in str(invoke_e):
#                     logger.warning(f"Query hit recursion limit.")
#                     final_answer = "[Final Answer]: Processing timeout"
#                     execution_trace = []
#                 else:
#                     raise invoke_e
            
#             # Visualization 处理
#             if question_type == 'Visualization':
#                 all_code = "\n".join([t.get("code", "") for t in execution_trace if t.get("code")])
#                 if "plt." not in final_answer and "plt." in all_code:
#                     final_answer = all_code
#                 if "plt.show()" not in final_answer:
#                     final_answer += "\nplt.show()"
            
#             # 更新 recorder
#             recorder.final_output = final_answer
#             recorder.execution_trace = execution_trace
            
#             # 发送完成事件
#             recorder._event_queue.put_nowait(
#                 ExcelAgentStreamEvent(
#                     name="excel_agent.answer.delta",
#                     item={
#                         "type": "answer_generation",
#                         "content": recorder.final_output
#                     }
#                 )
#             )
            
#         except Exception as e:
#             logger.error(f"Error processing task: {str(e)}")
#             recorder._event_queue.put_nowait(QueueCompleteSentinel())
#             recorder._is_complete = True
#             raise e
#         finally:
#             recorder._event_queue.put_nowait(QueueCompleteSentinel())
#             recorder._is_complete = True

# if __name__ == "__main__":

#     async def main():
#         os.environ["FILE_PATH"] = "ADG/benchmarks/realhitbench/tables/employment-table01.xlsx"
#         query = "What was the employment in non-agricultural industries in the year with the highest civilian labor force total?"
#         agent = ExcelAgent(config="configs/agents/ragref/excel/excel.yaml")
        
#         # 使用异步流式调用
#         rec = agent.run_streamed(query)
#         async for event in rec.stream_events():
#             print(f"Event: {event}")
        
#         print(f"\nFinal Answer:\n{rec.final_output}")
    
#     asyncio.run(main())