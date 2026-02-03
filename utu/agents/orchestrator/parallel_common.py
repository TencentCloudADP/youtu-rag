"""
Parallel Orchestrator 数据模型

定义并行编排所需的数据结构
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Any, Dict


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ParallelTask:
    """
    并行任务定义

    表示一个需要执行的子任务
    """
    agent_name: str                      # Agent 名称 (如 "ExcelQA", "KBSearch")
    task: str                            # 任务描述
    priority: int = 0                    # 优先级（资源限制时排序，数字越大优先级越高）

    # 运行时状态（由系统自动填充）
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None         # 任务结果
    error: Optional[str] = None          # 错误信息（如果失败）
    start_time: Optional[float] = None   # 开始时间
    end_time: Optional[float] = None     # 结束时间

    def __str__(self):
        return f"ParallelTask({self.agent_name}: {self.task[:50]}...)"

    def to_dict(self):
        """转换为字典"""
        return {
            "agent_name": self.agent_name,
            "task": self.task,
            "priority": self.priority,
            "status": self.status.value,
            "result": self.result,
            "error": self.error
        }


@dataclass
class ParallelGroup:
    """
    并行任务组

    组内的任务并行执行，组之间顺序执行
    """
    group_id: int                        # 组 ID
    tasks: List[ParallelTask]            # 任务列表
    description: str = ""                # 组描述（可选）

    def __str__(self):
        return f"ParallelGroup(id={self.group_id}, tasks={len(self.tasks)})"

    def to_dict(self):
        """转换为字典"""
        return {
            "group_id": self.group_id,
            "description": self.description,
            "tasks": [task.to_dict() for task in self.tasks]
        }


@dataclass
class ParallelPlan:
    """
    并行执行计划

    包含完整的任务规划信息
    """
    input: str                           # 原始用户输入
    analysis: str                        # 分析说明
    parallel_groups: List[ParallelGroup] # 并行任务组列表

    @property
    def tasks(self) -> List[ParallelTask]:
        """获取所有任务（扁平化）- 兼容 Recorder.add_plan()"""
        return self.get_all_tasks()

    def get_all_tasks(self) -> List[ParallelTask]:
        """获取所有任务（扁平化）"""
        return [task for group in self.parallel_groups for task in group.tasks]

    def get_total_task_count(self) -> int:
        """获取任务总数"""
        return len(self.get_all_tasks())

    def format_plan(self) -> str:
        """格式化计划为字符串 - 兼容 Recorder.add_plan()"""
        lines = [f"Analysis: {self.analysis}", ""]
        for group in self.parallel_groups:
            if len(self.parallel_groups) > 1:
                lines.append(f"Parallel Group {group.group_id}:")
            for i, task in enumerate(group.tasks, 1):
                lines.append(f"  {i}. [{task.agent_name}] {task.task}")
            lines.append("")
        return "\n".join(lines)

    def __str__(self):
        return (
            f"ParallelPlan(\n"
            f"  input='{self.input[:50]}...',\n"
            f"  groups={len(self.parallel_groups)},\n"
            f"  total_tasks={self.get_total_task_count()}\n"
            f")"
        )

    def to_dict(self):
        """转换为字典"""
        return {
            "input": self.input,
            "analysis": self.analysis,
            "parallel_groups": [group.to_dict() for group in self.parallel_groups]
        }


@dataclass
class ParallelOrchestratorEvent:
    """
    并行编排器事件

    用于流式输出中传递事件信息
    """
    name: str                                   # 事件名称
    agent_name: Optional[str] = None            # Agent 名称（单个任务事件）
    group_idx: Optional[int] = None             # 任务组索引
    task: Optional[ParallelTask] = None         # 任务对象
    tasks: Optional[List[ParallelTask]] = None  # 任务列表
    result: Optional[str] = None                # 结果
    data: Optional[Dict[str, Any]] = None       # 附加数据

    def to_dict(self):
        """转换为字典（用于 JSON 序列化）"""
        return {
            "type": self.name,
            "agent_name": self.agent_name,
            "group_idx": self.group_idx,
            "task": self.task.to_dict() if self.task else None,
            "tasks": [t.to_dict() for t in self.tasks] if self.tasks else None,
            "result": self.result,
            "data": self.data
        }
