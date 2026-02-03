"""
MCTS规划器（简化版 - 基于Dijkstra算法）
在ADG上进行路径规划，找到成本最优路径
"""

import heapq
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
import sys
from pathlib import Path

# 修正导入路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from model.adg_benchmark.adg.macro_operations import MacroOperation, OPERATIONS_BY_NAME
from model.adg_benchmark.adg.adg_generator import TaskPath
from utils.logger import logger


# ==============================
# Planning state
# ==============================

@dataclass(frozen=True, order=True)
class PlanningState:
    """规划状态（不可变以提升哈希性能）"""
    completed_operations: Tuple[str, ...] = field(default_factory=tuple)
    available_state: Tuple[str, ...] = field(default_factory=tuple)
    accumulated_cost: float = 0.0

    def extend(self, op_name: str, op_cost: float, new_state_keys: List[str]):
        """生成新状态（代替 copy+修改方式，更快）"""
        return PlanningState(
            completed_operations=self.completed_operations + (op_name,),
            available_state=tuple(sorted(set(self.available_state) | set(new_state_keys))),
            accumulated_cost=self.accumulated_cost + op_cost
        )


@dataclass(order=True)
class PathNode:
    """路径节点（用于Dijkstra）"""
    cost: float
    state: PlanningState


# ==============================
# MCTS Planner (Dijkstra-based)
# ==============================

class MCTSPlanner:
    """MCTS规划器（Dijkstra版本）"""
    
    def __init__(self):
        logger.info("MCTS Planner (Dijkstra-based) initialized")

    # -------- Utility --------

    def _can_execute_operation(self, op: MacroOperation, available_state: Tuple[str, ...]) -> bool:
        available = set(available_state)
        return all(pre.key in available for pre in op.preconditions)

    def _state_key(self, state: PlanningState) -> Tuple:
        """唯一键（tuple 更快且无需 join）"""
        return (state.completed_operations, state.available_state)

    # -------- Next Ops --------

    def _get_next_operations(
        self, current_state: PlanningState, target_ops: List[str]
    ) -> List[Tuple[MacroOperation, float]]:
        ops = []
        done = set(current_state.completed_operations)
        
        for name in target_ops:
            if name in done:
                continue
            
            op = OPERATIONS_BY_NAME.get(name)
            # op.postconditions 是 List[str]，不需要 .key
            if op and self._can_execute_operation(op, current_state.available_state):
                ops.append((op, op.estimated_cost))
                
        return ops

    # -------- Main Search --------

    def find_optimal_path(
        self, candidate_paths: List[TaskPath], initial_state: Set[str] = None
    ) -> TaskPath:
        if not candidate_paths:
            return TaskPath(operations=[], estimated_cost=0.0)
        
        if len(candidate_paths) == 1:
            return candidate_paths[0]
            
        if initial_state is None:
            initial_state = {"file_loaded"}
            
        best_path, best_cost = None, float("inf")
        
        for idx, path in enumerate(candidate_paths, 1):
            logger.info(f"Evaluating candidate path {idx}: {path.operations}")
            cost, exec_ops = self._dijkstra_search(path.operations, initial_state)
            
            if exec_ops and cost < best_cost:
                best_cost = cost
                best_path = TaskPath(
                    operations=exec_ops,
                    estimated_cost=best_cost,
                    reasoning=path.reasoning + f" (actual cost: {best_cost:.2f})",
                )
        
        if best_path is None:
            logger.warning("All candidate paths invalid, returning the first one")
            return candidate_paths[0]
            
        return best_path

    # -------- Dijkstra --------

    def _dijkstra_search(
        self, target_ops: List[str], initial_state: Set[str]
    ) -> Tuple[float, List[str]]:
        start = PlanningState(
            completed_operations=tuple(),
            available_state=tuple(sorted(initial_state)),
            accumulated_cost=0.0,
        )
        
        pq = [PathNode(cost=0.0, state=start)]
        visited = set()
        dist = {self._state_key(start): 0.0}
        
        best_partial = start
        
        while pq:
            node = heapq.heappop(pq)
            state = node.state
            
            key = self._state_key(state)
            if key in visited:
                continue
            visited.add(key)
            
            # 完成所有操作
            if set(state.completed_operations) == set(target_ops):
                return node.cost, list(state.completed_operations)
            
            # 更新最佳部分状态（完成最多操作）
            if len(state.completed_operations) > len(best_partial.completed_operations):
                best_partial = state
            
            for op, cost in self._get_next_operations(state, target_ops):
                # 修正：op.postconditions 已经是 List[str]
                new_state = state.extend(
                    op.name, cost, op.postconditions
                )
                new_key = self._state_key(new_state)
                new_cost = node.cost + cost
                
                if new_key not in dist or new_cost < dist[new_key]:
                    dist[new_key] = new_cost
                    heapq.heappush(pq, PathNode(cost=new_cost, state=new_state))
        
        # 返回尽量多的 partial path
        # Fallback: 如果找不到完整路径，尝试返回最长的前缀路径
        # 但原来的 fallback 逻辑是返回 target_ops（忽略前置条件）。
        # 这里我们保留 Dijkstra 的结果，但也添加一个 check
        
        final_ops = list(best_partial.completed_operations)
        final_cost = sum(
            OPERATIONS_BY_NAME[op].estimated_cost 
            for op in final_ops 
            if op in OPERATIONS_BY_NAME
        )
        
        # 恢复之前的 Fallback 逻辑：如果找到的路径比目标短太多，且看起来无法继续
        # 我们可以考虑是否信任 LLM 的原始路径。
        # 但这个 Dijkstra 是基于 preconditions 的严格搜索。
        # 如果严格搜索失败，我们应该返回什么？
        # 原代码逻辑：
        # logger.info("Fallback: Returning original candidate path (ignoring preconditions)")
        # return fallback_cost, target_operations
        
        if len(final_ops) < len(target_ops):
            logger.warning("Could not find complete path satisfying all preconditions")
            logger.info("Fallback: Returning original candidate path (ignoring preconditions)")
            fallback_cost = sum(
                OPERATIONS_BY_NAME[op].estimated_cost 
                for op in target_ops 
                if op in OPERATIONS_BY_NAME
            )
            return fallback_cost, target_ops
            
        return final_cost, final_ops

    # -------- Replan --------

    def replan(self, original_path: TaskPath, feedback: Dict[str, Any], initial_state: Set[str]) -> TaskPath:
        failed = feedback.get("failed_operation")
        if failed and original_path.operations and failed == original_path.operations[-1]:
            # 简单移除失败的最后一步？通常这不是好的 replan 策略。
            # 但为了保持代码一致性，我们保留它。
            new_ops = original_path.operations[:-1]
            new_cost = sum(
                OPERATIONS_BY_NAME[o].estimated_cost for o in new_ops if o in OPERATIONS_BY_NAME
            )
            return TaskPath(
                operations=new_ops,
                estimated_cost=new_cost,
                reasoning=f"Removed failed op: {failed}",
            )
        return original_path
