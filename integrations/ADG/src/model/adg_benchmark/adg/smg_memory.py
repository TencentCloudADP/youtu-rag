"""
SMG - 结构化记忆图（Structured Memory Graph）
压缩执行记忆，只保留关键的schema相关信息
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import pandas as pd
import sys
from pathlib import Path

# 修正导入路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from utils.logger import logger


@dataclass
class MemorySnapshot:
    """记忆快照"""
    operation_name: str
    schema_input: Dict[str, str]  # 输入schema：{列名: 类型}
    schema_output: Dict[str, str]  # 输出schema
    shape_change: tuple  # (行数变化, 列数变化)
    key_columns: List[str] = field(default_factory=list)  # 关键列（被操作的列）
    success: bool = True
    error_msg: str = ""
    
    def __repr__(self):
        return (f"Snapshot({self.operation_name}: "
                f"{self.schema_input.keys()} -> {self.schema_output.keys()}, "
                f"shape_change={self.shape_change})")


@dataclass
class CompressedMemory:
    """压缩后的记忆"""
    operation_sequence: List[str]  # 操作序列
    key_snapshots: List[MemorySnapshot]  # 关键快照（只保留schema变化的步骤）
    final_schema: Dict[str, str]  # 最终schema
    error_history: List[Dict[str, Any]] = field(default_factory=list)  # 错误记录
    
    def __repr__(self):
        return (f"CompressedMemory("
                f"ops={len(self.operation_sequence)}, "
                f"snapshots={len(self.key_snapshots)}, "
                f"final_cols={len(self.final_schema)})")


class SMGMemory:
    """SMG记忆管理器"""
    
    def __init__(self):
        self.snapshots: List[MemorySnapshot] = []
        self.current_schema: Dict[str, str] = {}
        self.current_shape: tuple = (0, 0)
        logger.info("SMG Memory initialized")
    
    def record_execution(
        self,
        operation_name: str,
        df_before: Optional[pd.DataFrame],
        df_after: Optional[pd.DataFrame],
        key_columns: List[str] = None,
        success: bool = True,
        error_msg: str = ""
    ):
        """记录执行步骤
        
        Args:
            operation_name: 操作名称
            df_before: 操作前的DataFrame
            df_after: 操作后的DataFrame
            key_columns: 被操作的关键列
            success: 是否成功
            error_msg: 错误信息
        """
        if key_columns is None:
            key_columns = []
        
        # 提取schema
        schema_input = {}
        schema_output = {}
        shape_before = (0, 0)
        shape_after = (0, 0)
        
        if df_before is not None and isinstance(df_before, pd.DataFrame):
            schema_input = {col: str(dtype) for col, dtype in df_before.dtypes.items()}
            shape_before = df_before.shape
        
        if df_after is not None and isinstance(df_after, pd.DataFrame):
            schema_output = {col: str(dtype) for col, dtype in df_after.dtypes.items()}
            shape_after = df_after.shape
        
        # 计算shape变化
        shape_change = (
            shape_after[0] - shape_before[0],  # 行数变化
            shape_after[1] - shape_before[1]   # 列数变化
        )
        
        # 创建快照
        snapshot = MemorySnapshot(
            operation_name=operation_name,
            schema_input=schema_input,
            schema_output=schema_output,
            shape_change=shape_change,
            key_columns=key_columns,
            success=success,
            error_msg=error_msg
        )
        
        self.snapshots.append(snapshot)
        
        # 更新当前状态
        if df_after is not None:
            self.current_schema = schema_output
            self.current_shape = shape_after
        
        logger.info(f"Recorded execution: {snapshot}")
    
    def compress(self) -> CompressedMemory:
        """压缩记忆
        
        压缩策略：
        1. 只保留schema变化的快照（列增删、类型变化、shape大幅变化）
        2. 折叠连续的相同类型操作
        3. 保留所有错误记录
        
        Returns:
            压缩后的记忆
        """
        logger.info(f"Compressing {len(self.snapshots)} snapshots...")
        
        if not self.snapshots:
            return CompressedMemory(
                operation_sequence=[],
                key_snapshots=[],
                final_schema={}
            )
        
        key_snapshots = []
        error_history = []
        prev_schema = {}
        
        for i, snapshot in enumerate(self.snapshots):
            # 记录错误
            if not snapshot.success:
                error_history.append({
                    "operation": snapshot.operation_name,
                    "error": snapshot.error_msg,
                    "position": i
                })
                key_snapshots.append(snapshot)  # 错误快照一定保留
                prev_schema = snapshot.schema_output
                continue
            
            # 检查是否有schema变化
            has_schema_change = False
            
            # 1. 列数量变化
            if len(snapshot.schema_output) != len(snapshot.schema_input):
                has_schema_change = True
            
            # 2. 列名变化
            elif set(snapshot.schema_output.keys()) != set(snapshot.schema_input.keys()):
                has_schema_change = True
            
            # 3. 类型变化
            elif snapshot.schema_output != snapshot.schema_input:
                has_schema_change = True
            
            # 4. shape大幅变化（行数变化超过20%）
            elif abs(snapshot.shape_change[0]) > max(10, self.current_shape[0] * 0.2):
                has_schema_change = True
            
            # 5. 第一个快照一定保留
            elif i == 0:
                has_schema_change = True
            
            # 6. 最后一个快照一定保留
            elif i == len(self.snapshots) - 1:
                has_schema_change = True
            
            if has_schema_change:
                key_snapshots.append(snapshot)
                prev_schema = snapshot.schema_output
        
        # 构建操作序列
        operation_sequence = [s.operation_name for s in self.snapshots]
        
        # 最终schema
        final_schema = self.snapshots[-1].schema_output if self.snapshots else {}
        
        compressed = CompressedMemory(
            operation_sequence=operation_sequence,
            key_snapshots=key_snapshots,
            final_schema=final_schema,
            error_history=error_history
        )
        
        logger.info(f"Compression complete: {len(self.snapshots)} -> {len(key_snapshots)} snapshots")
        logger.info(f"  Error count: {len(error_history)}")
        
        return compressed
    
    def get_state_summary(self) -> Dict[str, Any]:
        """获取当前状态摘要（供MCTS使用）"""
        return {
            "current_schema": self.current_schema,
            "current_shape": self.current_shape,
            "operation_count": len(self.snapshots),
            "success_count": sum(1 for s in self.snapshots if s.success),
            "error_count": sum(1 for s in self.snapshots if not s.success),
        }
    
    def clear(self):
        """清空记忆"""
        self.snapshots.clear()
        self.current_schema = {}
        self.current_shape = (0, 0)
        logger.info("Memory cleared")
    
    def get_execution_trace(self) -> List[Dict[str, Any]]:
        """获取执行轨迹（用于可视化）"""
        trace = []
        for i, snapshot in enumerate(self.snapshots):
            trace.append({
                "step": i + 1,
                "operation": snapshot.operation_name,
                "input_columns": list(snapshot.schema_input.keys()),
                "output_columns": list(snapshot.schema_output.keys()),
                "shape_change": snapshot.shape_change,
                "key_columns": snapshot.key_columns,
                "success": snapshot.success,
                "error": snapshot.error_msg
            })
        return trace


def main():
    """测试函数"""
    import pandas as pd
    
    # 创建记忆管理器
    memory = SMGMemory()
    
    # 模拟执行过程
    df1 = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'e'],
        'C': [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    
    # 步骤1: DetectSchema
    memory.record_execution(
        "DetectSchema",
        df_before=None,
        df_after=df1,
        success=True
    )
    
    # 步骤2: SelectColumns（选择A和C列）
    df2 = df1[['A', 'C']]
    memory.record_execution(
        "SelectColumns",
        df_before=df1,
        df_after=df2,
        key_columns=['A', 'C'],
        success=True
    )
    
    # 步骤3: FilterRows（过滤A>2的行）
    df3 = df2[df2['A'] > 2]
    memory.record_execution(
        "FilterRows",
        df_before=df2,
        df_after=df3,
        key_columns=['A'],
        success=True
    )
    
    # 步骤4: DeriveColumn（计算新列D）
    df4 = df3.copy()
    df4['D'] = df4['A'] * df4['C']
    memory.record_execution(
        "DeriveColumn",
        df_before=df3,
        df_after=df4,
        key_columns=['D'],
        success=True
    )
    
    # 步骤5: Aggregate（失败）
    memory.record_execution(
        "Aggregate",
        df_before=df4,
        df_after=None,
        success=False,
        error_msg="Column 'E' not found"
    )
    
    # 输出执行轨迹
    print("\n执行轨迹:")
    for trace_item in memory.get_execution_trace():
        print(f"  步骤 {trace_item['step']}: {trace_item['operation']}")
        print(f"    输入列: {trace_item['input_columns']}")
        print(f"    输出列: {trace_item['output_columns']}")
        print(f"    Shape变化: {trace_item['shape_change']}")
        if not trace_item['success']:
            print(f"    ❌ 错误: {trace_item['error']}")
    
    # 压缩记忆
    compressed = memory.compress()
    
    print(f"\n压缩结果:")
    print(f"  原始快照数: {len(memory.snapshots)}")
    print(f"  压缩后快照数: {len(compressed.key_snapshots)}")
    print(f"  操作序列: {' → '.join(compressed.operation_sequence)}")
    print(f"  最终schema: {list(compressed.final_schema.keys())}")
    print(f"  错误记录: {len(compressed.error_history)}")
    
    # 状态摘要
    summary = memory.get_state_summary()
    print(f"\n状态摘要:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

