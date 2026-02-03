"""
ADG生成器（Abstract/Macro Graph Generator）
根据用户问题和表格信息，使用LLM生成任务分解图和备选路径
"""

import json
import sys
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field

# 修正导入路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# 修改内部导入路径
from model.adg_benchmark.adg.macro_operations import ALL_OPERATIONS, OPERATIONS_BY_NAME, MacroOperation
from utils.call_llm import LLMCompletionCall
from utils.logger import logger


@dataclass
class TaskPath:
    """任务路径（操作序列）"""
    operations: List[str]  # 操作名称列表
    estimated_cost: float  # 预估总成本
    reasoning: str = ""  # 推理过程
    
    def __repr__(self):
        return f"Path({' -> '.join(self.operations)}, cost={self.estimated_cost:.2f})"


@dataclass
class ADGResult:
    """ADG生成结果"""
    user_question: str
    table_info: Dict[str, Any]
    candidate_paths: List[TaskPath] = field(default_factory=list)
    selected_path: Optional[TaskPath] = None
    raw_response: str = ""
    prompt: str = ""  # LLM 调用的完整 prompt
    response: str = ""  # LLM 的原始响应


class ADGGenerator:
    """ADG生成器"""
    
    def __init__(self):
        self.llm_client = LLMCompletionCall()
        logger.info("ADG Generator initialized")
    
    def _build_operation_catalog(self) -> str:
        """构建操作目录（供LLM参考）"""
        lines = ["Available Macro Operations:\n"]
        
        current_category = None
        for op in ALL_OPERATIONS:
            if op.category != current_category:
                current_category = op.category
                lines.append(f"\n## {current_category.value.replace('_', ' ').title()}")
            
            lines.append(f"\n{op.name} (Cost: {op.estimated_cost})")
            lines.append(f"  - Description: {op.description}")
            
            if op.preconditions:
                precond_str = ", ".join([c.description or c.key for c in op.preconditions])
                lines.append(f"  - Preconditions: {precond_str}")
            
            if op.postconditions:
                lines.append(f"  - Postconditions: {', '.join(op.postconditions)}")
        
        return "\n".join(lines)
    
    def _build_prompt(self, user_question: str, table_info: Dict[str, Any]) -> str:
        """构建LLM prompt - 【改进版】包含实际数据样本"""
        
        operation_catalog = self._build_operation_catalog()
        
        # 提取表格信息摘要
        schema_summary = ""
        
        # 【关键改进】优先使用实际加载的列名
        actual_columns = table_info.get("actual_columns", table_info.get("column_names", []))
        if actual_columns:
            schema_summary = f"Columns: {', '.join(str(c) for c in actual_columns[:20])}"
        
        if "row_count" in table_info:
            schema_summary += f"\nRow Count: {table_info['row_count']}"
        
        # 【关键改进】添加实际数据类型
        if "actual_dtypes" in table_info:
            dtypes = table_info["actual_dtypes"]
            dtype_summary = ", ".join([f"{k}: {v}" for k, v in list(dtypes.items())[:10]])
            schema_summary += f"\nColumn Types: {dtype_summary}"
        
        # 【核心改进】添加数据样本 - 让 LLM 看到实际数据！
        if "data_sample" in table_info and table_info["data_sample"]:
            schema_summary += f"\n\n**Data Sample (First 15 Rows):**\n{table_info['data_sample']}"
        
        # 添加额外的语义信息（如果有）
        if "summary_text" in table_info and table_info["summary_text"]:
             schema_summary += f"\n\nTable Summary: {table_info['summary_text']}"
        
        # Meta Graph Triplets (保留但降低优先级)
        if "meta_graph_triplets" in table_info and table_info["meta_graph_triplets"]:
            triplets = table_info["meta_graph_triplets"]
            child_relations = [t for t in triplets if "has_child" in t]
            other_relations = [t for t in triplets if "has_child" not in t]
            selected_triplets = (child_relations[:10] + other_relations)[:15]
            
            if selected_triplets:
                schema_summary += "\n\nSchema Semantics (Triplets):\n" + "\n".join([f"  - {t}" for t in selected_triplets])
        
        prompt = f"""You are an Excel data analysis task planning expert. Given a user question and **actual table data**, you need to:
1. Analyze the data sample to understand the table structure.
2. Decompose the task into a sequence of high-level macro operations.
3. Provide 2-3 candidate paths (different operation orders or methods).
4. Estimate the total cost for each path.

{operation_catalog}

# Task
User Question: {user_question}

# Table Information
{schema_summary}

# Requirements
1. **IMPORTANT**: Study the data sample carefully before planning operations.
2. Each path is a sequence of operations, e.g., ["DetectSchema", "FilterRows", "Aggregate"]
3. Choose operations that match the actual data types and structure you see.
4. For simple queries (value lookup, counting), prefer shorter paths.
5. For complex queries (aggregation, comparison), include necessary transformations.
6. **Hierarchy Header**: If you see merged/hierarchical headers in the data, include ExplodeOrFlatten.

# Example Output Format (Strict JSON)
```json
{{
  "candidate_paths": [
    {{
      "operations": ["FilterRows", "SelectColumns", "Aggregate"],
      "estimated_cost": 2.6,
      "reasoning": "Simple filtering task: filter by condition, select relevant columns, aggregate result."
    }},
    {{
      "operations": ["DetectSchema", "FilterRows", "SortValues"],
      "estimated_cost": 1.9,
      "reasoning": "For ranking/sorting query: detect schema, filter, then sort."
    }}
  ]
}}
```

Please output JSON:
"""
        return prompt
    
    def _parse_json_safely(self, raw_response: str) -> Optional[Dict]:
        """安全解析JSON"""
        if not raw_response:
            return None
        
        # 尝试提取JSON
        try:
            # 尝试直接解析
            return json.loads(raw_response)
        except json.JSONDecodeError:
            pass
        
        # 尝试提取```json ... ```中的内容
        import re
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 尝试提取第一个完整的JSON对象
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        logger.error(f"Failed to parse JSON from response: {raw_response[:200]}")
        return None
    
    def generate_adg(
        self, 
        user_question: str, 
        table_info: Dict[str, Any]
    ) -> ADGResult:
        """生成ADG（任务分解图）
        
        Args:
            user_question: 用户问题
            table_info: 表格信息（schema, column_names, row_count等）
        
        Returns:
            ADG生成结果
        """
        logger.info(f"Generating ADG for question: {user_question}...")
        
        # 构建prompt
        prompt = self._build_prompt(user_question, table_info)
        
        # 调用LLM
        try:
            raw_response = self.llm_client.call_api(prompt)
            logger.info(f"LLM response received: {len(raw_response)} chars")
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return ADGResult(
                user_question=user_question,
                table_info=table_info,
                raw_response=f"Error: {str(e)}",
                prompt=prompt,
                response=f"Error: {str(e)}"
            )
        
        # 解析结果
        parsed = self._parse_json_safely(raw_response)
        
        if not parsed or "candidate_paths" not in parsed:
            logger.error("Failed to parse LLM response")
            # 使用默认路径
            default_path = TaskPath(
                operations=["DetectSchema", "InspectColumn", "Aggregate"],
                estimated_cost=2.3,
                reasoning="默认路径：检测schema → 检查列 → 聚合统计"
            )
            return ADGResult(
                user_question=user_question,
                table_info=table_info,
                candidate_paths=[default_path],
                raw_response=raw_response,
                prompt=prompt,
                response=raw_response
            )
        
        # 转换为TaskPath对象
        candidate_paths = []
        for path_data in parsed["candidate_paths"]:
            if not isinstance(path_data, dict):
                continue
            
            operations = path_data.get("operations", [])
            estimated_cost = path_data.get("estimated_cost", 0.0)
            reasoning = path_data.get("reasoning", "")
            
            # 验证操作是否存在
            valid_operations = []
            for op_name in operations:
                if op_name in OPERATIONS_BY_NAME:
                    valid_operations.append(op_name)
                else:
                    logger.warning(f"Unknown operation: {op_name}")
            
            if valid_operations:
                candidate_paths.append(TaskPath(
                    operations=valid_operations,
                    estimated_cost=estimated_cost,
                    reasoning=reasoning
                ))
        
        if not candidate_paths:
            logger.warning("No valid paths found, using default")
            default_path = TaskPath(
                operations=["DetectSchema", "InspectColumn", "Aggregate"],
                estimated_cost=2.3,
                reasoning="默认路径"
            )
            candidate_paths = [default_path]
        
        logger.info(f"Generated {len(candidate_paths)} candidate paths")
        for i, path in enumerate(candidate_paths, 1):
            logger.info(f"  Path {i}: {path}")
        
        return ADGResult(
            user_question=user_question,
            table_info=table_info,
            candidate_paths=candidate_paths,
            raw_response=raw_response,
            prompt=prompt,
            response=raw_response
        )


def main():
    """测试函数"""
    generator = ADGGenerator()
    
    # 测试数据
    user_question = "计算2023年各部门的平均工资，并按降序排列"
    table_info = {
        "column_names": ["部门", "员工姓名", "工资", "年份"],
        "row_count": 1000,
        "column_types": {"部门": "text", "员工姓名": "text", "工资": "numeric", "年份": "numeric"}
    }
    
    result = generator.generate_adg(user_question, table_info)
    
    print(f"\n{'='*70}")
    print(f"用户问题: {result.user_question}")
    print(f"{'='*70}")
    
    print(f"\n候选路径 ({len(result.candidate_paths)} 条):")
    for i, path in enumerate(result.candidate_paths, 1):
        print(f"\n路径 {i}:")
        print(f"  操作序列: {' → '.join(path.operations)}")
        print(f"  预估成本: {path.estimated_cost:.2f}")
        print(f"  推理: {path.reasoning}")


if __name__ == "__main__":
    main()
