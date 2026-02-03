"""
Schema Linking - 根据问题选择相关的表格

核心功能：
1. 从 meta graph 提取三元组形式的表格摘要
2. 将摘要缓存到文件，避免重复计算
3. 使用 few-shot prompt 让 LLM 选择最相关的 top-3 表格
"""

import os
import json
import sys
import time
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent
sys.path.append(str(src_dir))

from utils.call_llm import LLMCompletionCall
from utils.logger import logger


@dataclass
class TableSummary:
    """表格摘要"""
    file_name: str
    triplets: List[str]
    summary_text: str = ""  # 表格主题摘要


@dataclass
class SchemaLinkingResult:
    """Schema Linking 结果"""
    query: str
    selected_tables: List[str]  # top-3 表格文件名
    raw_response: str = ""


class SchemaLinking:
    """Schema Linking - 表格检索"""
    
    def __init__(self, meta_graphs_dir: Optional[str] = None):
        """初始化
        
        Args:
            meta_graphs_dir: meta graph 文件所在目录，默认为 output/realhitbench/tables
        """
        self.llm_client = LLMCompletionCall()
        
        # 设置路径
        if meta_graphs_dir:
            self.meta_graphs_dir = Path(meta_graphs_dir)
        else:
            project_root = Path(__file__).parent.parent.parent
            self.meta_graphs_dir = project_root / "output" / "realhitbench" / "tables"

        
        self.cache_file = self.meta_graphs_dir.parent / "table_summaries_cache.json"
        
        # if not self.meta_graphs_dir.exists():
        #     raise FileNotFoundError(f"Meta graphs directory not found: {self.meta_graphs_dir}")
        
        logger.info(f"Schema Linking initialized with {self.meta_graphs_dir}")
    
    def _generate_table_summary(self, meta_graph: Dict[str, Any], triplets: List[str]) -> str:
        """生成表格的简短摘要文本
        
        Args:
            meta_graph: meta graph 数据
            triplets: 提取的三元组列表
        
        Returns:
            摘要文本（1-2句话描述表格内容）
        """
        meta_info = meta_graph.get("meta_info", {})
        
        # 提取列表头关键词
        col_keywords = []
        row_keywords = []
        
        for t in triplets[:30]:
            if "has_column_header" in t:
                # 提取三元组中的值
                match = re.search(r'"([^"]+)"', t)
                if match:
                    value = match.group(1)
                    # 提取关键词（去除常见词）
                    words = value.lower().split()
                    keywords = [w for w in words if len(w) > 3 and w not in ['table', 'data', 'annual', 'household']]
                    col_keywords.extend(keywords[:2])
            elif "has_row_header" in t:
                match = re.search(r'"([^"]+)"', t)
                if match:
                    value = match.group(1)
                    words = value.lower().split()
                    keywords = [w for w in words if len(w) > 3 and w not in ['table', 'data', 'total']]
                    row_keywords.extend(keywords[:2])
        
        # 去重
        col_keywords = list(dict.fromkeys(col_keywords))[:5]
        row_keywords = list(dict.fromkeys(row_keywords))[:5]
        
        # 生成摘要
        summary_parts = []
        
        if col_keywords:
            summary_parts.append(f"Columns: {', '.join(col_keywords)}")
        
        if row_keywords:
            summary_parts.append(f"Rows: {', '.join(row_keywords)}")
        
        summary = "; ".join(summary_parts) if summary_parts else "General data table"
        
        return summary
    
    def _extract_header_value(self, props: Dict[str, Any]) -> Optional[str]:
        """从属性中提取表头值

        Args:
            props: 实体属性字典

        Returns:
            清理后的表头值，如果无效则返回 None
        """
        value = props.get("value") or props.get("value_text") or props.get("value_norm") or props.get("text")
        if not value or not isinstance(value, str):
            return None

        value_stripped = value.strip()
        if not value_stripped or value_stripped.startswith('[EMPTY_'):
            return None

        return value_stripped.replace('\n', ' ').replace('  ', ' ')[:80]

    def _collect_headers_from_entities(self, entities: List[Dict[str, Any]]) -> tuple:
        """从实体列表中收集列标题和行标题

        Args:
            entities: 实体列表

        Returns:
            (col_headers, row_headers, col_values_seen, row_values_seen) 元组
        """
        col_headers = []
        row_headers = []
        col_values_seen = set()
        row_values_seen = set()

        for ent in entities:
            label = ent.get("label", "")
            props = ent.get("properties", {}) or {}

            value_clean = self._extract_header_value(props)
            if not value_clean:
                continue

            if label == "column_header" and value_clean not in col_values_seen:
                col_headers.append((ent.get("id"), props, value_clean))
                col_values_seen.add(value_clean)
            elif label == "row_header" and value_clean not in row_values_seen:
                row_headers.append((ent.get("id"), props, value_clean))
                row_values_seen.add(value_clean)

        return col_headers, row_headers, col_values_seen, row_values_seen

    def _add_header_triplets(self, col_headers: List[tuple], row_headers: List[tuple]) -> List[str]:
        """添加表格与表头的关系三元组

        Args:
            col_headers: 列标题列表 [(id, props, value_clean), ...]
            row_headers: 行标题列表 [(id, props, value_clean), ...]

        Returns:
            三元组列表
        """
        triplets = []

        for _, _, value_clean in col_headers[:30]:
            triplets.append(f'(table, has_column_header, "{value_clean}")')

        for _, _, value_clean in row_headers[:30]:
            triplets.append(f'(table, has_row_header, "{value_clean}")')

        return triplets

    def _supplement_headers_from_meta_info(
        self,
        meta_info: Dict[str, Any],
        col_headers: List[tuple],
        row_headers: List[tuple],
        col_values_seen: set,
        row_values_seen: set
    ) -> List[str]:
        """从 meta_info 中补充表头信息

        Args:
            meta_info: meta_info 字典
            col_headers: 已有的列标题列表
            row_headers: 已有的行标题列表
            col_values_seen: 已见过的列标题值集合
            row_values_seen: 已见过的行标题值集合

        Returns:
            补充的三元组列表
        """
        triplets = []

        # 补充列表头
        if len(col_headers) < 3:
            col_header_names = meta_info.get("col_header_names", [])
            for name in col_header_names[:20]:
                value_clean = self._extract_header_value({"value": name})
                if value_clean and value_clean not in col_values_seen:
                    triplets.append(f'(table, has_column_header, "{value_clean}")')
                    col_values_seen.add(value_clean)

        # 补充行表头
        if len(row_headers) < 3:
            row_header_names = meta_info.get("row_header_names", [])
            for name in row_header_names[:20]:
                value_clean = self._extract_header_value({"value": name})
                if value_clean and value_clean not in row_values_seen:
                    triplets.append(f'(table, has_row_header, "{value_clean}")')
                    row_values_seen.add(value_clean)

        return triplets

    def _build_parent_map(self, relationships: List[Dict[str, Any]]) -> Dict[str, str]:
        """构建表头父子关系映射

        Args:
            relationships: 关系列表

        Returns:
            child_id -> parent_id 的映射字典
        """
        header_parent_map = {}

        for rel in relationships:
            if rel.get("relation") == "has_child":
                start_id = rel.get("start_entity_id")
                end_id = rel.get("end_entity_id")
                if start_id and end_id:
                    header_parent_map[end_id] = start_id

        return header_parent_map

    def _add_hierarchy_triplets(
        self,
        headers: List[tuple],
        header_parent_map: Dict[str, str],
        entity_map: Dict[str, Dict[str, Any]],
        max_count: int = 20
    ) -> List[str]:
        """添加表头层级关系三元组

        Args:
            headers: 表头列表 [(id, props, value_clean), ...]
            header_parent_map: 父子关系映射
            entity_map: 实体ID到实体的映射
            max_count: 最大三元组数量

        Returns:
            层级关系三元组列表
        """
        triplets = []

        for header_id, props, value_clean in headers:
            if len(triplets) >= max_count:
                break

            level = props.get("level")
            if level is not None and level > 2:
                continue

            if header_id not in header_parent_map:
                continue

            parent_id = header_parent_map[header_id]
            parent_ent = entity_map.get(parent_id)
            if not parent_ent:
                continue

            parent_props = parent_ent.get("properties", {}) or {}
            parent_value_clean = self._extract_header_value(parent_props)
            if parent_value_clean:
                triplets.append(f'("{parent_value_clean}", has_child, "{value_clean}")')

        return triplets

    def _extract_triplets_from_meta_graph(self, meta_graph: Dict[str, Any]) -> List[str]:
        """从 meta graph 中提取三元组信息

        提取的三元组类型：
        1. 表格与表头的关系: (table, has_column_header, "header_value")
        2. 表格与行表头的关系: (table, has_row_header, "header_value")
        3. 表头层级关系: ("parent_header", has_child, "child_header")
        4. 如果没有行/列表头，从 meta_info 中提取

        Returns:
            三元组字符串列表
        """
        entities = meta_graph.get("entities", [])
        relationships = meta_graph.get("relationships", [])

        # 构建实体映射
        entity_map = {ent.get("id"): ent for ent in entities}

        # 收集列标题和行标题实体
        col_headers, row_headers, col_values_seen, row_values_seen = self._collect_headers_from_entities(entities)

        # 添加表格与表头的关系三元组
        triplets = self._add_header_triplets(col_headers, row_headers)

        # 如果表头数量太少，从 meta_info 中补充
        if len(col_headers) < 3 or len(row_headers) < 3:
            meta_info = meta_graph.get("meta_info", {})
            supplement_triplets = self._supplement_headers_from_meta_info(
                meta_info, col_headers, row_headers, col_values_seen, row_values_seen
            )
            triplets.extend(supplement_triplets)

        # 构建表头层级关系
        header_parent_map = self._build_parent_map(relationships)

        # 添加层级关系三元组
        hierarchy_triplets = self._add_hierarchy_triplets(
            col_headers + row_headers, header_parent_map, entity_map
        )
        triplets.extend(hierarchy_triplets)

        # 去重（保持顺序）
        unique_triplets = list(dict.fromkeys(triplets))
        return unique_triplets
    
    def generate_summaries(self, force_regenerate: bool = False) -> List[TableSummary]:
        """生成所有表格的摘要（支持缓存）
        
        Args:
            force_regenerate: 是否强制重新生成（忽略缓存）
        
        Returns:
            表格摘要列表
        """
        # 检查缓存
        if not force_regenerate and self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                summaries = [
                    TableSummary(
                        file_name=item["file_name"],
                        triplets=item["triplets"],
                        summary_text=item.get("summary_text", "")
                    )
                    for item in cache_data.get("summaries", [])
                ]
                logger.info(f"Loaded {len(summaries)} table summaries from cache")
                return summaries
            except Exception as e:
                logger.warning(f"Failed to load cache, will regenerate: {e}")
        
        # 生成摘要
        logger.info("Generating table summaries from meta graphs...")
        summaries = []
        meta_files = list(self.meta_graphs_dir.glob("*.json"))
        
        for idx, meta_file in enumerate(meta_files, 1):
            try:
                if idx % 50 == 0:
                    logger.info(f"Processing {idx}/{len(meta_files)}: {meta_file.name}")
                
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta_graph = json.load(f)
                
                # 提取三元组
                triplets = self._extract_triplets_from_meta_graph(meta_graph)
                
                # 限制三元组数量（避免摘要过长）
                triplets = triplets[:50]
                
                # 生成摘要文本
                summary_text = self._generate_table_summary(meta_graph, triplets)
                
                summaries.append(TableSummary(
                    file_name=meta_file.stem,
                    triplets=triplets,
                    summary_text=summary_text
                ))
                
            except Exception as e:
                logger.error(f"Failed to process {meta_file.name}: {e}")
                continue
        
        logger.info(f"Generated {len(summaries)} table summaries")
        
        # 保存缓存
        try:
            cache_data = {
                "total_tables": len(summaries),
                "summaries": [
                    {
                        "file_name": s.file_name,
                        "triplets": s.triplets,
                        "summary_text": s.summary_text
                    }
                    for s in summaries
                ]
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved table summaries cache to {self.cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
        
        return summaries
    
    def _format_summaries_for_prompt(self, summaries: List[TableSummary], max_tables: int = 500) -> str:
        """将摘要格式化为 prompt 文本
        
        Args:
            summaries: 表格摘要列表
            max_tables: 最多显示多少个表格（避免 prompt 过长）
        
        Returns:
            格式化后的文本
        """
        lines = []
        
        # 如果表格过多，只显示前 max_tables 个
        display_summaries = summaries[:max_tables]
        
        for idx, summary in enumerate(display_summaries, 1):
            lines.append(f"  [{idx}] {summary.file_name}")
            # 添加摘要文本（如果有）
            if summary.summary_text:
                lines.append(f"    Summary: {summary.summary_text}")
            
            # 智能选择三元组：优先显示短的、关键的
            selected_triplets = self._select_important_triplets(summary.triplets, max_count=15)
            for triplet in selected_triplets:
                lines.append(f"    {triplet}")
            lines.append("")  # 空行分隔
        
        if len(summaries) > max_tables:
            lines.append(f"  ... (还有 {len(summaries) - max_tables} 个表格未显示)")
        
        return "\n".join(lines)
    
    def _select_important_triplets(self, triplets: List[str], max_count: int = 15) -> List[str]:
        """智能选择重要的三元组
        
        优先级：
        1. 短的三元组（更具体）
        2. 包含关键信息（年份、数值、实体）的三元组
        3. 避免重复冗长的标题
        
        Args:
            triplets: 三元组列表
            max_count: 最多返回多少个
        
        Returns:
            选中的三元组列表
        """
        if len(triplets) <= max_count:
            return triplets
        
        # 按长度和重要性评分
        scored_triplets = []
        seen_values = set()
        
        for t in triplets:
            # 提取三元组中的值
            match = re.search(r'"([^"]+)"', t)
            if not match:
                continue
            
            value = match.group(1)
            value_lower = value.lower()
            
            # 跳过过长的重复标题
            if len(value) > 80 and value in seen_values:
                continue
            
            seen_values.add(value)
            
            # 评分规则
            score = 0
            
            # 1. 长度评分：短的更好（最多50分）
            length_score = max(0, 50 - len(value) // 2)
            score += length_score
            
            # 2. 关键词加分（各10分）
            keywords = ['year', 'age', 'rate', 'percent', 'total', 'employed', 'unemployed', 
                       'male', 'female', 'men', 'women', 'population', 'labor', 'force',
                       'agriculture', 'industry', 'occupation', 'education', 'income']
            for kw in keywords:
                if kw in value_lower:
                    score += 10
            
            # 3. 包含数字的加分
            if re.search(r'\d{4}', value):  # 年份
                score += 15
            elif re.search(r'\d+', value):  # 其他数字
                score += 5
            
            # 4. 避免过长的官方标题（扣分）
            if 'household data' in value_lower or 'annual averages' in value_lower:
                score -= 30
            
            scored_triplets.append((score, t))
        
        # 按分数排序，取前 max_count 个
        scored_triplets.sort(key=lambda x: -x[0])
        selected = [t for score, t in scored_triplets[:max_count]]
        
        return selected
    
    def _build_few_shot_prompt(self, query: str, summaries: List[TableSummary]) -> str:
        """构建 few-shot prompt
        
        Args:
            query: 用户问题
            summaries: 表格摘要列表
        
        Returns:
            完整的 prompt
        """
        summaries_text = self._format_summaries_for_prompt(summaries)
        
        prompt = f"""You are an expert in table retrieval. Given a question, select the TOP-3 most relevant tables from the candidates.

**IMPORTANT RULES**:
1. **Focus on TABLE CONTENT (triplets), NOT just the table file name**
2. Match the question's semantic meaning with the table's actual columns and rows
3. A table about "employment" might contain agriculture/industry data even if not named "agriculture-table"
4. Look for specific column headers and row headers that answer the question

# Example 1 (Positive)
Question: What is the average time men and women spend on work-related activities?

Candidate tables:
  [1] activitytime-table01
    Summary: Columns: hours, average, spent; Rows: work, activities, leisure
    (table, has_column_header, "Men")  
    (table, has_column_header, "Women")
    (table, has_row_header, "Work and work-related activities")
    ("Work and work-related activities", has_child, "Working")
  
  [2] labor-table50
    Summary: Columns: company; Rows: company
    (table, has_row_header, "Company 1")
    (table, has_row_header, "Company 2")
    
Selected: ["activitytime-table01"]
Reason: Table 1 has "Men/Women" columns AND "Work-related activities" rows. Table 2 only has company names, irrelevant.

# Example 2 (Common Mistake to Avoid)
Question: What was the employment in agriculture in 1955?

Candidate tables:
  [1] employment-table01  
    Summary: Columns: year, employed, civilian, labor; Rows: none
    (table, has_column_header, "Year")
    (table, has_column_header, "Employed")
    (table, has_column_header, "Agriculture")  ← KEY: Contains "Agriculture" column!
    (table, has_column_header, "Nonagricultural industries")
  
  [2] agriculture-table09
    Summary: Columns: production, yield; Rows: crops, wheat, corn
    (table, has_column_header, "Production")
    (table, has_row_header, "Wheat")
    (table, has_row_header, "Corn")
    
Selected: ["employment-table01"]
Reason: Table 1 has "Year", "Employed", and "Agriculture" columns - directly answers the question. 
Table 2 is about crop production, NOT employment. **Don't be misled by the file name "agriculture-table"!**

# Example 3 (Multi-dimensional Query)
Question: What is the unemployment rate for people aged 16-24 in 2022?

Candidate tables:
  [1] employment-table06
    Summary: Columns: civilian, labor, force; Rows: employment, status, years
    (table, has_column_header, "Civilian labor force")
    (table, has_row_header, "16 to 19 years")
    (table, has_row_header, "20 to 24 years")
    (table, has_row_header, "Unemployed")
  
  [2] labor-table64
    Summary: Columns: rate; Rows: years, company
    (table, has_column_header, "Rate")
    (table, has_row_header, "16 years and over")
    
Selected: ["employment-table06", "labor-table64"]
Reason: Table 1 has specific age groups (16-19, 20-24) and unemployment status. Table 2 provides rate data.

# Current Task
Question: {query}

Candidate tables:
{summaries_text}

**Remember**: 
- Analyze the CONTENT (column/row headers) carefully
- Match the question's requirements (age groups, years, metrics, etc.)
- Don't rely solely on file names

Please output in STRICT JSON format (select TOP-3 most relevant tables):
```json
{{
  "top_tables": ["table1_name", "table2_name", "table3_name"]
}}
```

Output JSON:
"""
        return prompt
    
    def _parse_json_safely(self, text: str) -> Optional[Dict[str, Any]]:
        """安全解析 JSON"""
        try:
            return json.loads(text)
        except Exception:
            pass
        
        # 尝试提取第一个 {...} 块
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end+1])
        except Exception:
            return None
        
        return None
    
    def select_tables(self, query: str, force_regenerate_cache: bool = False, max_retries: int = 3) -> SchemaLinkingResult:
        """根据 query 选择最相关的 top-3 表格（带 retry 机制）
        
        Args:
            query: 用户问题
            force_regenerate_cache: 是否强制重新生成缓存
            max_retries: 最大重试次数
        
        Returns:
            Schema Linking 结果
        """
        # 1. 生成或加载摘要
        summaries = self.generate_summaries(force_regenerate=force_regenerate_cache)
        
        if not summaries:
            logger.error("No table summaries available")
            return SchemaLinkingResult(
                query=query,
                selected_tables=[]
            )
        
        # 2. 构建 few-shot prompt
        prompt = self._build_few_shot_prompt(query, summaries)
        
        # 3. 调用 LLM（带 retry）
        raw_response = None
        last_error = None
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = 2 ** attempt  # 指数退避：2s, 4s, 8s
                    logger.warning(f"Retry {attempt}/{max_retries} after {wait_time}s wait...")
                    time.sleep(wait_time)
                
                logger.info(f"Calling LLM for table selection (query: {query[:80]}...)")
                raw_response = self.llm_client.call_api(prompt)
                break  # 成功则跳出
                
            except Exception as e:
                last_error = e
                logger.error(f"LLM API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} attempts failed for query: {query[:80]}")
                    return SchemaLinkingResult(
                        query=query,
                        selected_tables=[],
                        raw_response=f"Error after {max_retries} retries: {str(last_error)}"
                    )
        
        # 4. 解析结果
        parsed = self._parse_json_safely(raw_response)
        
        if not parsed:
            logger.error(f"Failed to parse LLM response: {raw_response[:200] if raw_response else 'None'}")
            return SchemaLinkingResult(
                query=query,
                selected_tables=[],
                raw_response=raw_response or ""
            )
        
        top_tables = parsed.get("top_tables", [])
        
        # 确保是列表且最多3个
        if not isinstance(top_tables, list):
            top_tables = []
        
        top_tables = top_tables[:3]
        
        logger.info(f"Selected tables: {top_tables}")
        
        return SchemaLinkingResult(
            query=query,
            selected_tables=top_tables,
            raw_response=raw_response or ""
        )
    
    def get_meta_graph(self, table_name: str) -> Optional[Dict[str, Any]]:
        """根据表格名称加载对应的 meta graph
        
        Args:
            table_name: 表格文件名（不带后缀）
        
        Returns:
            Meta graph 数据，如果不存在返回 None
        """
        meta_file = self.meta_graphs_dir / f"{table_name}.json"
        
        if not meta_file.exists():
            logger.warning(f"Meta graph file not found: {meta_file}")
            return None
        
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load meta graph {table_name}: {e}")
            return None


def main():
    """测试函数"""
    # 初始化
    schema_linking = SchemaLinking()
    
    # 测试问题
    test_queries = [
        "What is the average time men and women spend on work-related activities?",
        "What is the unemployment rate for different age groups?",
        "2019年不同行业的平均工资是多少？"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        result = schema_linking.select_tables(query)
        
        print(f"\nSelected tables:")
        for idx, table_name in enumerate(result.selected_tables, 1):
            print(f"  {idx}. {table_name}")
        
        print(f"\nRaw response preview:")
        print(result.raw_response[:300])


if __name__ == "__main__":
    main()

