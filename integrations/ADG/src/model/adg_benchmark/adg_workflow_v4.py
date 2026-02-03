"""
ADG Workflow V4 - 分步执行 + 丰富上下文

核心设计:
1. 保持 ADG 分步执行流程（每个操作单独生成代码）
2. 每步代码生成时提供丰富上下文:
   - Query（用户问题）
   - Meta Graph 语义信息
   - Excel 数据样本（100-200行）
   - 当前 DataFrame 状态
3. SMG 记忆错误，支持智能重试
4. 自动导入常用库（numpy, pandas, matplotlib）
"""

import pandas as pd
import numpy as np
import sys
import os
import re
from typing import Dict, Any, List, Optional, Tuple
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from langgraph.graph import StateGraph, START, END

from src.model.adg_benchmark.adg.adg_generator import ADGGenerator, TaskPath
from src.model.adg_benchmark.adg.mcts_planner import MCTSPlanner
from src.model.adg_benchmark.adg.smg_memory import SMGMemory
from src.model.adg_benchmark.adg.macro_operations import OPERATIONS_BY_NAME, MacroOperation
from src.model.adg_benchmark.schema_loader import SchemaLoader
from utils.call_llm import LLMCompletionCall
from utils.logger import logger


class ADGWorkflowStateV4(TypedDict):
    """V4 工作流状态"""
    # 基础信息
    user_question: str
    question_type: str
    current_table: Dict[str, Any]
    
    # Schema + Data
    table_info: Optional[Dict[str, Any]]
    dataframe: Optional[pd.DataFrame]
    original_dataframe: Optional[pd.DataFrame]  # 保留原始数据用于上下文
    
    # ADG 规划
    candidate_paths: Optional[List[TaskPath]]
    selected_path: Optional[TaskPath]
    
    # 分步执行
    current_op_index: int
    execution_trace: Optional[List[Dict]]
    
    # 输出
    final_answer: str
    
    # 控制
    replan_count: int
    max_replan: int
    
    # LLM 调用历史（用于训练数据收集）
    llm_call_history: Optional[List[Dict[str, Any]]]  # 格式: [{"type": "adg"/"code"/"answer", "prompt": "...", "response": "...", "context": {...}}, ...]


class ADGWorkflowV4:
    """ADG Workflow V4 - 分步执行 + 丰富上下文"""
    
    def __init__(self, meta_graphs_dir: Optional[str] = None, data_sample_rows: int = 150, event_callback=None):
        """
        Args:
            meta_graphs_dir: Meta Graph 目录
            data_sample_rows: 提供给 LLM 的数据样本行数（默认150行）
            event_callback: 事件回调函数，用于流式输出 LLM 响应
        """
        self.adg_generator = ADGGenerator()
        self.mcts_planner = MCTSPlanner()
        self.llm_client = LLMCompletionCall()
        self.schema_loader = SchemaLoader(meta_graphs_dir)
        self.data_sample_rows = data_sample_rows
        self.event_callback = event_callback  # 添加事件回调
        logger.info(f"ADG Workflow V4 initialized (data_sample_rows={data_sample_rows})")
    
    def _emit_event(self, name: str, event_data: Dict[str, Any]):
        """发送事件到回调函数"""
        if self.event_callback:
            try:
                self.event_callback(name, event_data)
            except Exception as e:
                logger.warning(f"Failed to emit event: {e}")
    
    # ==================== Helper Functions ====================
    
    def _clean_col_name(self, name) -> str:
        if not isinstance(name, str):
            return str(name)
        clean = name.replace('\n', ' ').replace('\r', ' ')
        return ' '.join(clean.split()).strip()

    def _read_xls_sheets(self, file_path: str) -> Tuple[List[pd.DataFrame], List[str]]:
        """Read all sheets from .xls file using xlrd"""
        df_list = []
        sheet_names = []
        try:
            import xlrd
            workbook = xlrd.open_workbook(file_path)
            logger.info(f"Reading {workbook.nsheets} sheets from .xls file: {workbook.sheet_names()}")

            for sheet_idx in range(workbook.nsheets):
                try:
                    sheet = workbook.sheet_by_index(sheet_idx)
                    sheet_name = workbook.sheet_names()[sheet_idx]
                    if sheet.nrows > 0:
                        data = []
                        for row_idx in range(sheet.nrows):
                            row = sheet.row_values(row_idx)
                            data.append(row)
                        df_temp = pd.DataFrame(data)
                        if df_temp is not None and not df_temp.empty:
                            df_list.append(df_temp)
                            sheet_names.append(sheet_name)
                            logger.info(f"Successfully read sheet {sheet_idx} '{sheet_name}' with shape {df_temp.shape}")
                except Exception as e_sheet:
                    logger.warning(f"Failed to read sheet {sheet_idx}: {e_sheet}")
                    continue

            if not df_list:
                logger.error(f"All sheets failed to read for .xls file: {file_path}")
                return [], []
        except Exception as e:
            logger.error(f"Failed to read .xls file: {e}")
            return [], []

        return df_list, sheet_names

    def _read_xlsx_sheets(self, file_path: str) -> Tuple[List[pd.DataFrame], List[str]]:
        """Read all sheets from .xlsx file using pandas/openpyxl"""
        df_list = []
        sheet_names = []

        # Determine available engines
        engines_to_try = []
        try:
            import openpyxl
            engines_to_try.append('openpyxl')
        except ImportError:
            pass
        try:
            import xlrd
            engines_to_try.append('xlrd')
        except ImportError:
            pass
        if not engines_to_try:
            engines_to_try.append(None)

        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names_all = excel_file.sheet_names
            logger.info(f"Reading {len(sheet_names_all)} sheets from .xlsx file: {sheet_names_all}")

            for sheet_name in sheet_names_all:
                df_temp = None
                # Try with header=None first
                for engine in engines_to_try:
                    try:
                        df_temp = pd.read_excel(file_path, sheet_name=sheet_name, engine=engine, header=None)
                        if df_temp is not None and not df_temp.empty:
                            break
                    except Exception as e:
                        logger.debug(f"Failed to read sheet '{sheet_name}' with engine {engine} (header=None): {e}")
                        continue

                # If header=None failed, try without it
                if df_temp is None or df_temp.empty:
                    for engine in engines_to_try:
                        try:
                            df_temp = pd.read_excel(file_path, sheet_name=sheet_name, engine=engine)
                            if df_temp is not None and not df_temp.empty:
                                break
                        except Exception as e:
                            logger.debug(f"Failed to read sheet '{sheet_name}' with engine {engine}: {e}")
                            continue

                if df_temp is not None and not df_temp.empty:
                    df_list.append(df_temp)
                    sheet_names.append(sheet_name)
                    logger.info(f"Successfully read sheet '{sheet_name}' with shape {df_temp.shape}")
                else:
                    logger.warning(f"Failed to read sheet '{sheet_name}'")

            if not df_list:
                logger.error(f"All sheets failed to read for .xlsx file: {file_path}")
                return [], []
        except Exception as e:
            logger.error(f"Failed to read .xlsx file: {e}", exc_info=True)
            return [], []

        return df_list, sheet_names

    def _merge_dataframes(self, df_list: List[pd.DataFrame], sheet_names: List[str]) -> Optional[pd.DataFrame]:
        """Merge multiple DataFrames with column alignment, adding _sheet_name column"""
        if not df_list:
            logger.warning("_merge_dataframes: df_list is empty")
            return None

        merged_dfs = []
        for df, sheet_name in zip(df_list, sheet_names):
            if df is not None and not df.empty:
                try:
                    df_copy = df.copy()
                    df_copy['_sheet_name'] = sheet_name
                    merged_dfs.append(df_copy)
                    logger.debug(f"Added sheet '{sheet_name}' with shape {df.shape}")
                except Exception as e:
                    logger.error(f"Failed to process sheet '{sheet_name}': {e}")
            else:
                logger.warning(f"Sheet '{sheet_name}' is None or empty")

        if not merged_dfs:
            logger.error("_merge_dataframes: No valid dataframes after processing")
            return None

        # If only one sheet, return it directly
        if len(merged_dfs) == 1:
            result = merged_dfs[0]
            logger.debug(f"Single sheet, returning with shape {result.shape}")
            return result

        # Merge multiple sheets vertically
        try:
            # Get union of all column names
            all_columns = set()
            for df in merged_dfs:
                all_columns.update([str(c) for c in df.columns])
            all_columns = sorted([str(c) for c in all_columns])

            # Align columns for each DataFrame
            aligned_dfs = []
            for df in merged_dfs:
                df_aligned = df.copy()
                df_aligned.columns = [str(c) for c in df_aligned.columns]

                # Add missing columns
                missing_cols = [c for c in all_columns if c not in df_aligned.columns]
                if missing_cols:
                    missing_df = pd.DataFrame({col: None for col in missing_cols}, index=df_aligned.index)
                    df_aligned = pd.concat([df_aligned, missing_df], axis=1)

                # Reorder columns (_sheet_name at the end)
                col_order = [c for c in all_columns if c != '_sheet_name'] + ['_sheet_name']
                df_aligned = df_aligned[col_order]
                aligned_dfs.append(df_aligned)

            # Vertical concatenation
            merged = pd.concat(aligned_dfs, ignore_index=True)
            logger.info(f"Merged {len(merged_dfs)} sheets into one DataFrame with shape {merged.shape}")
            return merged
        except Exception as e:
            logger.warning(f"Failed to merge DataFrames: {e}, using first sheet only")
            return merged_dfs[0] if merged_dfs else None

    def _detect_header_row(self, df: pd.DataFrame, expected_cols: List[str]) -> int:
        """Detect which row contains the header based on expected column names"""
        best_row_idx, best_match_count = -1, -1
        expected_keywords = set()
        for col in expected_cols:
            expected_keywords.update(str(col).lower().split())
        logger.info(f"Expected keywords: {list(expected_keywords)[:10]}... (total: {len(expected_keywords)})")

        for idx in range(min(10, len(df))):
            row_values = df.iloc[idx].astype(str).tolist()
            match_count = sum(
                2 if self._clean_col_name(v).lower() in expected_keywords else
                (1 if set(self._clean_col_name(v).lower().split()) & expected_keywords else 0)
                for v in row_values
            )
            if match_count > best_match_count:
                best_match_count = match_count
                best_row_idx = idx
            logger.debug(f"Row {idx} match_count: {match_count}, best_match_count: {best_match_count}")

        logger.info(f"Best match: row_idx={best_row_idx}, match_count={best_match_count}")
        return best_row_idx

    def _apply_header(self, df: pd.DataFrame, header_row_idx: int) -> pd.DataFrame:
        """Apply header row to DataFrame and handle duplicate column names"""
        if header_row_idx < 0 or header_row_idx >= len(df):
            logger.warning(f"Invalid header_row_idx={header_row_idx}, returning df as-is")
            return df

        logger.info(f"Using row {header_row_idx} as header, will take rows {header_row_idx + 1} onwards")
        headers = [self._clean_col_name(h) for h in df.iloc[header_row_idx].tolist()]

        # Handle duplicate column names
        seen = {}
        new_headers = []
        for h in headers:
            if h in seen:
                seen[h] += 1
                new_headers.append(f"{h}.{seen[h]}")
            else:
                seen[h] = 0
                new_headers.append(h)

        # Check if there are remaining rows after header
        if header_row_idx + 1 >= len(df):
            logger.warning(f"header_row_idx={header_row_idx}, df has {len(df)} rows, no data rows after header")
            df_result = df.copy()
            df_result.columns = [self._clean_col_name(str(c)) for c in df.columns]
            return df_result

        df_result = df.iloc[header_row_idx + 1:].copy()
        df_result.columns = new_headers
        df_result.reset_index(drop=True, inplace=True)
        logger.info(f"Created df from rows {header_row_idx + 1} onwards, shape: {df_result.shape}")
        return df_result

    def _process_df_without_expected_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process DataFrame when no expected columns provided - detect if first row is header"""
        logger.info("Processing df without expected columns")
        if df is None or df.empty:
            logger.error("df is None or empty")
            return df

        df_result = df.copy()
        logger.info(f"Processing df with shape {df_result.shape}")

        if len(df_result) > 0:
            # Check if first row looks like a header
            first_row = df_result.iloc[0].astype(str).tolist()
            text_count = sum(
                1 for v in first_row
                if not str(v).replace('.', '').replace('-', '').isdigit() and str(v) != 'nan'
            )
            if text_count > len(first_row) * 0.5:
                logger.info(f"Using first row as header (text_count={text_count}/{len(first_row)})")
                df_result.columns = [self._clean_col_name(str(c)) for c in first_row]
                df_result = df_result[1:].reset_index(drop=True)
            else:
                logger.info("First row is not header, cleaning column names")
                df_result.columns = [self._clean_col_name(str(c)) for c in df_result.columns]
        else:
            logger.warning("df has 0 rows, cannot process header")
            df_result.columns = [self._clean_col_name(str(c)) for c in df_result.columns]

        logger.info(f"After header processing, df shape: {df_result.shape}")
        return df_result

    def _process_df_with_expected_cols(self, df: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
        """Process DataFrame with expected column matching"""
        logger.info(f"Processing df with expected columns ({len(expected_cols)})")
        if df is None or df.empty:
            logger.error("df is None or empty")
            return df

        best_row_idx = self._detect_header_row(df, expected_cols)

        if best_row_idx >= 0:
            df_result = self._apply_header(df, best_row_idx)
        else:
            logger.warning(f"No matching row found (best_row_idx={best_row_idx}), using df as fallback")
            df_result = df.copy()
            df_result.columns = [self._clean_col_name(str(c)) for c in df_result.columns]
            logger.info(f"Using df as fallback, shape: {df_result.shape}")

        return df_result

    def _load_excel_fallback(self, file_path: str) -> Optional[pd.DataFrame]:
        """Fallback loading without column matching"""
        logger.info(f"Fallback loading for {file_path}")
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.xls':
            df_list, sheet_names = self._read_xls_sheets(file_path)
        else:
            df_list, sheet_names = self._read_xlsx_sheets(file_path)

        if not df_list:
            logger.error(f"Fallback: All sheets failed to read for {file_path}")
            return None

        df = self._merge_dataframes(df_list, sheet_names)
        if df is None or df.empty:
            logger.error(f"Fallback: Failed to merge sheets for {file_path}")
            return None

        df.columns = [self._clean_col_name(str(c)) for c in df.columns]
        return df

    def _smart_load_excel(self, file_path: str, table_info: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """智能加载 Excel（支持 .xlsx 和 .xls 格式，支持多sheet，合并所有sheet的数据）"""
        logger.info(f"_smart_load_excel called for {file_path}")

        try:
            # Read all sheets based on file extension
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.xls':
                df_list, sheet_names = self._read_xls_sheets(file_path)
            else:
                df_list, sheet_names = self._read_xlsx_sheets(file_path)

            # Check if reading was successful
            if not df_list:
                logger.error(f"Failed to read any sheets from {file_path}")
                return None

            # Merge all sheets
            df_raw = self._merge_dataframes(df_list, sheet_names)
            if df_raw is None or df_raw.empty:
                logger.error(f"Failed to merge sheets for {file_path}")
                return None

            logger.info(f"Successfully read and merged sheets, df_raw shape: {df_raw.shape}")

            # Process DataFrame based on whether expected columns are provided
            expected_cols = table_info.get("column_names", [])
            logger.info(f"Expected columns: {expected_cols}, df_raw shape: {df_raw.shape}")

            if not expected_cols:
                # No expected columns - use heuristic header detection
                df = self._process_df_without_expected_cols(df_raw)
            else:
                # Expected columns provided - match against them
                df = self._process_df_with_expected_cols(df_raw, expected_cols)

            if df is not None and not df.empty:
                logger.info(f"Successfully loaded DataFrame with shape {df.shape} from {file_path}")
                return df
            else:
                logger.error(f"Final df is None or empty after processing for {file_path}")
                return None

        except Exception as e:
            logger.error(f"Smart load failed with exception: {e}", exc_info=True)
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")

            # Fallback: simple loading without column matching
            try:
                df = self._load_excel_fallback(file_path)
                return df if df is not None and not df.empty else None
            except Exception as e2:
                logger.error(f"Fallback load also failed: {e2}")
                import traceback
                logger.error(traceback.format_exc())
                return None
    
    def _build_meta_context(self, table_info: Dict[str, Any]) -> str:
        """构建 Meta Graph 语义上下文"""
        context_parts = []
        
        # Meta Graph 语义信息（帮助理解表格含义）
        if "meta_graph_triplets" in table_info:
            triplets = table_info["meta_graph_triplets"]
            child_rels = [t for t in triplets if "has_child" in t]
            col_rels = [t for t in triplets if "has_column_header" in t]
            
            if child_rels or col_rels:
                context_parts.append("# Table Semantics (from Meta Graph)")
                if child_rels:
                    context_parts.append("Hierarchical Structure:")
                    for t in child_rels[:15]:
                        context_parts.append(f"  {t}")
                if col_rels:
                    context_parts.append("Column Headers:")
                    for t in col_rels[:10]:
                        context_parts.append(f"  {t}")
        
        # 摘要
        if table_info.get("summary_text"):
            context_parts.append(f"\nSummary: {table_info['summary_text']}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _build_data_context(self, df: pd.DataFrame, original_df: pd.DataFrame = None) -> str:
        """构建数据上下文（使用原始数据的前 N 行）"""
        # 优先使用原始数据，以便 LLM 看到完整的表格结构
        sample_df = original_df if original_df is not None else df
        
        if sample_df is None or (hasattr(sample_df, 'empty') and sample_df.empty):
            return "Data Sample: No data available\nCurrent DataFrame Status: Empty or None"
        
        sample_rows = min(self.data_sample_rows, len(sample_df))
        
        # 确保 df 有效
        if df is None or (hasattr(df, 'empty') and df.empty):
            df = sample_df
        
        context = f"""Data Sample ({sample_rows} rows from original table):
{sample_df.head(sample_rows).to_string()}

Current DataFrame Status:
- Shape: {df.shape if df is not None else 'N/A'}
- Columns: {list(df.columns) if df is not None else 'N/A'}
- Types: {', '.join([f'{c}: {t}' for c, t in list(df.dtypes.items())[:15]]) if df is not None else 'N/A'}
"""
        return context

    # ==================== Step 1: Load Data ====================
    
    def load_data(self, state: ADGWorkflowStateV4) -> Dict[str, Any]:
        """Step 1: 加载 Schema 和 DataFrame"""
        # print(f"\n  [Step 1] Load Data")
        
        current_table = state.get("current_table", {}) or {}
        file_name = current_table.get("file_name", "")
        file_path = current_table.get("file_path", "")
        
        if not file_name:
            return {"error": "No file name provided"}
        
        table_name = os.path.splitext(os.path.basename(file_name))[0]
        # print(f"    Table: {table_name}")
        
        try:
            # 从 Meta Graph 获取 Schema（如果找不到会自动生成）
            # 传入 file_path 以便自动生成 meta graph
            table_info = self.schema_loader.get_table_info(table_name, excel_path=file_path)
            if file_path:
                table_info["file_path"] = file_path
            
            # 加载 DataFrame - 如果文件不存在，尝试模糊匹配
            df = None
            if file_path and os.path.exists(file_path):
                try:
                    df = self._smart_load_excel(file_path, table_info)
                except Exception as e:
                    logger.error(f"Exception in _smart_load_excel for {file_path}: {e}", exc_info=True)
                    df = None
            elif file_path:
                # 文件不存在，尝试通过文件ID前缀进行模糊匹配
                import glob
                file_dir = os.path.dirname(file_path)
                # 提取文件ID（文件名中"-"之前的部分）
                file_id = file_name.split('-')[0] if '-' in file_name else os.path.splitext(file_name)[0]
                # 尝试匹配相同ID的文件
                pattern = os.path.join(file_dir, f"{file_id}*")
                matches = glob.glob(pattern)
                if matches:
                    # 使用第一个匹配的文件
                    matched_file = matches[0]
                    logger.info(f"File not found: {file_path}, using matched file: {matched_file}")
                    df = self._smart_load_excel(matched_file, table_info)
                else:
                    logger.warning(f"File not found and no matches: {file_path}")
            
            if df is not None and not df.empty:
                # print(f"    ✅ Loaded: {df.shape}")
                
                # 更新 table_info
                table_info["actual_columns"] = list(df.columns)
                table_info["actual_dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
                table_info["data_sample"] = df.head(self.data_sample_rows).to_string()
            else:
                if file_path and os.path.exists(file_path):
                    logger.warning(f"Failed to load DataFrame from {file_path} (file exists but df is None or empty)")
                else:
                    logger.warning(f"Failed to load DataFrame from {file_path} (file does not exist)")
                df = None
            
            # 安全地创建 original_dataframe 的副本
            original_df = None
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                try:
                    original_df = df.copy()
                except Exception as e:
                    logger.warning(f"Failed to copy DataFrame: {e}")
                    original_df = None
            
            return {
                "table_info": table_info,
                "dataframe": df,
                "original_dataframe": original_df,
                "current_op_index": 0,
                "execution_trace": [],
                "replan_count": 0,
                "max_replan": 3,
                "llm_call_history": []
            }
        except Exception as e:
            logger.error(f"Failed to load: {e}")
            return {"error": str(e)}
    
    # ==================== Step 2: ADG Planning ====================
    
    def generate_adg(self, state: ADGWorkflowStateV4) -> Dict[str, Any]:
        """Step 2: 生成 ADG 规划"""
        print(f"\n  [Step 2] Generate ADG")
        
        user_question = state.get("user_question", "")
        table_info = state.get("table_info", {})
        
        # 调用 ADG 生成器
        adg_result = self.adg_generator.generate_adg(user_question, table_info)
        
        print(f"    ✅ Generated {len(adg_result.candidate_paths)} candidate paths")
        
        # 记录 LLM 调用历史
        llm_history = state.get("llm_call_history", []) or []
        history_entry = {
            "type": "adg_generation",
            "prompt": adg_result.prompt,
            "response": adg_result.response,
            "context": {
                "user_question": user_question,
                "table_info_summary": {
                    "columns": table_info.get("column_names", [])[:10],
                    "row_count": table_info.get("row_count"),
                }
            }
        }
        llm_history.append(history_entry)
        
        self._emit_event(
            name="excel_agent.plan.delta",
            event_data={
                "type": "adg_generation",
                "step": "llm_response",
                "content": f"生成任务分解路径：\n{adg_result.response}",
                "prompt_preview": adg_result.prompt[:200] if adg_result.prompt else ""
            }
        )

        self._emit_event(
            name="excel_agent.plan.done",
            event_data={
                "type": "adg_generation",
                "step": "plan_done",
                "content": "<plan_done>"
            }
        )
        
        return {
            "candidate_paths": adg_result.candidate_paths,
            "llm_call_history": llm_history
        }
    
    # ==================== Step 3: MCTS Planning ====================
    
    def plan_with_mcts(self, state: ADGWorkflowStateV4) -> Dict[str, Any]:
        """Step 3: MCTS 路径选择"""
        print(f"\n  [Step 3] MCTS Planning")
        
        candidate_paths = state.get("candidate_paths", [])

        self._emit_event(
            name="excel_agent.task.start",
            event_data={
                "type": "mcts_planning",
                "operation": "MCTS Planning",
                "content": "<mcts_start>"
            }
        )
        
        optimal_path = self.mcts_planner.find_optimal_path(
            candidate_paths,
            initial_state={"file_loaded", "schema", "data_loaded"}
        )
        
        ops = " -> ".join(optimal_path.operations)
        print(f"    ✅ Path: {ops}")

        self._emit_event(
            name="excel_agent.task.delta",
            event_data={
                "type": "mcts_planning",
                "operation": "MCTS Planning",
                "content": f"✅ Path: {ops}",
                "clean": True
            }
        )
        
        # 重置执行索引
        replan_count = state.get("replan_count", 0)
        if state.get("execution_trace"):
            replan_count += 1

        self._emit_event(
            name="excel_agent.task.done",
            event_data={
                "type": "mcts_planning",
                "operation": f"MCTS Planning | 📈 {ops}",
                "content": "<mcts_done>"
            }
        )
        
        return {
            "selected_path": optimal_path,
            "current_op_index": 0,
            "replan_count": replan_count
        }
    
    # ==================== Step 4: 分步执行 ====================
    
    def execute_step(self, state: ADGWorkflowStateV4) -> Dict[str, Any]:
        """Step 4: 分步执行操作"""
        selected_path = state.get("selected_path")
        current_idx = state.get("current_op_index", 0)
        df = state.get("dataframe")
        original_df = state.get("original_dataframe")
        execution_trace = state.get("execution_trace", []) or []
        
        if not selected_path or current_idx >= len(selected_path.operations):
            # print(f"\n  [Step 4] All operations completed")
            return {}
        
        op_name = selected_path.operations[current_idx]
        operation = OPERATIONS_BY_NAME.get(op_name)
        
        # print(f"\n  [Step 4] Execute: {op_name} ({current_idx + 1}/{len(selected_path.operations)})")
        
        if not operation:
            print(f"    ⚠️ Unknown operation: {op_name}, skipping")
            execution_trace.append({
                "operation": op_name,
                "success": True,
                "skipped": True
            })
            return {
                "current_op_index": current_idx + 1,
                "execution_trace": execution_trace
            }
        
        self._emit_event(
            name="excel_agent.task.start",
            event_data={
                "type": "code_generation",
                "operation": op_name,
                "step_index": current_idx,
                "content": "<code_start>"
            }
        )
        
        # 生成代码
        code_result = self._generate_step_code(
            operation=operation,
            df=df,
            original_df=original_df,
            user_question=state.get("user_question", ""),
            question_type=state.get("question_type", ""),
            table_info=state.get("table_info", {}),
            execution_trace=execution_trace
        )
        
        # 处理返回值（可能是 code 或 (code, prompt, response)）
        if isinstance(code_result, tuple) and len(code_result) == 3:
            code, prompt, raw_response = code_result
            # 记录 LLM 调用历史
            llm_history = state.get("llm_call_history", []) or []
            history_entry = {
                "type": "code_generation",
                "operation": op_name,
                "step_index": current_idx,
                "prompt": prompt,
                "response": raw_response,
                "context": {
                    "user_question": state.get("user_question", ""),
                    "df_shape": df.shape if df is not None else None,
                    "previous_operations": [t.get("operation") for t in execution_trace[-3:]]
                }
            }
            llm_history.append(history_entry)


            self._emit_event(
                name="excel_agent.task.delta",
                event_data={
                    "type": "code_generation",
                    "operation": op_name,
                    "step_index": current_idx,
                    "content": code,
                    "mode": "code",
                    "prompt_preview": prompt[:200] if prompt else "",
                    "clean": True,
                }
            )
            
            state_updates = {"llm_call_history": llm_history}
        else:
            code = code_result
            state_updates = {}
        
        # 执行代码
        # 安全地创建 df 的副本
        df_before = None
        if df is not None and isinstance(df, pd.DataFrame):
            try:
                df_before = df.copy()
            except Exception as e:
                logger.warning(f"Failed to copy DataFrame before execution: {e}")
                df_before = df  # 如果 copy 失败，至少保留原始引用
        
        exec_result = self._execute_code(code, df)
        
        if exec_result.get("success"):
            result_df = exec_result.get("dataframe", df)
            # print(f"    ✅ Success (Shape: {result_df.shape if isinstance(result_df, pd.DataFrame) else 'N/A'})")
            
            execution_trace.append({
                "operation": op_name,
                "code": code,
                "success": True,
                "shape": result_df.shape if isinstance(result_df, pd.DataFrame) else None
            })

            self._emit_event(
                name="excel_agent.task.done",
                event_data={
                    "type": "code_execution",
                    "operation": f"{op_name} | ✅ Execution Success",
                    "step_index": current_idx,
                    "content": f"✅ Execution Success: (Shape: {result_df.shape if isinstance(result_df, pd.DataFrame) else 'N/A'})"
                }
            )
            
            return {
                "dataframe": result_df,
                "current_op_index": current_idx + 1,
                "execution_trace": execution_trace,
                **state_updates
            }
        else:
            error_msg = exec_result.get("error", "Unknown error")
            # print(f"    ❌ Failed: {error_msg}")
            
            execution_trace.append({
                "operation": op_name,
                "code": code,
                "success": False,
                "error": error_msg
            })

            self._emit_event(
                name="excel_agent.task.done",
                event_data={
                    "type": "code_execution",
                    "operation": f"{op_name} | ❌ Execution Failed:「{error_msg[:50]}...」",
                    "step_index": current_idx,
                    "content": f"❌ Execution Failed: {error_msg}"
                }
            )

            return {
                "current_op_index": current_idx + 1,  # 继续下一步，不要卡住
                "execution_trace": execution_trace,
                **state_updates
            }
    
    def _generate_step_code(
        self, 
        operation: MacroOperation,
        df: pd.DataFrame,
        original_df: pd.DataFrame,
        user_question: str,
        question_type: str,
        table_info: Dict[str, Any],
        execution_trace: List[Dict]
    ) -> str:
        """为单个操作生成代码"""
        
        # 检查 DataFrame 是否有效
        if df is None:
            df = original_df
        if df is None or (hasattr(df, 'empty') and df.empty):
            return "# Error: DataFrame is None or empty\npass"
        
        actual_columns = list(df.columns)
        
        # Meta Graph 语义信息
        meta_context = self._build_meta_context(table_info)
        
        # 直接展示 150 行原始数据（保留层级信息）
        sample_df = original_df if original_df is not None else df
        if sample_df is None or (hasattr(sample_df, 'empty') and sample_df.empty):
            data_sample = "No data available"
        else:
            data_sample = sample_df.head(self.data_sample_rows).to_string()
        
        # 当前 df 状态
        if df is None or (hasattr(df, 'empty') and df.empty):
            current_preview = "Empty"
        else:
            current_preview = df.head(20).to_string() if len(df) > 0 else "Empty"
        
        # 执行历史
        history = ""
        if execution_trace:
            history = "# Previous Operations\n"
            for i, t in enumerate(execution_trace[-3:], 1):
                status = "✓" if t.get("success") else "✗"
                history += f"{i}. {t['operation']} [{status}]"
                if not t.get("success") and t.get("error"):
                    history += f" - {t['error'][:40]}"
                history += "\n"
        
        prompt = f"""Generate Python/Pandas code for this operation.

# Question
{user_question}

# Operation: {operation.name}
{operation.description}

{meta_context}

# Table Data ({self.data_sample_rows} rows)
{data_sample}

# Current df (Shape: {df.shape})
{current_preview}

{history}

# Requirements
- df is already loaded, do NOT read files
- Output ONLY executable Python code with markdown format, e.g., ```python\n...```
- Keep result in df
- Available: pd, np

Code: 
"""
        try:
            raw_response = self.llm_client.call_api(prompt)
            code = self._clean_code(raw_response)
            code = self._validate_and_fix_column_names(code, actual_columns)
            return code, prompt, raw_response  # 返回 prompt 和原始响应
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return "pass"
    
    def _validate_and_fix_column_names(self, code: str, actual_columns: List[str]) -> str:
        """验证并尝试修复代码中的列名"""
        # 简单的列名检查 - 如果发现明显错误的列名引用，尝试找到最接近的匹配
        import difflib
        
        # 提取代码中可能的列名引用 (在引号内的字符串)
        potential_cols = re.findall(r"['\"]([^'\"]+)['\"]", code)
        
        for col in potential_cols:
            if col not in actual_columns and len(col) > 2:
                # 尝试找到最接近的列名
                close_matches = difflib.get_close_matches(col, actual_columns, n=1, cutoff=0.6)
                if close_matches:
                    # 替换为正确的列名
                    code = code.replace(f"'{col}'", f"'{close_matches[0]}'")
                    code = code.replace(f'"{col}"', f'"{close_matches[0]}"')
                    logger.info(f"Fixed column name: '{col}' -> '{close_matches[0]}'")
        
        return code
    
    def _clean_code(self, code: str) -> str:
        """清理代码"""
        code = code.strip()
        if code.startswith("```python"):
            code = code[9:].strip()
        if code.startswith("```"):
            code = code[3:].strip()
        if code.endswith("```"):
            code = code[:-3].strip()
        
        # 移除文件读取
        if "pd.read_excel" in code or "pd.read_csv" in code:
            lines = code.split('\n')
            code = '\n'.join([l for l in lines if 'read_excel' not in l and 'read_csv' not in l])
        
        return code
    
    def _execute_code(self, code: str, df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """执行代码（包含常用库）"""
        try:
            # 如果 df 为 None，返回错误
            if df is None or not isinstance(df, pd.DataFrame):
                return {"success": False, "error": "DataFrame is None or invalid"}
            
            # 安全检查
            danger_keywords = ["exit(", "quit(", "sys.exit", "os.system", "subprocess", "__import__"]
            for kw in danger_keywords:
                if kw in code:
                    return {"success": False, "error": f"Forbidden: {kw}"}
            
            # 准备执行环境（包含常用库）
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # 安全地创建 df 的副本
            df_copy = df
            try:
                df_copy = df.copy()
            except Exception as e:
                logger.warning(f"Failed to copy DataFrame in _execute_code: {e}")
                df_copy = df  # 如果 copy 失败，使用原始引用
            
            local_vars = {"df": df_copy, "pd": pd, "np": np, "plt": plt}
            global_vars = {
                "pd": pd, 
                "np": np, 
                "plt": plt,
                "__builtins__": __builtins__
            }
            
            exec(code, global_vars, local_vars)
            result_df = local_vars.get("df", df)
            
            # 确保返回 DataFrame
            if not isinstance(result_df, pd.DataFrame):
                return {"success": True, "dataframe": df}
            
            return {"success": True, "dataframe": result_df}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ==================== Step 5: 判断是否继续/重规划 ====================
    
    def should_continue(self, state: ADGWorkflowStateV4) -> str:
        """判断下一步动作"""
        selected_path = state.get("selected_path")
        current_idx = state.get("current_op_index", 0)
        execution_trace = state.get("execution_trace", [])
        replan_count = state.get("replan_count", 0)
        max_replan = state.get("max_replan", 3)
        
        # 检查是否所有操作都完成
        if selected_path and current_idx >= len(selected_path.operations):
            return "generate_answer"
        
        # 检查最近的执行是否失败
        if execution_trace:
            recent_failures = sum(1 for t in execution_trace[-3:] if not t.get("success", True))
            if recent_failures >= 2 and replan_count < max_replan:
                print(f"    ⚠️ Multiple failures, triggering replan ({replan_count + 1}/{max_replan})")
                return "replan"
        
        return "continue"
    
    # ==================== Step 6: 生成答案 ====================
    
    def generate_answer(self, state: ADGWorkflowStateV4) -> Dict[str, Any]:
        """Step 6: 生成最终答案"""
        print(f"\n  [Step 5] Generate Answer")
        
        df = state.get("dataframe")
        original_df = state.get("original_dataframe")
        user_question = state.get("user_question", "")
        question_type = state.get("question_type", "")
        execution_trace = state.get("execution_trace", [])
        table_info = state.get("table_info", {})
        
        # Visualization 类型
        if question_type == "Visualization":
            return self._generate_visualization_answer(df, original_df, user_question, table_info, state)
        
        # 检查 DataFrame 状态
        if df is None or (hasattr(df, 'empty') and df.empty):
            print("    ⚠️ DataFrame empty, using original data")
            df = original_df
        
        if df is None or (hasattr(df, 'empty') and df.empty):
            return {"final_answer": "[Final Answer]: Unable to process data - DataFrame is None or empty"}
        
        actual_columns = list(df.columns)
        
        # Meta Graph 语义
        meta_context = self._build_meta_context(table_info)
        
        # 展示数据
        data_sample = df.head(100).to_string()
        
        # 执行历史
        ops_history = "\n".join([
            f"{i+1}. {t['operation']} {'✓' if t.get('success') else '✗'}" 
            for i, t in enumerate(execution_trace)
        ])

        self._emit_event(
            name="excel_agent.answer.start",
            event_data={
                "type": "answer_generation",
                "content": "<answer_start>"
            }
        )
        
        prompt = f"""Answer the question based on the data.

# Question
{user_question}

{meta_context}

# Operations Executed
{ops_history}

# Data (Shape: {df.shape})
{data_sample}

# Instructions
- Analyze the data to answer the question
- If calculation needed, write Python code (df is available)
- **End with**: [Final Answer]: your_answer

Response:
"""
        try:
            response = self.llm_client.call_api(prompt)
            
            # 记录 LLM 调用历史
            llm_history = state.get("llm_call_history", []) or []
            history_entry = {
                "type": "answer_generation",
                "prompt": prompt,
                "response": response,
                "context": {
                    "user_question": user_question,
                    "df_shape": df.shape if df is not None else None,
                    "execution_trace_summary": ops_history
                }
            }
            llm_history.append(history_entry)
            
            # 尝试执行代码块
            if "```python" in response:
                code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
                if code_match:
                    code = code_match.group(1)
                    code = self._validate_and_fix_column_names(code, actual_columns)
                    try:
                        # 安全地创建 df 的副本
                        df_copy = df
                        if df is not None and isinstance(df, pd.DataFrame):
                            try:
                                df_copy = df.copy()
                            except Exception as e:
                                logger.warning(f"Failed to copy DataFrame in generate_final_answer: {e}")
                                df_copy = df  # 如果 copy 失败，使用原始引用
                        
                        local_vars = {"df": df_copy, "pd": pd, "np": np}
                        exec(code, {"pd": pd, "np": np, "__builtins__": __builtins__}, local_vars)
                        if "result" in local_vars:
                            print(f"    Code result: {local_vars['result']}")
                    except Exception as e:
                        logger.warning(f"Code execution in answer: {e}")
            
            # 提取 Final Answer
            if "[Final Answer]" not in response:
                lines = response.strip().split('\n')
                last_meaningful = [l for l in lines if l.strip()][-1] if lines else response
                response = f"{response}\n[Final Answer]: {last_meaningful}"
            
            print("    ✅ Answer Generated")
            llm_history = state.get("llm_call_history", []) or []
            return {
                "final_answer": response,
                "llm_call_history": llm_history
            }
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            llm_history = state.get("llm_call_history", []) or []
            return {
                "final_answer": f"[Final Answer]: Error generating answer: {e}",
                "llm_call_history": llm_history
            }
    
    def _generate_visualization_answer(
        self, 
        df: pd.DataFrame, 
        original_df: pd.DataFrame,
        user_question: str,
        table_info: Dict[str, Any],
        state: Optional[ADGWorkflowStateV4] = None
    ) -> Dict[str, Any]:
        """生成可视化代码（独立可执行，使用 iloc 避免列名问题）"""
        sample_df = original_df if original_df is not None else df
        
        if sample_df is None or (hasattr(sample_df, 'empty') and sample_df.empty):
            return {"final_answer": "import matplotlib.pyplot as plt\nplt.figure()\nplt.title('No Data')\nplt.show()"}
        
        actual_columns = list(sample_df.columns)
        
        # 使用 header=None 加载时的数据预览，让 LLM 看到真实的数据结构
        # 同时提供列索引映射
        col_mapping = "\n".join([f"  Column {i}: {str(col)[:50]}" for i, col in enumerate(actual_columns[:20])])
        
        # 显示带行号的数据，方便 LLM 理解数据位置
        data_preview_lines = []
        for idx in range(min(50, len(sample_df))):
            row = sample_df.iloc[idx]
            row_str = "  ".join([f"{v}" for v in row.values[:10]])
            data_preview_lines.append(f"Row {idx}: {row_str[:150]}")
        data_preview = "\n".join(data_preview_lines)
        
        prompt = f"""Generate Python visualization code for the following task.

# Task
{user_question}

# Data Structure
- Shape: {sample_df.shape} (rows, columns)
- Column indices and names:
{col_mapping}

# Data Preview (first 50 rows, first 10 columns)
{data_preview}

# CRITICAL REQUIREMENTS
1. Use df.iloc[row, col] for data access - DO NOT use column names directly
2. Column names may contain special characters or be 'nan', so always use iloc with numeric indices
3. Extract numeric values properly, handle potential NaN values

# Code Template (MUST follow this exact structure):
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_excel('table.xlsx')

# Extract data using iloc (row_index, column_index)
# Example: df.iloc[0, 1] gets value at row 0, column 1
# Example: df.iloc[1:10, 2] gets rows 1-9 from column 2
# Example: df.iloc[:, 0].tolist() gets all values from column 0

# [Your data extraction code here - use iloc!]

# [Your plotting code here]

plt.tight_layout()
plt.show()
```

# Output ONLY the complete Python code, nothing else.

Code:
"""
        try:
            raw_response = self.llm_client.call_api(prompt)
            code = self._clean_code(raw_response)
            
            # 记录 LLM 调用历史
            if state is not None:
                llm_history = state.get("llm_call_history", []) or []
                history_entry = {
                    "type": "visualization_code_generation",
                    "prompt": prompt,
                    "response": raw_response,
                    "context": {
                        "user_question": user_question,
                        "df_shape": df.shape if df is not None else None,
                    }
                }
                llm_history.append(history_entry)
                
                # 发送流式事件
                self._emit_event({
                    "type": "llm_response",
                    "step": "visualization_code_generation",
                    "response": raw_response,
                    "prompt_preview": prompt[:200] if prompt else ""
                })
            
            # 确保代码包含必要的导入和数据加载
            if "import pandas" not in code:
                code = "import pandas as pd\n" + code
            if "import matplotlib" not in code:
                lines = code.split('\n')
                # 在 pandas import 后添加 matplotlib import
                for i, line in enumerate(lines):
                    if 'import pandas' in line:
                        lines.insert(i + 1, "import matplotlib.pyplot as plt")
                        break
                else:
                    code = "import matplotlib.pyplot as plt\n" + code
                code = '\n'.join(lines)
            
            if "pd.read_excel" not in code and "pd.read_csv" not in code:
                # 在 import 语句后添加数据加载
                lines = code.split('\n')
                insert_idx = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('import') or line.strip().startswith('from'):
                        insert_idx = i + 1
                code_lines = lines[:insert_idx] + ["\ndf = pd.read_excel('table.xlsx')\n"] + lines[insert_idx:]
                code = '\n'.join(code_lines)
            
            if "plt.show()" not in code:
                code += "\nplt.show()"
            
            # 修复常见问题：将列名访问改为 iloc
            code = self._fix_column_access(code, actual_columns)
            
            print("    ✅ Visualization Code Generated")
            result = {"final_answer": code}
            if state is not None:
                llm_history = state.get("llm_call_history", []) or []
                result["llm_call_history"] = llm_history
            return result
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            return {"final_answer": f"import pandas as pd\nimport matplotlib.pyplot as plt\ndf = pd.read_excel('table.xlsx')\nplt.figure()\nplt.title('Error')\nplt.show()"}
    
    def _fix_column_access(self, code: str, columns: List[str]) -> str:
        """修复代码中的列名访问问题，将 df['colname'] 转换为 df.iloc[:, idx]"""
        import re
        
        # 创建列名到索引的映射
        col_to_idx = {str(col): i for i, col in enumerate(columns)}
        
        # 修复 df['nan'] 这种明显错误的列名
        code = re.sub(r"df\['nan'\]", "df.iloc[:, 0]", code)
        code = re.sub(r"df\['nan\.\d+'\]", "df.iloc[:, 0]", code)
        
        # 修复 df['Unnamed: X'] 模式
        def replace_unnamed(match):
            try:
                idx = int(match.group(1))
                return f"df.iloc[:, {idx}]"
            except:
                return match.group(0)
        code = re.sub(r"df\['Unnamed: (\d+)'\]", replace_unnamed, code)
        
        return code


def build_adg_workflow_v4(meta_graphs_dir: Optional[str] = None, data_sample_rows: int = 150, event_callback=None) -> Tuple[StateGraph, ADGWorkflowV4]:
    """构建 ADG Workflow V4
    
    流程: Load → ADG → MCTS → [Execute Step]* → Answer
    
    Args:
        meta_graphs_dir: Meta Graph 目录
        data_sample_rows: 提供给 LLM 的数据样本行数
        event_callback: 事件回调函数，用于流式输出 LLM 响应
    """
    workflow = ADGWorkflowV4(meta_graphs_dir, data_sample_rows=data_sample_rows, event_callback=event_callback)
    
    graph = StateGraph(ADGWorkflowStateV4)
    
    # 添加节点
    graph.add_node("load_data", workflow.load_data)
    graph.add_node("generate_adg", workflow.generate_adg)
    graph.add_node("plan_with_mcts", workflow.plan_with_mcts)
    graph.add_node("execute_step", workflow.execute_step)
    graph.add_node("generate_answer", workflow.generate_answer)
    
    # 添加边
    graph.add_edge(START, "load_data")
    graph.add_edge("load_data", "generate_adg")
    graph.add_edge("generate_adg", "plan_with_mcts")
    graph.add_edge("plan_with_mcts", "execute_step")
    
    # 条件边：判断继续执行还是重规划还是生成答案
    graph.add_conditional_edges(
        "execute_step",
        workflow.should_continue,
        {
            "continue": "execute_step",
            "replan": "plan_with_mcts",
            "generate_answer": "generate_answer"
        }
    )
    
    graph.add_edge("generate_answer", END)
    
    print("  [ADG Workflow V4 Built] Load → ADG → MCTS → Execute Steps → Answer")
    
    return graph.compile(), workflow

