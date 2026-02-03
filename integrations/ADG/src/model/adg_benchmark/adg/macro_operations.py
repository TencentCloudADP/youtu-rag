"""
宏操作定义（ADG - Abstract/Macro Graph）
定义 Excel Agent 的高层语义操作，每个操作包含前置条件和后置条件
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class OperationCategory(Enum):
    """操作分类"""
    DATA_UNDERSTANDING = "data_understanding"
    DATA_CLEANING = "data_cleaning"
    FILTERING_EXTRACTION = "filtering_extraction"
    TRANSFORMATION = "transformation"
    AGGREGATION = "aggregation"
    MULTI_TABLE = "multi_table"
    VALIDATION = "validation"


@dataclass
class StateCondition:
    """状态条件（前置或后置）"""
    key: str  # 状态键名
    required_value: Any = None  # 期望值（None表示只需要存在）
    description: str = ""
    
    def is_satisfied(self, state: Dict[str, Any]) -> bool:
        """检查条件是否满足"""
        if self.key not in state:
            return False
        if self.required_value is None:
            return True
        return state[self.key] == self.required_value


@dataclass
class MacroOperation:
    """宏操作定义"""
    name: str
    category: OperationCategory
    description: str
    preconditions: List[StateCondition] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)  # 输出的状态键
    estimated_cost: float = 1.0  # 预估成本（默认为1，相对值）
    code_template: str = ""  # 代码生成模板
    
    def can_execute(self, state: Dict[str, Any]) -> bool:
        """检查是否可以执行"""
        return all(cond.is_satisfied(state) for cond in self.preconditions)
    
    def get_dependencies(self) -> Set[str]:
        """获取依赖的状态键"""
        return {cond.key for cond in self.preconditions}


# ============================================================================
# I. 数据理解与结构识别（Data Understanding）
# ============================================================================

DETECT_SCHEMA = MacroOperation(
    name="DetectSchema",
    category=OperationCategory.DATA_UNDERSTANDING,
    description="Detect table structure: identify column names, types, ranges, and missing values.",
    preconditions=[
        StateCondition("file_loaded", True, "File loaded")
    ],
    postconditions=["schema", "column_names", "column_types", "row_count", "null_info"],
    estimated_cost=0.5,
    code_template="""
# DetectSchema
df_info = {{
    'schema': df.dtypes.to_dict(),
    'column_names': list(df.columns),
    'column_types': {{col: str(dtype) for col, dtype in df.dtypes.items()}},
    'row_count': len(df),
    'null_info': df.isnull().sum().to_dict()
}}
"""
)

INSPECT_COLUMN = MacroOperation(
    name="InspectColumn",
    category=OperationCategory.DATA_UNDERSTANDING,
    description="Inspect column: statistics, type inference, and outlier detection for a specific column.",
    preconditions=[
        StateCondition("schema", description="Schema detected")
    ],
    postconditions=["column_profiles"],
    estimated_cost=0.8,
    code_template="""
# InspectColumn: {column_name}
profile = {{
    'unique_count': df['{column_name}'].nunique(),
    'null_count': df['{column_name}'].isnull().sum(),
    'dtype': str(df['{column_name}'].dtype),
    'sample_values': df['{column_name}'].dropna().head(5).tolist()
}}
"""
)

INFER_ENTITY_TYPE = MacroOperation(
    name="InferEntityType",
    category=OperationCategory.DATA_UNDERSTANDING,
    description="Infer entity type: identify if a column is ID, Category, Date, Numeric, or Text.",
    preconditions=[
        StateCondition("column_profiles", description="Columns inspected")
    ],
    postconditions=["entity_types"],
    estimated_cost=1.0,
    code_template="""
# InferEntityType
entity_types = {{}}
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        entity_types[col] = 'numeric'
    elif pd.api.types.is_datetime64_any_dtype(df[col]):
        entity_types[col] = 'date'
    elif df[col].nunique() < 10:
        entity_types[col] = 'category'
    else:
        entity_types[col] = 'text'
"""
)

# ============================================================================
# II. 数据清理（Data Cleaning）
# ============================================================================

CLEAN_MISSING = MacroOperation(
    name="CleanMissing",
    category=OperationCategory.DATA_CLEANING,
    description="Handle missing values: Drop rows or fill NaN with mean/mode/value.",
    preconditions=[
        StateCondition("column_profiles", description="Columns inspected")
    ],
    postconditions=["cleaned_missing"],
    estimated_cost=1.2,
    code_template="""
# CleanMissing: {strategy}
if '{strategy}' == 'drop':
    df = df.dropna(subset=['{column_name}'])
elif '{strategy}' == 'fill_mean':
    df['{column_name}'].fillna(df['{column_name}'].mean(), inplace=True)
elif '{strategy}' == 'fill_mode':
    df['{column_name}'].fillna(df['{column_name}'].mode()[0], inplace=True)
"""
)

NORMALIZE_FORMAT = MacroOperation(
    name="NormalizeFormat",
    category=OperationCategory.DATA_CLEANING,
    description="Normalize format: Convert date formats, numeric formats, or trim text.",
    preconditions=[
        StateCondition("entity_types", description="Entity types inferred")
    ],
    postconditions=["normalized_format"],
    estimated_cost=1.0,
    code_template="""
# NormalizeFormat
if entity_types['{column_name}'] == 'date':
    df['{column_name}'] = pd.to_datetime(df['{column_name}'], errors='coerce')
elif entity_types['{column_name}'] == 'text':
    df['{column_name}'] = df['{column_name}'].str.strip()
"""
)

DEDUPLICATE_ROWS = MacroOperation(
    name="DeduplicateRows",
    category=OperationCategory.DATA_CLEANING,
    description="Deduplicate rows: Remove duplicate rows based on primary key or specific columns.",
    preconditions=[
        StateCondition("schema", description="Schema detected")
    ],
    postconditions=["deduplicated"],
    estimated_cost=1.5,
    code_template="""
# DeduplicateRows
df = df.drop_duplicates(subset={columns}, keep='first')
"""
)

# ============================================================================
# III. 数据过滤与提取（Filtering / Extraction）
# ============================================================================

FILTER_ROWS = MacroOperation(
    name="FilterRows",
    category=OperationCategory.FILTERING_EXTRACTION,
    description="Filter rows: Filter dataframe based on conditions (>, <, =, contains).",
    preconditions=[
        StateCondition("schema", description="Schema detected")
    ],
    postconditions=["filtered_df"],
    estimated_cost=0.8,
    code_template="""
# FilterRows: {condition}
df = df[{condition}]
"""
)

SELECT_COLUMNS = MacroOperation(
    name="SelectColumns",
    category=OperationCategory.FILTERING_EXTRACTION,
    description="Select columns: Keep only relevant columns for the task.",
    preconditions=[
        StateCondition("schema", description="Schema detected")
    ],
    postconditions=["selected_columns"],
    estimated_cost=0.3,
    code_template="""
# SelectColumns
df = df[{column_list}]
"""
)

SORT_VALUES = MacroOperation(
    name="SortValues",
    category=OperationCategory.FILTERING_EXTRACTION,
    description="Sort values: Sort dataframe by a column (ascending/descending).",
    preconditions=[
        StateCondition("column_profiles", description="Columns inspected")
    ],
    postconditions=["sorted_df"],
    estimated_cost=0.6,
    code_template="""
# SortValues
df = df.sort_values(by='{column_name}', ascending={ascending})
"""
)

# ============================================================================
# IV. 数据转换（Transformations）
# ============================================================================

DERIVE_COLUMN = MacroOperation(
    name="DeriveColumn",
    category=OperationCategory.TRANSFORMATION,
    description="Derive column: Create a new column using a formula (arithmetic, string ops).",
    preconditions=[
        StateCondition("column_profiles", description="Columns inspected")
    ],
    postconditions=["derived_column"],
    estimated_cost=1.0,
    code_template="""
# DeriveColumn: {new_column_name}
df['{new_column_name}'] = {formula}
"""
)

BIN_OR_GROUP = MacroOperation(
    name="BinOrGroup",
    category=OperationCategory.TRANSFORMATION,
    description="Binning/Grouping: Categorize continuous values into bins or group text.",
    preconditions=[
        StateCondition("entity_types", description="Entity types inferred")
    ],
    postconditions=["binned_column"],
    estimated_cost=1.2,
    code_template="""
# BinOrGroup: {column_name}
df['{new_column_name}'] = pd.cut(df['{column_name}'], bins={bins}, labels={labels})
"""
)

EXPLODE_OR_FLATTEN = MacroOperation(
    name="ExplodeOrFlatten",
    category=OperationCategory.TRANSFORMATION,
    description="Explode/Flatten: Split text column by delimiter and explode to rows.",
    preconditions=[
        StateCondition("column_profiles", description="Columns inspected")
    ],
    postconditions=["exploded_column"],
    estimated_cost=1.5,
    code_template="""
# ExplodeOrFlatten: {column_name}
df['{column_name}_split'] = df['{column_name}'].str.split('{delimiter}')
df = df.explode('{column_name}_split')
"""
)

# ============================================================================
# V. 聚合与统计（Aggregation / Analytics）
# ============================================================================

AGGREGATE = MacroOperation(
    name="Aggregate",
    category=OperationCategory.AGGREGATION,
    description="Aggregate: Compute statistics (Sum, Avg, Count, Median, Mode).",
    preconditions=[
        StateCondition("entity_types", description="Entity types inferred")
    ],
    postconditions=["aggregated_result"],
    estimated_cost=1.5,
    code_template="""
# Aggregate: {agg_func}
result = df.agg({{'{column_name}': '{agg_func}'}})
"""
)

GROUP_BY = MacroOperation(
    name="GroupBy",
    category=OperationCategory.AGGREGATION,
    description="GroupBy: Group by columns and compute statistics.",
    preconditions=[
        StateCondition("entity_types", description="Entity types inferred")
    ],
    postconditions=["grouped_df"],
    estimated_cost=1.8,
    code_template="""
# GroupBy: {group_columns}
grouped = df.groupby({group_columns})
"""
)

RANK = MacroOperation(
    name="Rank",
    category=OperationCategory.AGGREGATION,
    description="Rank: Assign ranks to data based on a column.",
    preconditions=[
        StateCondition("aggregated_result", description="Aggregated")
    ],
    postconditions=["ranked_result"],
    estimated_cost=0.8,
    code_template="""
# Rank
df['rank'] = df['{column_name}'].rank(ascending={ascending})
"""
)

VALIDATE_AGGREGATION_RESULT = MacroOperation(
    name="ValidateAggregationResult",
    category=OperationCategory.VALIDATION,
    description="Validate Result: Check if aggregation result contains anomalies (NaN, Zero).",
    preconditions=[
        StateCondition("aggregated_result", description="Aggregated")
    ],
    postconditions=["validation_result"],
    estimated_cost=0.5,
    code_template="""
# ValidateAggregationResult
validation = {{
    'has_null': result.isnull().any(),
    'has_zero': (result == 0).any() if pd.api.types.is_numeric_dtype(result) else False
}}
"""
)

# ============================================================================
# VI. 多表操作（Multi-table Integration）
# ============================================================================

JOIN_TABLES = MacroOperation(
    name="JoinTables",
    category=OperationCategory.MULTI_TABLE,
    description="Join Tables: Merge two tables (inner, left, outer) on a key.",
    preconditions=[
        StateCondition("entity_types", description="Entity types inferred"),
        StateCondition("multiple_tables", True, "Multiple tables present")
    ],
    postconditions=["joined_df"],
    estimated_cost=2.5,
    code_template="""
# JoinTables: {join_type}
df = pd.merge(df1, df2, on='{key_column}', how='{join_type}')
"""
)

MERGE_FILES = MacroOperation(
    name="MergeFiles",
    category=OperationCategory.MULTI_TABLE,
    description="Merge Files: Concatenate rows from multiple tables with same schema.",
    preconditions=[
        StateCondition("multiple_tables", True, "Multiple tables present")
    ],
    postconditions=["merged_df"],
    estimated_cost=2.0,
    code_template="""
# MergeFiles
df = pd.concat([df1, df2], ignore_index=True)
"""
)

# ============================================================================
# VII. 执行辅助（Validation / Error Handling）
# ============================================================================

VALIDATE_FORMULA = MacroOperation(
    name="ValidateFormula",
    category=OperationCategory.VALIDATION,
    description="Validate Formula: Check if a derived column or aggregation formula is valid.",
    preconditions=[
        StateCondition("derived_column", description="Column derived")
    ],
    postconditions=["formula_valid"],
    estimated_cost=0.3,
    code_template="""
# ValidateFormula
try:
    test_result = {formula}
    formula_valid = True
except Exception as e:
    formula_valid = False
    error_msg = str(e)
"""
)

SNAPSHOT_STATE = MacroOperation(
    name="SnapshotState",
    category=OperationCategory.VALIDATION,
    description="Snapshot State: Record dataframe schema and shape for memory.",
    preconditions=[],  # Can be called anytime
    postconditions=["snapshot"],
    estimated_cost=0.2,
    code_template="""
# SnapshotState
snapshot = {{
    'schema': df.dtypes.to_dict(),
    'shape': df.shape,
    'columns': list(df.columns),
    'head': df.head(3).to_dict('records')
}}
"""
)


# ============================================================================
# 操作注册表
# ============================================================================

ALL_OPERATIONS = [
    DETECT_SCHEMA,
    INSPECT_COLUMN,
    INFER_ENTITY_TYPE,
    CLEAN_MISSING,
    NORMALIZE_FORMAT,
    DEDUPLICATE_ROWS,
    FILTER_ROWS,
    SELECT_COLUMNS,
    SORT_VALUES,
    DERIVE_COLUMN,
    BIN_OR_GROUP,
    EXPLODE_OR_FLATTEN,
    AGGREGATE,
    GROUP_BY,
    RANK,
    VALIDATE_AGGREGATION_RESULT,
    JOIN_TABLES,
    MERGE_FILES,
    VALIDATE_FORMULA,
    SNAPSHOT_STATE,
]

OPERATIONS_BY_NAME = {op.name: op for op in ALL_OPERATIONS}


def get_operation(name: str) -> Optional[MacroOperation]:
    """根据名称获取操作"""
    return OPERATIONS_BY_NAME.get(name)


def get_operations_by_category(category: OperationCategory) -> List[MacroOperation]:
    """根据分类获取操作"""
    return [op for op in ALL_OPERATIONS if op.category == category]
