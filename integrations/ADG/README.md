# ADG Benchmark V4 部署包

## 文件结构

此包包含运行 `run_benchmark_v4.py` 所需的所有文件。

## 主要文件

- `src/model/adg_benchmark/run_benchmark_v4.py` - 主入口文件
- `src/model/adg_benchmark/adg_workflow_v4.py` - ADG 工作流
- `src/model/adg_benchmark/adg/` - ADG 核心模块
- `src/utils/` - 工具模块（logger, call_llm）
- `env.env` - 环境变量配置

## 依赖

请安装以下 Python 包：

```bash
pip install pandas numpy openpyxl langgraph tqdm python-dotenv requests
```

## 使用方法

1. 配置 `env.env` 文件，设置 LLM API 信息
2. 确保数据集和 meta graphs 目录存在（或使用自动生成功能）
3. 运行：

```bash
python src/model/adg_benchmark/run_benchmark_v4.py --dataset final --max_questions 10
```

## 参数说明

- `--dataset`: 数据集名称 ('final' or 'long')
- `--max_questions`: 最大问题数
- `--max_workers`: 并发线程数（默认 10）
- `--data_sample_rows`: 展示给 LLM 的数据行数（默认 150）
- `--question_types`: 题型过滤，如 `--question_types Visualization "Structure Comprehending"`
- `--start_id`: 起始问题ID

## 打包信息

- 打包时间: 2025-12-16 11:00:22
- 文件数量: 13
