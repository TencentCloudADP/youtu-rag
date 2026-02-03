"""Query Parser for Timeliness."""

import json
import os
import logging
import time
import datetime
import asyncio
from typing import Any, Optional, List
from jinja2 import Template
from ...utils import SimplifiedAsyncOpenAI, LLMOutputParser

logger = logging.getLogger(__name__)

PROMPT_1 = """You are an intent analysis expert. Your task is to analyze user queries, determine their timeliness requirements, and output standardized retrieval filter parameters.

# ⏰ Current Time Reference (CRITICAL)
**Today is: {{ current_day }} ({{ current_day_of_week }})**
- Current year: {{ current_year }}
- Current quarter: {{ current_quarter }}
- Current half-year: {{ current_half_year }}
- Last month: {{ last_month }}
- Last year: {{ last_year }}

# 🚨 Time Judgment Rules (MUST FOLLOW)
**When determining time_orientation, you MUST follow this logic:**
1. **past**: Year < {{ current_year }} or explicitly refers to a time period that has already passed
   - Examples: "2025", "last year", "last month", "FY25 mid-year report"
2. **present**: Explicitly refers to an ongoing time period (e.g., "this month", "this quarter")
3. **future**: Year > {{ current_year }} or explicitly refers to a time period that has not yet arrived
   - Examples: "2027", "next year", "next quarter"
4. **range**: Spans multiple different time periods
5. **latest**: User does not specify a specific time but needs the most recent information
6. **none**: Does not involve any time concept

**⚠️ Special Note:**
- If "2025" appears in the query → relative to current time (2026), it belongs to **past**, NOT future!
- If "2024" appears in the query → relative to current time (2026), it belongs to **past**
- If "2027" appears in the query → relative to current time (2026), it belongs to **future**

# Output Format
Return only a pure JSON string containing the following fields:
1. `is_temporal`: boolean, whether it involves a specific time point, time period, or timeliness (e.g., "latest", "last year").
2. `time_orientation`: string, options: ["past", "present", "future", "range", "latest", "none"].
3. `standard_tags`: list, normalized time tags (must conform to: YYYY, YYYY-MM, YYYY-QX, YYYY-HX, YYYY-MM-DD).
4. `match_strategy`: string, options:
    - "publish_date" (for publish dates like "when was it released", "last month's news")
    - "key_timepoints" (for content attributes like "2026 financial report", "FY25 plan")
    - "both" (default, or for queries like "data from 2024 to 2025")
5. `reasoning`: string, brief explanation of the parsing logic, **must include comparison judgment with current time**.

# Standard Mapping Rules
- Mid-year report/first half -> YYYY-H1
- Annual report/full year -> YYYY
- Q3 report/Q3 -> YYYY-Q3
- Latest/recent/current status -> set time_orientation to "latest", standard_tags to current time period.
- Relative time -> calculate based on current time (e.g., "last month" is {{ last_month }}, "last year" is {{ last_year }})

# Examples (based on current time {{ current_day }})
Query: What gold mine information was disclosed in Zhaojin Gold's FY25 mid-year report?
Output: 
{
  "is_temporal": true,
  "time_orientation": "past",
  "standard_tags": ["2025-H1"],
  "match_strategy": "both",
  "reasoning": "User is asking about the 2025 mid-year report. Since the current year is {{ current_year }}, 2025 has already passed, so time_orientation is past."
}

Query: "What are Xiaomi's financial report profits for Q3 2024 and Q3 2025 respectively?"
Output:
{
  "is_temporal": true,
  "time_orientation": "past",
  "standard_tags": ["2024-Q3", "2025-Q3"],
  "match_strategy": "key_timepoints",
  "reasoning": "User is asking about 2024 Q3 and 2025 Q3. Relative to current time ({{ current_year }}), both belong to the past, so time_orientation is past."
}

Query: "Practical methods in news operations in 2025"
Output:
{
  "is_temporal": true,
  "time_orientation": "past",
  "standard_tags": ["2025"],
  "match_strategy": "both",
  "reasoning": "User is asking about 2025 content. Current year is {{ current_year }}, 2025 has already passed, so time_orientation is past."
}

Query: "What new products are there?"
Output:
{
  "is_temporal": false,
  "time_orientation": "latest",
  "standard_tags": ["{{ current_quarter }}"],
  "match_strategy": "both",
  "reasoning": "User is asking about new products. Default to querying the most recent time range, using the current quarter."
}

Query: "Market forecast for 2027"
Output:
{
  "is_temporal": true,
  "time_orientation": "future",
  "standard_tags": ["2027"],
  "match_strategy": "key_timepoints",
  "reasoning": "User is asking about 2027 forecast. Relative to current time ({{ current_year }}), it belongs to the future, so time_orientation is future."
}

# User Query to Process:
{{ query }}

Output:
"""

class TimeParser:

    def __init__(self):
        """Initialize the parser."""

        self.llm = SimplifiedAsyncOpenAI(
            type=os.environ.get("UTU_LLM_TYPE"),
            api_key=os.environ.get("UTU_LLM_API_KEY"),
            base_url=os.environ.get("UTU_LLM_BASE_URL"),
            model=os.environ.get("UTU_LLM_MODEL"),
        )

        self.current_date = datetime.datetime.now()
        
        self.current_day = self.current_date.strftime("%Y-%m-%d")
        
        self.current_day_of_week = self.current_date.strftime("%A")
        
        self.current_month = self.current_date.strftime("%Y-%m")
        self.last_month = (self.current_date - datetime.timedelta(days=30)).strftime("%Y-%m")

        self.current_year = str(self.current_date.year)
        self.last_year = str(self.current_date.year - 1)
        
        quarter = (self.current_date.month - 1) // 3 + 1
        self.current_quarter = f"{self.current_date.year}-Q{quarter}"
        
        quarter_start_month = (quarter - 1) * 3 + 1
        quarter_end_month = quarter * 3
        quarter_start = f"{self.current_date.year}-{quarter_start_month:02d}-01"
        if quarter_end_month == 12:
            quarter_end = f"{self.current_date.year}-12-31"
        else:
            next_month = datetime.date(self.current_date.year, quarter_end_month + 1, 1)
            last_day = (next_month - datetime.timedelta(days=1)).day
            quarter_end = f"{self.current_date.year}-{quarter_end_month:02d}-{last_day}"
        self.current_quarter_time_range = f'["{quarter_start}", "{quarter_end}"]'
        
        half_year = 1 if self.current_date.month <= 6 else 2
        self.current_half_year = f"{self.current_date.year}-H{half_year}"
        
        self.current_year = str(self.current_date.year)

    async def parse(self, query: str) -> str:

        template = Template(PROMPT_1)
        input_str = template.render(
            query=query,
            current_day=self.current_day,
            current_day_of_week=self.current_day_of_week,
            current_year=self.current_year,
            last_month=self.last_month,
            last_year=self.last_year,
            current_quarter=self.current_quarter,
            current_half_year=self.current_half_year,
            current_quarter_time_range=self.current_quarter_time_range,
        )

        messages = [{"role": "user", "content": input_str}]
        response = await self.llm.query_one(messages=messages)
        res = LLMOutputParser.extract_code_json(response, "json")
        standard_tags = res["standard_tags"]
        return res

# async def main():
#     """Main function."""

#     parser = TimeParser()
#     query = "什么时候开学"
#     # query = "整理2014-2025年塔山煤矿和色连煤矿的产量和商品煤销量，以表格呈现"
#     # query = "给我找找刘德华的基本信息"
#     query = "请结合7月进行的连续血糖测试及“2025年7月使用辅理善测量血糖记录”中对相应食物种类、数量及运动、用药的记录，建立对应关系，并分析评估是否属糖尿病前期，是否需要药物干预？"
#     # query = "有什么新产品"
#     # query = "基于以下内容，逐项分析腾讯这家公司的最近的财务情况：第一层：生存能力 - 它会不会倒下？（安全与健康） \n这是分析的底线，主要评估公司的财务韧性和短期偿债风险。\r\n\r\n看现金流：经营现金流是否健康？能否覆盖日常运营和利息支出？一家公司可以暂时亏损，但不能没有现金。\r\n看偿债能力：资产负债率是否合理？短期债务压力大不大？流动资产能否覆盖流动负债？（常用指标：流动比率、速动比率）\r\n看资产质量：资产结构是否健康？有没有大量难以变现的无效资产或高额商誉？\r\n\r\n第二层：盈利能力 - 它赚钱的效率高吗？（能力与效率）\r\n这是分析的核心，主要评估公司将资源转化为利润的效率和可持续性。\r\n看盈利规模与质量：净利润是否来自主营业务？还是靠政府补贴或资产出售等一次性收益？\r\n看利润率：毛利率反映产品的竞争力和定价权；净利率反映公司的整体费用控制能力和经营效率。\r\n看回报率：这是最关键的部分。净资产收益率（ROE） 衡量公司用股东投入的钱创造了多少回报；总资产收益率（ROA） 衡量公司利用所有资产赚钱的效率。\r\n\r\n第三层：成长能力 - 它的未来会更好吗？（动力与潜力）\r\n这决定了公司的未来价值和市场想象力，主要评估其发展潜力和增长驱动因素。\r\n看增长曲线：营业收入、净利润是否持续增长？增长是加速还是放缓？\r\n看增长驱动：增长来自哪里？是行业红利、市场份额提升、产品提价还是收购兼并？\r\n看研发与投入：公司是否在为未来投资？（如研发费用占收入比、资本开支等）\r\n第四层：治理与前景 - 它由谁掌舵，航向何方？（人与环境）\r\n这是定性分析的层面，决定了以上三层能力的稳定性和上限。\r\n\r\n看管理层与公司治理：管理层是否诚信、有能力？股权结构是否清晰？对小股东是否友好？\r\n看行业前景与竞争格局：公司所处行业是朝阳行业还是夕阳行业？它在行业中处于什么地位？（领导者、挑战者还是追随者？）\r\n看商业模式与企业文化：它的业务模式是否容易理解且难以复制？企业文化是否积极向上？"
#     # query = "最新的养老政策有哪些"
#     # query = "预测中国互联网企业未来的增长情况"
#     # query = "阿里过去几年财务状况如何"
#     # query = "人形机器人行业最新的进展及相关公司营收"
#     query = "中恒电气有希望进入海外供应链嘛？过去没有进入的主要瓶颈是什么，是管理层意愿嘛"
#     # query = "储能行业现状"
#     # query = "详细做一份关于煤炭行业的调研报告，分析煤炭行业现状，历史周期和原因，预判后面煤炭价格"
#     # query = "钧达股份分析公司：\r\n1、公司产品竞争力分析（连带介绍下行业第一的公司和产品），是否有新产品上市或在研。\r\n2、公司所在行业分析，主营产品行业分析（市场规模、增速等关键数据要有）\r\n3、根据行业和公司竞争力市占率预测营收和净利。\r\n4、上下游分析，是否有要关注的点。\r\n5、行业横向对比，要包含估值、毛利、市销率等关键指标，以财务稳健度、增速等，估算公司合理估值（可参考研报目标价）。\r\n6、是否有全球或者国内，独家的产品或技术。\r\n7、未来的增长点在那里，公司成长性如何。\r\n8、公司的潜在风险点和需要预防的点。\r\n9、结合最匹配的估值模型，计算合理市值。\r\n10、公司最近有什么舆情和资讯值得关注。"
#     # query = "华中科技大学3月和9月份都有哪些校园活动"
#     res = await parser.parse(query=query)
#     print(res)


# if __name__ == "__main__":

#     asyncio.run(main())