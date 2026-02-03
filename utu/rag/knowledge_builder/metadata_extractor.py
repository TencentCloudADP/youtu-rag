"""AI-powered metadata extraction from documents."""

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Optional

from openai import AsyncOpenAI

from ..utils import date_to_time_range, strf_to_timestamp

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extract metadata from documents using AI models."""

    def __init__(self, metadata_config: Optional[dict[str, Any]] = None):
        """Initialize metadata extractor.

        Args:
            metadata_config: Configuration from file_management.yaml's metadata_extraction section.
                            If None, uses default hardcoded values.
        """
        self.config = metadata_config or {}
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv("UTU_LLM_API_KEY"),
            base_url=os.getenv("UTU_LLM_BASE_URL"),
        )
        self.model = os.getenv("UTU_LLM_MODEL")

        # Load configuration values with defaults
        self.preview_length = self.config.get("preview_length", 500)
        model_params = self.config.get("model_params", {})
        self.temperature = model_params.get("temperature", 0.1)
        self.top_p = model_params.get("top_p", 0.95)
        self.system_prompt = self.config.get(
            "system_prompt",
            "你是一个专业的文档分析助手，擅长从文本中提取元数据信息。请严格按照JSON格式返回结果。"
        )
        self.extraction_prompt_template = self.config.get("extraction_prompt", self._get_default_prompt())

    def _get_default_prompt(self) -> str:
        """Get default extraction prompt template."""
        return """你是一个专业的数据标注与信息提取专家，擅长从非结构化文本中精准提取元数据。
请分析提供的文件名和文章片段，提取关键元数据用于构建检索索引。

# Inputs
- 文件名：{title}
- 内容片段：{preview_text}

# Extraction Rules
1. publish_date (发布日期):
   - 优先从文件名中寻找日期模式（如 PPT-20230501.ppt。乱码文件名不进行提取）。
   - 若文件名无日期，从正文开头寻找"发布于"、"日期："等关键词。
   - 统一格式为 YYYY-MM-DD，无法确定则返回 null。

2. key_timepoints (关键时间点):
   - 识别文中具有时效性意义的时间点。
   - 时间点标准化转化要求：
     [具体日] YYYY-MM-DD
     [具体月] YYYY-MM
     [季度制] YYYY-Q1, YYYY-Q2, YYYY-Q3, YYYY-Q4
     [半年制] YYYY-H1, YYYY-H2
     [年份制] YYYY
   - 排除掉过于细碎的日期（如"昨天"、"刚才"），仅保留具有里程碑意义的时间。

3. authors (作者/来源):
   - 提取个人作者、工作室或发布机构名称。

4. summary (摘要):
   - 提取核心观点及结论。
   - 字数严格控制在100字以内。

# Output Format
只返回纯JSON字符串，不要包含任何其他文字。

{{
    "publish_date": "YYYY-MM-DD" | null,
    "key_timepoints": ["string"],
    "authors": ["string"],
    "summary": "string" | null
}}
"""

    async def extract_metadata(self, text: str, filename: str) -> dict[str, Any]:
        """Extract comprehensive metadata from document text.

        Args:
            text: Full document text
            filename: Original filename

        Returns:
            Dictionary containing extracted metadata with:
            - char_length: Total character count (always included)
            - Other fields extracted by LLM based on the configured prompt
        """
        # char_length is always calculated directly
        metadata = {
            "char_length": str(len(text)),
        }

        # If text is too short, skip AI extraction
        if len(text) < 10:
            logger.warning(f"Text too short for metadata extraction: {filename}")
            # Add default empty values for common fields
            metadata.update({
                "publish_date": None,
                "key_timepoints": [],
                "summary": None,
            })
            return metadata

        title = filename
        preview_text = text[:self.preview_length]

        # Use AI to extract metadata based on configured prompt
        try:
            extracted_info = await self._extract_with_llm(title, preview_text)
            metadata.update(extracted_info)
        except Exception as e:
            logger.error(f"Error in AI metadata extraction: {e}")
            # Add default empty values for common fields
            metadata.update({
                "publish_date": None,
                "key_timepoints": [],
                "summary": None,
            })

        return metadata


    async def _extract_with_llm(self, title: str, preview_text: str) -> dict[str, Any]:
        """Use LLM to extract metadata from document using configured prompt.

        Args:
            title: Document title/filename
            preview_text: Text preview (length determined by config)

        Returns:
            Dictionary with fields defined in the extraction prompt
        """
        # Format the prompt template with actual values
        prompt = self.extraction_prompt_template.format(
            title=title,
            preview_text=preview_text
        )

        try:
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
            )

            # Parse the response
            result_text = response.choices[0].message.content.strip()

            # Try to extract JSON from the response
            json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(result_text)

            logger.info(f"LLM extraction result: {result_text}")

            # Post-process common fields if present
            if "publish_date" in result and result["publish_date"]:
                publish_date = self._validate_date(result["publish_date"])
                result["publish_date"] = publish_date
                result["publish_date_min_stamp"], result["publish_date_max_stamp"] = date_to_time_range(publish_date)

            if "key_timepoints" in result and result["key_timepoints"]:
                min_stamps, max_stamps = [], []
                for timepoint in result["key_timepoints"]:
                    min_stamp, max_stamp = date_to_time_range(timepoint)
                    min_stamps.append(min_stamp)
                    max_stamps.append(max_stamp)
                result["key_timepoints_min_stamp"] = min(min_stamps)
                result["key_timepoints_max_stamp"] = max(max_stamps)

            if "summary" in result and result["summary"] and len(result["summary"]) > 100:
                result["summary"] = result["summary"][:100]

            # Return all fields from LLM response dynamically
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {result_text}")
            return {}

        except Exception as e:
            logger.error(f"Error in LLM metadata extraction: {e}")
            return {}

    def _validate_date(self, date_str: str) -> Optional[str]:
        """Validate and normalize date string.

        Args:
            date_str: Date string in various formats

        Returns:
            Normalized date string or None if invalid
        """
        if not date_str or date_str == "null":
            return None

        # Try different date formats
        formats = [
            "%Y-%m-%d",  # 2024-01-15
            "%Y/%m/%d", # 2024/01/15
            "%Y-%m",  # 2024-01
            "%Y/%m", # 2024/01
            "%Y",  # 2024
            "%Y年%m月%d日",  # 2024年01月15日
            "%Y年%m月",  # 2024年01月
            "%Y年",  # 2024年
        ]

        for fmt in formats:
            try:
                date_obj = datetime.strptime(date_str.strip(), fmt)
                # Return in standardized format
                if fmt == "%Y" or fmt == "%Y年":
                    return date_obj.strftime("%Y")
                elif fmt == "%Y-%m" or fmt == "%Y年%m月" or fmt == "%Y/%m":
                    return date_obj.strftime("%Y-%m")
                else:
                    return date_obj.strftime("%Y-%m-%d")
            except ValueError:
                continue

        # If no format matches, return None
        logger.warning(f"Could not parse date: {date_str}")
        return None

    async def batch_extract_metadata(
        self, texts_and_filenames: list[tuple[str, str]]
    ) -> list[dict[str, Any]]:
        """Extract metadata from multiple documents in parallel.

        Args:
            texts_and_filenames: List of (text, filename) tuples

        Returns:
            List of metadata dictionaries
        """
        tasks = [self.extract_metadata(text, filename) for text, filename in texts_and_filenames]
        results = await asyncio.gather(*tasks)
        return results
