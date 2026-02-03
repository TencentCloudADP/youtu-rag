"""Excel document loader."""

import io
import logging

import pandas as pd

from .base_loader import BaseDocumentLoader

logger = logging.getLogger(__name__)


class ExcelLoader(BaseDocumentLoader):
    """Load and extract text from Excel files (.xlsx, .xls).
    """

    def load(self, file_data: bytes, filename: str) -> str:
        """Extract text from Excel file and convert to Markdown format.

        Strategy:
        - Read all sheets from the Excel file
        - For each sheet:
          1. Forward-fill merged cells (replicate values)
          2. Convert entire table to Markdown format
          3. Add metadata (filename, sheet name, dimensions)
        - Join all sheets' content

        Args:
            file_data: Excel file content as bytes
            filename: Original filename

        Returns:
            Extracted and formatted text content in Markdown
        """
        try:
            excel_file = pd.ExcelFile(io.BytesIO(file_data))
            all_text_parts = []
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    if df.empty:
                        continue
                    df = df.ffill()  # Forward fill 
                    df = df.fillna('')

                    sheet_text_parts = [
                        f"# 文件: {filename}",
                        f"## Sheet: {sheet_name}",
                        f"",
                        f"**总行数**: {len(df)} | **总列数**: {len(df.columns)}",
                        f"",
                    ]

                    markdown_table = self._df_to_markdown(df)
                    sheet_text_parts.append(markdown_table)

                    sheet_text = "\n".join(sheet_text_parts)
                    all_text_parts.append(sheet_text)

                except Exception as e:
                    logger.warning(f"Error reading sheet '{sheet_name}' from {filename}: {e}")
                    continue

            if not all_text_parts:
                logger.warning(f"No readable content found in Excel file: {filename}")
                return ""

            separator = "\n\n" + "="*80 + "\n\n"
            return separator.join(all_text_parts)

        except Exception as e:
            logger.error(f"Error loading Excel file {filename}: {e}")
            return ""

    def _df_to_markdown(self, df: pd.DataFrame) -> str:
        return self._manual_markdown_table(df)

    def _manual_markdown_table(self, df: pd.DataFrame) -> str:
        """Manually construct compact Markdown table (no alignment padding)."""
        lines = []

        headers = [str(col).strip() for col in df.columns]
        lines.append("|" + "|".join(headers) + "|")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")

        for _, row in df.iterrows():
            values = [str(val).strip().replace("|", "\\|") for val in row]
            lines.append("|" + "|".join(values) + "|")

        return "\n".join(lines)
