"""DOCX document loader."""

import logging
from io import BytesIO

import docx

from .base_loader import BaseDocumentLoader

logger = logging.getLogger(__name__)


class DOCXLoader(BaseDocumentLoader):
    """Load and extract text from DOCX files."""

    def load(self, file_data: bytes, filename: str) -> str:
        """Extract text from DOCX file.

        Args:
            file_data: DOCX file content as bytes
            filename: Original filename

        Returns:
            Extracted text content
        """
        try:
            docx_file = BytesIO(file_data)
            doc = docx.Document(docx_file)

            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"

            return text

        except Exception as e:
            logger.error(f"Error extracting text from DOCX {filename}: {e}")
            return ""
