"""Text document loader."""

import logging

from .base_loader import BaseDocumentLoader

logger = logging.getLogger(__name__)


class TextLoader(BaseDocumentLoader):
    """Load and extract text from plain text files."""

    def load(self, file_data: bytes, filename: str) -> str:
        """Extract text from plain text file."""
        try:
            return file_data.decode('utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"Error decoding text file {filename}: {e}")
            return ""
