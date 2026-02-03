"""PDF document loader."""

import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import PyPDF2
import fitz 

from .base_loader import BaseDocumentLoader
from .image_ocr_loader import ImageOCRLoader

logger = logging.getLogger(__name__)


class PDFLoader(BaseDocumentLoader):
    """Load and extract text from PDF files."""

    def __init__(self, ocr_config: Optional[Dict[str, Any]] = None):
        """Initialize PDF loader with optional OCR support.

        Args:
            ocr_config: OCR configuration dictionary with 'enabled', 'base_url', etc.
        """
        self.ocr_config = ocr_config or {}
        self.ocr_enabled = self.ocr_config.get("enabled", False)
        self.derived_files = None
        self.ocr_results_per_page = [] 

    def load(self, file_data: bytes, filename: str) -> str:
        """Extract text from PDF file.

        If OCR is enabled, convert each page to image and process via OCR.
        Otherwise, use PyPDF2 for traditional text extraction.

        Args:
            file_data: PDF file content as bytes
            filename: Original filename

        Returns:
            Extracted text content (markdown format if OCR enabled)
        """
        if self.ocr_enabled:
            return self._load_with_ocr(file_data, filename)
        else:
            return self._load_with_pypdf2(file_data, filename)

    def _load_with_pypdf2(self, file_data: bytes, filename: str) -> str:
        """Traditional PDF text extraction using PyPDF2.

        Args:
            file_data: PDF file content as bytes
            filename: Original filename

        Returns:
            Extracted text content
        """
        try:
            pdf_file = BytesIO(file_data)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"

            return text

        except Exception as e:
            logger.error(f"Error extracting text from PDF {filename}: {e}")
            return ""

    def _load_with_ocr(self, file_data: bytes, filename: str) -> str:
        """Extract text from PDF using OCR on page images.

        Converts each PDF page to PNG image, processes via OCR,
        and combines results from all pages.

        Args:
            file_data: PDF file content as bytes
            filename: Original filename

        Returns:
            Combined markdown text from all pages
        """
        try:
            pdf_document = fitz.open(stream=file_data, filetype="pdf")
            total_pages = len(pdf_document)

            logger.info(f"Processing PDF {filename} with OCR: {total_pages} pages")

            all_markdown_text = []
            self.ocr_results_per_page = []
            all_derived_files = {}

            for page_num in range(total_pages):
                page = pdf_document[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                img_data = pix.tobytes("png")

                logger.info(f"Page {page_num + 1}/{total_pages}: Converted to PNG ({len(img_data)} bytes)")

                ocr_loader = ImageOCRLoader(ocr_config=self.ocr_config)

                page_filename = f"{Path(filename).stem}.png"
                page_markdown = ocr_loader.load(img_data, page_filename)

                if page_markdown:
                    page_header = f"## Page {page_num + 1}\n\n"
                    all_markdown_text.append(page_header + page_markdown)

                    self.ocr_results_per_page.append({
                        "page_num": page_num + 1,
                        "ocr_json_result": ocr_loader.ocr_json_result,
                        "markdown_text": page_markdown
                    })

                    page_derived_files = ocr_loader.get_derived_files()
                    if page_derived_files:
                        for derived_filename, (content, content_type) in page_derived_files.items():
                            prefixed_name = f"page_{page_num + 1}_{derived_filename}"
                            all_derived_files[prefixed_name] = (content, content_type)

                    logger.info(f"Page {page_num + 1}/{total_pages}: OCR successful, extracted {len(page_markdown)} chars")
                else:
                    logger.warning(f"Page {page_num + 1}/{total_pages}: OCR failed, no text extracted")

            pdf_document.close()

            combined_markdown = "\n\n---\n\n".join(all_markdown_text)

            self.derived_files = all_derived_files

            logger.info(f"PDF OCR completed: {total_pages} pages, {len(combined_markdown)} total chars")
            return combined_markdown

        except Exception as e:
            logger.error(f"Error processing PDF with OCR {filename}: {e}")
            logger.info(f"Falling back to PyPDF2 text extraction for {filename}")
            return self._load_with_pypdf2(file_data, filename)

    def get_derived_files(self) -> Optional[Dict[str, Tuple[bytes, str]]]:
        """Get derived files generated during OCR processing.

        Returns:
            Dictionary mapping filenames to (content_bytes, content_type) tuples
        """
        return self.derived_files
