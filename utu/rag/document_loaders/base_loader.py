"""Base class for document loaders."""

from abc import ABC, abstractmethod


class BaseDocumentLoader(ABC):
    """Base class for all document loaders."""

    @abstractmethod
    def load(self, file_data: bytes, filename: str) -> str:
        pass

    @classmethod
    def from_extension(cls, extension: str, ocr_config=None) -> "BaseDocumentLoader":
        from .pdf_loader import PDFLoader
        from .docx_loader import DOCXLoader
        from .text_loader import TextLoader
        from .excel_loader import ExcelLoader
        from .image_ocr_loader import ImageOCRLoader

        extension = extension.lower().lstrip('.')

        # Image formats that require OCR
        image_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

        if extension in image_extensions:
            return ImageOCRLoader(ocr_config=ocr_config)

        # PDF can also use OCR when enabled
        if extension == 'pdf':
            return PDFLoader(ocr_config=ocr_config)

        loaders = {
            'doc': DOCXLoader,
            'docx': DOCXLoader,
            'txt': TextLoader,
            'md': TextLoader,
            'xls': ExcelLoader,
            'xlsx': ExcelLoader,
        }

        loader_class = loaders.get(extension, TextLoader)
        return loader_class()
