import uuid
from datetime import datetime

from ...utils import CACHE_DIR, EnvUtils, get_logger

logger = get_logger(__name__)


class PDFParser:
    def __init__(self, config: dict) -> None:
        self.cache_dir = CACHE_DIR / "documents"
        EnvUtils.ensure_package("pymupdf")

    async def parse(self, path: str) -> str:
        """Convert PDF to Markdown format with image extraction and return the processed text."""
        import fitz  # pymupdf

        # Create unique directory with date and ID for images
        unique_id = str(uuid.uuid4())[:8]  # First 8 characters of UUID
        output_dir = self.cache_dir / datetime.now().strftime("%Y-%m-%d") / f"pdf_images_{unique_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(path)
        markdown_content = ""

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)

            text = page.get_text()
            markdown_content += f"## Page {page_num + 1}\n\n"
            if text.strip():
                markdown_content += text.strip() + "\n\n"
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n < 5:  # GRAY or RGB
                        img_filename = f"page_{page_num}_img_{img_index}.png"
                        img_path = output_dir / img_filename
                        pix.save(img_path)
                        markdown_content += f"![Image]({img_path})\n\n"
                    pix = None
                except Exception as img_e:
                    logger.warning(f"Failed to extract image {img_index} from page {page_num}: {img_e}")
                    continue

        doc.close()
        return markdown_content
