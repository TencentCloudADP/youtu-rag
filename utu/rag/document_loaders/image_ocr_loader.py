"""Image OCR Loader - Process images and extract text via OCR"""

import io
import json
import logging
import base64
import requests
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from PIL import Image, ImageDraw, ImageFont

from .base_loader import BaseDocumentLoader

logger = logging.getLogger(__name__)


class ImageOCRLoader(BaseDocumentLoader):
    """Loader for extracting text from images using OCR"""

    def __init__(self, ocr_config: Optional[Dict[str, Any]] = None):
        self.ocr_config = ocr_config or {}
        self.ocr_enabled = self.ocr_config.get("enabled", False)

        self.derived_files = None
        self.ocr_json_result = None

    def load(self, file_data: bytes, filename: str) -> str:
        if not self.ocr_enabled:
            logger.warning(f"OCR is not enabled, cannot extract text from image: {filename}")
            return ""

        try:
            extracted_text = self._call_ocr_api(file_data, filename)
            self.derived_files = self.generate_derived_files(file_data, filename, extracted_text)

            return extracted_text

        except Exception as e:
            logger.error(f"OCR processing failed for {filename}: {e}")
            return ""

    def _call_ocr_api(self, file_data: bytes, filename: str, max_retries: int = 3, retry_delay: int = 2) -> str:
        """Call OCR API with retry mechanism."""
        api_url = self.ocr_config.get("base_url")

        if not api_url:
            logger.error(f"OCR base_url not configured in ocr_config")
            return ""

        logger.info(f"OCR start: {filename}")

        img_base64 = base64.b64encode(file_data).decode('utf-8')
        data = {"image": img_base64}

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    api_url,
                    data=json.dumps(data),
                    headers={'Content-Type': 'application/json'},
                    timeout=300
                )

                if response.status_code == 503:
                    if attempt < max_retries - 1:
                        logger.warning(f"OCR service unavailable (503) for {filename}, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                        import time
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"OCR service unavailable after {max_retries} retries for {filename}")
                        return ""

                if response.status_code != 200:
                    logger.error(f"OCR API returned status {response.status_code}: {response.text}")
                    return ""

                ocr_result = response.json()

                # Validate that ocr_result is a list
                if not isinstance(ocr_result, list):
                    logger.error(f"OCR API returned unexpected format for {filename}: expected list, got {type(ocr_result).__name__}")
                    logger.debug(f"OCR response: {ocr_result}")
                    return ""

                self.ocr_json_result = ocr_result
                markdown_text = self._json_to_markdown(ocr_result)

                logger.info(f"OCR successfully processed {filename}, extracted {len(ocr_result)} blocks")
                return markdown_text

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    logger.warning(f"OCR API timeout for {filename}, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    import time
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"OCR API timeout after {max_retries} retries for {filename}")
                    return ""

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    logger.warning(f"OCR API request failed for {filename}: {e}, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    import time
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"OCR API request failed after {max_retries} retries for {filename}: {e}")
                    return ""

            except Exception as e:
                logger.error(f"OCR processing error for {filename}: {e}")
                return ""

        logger.error(f"Failed to process OCR for {filename} after {max_retries} retries")
        return ""

    def _json_to_markdown(self, ocr_result: List[Dict[str, Any]]) -> str:
        if not ocr_result:
            return ""

        # Validate input type
        if not isinstance(ocr_result, list):
            logger.error(f"Invalid ocr_result type: expected list, got {type(ocr_result).__name__}")
            return ""

        markdown_lines = []

        for block in ocr_result:
            block_type = block.get("type", "Text")
            content = block.get("content", "").strip()

            if not content:
                if block_type == "Figure":
                    continue
                continue

            if block_type == "Title":
                if content.startswith("#"):
                    markdown_lines.append(content)
                else:
                    markdown_lines.append(f"## {content}")
            elif block_type == "Header":
                markdown_lines.append(f"**{content}**")
            elif block_type == "Footer":
                markdown_lines.append(f"_{content}_")
            elif block_type == "Figure":
                markdown_lines.append(f"[Image Area]")
            else:
                markdown_lines.append(content)
            markdown_lines.append("")

        return "\n".join(markdown_lines)

    def _draw_layout_image(self, file_data: bytes, ocr_result: List[Dict[str, Any]]) -> bytes:
        try:
            if not isinstance(ocr_result, list):
                logger.error(f"Invalid ocr_result type: expected list, got {type(ocr_result).__name__}")
                return file_data

            image = Image.open(io.BytesIO(file_data))
            original_format = image.format or 'PNG'

            if image.mode != 'RGBA':
                image = image.convert('RGBA')

            overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            draw = ImageDraw.Draw(image)

            color_map = {
                "Title": {
                    "border": (255, 0, 0),           # Red
                    "fill": (255, 0, 0, 30)          # Light red with transparency
                },
                "Header": {
                    "border": (0, 128, 255),         # Blue
                    "fill": (0, 128, 255, 30)        # Light blue with transparency
                },
                "Footer": {
                    "border": (128, 128, 128),       # Gray
                    "fill": (128, 128, 128, 30)      # Light gray with transparency
                },
                "Text": {
                    "border": (0, 255, 0),           # Green
                    "fill": (0, 255, 0, 30)          # Light green with transparency
                },
                "Figure": {
                    "border": (255, 165, 0),         # Orange
                    "fill": (255, 165, 0, 30)        # Light orange with transparency
                },
            }

            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
                except:
                    font = ImageFont.load_default()

            for block in ocr_result:
                bbox = block.get("bbox")
                block_type = block.get("type", "Text")

                if not bbox or len(bbox) < 8:
                    continue
  
                points = [
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (bbox[4], bbox[5]),
                    (bbox[6], bbox[7])
                ]

                colors = color_map.get(block_type, {
                    "border": (0, 255, 0),
                    "fill": (0, 255, 0, 30)
                })

                overlay_draw.polygon(points, fill=colors["fill"])
                draw.polygon(points, outline=colors["border"], width=3)

                # Add type label at top-right corner of bbox
                x_coords = [bbox[0], bbox[2], bbox[4], bbox[6]]
                y_coords = [bbox[1], bbox[3], bbox[5], bbox[7]]

                # Find top-right corner (max x, min y)
                label_x = max(x_coords) - 5  # Offset from right edge
                label_y = min(y_coords) + 2  # Offset from top edge

                label_text = block_type
                try:
                    bbox_text = draw.textbbox((label_x, label_y), label_text, font=font)
                    label_bg = [
                        bbox_text[0] - 2,
                        bbox_text[1] - 1,
                        bbox_text[2] + 2,
                        bbox_text[3] + 1
                    ]
                except:
                    text_width, text_height = draw.textsize(label_text, font=font)
                    label_bg = [
                        label_x - 2,
                        label_y - 1,
                        label_x + text_width + 2,
                        label_y + text_height + 1
                    ]

                overlay_draw.rectangle(label_bg, fill=(255, 255, 255, 180))

                draw.text(
                    (label_x, label_y),
                    label_text,
                    fill=colors["border"],
                    font=font
                )

            image = Image.alpha_composite(image, overlay)

            if original_format in ['JPEG', 'JPG']:
                image = image.convert('RGB')

            output = io.BytesIO()
            save_format = 'JPEG' if original_format in ['JPEG', 'JPG'] else original_format
            image.save(output, format=save_format, quality=95)
            return output.getvalue()

        except Exception as e:
            logger.error(f"Failed to draw layout image: {e}")
            return file_data

    def generate_derived_files(
        self, file_data: bytes, filename: str, extracted_text: str
    ) -> Dict[str, Tuple[bytes, str]]:
        """Generate OCR processed derived files."""
        name_without_ext = Path(filename).stem
        derived_files = {}

        if not self.ocr_json_result:
            logger.warning(f"No OCR result available for {filename}, using placeholder")
            try:
                original_image = Image.open(io.BytesIO(file_data))
                image_format = (original_image.format or 'PNG').lower()
                layout_ext = 'jpg' if image_format in ['jpeg', 'jpg'] else image_format
                content_type = f"image/{image_format}" if image_format != 'jpg' else "image/jpeg"
            except:
                layout_ext = 'png'
                content_type = "image/png"

            layout_filename = f"{name_without_ext}_layout.{layout_ext}"
            derived_files[layout_filename] = (file_data, content_type)

            md_filename = f"{name_without_ext}.md"
            md_content = extracted_text or f"# OCR Result\n\nSource image: {filename}\n\n(OCR processing failed)"
            derived_files[md_filename] = (md_content.encode('utf-8'), "text/markdown")

            json_filename = f"{name_without_ext}.json"
            json_data = []
            derived_files[json_filename] = (json.dumps(json_data, ensure_ascii=False, indent=2).encode('utf-8'), "application/json")

            return derived_files

        try:
            # 1. Generate layout image - draw text boxes on original image
            original_image = Image.open(io.BytesIO(file_data))
            image_format = (original_image.format or 'PNG').lower()
            layout_ext = 'jpg' if image_format in ['jpeg', 'jpg'] else image_format

            layout_filename = f"{name_without_ext}_layout.{layout_ext}"
            layout_image_data = self._draw_layout_image(file_data, self.ocr_json_result)
            content_type = f"image/{image_format}" if image_format != 'jpg' else "image/jpeg"
            derived_files[layout_filename] = (layout_image_data, content_type)
            logger.info(f"Generated layout image: {layout_filename}")

        except Exception as e:
            logger.error(f"Failed to generate layout image: {e}")
            layout_filename = f"{name_without_ext}_layout.png"
            derived_files[layout_filename] = (file_data, "image/png")

        # 2. Generate Markdown file
        md_filename = f"{name_without_ext}.md"
        md_content = extracted_text or f"# OCR Result\n\nSource image: {filename}\n\n(No content)"
        derived_files[md_filename] = (md_content.encode('utf-8'), "text/markdown")
        logger.info(f"Generated markdown file: {md_filename}")

        # 3. Generate JSON file - use OCR returned raw data
        json_filename = f"{name_without_ext}.json"
        json_content = json.dumps(self.ocr_json_result, ensure_ascii=False, indent=2)
        derived_files[json_filename] = (json_content.encode('utf-8'), "application/json")
        logger.info(f"Generated JSON file: {json_filename}")

        return derived_files

    def get_derived_files(self) -> Optional[Dict[str, Tuple[bytes, str]]]:
        return self.derived_files
