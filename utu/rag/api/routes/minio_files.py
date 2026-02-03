import logging
import os
import io
import zipfile
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from urllib.parse import quote

from ..database import get_db, KnowledgeBase, KBSourceConfig, KBExcelTable
from ..minio_client import minio_client
from ...document_loaders.base_loader import BaseDocumentLoader
from ..upload_progress import upload_tracker, UploadStatus

logger = logging.getLogger(__name__)
router = APIRouter()

OCR_SEMAPHORE = asyncio.Semaphore(2)  
CHUNK_SEMAPHORE = asyncio.Semaphore(2) 

class MetadataImportRow(BaseModel):
    filename: str
    etag: str
    metadata: Dict[str, Any]


class MetadataImportResult(BaseModel):
    total_rows: int
    successful: int
    failed: int
    errors: List[Dict[str, Any]]


@router.get("/check-exists/{filename}")
async def check_file_exists(filename: str):
    exists = minio_client.file_exists(filename)

    if exists:
        try:
            stat = minio_client.get_file_stat(filename)
            return {
                "exists": True,
                "filename": filename,
                "size": stat.size,
                "last_modified": stat.last_modified.isoformat(),
                "etag": stat.etag
            }
        except Exception as e:
            logger.warning(f"File exists but failed to get stats: {e}")
            return {
                "exists": True,
                "filename": filename
            }
    else:
        return {
            "exists": False,
            "filename": filename
        }


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload file to MinIO and automatically extract metadata

    For image files (png, jpg, etc.), if OCR is available and enabled:
    1. First perform OCR processing to extract text
    2. Extract metadata from the OCR-extracted text

    For non-image files or when OCR is unavailable:
    - Use the original document loader to extract text
    - For image files when OCR is unavailable, metadata fields are empty
    """
    try:
        logger.info(f"Uploading file: {file.filename}")

        content = await file.read()
        logger.info(f"File content read: {len(content)} bytes")

        file_ext = file.filename.split('.')[-1].lower() if '.' in file.filename else 'txt'

        _validate_file_content(content, file.filename)

        from ..services.kb_config_service import KBConfigService
        config = KBConfigService.load_yaml_config("file_management")
        logger.info(f"[DEBUG] Loaded config keys: {list(config.keys())}")

        ocr_config = config.get("ocr", {})
        metadata_config = config.get("metadata_extraction", {})
        chunk_config = config.get("chunk", {})

        file_text, ocr_used, derived_files_dict = await _process_ocr(
            content, file.filename, file_ext, ocr_config
        )

        image_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
        is_image = file_ext in image_extensions
        is_pdf = file_ext == 'pdf'

        logger.info(f"[DEBUG] About to process chunk for {file.filename}")
        chunk_processed, chunk_file_uploaded, chunked_text = await _process_chunk(
            file_text, file.filename, file_ext, chunk_config,
            is_image, is_pdf, ocr_used, derived_files_dict
        )

        normalized_metadata = await _extract_and_normalize_metadata(
            file_text, file.filename, metadata_config,
            is_image, is_pdf, ocr_used, derived_files_dict, chunk_processed
        )

        metadata = {k: v for k, v in normalized_metadata.items()
                   if k not in ['ocr_processed', 'chunk_processed']}

        _validate_file_content(content, file.filename)

        success, source_etag_value = minio_client.upload_file(
            file_data=content,
            object_name=file.filename,
            content_type=file.content_type or "application/octet-stream",
            metadata=normalized_metadata
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to upload file to MinIO")

        logger.info(f"File uploaded successfully: {file.filename} (OCR used: {ocr_used}, ETag: {source_etag_value})")

        sys_bucket = os.getenv("MINIO_BUCKET_SYS", "sysfile")
        derived_files_uploaded = []

        if derived_files_dict and ocr_used:
            derived_files_uploaded = await _upload_derived_files(
                derived_files_dict, file.filename, source_etag_value,
                minio_client, sys_bucket, None
            )

        chunk_file_uploaded = None
        if chunk_processed == 'chunk_success' and chunked_text:
            chunk_file_uploaded = f"{Path(file.filename).stem}_chunklevel.md"
            chunk_success, _ = await _upload_chunk_file(
                chunked_text, chunk_file_uploaded, file.filename,
                source_etag_value, minio_client, sys_bucket, None
            )

            if not chunk_success:
                normalized_metadata['chunk_processed'] = 'chunk_failed'
                chunk_file_uploaded = None

        return {
            "message": "File uploaded successfully",
            "filename": file.filename,
            "metadata": metadata,
            "ocr_processed": normalized_metadata.get('ocr_processed', 'ocr_failed'),
            "chunk_processed": normalized_metadata.get('chunk_processed', 'chunk_failed'),
            "derived_files": derived_files_uploaded if derived_files_uploaded else None,
            "chunk_file": chunk_file_uploaded
        }

    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _validate_content(content: bytes, filename: str, task_id: str) -> bytes:
    if isinstance(content, str):
        logger.warning(f"Content is str (unexpected), converting to bytes for {filename}")
        return content.encode('utf-8')
    elif not isinstance(content, bytes):
        error_msg = f"Content has unexpected type {type(content)} for {filename}"
        logger.error(error_msg)
        upload_tracker.update_progress(task_id, UploadStatus.FAILED, 0, error_msg, error=error_msg)
        raise TypeError(error_msg)
    return content


async def _process_ocr(
    content: bytes,
    filename: str,
    file_ext: str,
    ocr_config: dict,
    task_id: str = None
) -> tuple[str, bool, dict]:
    """ OCR Process """
    image_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
    is_image = file_ext in image_extensions
    is_pdf = file_ext == 'pdf'
    ocr_enabled = ocr_config.get("enabled", False)

    file_text = ""
    ocr_used = False
    derived_files_dict = None

    if not (is_image or is_pdf):
        if task_id:
            upload_tracker.update_progress(task_id, UploadStatus.UPLOADING, 30, "Extracting text...")
        loader = BaseDocumentLoader.from_extension(file_ext)
        file_text = await asyncio.to_thread(loader.load, content, filename)
        return file_text, ocr_used, derived_files_dict

    if not ocr_enabled:
        if is_image:
            logger.info(f"OCR is disabled, image will be uploaded without text extraction: {filename}")
            file_text = ""
            if task_id:
                upload_tracker.update_progress(task_id, UploadStatus.UPLOADING, 30, "Image file (OCR not enabled)")
        else:
            logger.info(f"OCR is disabled, using traditional PDF text extraction: {filename}")
            loader = BaseDocumentLoader.from_extension(file_ext)
            file_text = await asyncio.to_thread(loader.load, content, filename)
            if task_id:
                upload_tracker.update_progress(task_id, UploadStatus.UPLOADING, 30, "PDF text extraction completed")
        return file_text, ocr_used, derived_files_dict

    logger.info(f"Starting OCR processing for {filename}")
    if task_id:
        upload_tracker.update_progress(
            task_id, UploadStatus.OCR_PROCESSING, 15,
            f"Preparing OCR recognition ({'Image' if is_image else 'PDF'})..."
        )
        upload_tracker.update_progress(
            task_id, UploadStatus.OCR_PROCESSING, 20,
            f"Performing {'Image' if is_image else 'PDF'} OCR recognition..."
        )

    ocr_progress_stop = asyncio.Event() if task_id else None
    progress_task = None

    if task_id:
        async def simulate_ocr_progress():
            """Simulate progress updates during OCR processing (20% → 38%)"""
            progress = 20
            while not ocr_progress_stop.is_set():
                await asyncio.sleep(3)
                if not ocr_progress_stop.is_set() and progress < 38:
                    progress = min(progress + 3, 38)
                    upload_tracker.update_progress(
                        task_id, UploadStatus.OCR_PROCESSING, progress,
                        f"Performing {'Image' if is_image else 'PDF'} OCR recognition...({progress}%)"
                    )
                    logger.debug(f"Task {task_id}: OCR processing {progress}%")

        progress_task = asyncio.create_task(simulate_ocr_progress())

    logger.info(f"[SEMAPHORE] OCR waiting for semaphore, current value: {OCR_SEMAPHORE._value}, filename: {filename}")
    async with OCR_SEMAPHORE:
        logger.info(f"[SEMAPHORE] OCR acquired semaphore, remaining: {OCR_SEMAPHORE._value}, filename: {filename}")
        loader = BaseDocumentLoader.from_extension(file_ext, ocr_config=ocr_config)
        file_text = await asyncio.to_thread(loader.load, content, filename)
        ocr_used = True
    logger.info(f"[SEMAPHORE] OCR released semaphore, current value: {OCR_SEMAPHORE._value}, filename: {filename}")

    if task_id and progress_task:
        ocr_progress_stop.set()
        try:
            await asyncio.wait_for(progress_task, timeout=1.0)
        except asyncio.TimeoutError:
            pass

        upload_tracker.update_progress(
            task_id, UploadStatus.OCR_PROCESSING, 40, "OCR recognition completed, processing results..."
        )

    if hasattr(loader, 'get_derived_files'):
        derived_files_dict = loader.get_derived_files()
        logger.info(f"Generated {len(derived_files_dict) if derived_files_dict else 0} derived files")

    if task_id:
        upload_tracker.update_progress(
            task_id, UploadStatus.OCR_PROCESSING, 50, "✓ OCR processing completed"
        )

    return file_text, ocr_used, derived_files_dict


async def _process_chunk(
    file_text: str,
    filename: str,
    file_ext: str,
    chunk_config: dict,
    is_image: bool,
    is_pdf: bool,
    ocr_used: bool,
    derived_files_dict: dict,
    task_id: str = None
) -> tuple[str, str, str]:
    """Chunk Process"""
    chunk_processed = 'chunk_skipped'
    chunk_file_uploaded = None
    chunked_text = None
    is_excel_file = file_ext in {'xls', 'xlsx'}
    chunk_enabled = chunk_config.get("enabled", False)

    logger.info(f"[DEBUG] Chunk config: {chunk_config}")
    logger.info(f"[DEBUG] Chunk enabled: {chunk_enabled}")
    logger.info(f"[DEBUG] Is Excel file: {is_excel_file}")

    if is_excel_file:
        logger.info(f"Chunk processing skipped for Excel file: {filename}")
        if task_id:
            upload_tracker.update_progress(task_id, UploadStatus.UPLOADING, 60, "Excel file skipped Chunk processing")
        return chunk_processed, chunk_file_uploaded, chunked_text

    if not chunk_enabled:
        logger.info(f"Chunk processing disabled for: {filename}")
        if task_id:
            upload_tracker.update_progress(task_id, UploadStatus.UPLOADING, 60, "Chunk processing not enabled")
        return chunk_processed, chunk_file_uploaded, chunked_text

    logger.info(f"Chunk processing enabled for: {filename}")
    if task_id:
        upload_tracker.update_progress(task_id, UploadStatus.CHUNK_PROCESSING, 50, "Preparing Chunk level recognition...")

    try:
        temp_ocr_processed = 'ocr_success' if ((is_image or is_pdf) and ocr_used) else 'ocr_skipped'
        chunk_input_text = ""

        if temp_ocr_processed == 'ocr_success' and derived_files_dict:
            logger.info(f"Loading OCR-derived markdown from memory for chunk: {filename}")
            if task_id:
                upload_tracker.update_progress(
                    task_id, UploadStatus.CHUNK_PROCESSING, 50.5, "Loading OCR results for Chunk processing..."
                )

            import re
            md_contents = []

            for derived_filename, derived_data in derived_files_dict.items():
                if derived_filename.endswith('.md'):
                    if isinstance(derived_data, tuple) and len(derived_data) == 2:
                        derived_content, _ = derived_data
                    else:
                        derived_content = derived_data

                    page_match = re.match(r'page_(\d+)_.*\.md', derived_filename)
                    if page_match:
                        page_num = int(page_match.group(1))
                        md_contents.append((page_num, derived_content.decode('utf-8')))
                    else:
                        md_contents.append((0, derived_content.decode('utf-8')))

            if md_contents:
                md_contents.sort(key=lambda x: x[0])
                merged_parts = []
                for idx, (page_num, page_md_content) in enumerate(md_contents):
                    if idx > 0:
                        merged_parts.append("\n---\n")
                    if page_num > 0:
                        merged_parts.append(f"\n")
                    merged_parts.append(page_md_content)
                chunk_input_text = ''.join(merged_parts)
                logger.info(f"Merged {len(md_contents)} markdown file(s) from memory, total {len(chunk_input_text)} chars")
            else:
                logger.warning(f"No markdown files found in derived_files_dict for {filename}")
                chunk_input_text = file_text
        else:
            chunk_input_text = file_text

        if not chunk_input_text or len(chunk_input_text.strip()) < 50:
            logger.warning(
                f"Skipping chunk for {filename}: "
                f"text too short ({len(chunk_input_text) if chunk_input_text else 0} chars)"
            )
            chunk_processed = 'chunk_skipped'
            if task_id:
                upload_tracker.update_progress(
                    task_id, UploadStatus.CHUNK_PROCESSING, 60, "Skipped Chunk processing (text too short)"
                )
            return chunk_processed, chunk_file_uploaded, chunked_text

        logger.info(f"Processing chunk for {filename} ({len(chunk_input_text)} chars)")
        if task_id:
            upload_tracker.update_progress(
                task_id, UploadStatus.CHUNK_PROCESSING, 51,
                f"Performing Chunk level recognition ({len(chunk_input_text)} characters)..."
            )

        chunk_progress_stop = asyncio.Event() if task_id else None
        progress_task = None

        if task_id:
            async def simulate_chunk_progress():
                """Simulate progress updates during Chunk processing (51% → 58%)"""
                progress = 51
                while not chunk_progress_stop.is_set():
                    await asyncio.sleep(2)
                    if not chunk_progress_stop.is_set() and progress < 58:
                        progress = min(progress + 1, 58)
                        upload_tracker.update_progress(
                            task_id, UploadStatus.CHUNK_PROCESSING, progress,
                            f"Performing Chunk level recognition...({progress}%)"
                        )
                        logger.debug(f"Task {task_id}: Chunk processing {progress}%")

            progress_task = asyncio.create_task(simulate_chunk_progress())

        from ...knowledge_builder.chunk_processor import ChunkProcessor

        logger.info(f"[SEMAPHORE] Chunk waiting for semaphore, current value: {CHUNK_SEMAPHORE._value}, filename: {filename}")
        async with CHUNK_SEMAPHORE:
            logger.info(f"[SEMAPHORE] Chunk acquired semaphore, remaining: {CHUNK_SEMAPHORE._value}, filename: {filename}")
            chunk_processor = ChunkProcessor(chunk_config)
            chunked_text = await chunk_processor.chunk_document(chunk_input_text)
        logger.info(f"[SEMAPHORE] Chunk released semaphore, current value: {CHUNK_SEMAPHORE._value}, filename: {filename}")

        if task_id and progress_task:
            chunk_progress_stop.set()
            try:
                await asyncio.wait_for(progress_task, timeout=1.0)
            except asyncio.TimeoutError:
                pass

        if chunked_text:
            chunk_processed = 'chunk_success'
            chunk_file_uploaded = f"{Path(filename).stem}_chunklevel.md"
            logger.info(f"✓ Chunk processing completed, will upload later: {chunk_file_uploaded}")
            if task_id:
                upload_tracker.update_progress(
                    task_id, UploadStatus.CHUNK_PROCESSING, 60, f"✓ Chunk processing completed"
                )
        else:
            chunk_processed = 'chunk_failed'
            logger.error(f"✗ Chunk processing returned empty result for {filename}")
            if task_id:
                upload_tracker.update_progress(
                    task_id, UploadStatus.CHUNK_PROCESSING, 60, "✗ Chunk processing failed"
                )

    except Exception as e:
        logger.error(f"Chunk processing error for {filename}: {str(e)}", exc_info=True)
        chunk_processed = 'chunk_failed'
        if task_id:
            upload_tracker.update_progress(
                task_id, UploadStatus.CHUNK_PROCESSING, 60, f"✗ Chunk processing exception: {str(e)}"
            )

    return chunk_processed, chunk_file_uploaded, chunked_text


async def _extract_and_normalize_metadata(
    file_text: str,
    filename: str,
    metadata_config: dict,
    is_image: bool,
    is_pdf: bool,
    ocr_used: bool,
    derived_files_dict: dict,
    chunk_processed: str,
    task_id: str = None
) -> dict:
    """Metadata Process"""
    metadata_extraction_enabled = metadata_config.get("enabled", True)

    if task_id:
        logger.info(f"Starting metadata processing...")

    if metadata_extraction_enabled and file_text:
        logger.info(f"Metadata extraction enabled, extracting metadata for: {filename}")
        if task_id:
            upload_tracker.update_progress(task_id, UploadStatus.METADATA_EXTRACTING, 62, "Preparing to extract metadata...")
            upload_tracker.update_progress(task_id, UploadStatus.METADATA_EXTRACTING, 65, "Extracting metadata using AI...")

        from ...knowledge_builder.metadata_extractor import MetadataExtractor
        metadata_extractor = MetadataExtractor(metadata_config=metadata_config)
        metadata = await metadata_extractor.extract_metadata(
            text=file_text,
            filename=filename
        )

        if task_id:
            upload_tracker.update_progress(task_id, UploadStatus.METADATA_EXTRACTING, 70, "✓ Metadata extraction completed")
    else:
        metadata = {
            "char_length": str(len(file_text)) if file_text else "0",
            "publish_date": None,
            "key_timepoints": [],
            "summary": None,
        }
        if not metadata_extraction_enabled:
            logger.info(f"Metadata extraction disabled, using basic metadata for: {filename}")
        elif not file_text:
            logger.info(f"No text content, using basic metadata for: {filename}")

        if task_id:
            upload_tracker.update_progress(task_id, UploadStatus.UPLOADING, 70, "Skipped metadata extraction")

    normalized_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            normalized_metadata[key] = ';'.join(str(v) for v in value) if value else ''
        else:
            normalized_metadata[key] = value

    if (is_image or is_pdf) and ocr_used:
        if derived_files_dict and len(derived_files_dict) > 0:
            normalized_metadata['ocr_processed'] = 'ocr_success'
        else:
            normalized_metadata['ocr_processed'] = 'ocr_failed'
            logger.warning(f"OCR was attempted for {filename} but failed (no derived files generated)")
    else:
        normalized_metadata['ocr_processed'] = 'ocr_skipped'

    normalized_metadata['chunk_processed'] = chunk_processed

    return normalized_metadata


def _validate_file_content(content: bytes, filename: str, task_id: str = None):
    """
    Validate file content integrity (supports optional progress tracking)

    Args:
        content: File content
        filename: File name
        task_id: Task ID (optional, for progress tracking)
    """
    file_ext = filename.split('.')[-1].lower() if '.' in filename else ''

    if not isinstance(content, bytes):
        error_msg = f"CRITICAL: File content is not bytes! Type: {type(content)}, File: {filename}"
        logger.error(error_msg)
        logger.error(f"Content preview (first 200 chars): {str(content)[:200] if content else 'None'}")

        if task_id:
            upload_tracker.update_progress(
                task_id, UploadStatus.FAILED, 0,
                f"Critical error: File content unexpectedly modified (type: {type(content).__name__})",
                error=error_msg
            )
            raise TypeError(f"Content must be bytes, got {type(content).__name__}")
        else:
            raise HTTPException(
                status_code=400,
                detail=f"File read error: Content type exception (expected bytes, got {type(content).__name__})"
            )

    if file_ext == 'pdf':
        if not content.startswith(b'%PDF'):
            error_msg = f"PDF file validation failed: {filename}, size: {len(content)} bytes"
            logger.error(error_msg)
            logger.error(f"File header (first 100 bytes): {content[:100]}")

            if task_id:
                upload_tracker.update_progress(
                    task_id, UploadStatus.FAILED, 0,
                    f"PDF file format error: Incorrect file header",
                    error=error_msg
                )
                raise ValueError(f"Invalid PDF file header for {filename}")
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"PDF file format error: Incorrect file header (expected %PDF, got {content[:10]})"
                )
        logger.info(f"✓ PDF file validation passed: {filename}")


async def _upload_derived_files(
    derived_files_dict: dict,
    filename: str,
    source_etag_value: str,
    minio_client,
    sys_bucket: str,
    task_id: str
) -> list:
    """Upload OCR derived files"""
    derived_files_uploaded = []

    if not derived_files_dict:
        return derived_files_uploaded

    upload_tracker.update_progress(
        task_id, UploadStatus.UPLOADING_TO_MINIO, 91,
        f"Uploading {len(derived_files_dict)} OCR derived files..."
    )

    for derived_filename, derived_data in derived_files_dict.items():
        try:
            if isinstance(derived_data, tuple) and len(derived_data) == 2:
                file_data, content_type = derived_data
            else:
                file_data = derived_data
                content_type = "application/octet-stream"

            derived_metadata = {
                "source_image": filename,
                "source_image_bucket": minio_client.bucket_name,
                "source_image_path": f"{minio_client.bucket_name}/{filename}",
                "source_image_etag": source_etag_value,
                "file_type": "derived_file",
                "derived_type": derived_filename.split('.')[-1],
                "description": f"{derived_filename} - OCR processed from {filename}"
            }

            derived_success, _ = minio_client.upload_file(
                file_data=file_data,
                object_name=derived_filename,
                content_type=content_type,
                metadata=derived_metadata,
                bucket_name=sys_bucket
            )

            if derived_success:
                derived_files_uploaded.append(derived_filename)
                logger.info(f"Uploaded derived file: {derived_filename} to {sys_bucket}")
            else:
                logger.error(f"Failed to upload derived file: {derived_filename}")

        except Exception as e:
            logger.error(f"Error uploading derived file {derived_filename}: {e}")

    upload_tracker.update_progress(
        task_id, UploadStatus.UPLOADING_TO_MINIO, 93,
        f"✓ Uploaded {len(derived_files_uploaded)} OCR derived files"
    )

    return derived_files_uploaded


async def _upload_chunk_file(
    chunk_processed: str,
    chunk_file_uploaded: str,
    chunked_text: str,
    filename: str,
    source_etag_value: str,
    minio_client,
    sys_bucket: str,
    task_id: str
) -> tuple[str, str]:
    """Upload Chunk file"""
    if chunk_processed != 'chunk_success' or not chunk_file_uploaded or not chunked_text:
        return chunk_processed, chunk_file_uploaded

    try:
        upload_tracker.update_progress(
            task_id, UploadStatus.UPLOADING_TO_MINIO, 94, f"Uploading Chunk file..."
        )

        chunked_metadata = {
            "source_image": filename,
            "source_image_bucket": minio_client.bucket_name,
            "source_image_etag": source_etag_value,
            "file_type": "chunk_derived_file",
            "derived_type": "chunked_markdown",
            "description": f"Chunk level identification from {filename}"
        }

        chunk_success, _ = minio_client.upload_file(
            file_data=chunked_text.encode('utf-8'),
            object_name=chunk_file_uploaded,
            content_type="text/markdown",
            metadata=chunked_metadata,
            bucket_name=sys_bucket
        )

        if chunk_success:
            logger.info(f"✓ Chunk file uploaded: {chunk_file_uploaded}")
            upload_tracker.update_progress(
                task_id, UploadStatus.UPLOADING_TO_MINIO, 95, f"✓ Chunk file uploaded: {chunk_file_uploaded}"
            )
        else:
            logger.error(f"✗ Failed to upload chunk file: {chunk_file_uploaded}")
            chunk_processed = 'chunk_failed'
            chunk_file_uploaded = None
    except Exception as e:
        logger.error(f"Error uploading chunk file {chunk_file_uploaded}: {e}")
        chunk_processed = 'chunk_failed'
        chunk_file_uploaded = None

    return chunk_processed, chunk_file_uploaded


async def _process_file_with_progress(task_id: str, content: bytes, filename: str):
    """
    Background task: Process file and update progress
    """
    logger.info(f"Background task started for task: {task_id}, file: {filename}")

    try:
        content = _validate_content(content, filename, task_id)
    except TypeError:
        return

    try:
        # 1. init (10%)
        upload_tracker.update_progress(
            task_id, UploadStatus.UPLOADING, 10, "Starting file processing..."
        )

        file_ext = filename.split('.')[-1].lower() if '.' in filename else 'txt'
        image_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
        is_image = file_ext in image_extensions
        is_pdf = file_ext == 'pdf'

        from ..services.kb_config_service import KBConfigService
        config = KBConfigService.load_yaml_config("file_management")
        ocr_config = config.get("ocr", {})
        metadata_config = config.get("metadata_extraction", {})
        chunk_config = config.get("chunk", {})
        sys_bucket = os.getenv("MINIO_BUCKET_SYS", "sysfile")

        # 2. OCR (10% -> 50%)
        file_text, ocr_used, derived_files_dict = await _process_ocr(
            content, filename, file_ext, ocr_config, task_id
        )

        # 3. Chunk (50% -> 60%)
        chunk_processed, chunk_file_uploaded, chunked_text = await _process_chunk(
            file_text, filename, file_ext, chunk_config,
            is_image, is_pdf, ocr_used, derived_files_dict, task_id
        )

        # 4. metadata (60% -> 70%)
        normalized_metadata = await _extract_and_normalize_metadata(
            file_text, filename, metadata_config, is_image, is_pdf,
            ocr_used, derived_files_dict, chunk_processed, task_id
        )

        # 5. upload MinIO (70% -> 90%)
        upload_tracker.update_progress(
            task_id, UploadStatus.UPLOADING_TO_MINIO, 72, "Preparing to upload file..."
        )

        upload_tracker.update_progress(
            task_id, UploadStatus.UPLOADING_TO_MINIO, 75, f"Uploading {filename}..."
        )

        _validate_file_content(content, filename, task_id)

        success, source_etag_value = minio_client.upload_file(
            file_data=content,
            object_name=filename,
            content_type="application/octet-stream",
            metadata=normalized_metadata
        )

        if not success:
            raise Exception("Failed to upload file to MinIO")

        upload_tracker.update_progress(
            task_id, UploadStatus.UPLOADING_TO_MINIO, 90, f"✓ Original file uploaded (ETag: {source_etag_value})"
        )

        # 6. sys bucket (90% -> 95%)
        derived_files_uploaded = await _upload_derived_files(
            derived_files_dict, filename, source_etag_value, minio_client, sys_bucket, task_id
        )

        chunk_processed, chunk_file_uploaded = await _upload_chunk_file(
            chunk_processed, chunk_file_uploaded, chunked_text, filename,
            source_etag_value, minio_client, sys_bucket, task_id
        )

        if not derived_files_dict and chunk_processed != 'chunk_success':
            upload_tracker.update_progress(
                task_id, UploadStatus.UPLOADING_TO_MINIO, 95,
                "✓ File upload completed"
            )

        # 7. success (100%)
        result = {
            "message": "File uploaded successfully",
            "filename": filename,
            "metadata": normalized_metadata,
            "ocr_processed": normalized_metadata.get('ocr_processed', 'ocr_failed'),
            "chunk_processed": normalized_metadata.get('chunk_processed', 'chunk_failed'),
            "derived_files": derived_files_uploaded if derived_files_uploaded else None,
            "chunk_file": chunk_file_uploaded
        }

        upload_tracker.update_progress(
            task_id, UploadStatus.COMPLETED, 100, f"✓ {filename} upload completed!", result=result
        )

    except Exception as e:
        logger.error(f"Background upload error for task {task_id}: {str(e)}", exc_info=True)
        upload_tracker.update_progress(
            task_id, UploadStatus.FAILED, 0, f"Processing failed: {str(e)}", error=str(e)
        )


@router.post("/upload-with-progress")
async def upload_file_with_progress(
    file: UploadFile = File(...)
):
    """
    Upload file (with progress tracking)
    """
    try:
        logger.info(f"Creating upload task for file: {file.filename}")
        task_id = upload_tracker.create_task(file.filename)
        logger.debug(f"Created task_id: {task_id}")

        upload_tracker.update_progress(
            task_id, UploadStatus.UPLOADING, 5, "Task created, preparing to process..."
        )

        content = await file.read()

        if not isinstance(content, bytes):
            error_msg = f"CRITICAL: File content is not bytes! Type: {type(content)}, File: {file.filename}"
            logger.error(error_msg)
            upload_tracker.update_progress(
                task_id, UploadStatus.FAILED, 0,
                f"File read error: Content type exception (expected bytes, got {type(content).__name__})",
                error=error_msg
            )
            raise HTTPException(
                status_code=400,
                detail=f"File read error: Content type exception (expected bytes, got {type(content).__name__})"
            )

        logger.debug(f"Task {task_id}: File content read, size {len(content)} bytes")

        asyncio.create_task(_process_file_with_progress(task_id, content, file.filename))

        return {
            "task_id": task_id,
            "message": "File upload task created",
            "filename": file.filename
        }

    except Exception as e:
        logger.error(f"Upload task creation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/upload-progress/{task_id}")
async def get_upload_progress(task_id: str):
    """
    Query upload progress
    """
    progress = upload_tracker.get_progress(task_id)

    if not progress:
        logger.warning(f"Task not found: {task_id}")
        raise HTTPException(status_code=404, detail="Task not found")

    if progress.get('status') == 'chunk_processing' or progress.get('progress', 0) >= 90:
        import json
        logger.info(f"[API Response] Task {task_id[:8]}: status={progress.get('status')}, progress={progress.get('progress')}%, message={progress.get('message')}")

    return progress


@router.get("/active-uploads")
async def get_active_uploads():
    """
    Get all active upload tasks

    Returns:
        List of active upload tasks with their current status
    """
    try:
        active_tasks = []

        for task_id, task_data in upload_tracker._tasks.items():
            if task_data.get('status') not in [UploadStatus.COMPLETED, UploadStatus.FAILED]:
                active_tasks.append({
                    'task_id': task_id,
                    'filename': task_data.get('filename'),
                    'status': task_data.get('status'),
                    'progress': task_data.get('progress', 0),
                    'message': task_data.get('message', ''),
                    'created_at': task_data.get('created_at'),
                    'updated_at': task_data.get('updated_at')
                })

        logger.info(f"Found {len(active_tasks)} active upload task(s)")
        return {
            'active_tasks': active_tasks,
            'total': len(active_tasks)
        }

    except Exception as e:
        logger.error(f"Get active uploads error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_files():
    """
    List all files and their metadata in MinIO bucket
    """
    try:
        files = minio_client.list_files()
        
        result = []
        for file_obj in files:
            all_metadata = minio_client.get_file_metadata(file_obj.object_name) or {}
            
            result.append({
                "name": file_obj.object_name,
                "size": file_obj.size,
                "last_modified": file_obj.last_modified.isoformat(),
                "etag": file_obj.etag,
                "metadata": all_metadata
            })
        
        return result
    
    except Exception as e:
        logger.error(f"List files error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list-local-sqlite")
async def list_local_sqlite_files():
    """
    List SQLite database files in local filesystem
    Search rag_data/relational_database/ directory
    """
    try:
        from pathlib import Path
        from ..config import settings
        
        db_dir = settings.PROJECT_ROOT / "rag_data" / "relational_database"
        
        db_dir.mkdir(parents=True, exist_ok=True)
        
        result = []
        sqlite_extensions = ['.sqlite', '.sqlite3', '.db']
        
        for ext in sqlite_extensions:
            for file_path in db_dir.glob(f"*{ext}"):
                if file_path.is_file():
                    stat = file_path.stat()
                    # Normalize path to use forward slashes for cross-platform compatibility
                    # This ensures Windows paths (C:\path) are converted to (C:/path)
                    normalized_path = str(file_path.absolute()).replace('\\', '/')
                    result.append({
                        "name": file_path.name,
                        "path": normalized_path,
                        "size": stat.st_size,
                        "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })

        result.sort(key=lambda x: x["name"])
        
        logger.info(f"Found {len(result)} SQLite file(s) in {db_dir}")
        return result
    
    except Exception as e:
        logger.error(f"List local SQLite files error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metadata/{filename}")
async def get_file_metadata(filename: str):
    """
    Get metadata for a specific file, including all custom tags
    """
    try:
        all_metadata = minio_client.get_file_metadata(filename)
        
        if all_metadata is None:
            raise HTTPException(status_code=404, detail="File not found")
        
        stat = minio_client.get_file_stat(filename)
        
        all_metadata = {k: v for k, v in all_metadata.items() if not k.endswith('_stamp')}

        result_metadata = dict(all_metadata)
        result_metadata["_file_size"] = stat.size
        result_metadata["_last_modified"] = stat.last_modified.isoformat()
        result_metadata["_content_type"] = stat.content_type
        result_metadata["_etag"] = stat.etag
        
        return result_metadata
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get metadata error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/check-references/{filename}")
async def check_file_references(filename: str, db: Session = Depends(get_db)):
    """
    Check file reference status, returns which knowledge bases reference this file
    """
    try:
        source_configs = db.query(KBSourceConfig).filter(
            KBSourceConfig.source_identifier == filename,
            KBSourceConfig.source_type.in_(["minio_file", "qa_file"])
        ).all()

        if not source_configs:
            return {
                "is_referenced": False,
                "knowledge_bases": [],
                "total_references": 0
            }

        kb_info = []
        for config in source_configs:
            kb = db.query(KnowledgeBase).filter(
                KnowledgeBase.id == config.knowledge_base_id
            ).first()
            if kb:
                kb_info.append({
                    "id": kb.id,
                    "name": kb.name,
                    "collection_name": kb.collection_name,
                    "chunks_created": config.chunks_created or 0
                })

        return {
            "is_referenced": True,
            "knowledge_bases": kb_info,
            "total_references": len(source_configs)
        }

    except Exception as e:
        logger.error(f"Check file references error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete/{filename}")
async def delete_file(filename: str, db: Session = Depends(get_db)):
    """
    Delete file from MinIO and clean up all associated data:
    - Derived files (layout images, md text, json data, etc.)
    - Embedding data in vector store
    - Knowledge base source configurations
    - File-knowledge base mapping relationships
    - Excel table mappings (if Excel file)
    """
    try:
        sys_bucket = os.getenv("MINIO_BUCKET_SYS", "sysfile")
        derived_files = minio_client.find_derived_files(filename, sys_bucket)
        deleted_derived_count = 0

        if derived_files:
            logger.info(f"Found {len(derived_files)} derived file(s) for '{filename}': {derived_files}")
            for derived_file in derived_files:
                try:
                    if minio_client.delete_file(derived_file, bucket_name=sys_bucket):
                        deleted_derived_count += 1
                        logger.info(f"Deleted derived file: {derived_file} from {sys_bucket}")
                    else:
                        logger.warning(f"Failed to delete derived file: {derived_file}")
                except Exception as e:
                    logger.error(f"Error deleting derived file {derived_file}: {e}")
        else:
            logger.info(f"No derived files found for '{filename}'")

        source_configs = db.query(KBSourceConfig).filter(
            KBSourceConfig.source_identifier == filename,
            KBSourceConfig.source_type.in_(["minio_file", "qa_file"])
        ).all()

        affected_kb_ids = set()

        deleted_chunks_total = 0
        for source_config in source_configs:
            affected_kb_ids.add(source_config.knowledge_base_id)
            try:
                kb = db.query(KnowledgeBase).filter(
                    KnowledgeBase.id == source_config.knowledge_base_id
                ).first()

                if kb:
                    from ...storage.implementations.chroma_store import ChromaVectorStore
                    from ...config import VectorStoreConfig

                    vector_config = VectorStoreConfig(
                        collection_name=kb.collection_name,
                        persist_directory=os.getenv("VECTOR_STORE_PATH", "rag_data/vector_store")
                    )
                    vector_store = ChromaVectorStore(config=vector_config)

                    deleted_count = await vector_store.delete_by_document_id(filename)
                    deleted_chunks_total += deleted_count
                    logger.info(f"Deleted {deleted_count} chunks from KB '{kb.name}' (collection: {kb.collection_name})")

            except Exception as e:
                logger.error(f"Failed to delete vector data for KB {source_config.knowledge_base_id}: {e}")

        deleted_excel_tables = 0
        if filename.endswith(('.xlsx', '.xls')):
            excel_tables = db.query(KBExcelTable).filter(
                KBExcelTable.source_file == filename
            ).all()

            table_names = [table.table_name for table in excel_tables]

            deleted_excel_tables = db.query(KBExcelTable).filter(
                KBExcelTable.source_file == filename
            ).delete()

            if deleted_excel_tables > 0:
                logger.info(f"Deleted {deleted_excel_tables} Excel table mapping(s) for file '{filename}': {', '.join(table_names)}")

        deleted_configs = db.query(KBSourceConfig).filter(
            KBSourceConfig.source_identifier == filename,
            KBSourceConfig.source_type.in_(["minio_file", "qa_file"])
        ).delete()

        if deleted_configs > 0:
            logger.info(f"Deleted {deleted_configs} source config(s) for file '{filename}'")
        db.commit()

        success = minio_client.delete_file(filename)

        if not success:
            raise HTTPException(status_code=404, detail="File not found in storage")

        logger.info(f"Successfully deleted file '{filename}' and all associated data")

        return {
            "message": f"File '{filename}' and all associated data deleted successfully",
            "derived_files_deleted": deleted_derived_count,
            "vector_chunks_deleted": deleted_chunks_total,
            "knowledge_bases_affected": len(affected_kb_ids),
            "config_entries_deleted": deleted_configs,
            "excel_tables_deleted": deleted_excel_tables
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete file error: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{filename}")
async def download_file(filename: str):
    """
    Download file from MinIO
    """
    try:
        file_data = minio_client.download_file(filename)
        
        if file_data is None:
            raise HTTPException(status_code=404, detail="File not found")
        
        stat = minio_client.get_file_stat(filename)
        encoded_filename = quote(filename, safe='')

        content_disposition = f"attachment; filename=\"{filename.encode('ascii', 'ignore').decode('ascii') or 'download'}\"; filename*=UTF-8''{encoded_filename}"
        
        return StreamingResponse(
            file_data,
            media_type=stat.content_type or "application/octet-stream",
            headers={
                "Content-Disposition": content_disposition
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download file error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download-with-ocr/{filename}")
async def download_file_with_ocr(filename: str):
    """
    Download OCR-processed file (package download of original file and all derived files)

    Applicable to files with ocr_processed="ocr_success"
    - Original file (downloaded from MINIO_BUCKET)
    - Derived md multi-page merged file
    - All derived files (downloaded from MINIO_BUCKET_SYS)
      - .md files (markdown text)
      - .json files (structured data)
      - _layout files (annotated layout images, same format as original image: png/jpg/bmp/webp)

    Returns a ZIP archive
    """
    try:
        file_data = minio_client.download_file(filename)
        if file_data is None:
            raise HTTPException(status_code=404, detail="Original file not found")

        metadata = minio_client.get_file_metadata(filename)
        logger.info(f"File metadata for {filename}: {metadata}")

        ocr_processed_value = metadata.get("ocr_processed") if metadata else None
        logger.info(f"ocr_processed value: '{ocr_processed_value}' (type: {type(ocr_processed_value).__name__})")

        if not metadata or metadata.get("ocr_processed") != "ocr_success":
            logger.info(f"File {filename} is not OCR-processed (metadata={metadata}), downloading original file only")
            stat = minio_client.get_file_stat(filename)
            encoded_filename = quote(filename, safe='')
            content_disposition = f"attachment; filename=\"{filename.encode('ascii', 'ignore').decode('ascii') or 'download'}\"; filename*=UTF-8''{encoded_filename}"

            return StreamingResponse(
                file_data,
                media_type=stat.content_type or "application/octet-stream",
                headers={"Content-Disposition": content_disposition}
            )

        logger.info(f"Creating ZIP package for OCR-processed file: {filename}")
        sys_bucket = os.getenv("MINIO_BUCKET_SYS", "sysfile")
        base_name = Path(filename).stem

        derived_files = minio_client.find_derived_files(filename, sys_bucket)
        logger.info(f"Found {len(derived_files)} derived files for {filename}")

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            file_data.seek(0) 
            zip_file.writestr(filename, file_data.read())
            logger.info(f"Added original file to ZIP: {filename}")
            md_contents = [] 

            if derived_files:
                for derived_filename in derived_files:
                    if derived_filename.endswith('_chunklevel.md'):
                        logger.debug(f"Skipping _chunklevel.md file in OCR derived: {derived_filename}")
                        continue
                    try:
                        derived_data = minio_client.download_file(derived_filename, bucket_name=sys_bucket)
                        if derived_data:
                            zip_path = f"ocr_derived/{derived_filename}"
                            content = derived_data.read()
                            zip_file.writestr(zip_path, content)
                            logger.debug(f"Added derived file to ZIP: {zip_path}")
                            if derived_filename.endswith('.md'):
                                import re
                                page_match = re.match(r'page_(\d+)_.*\.md', derived_filename)
                                if page_match:
                                    page_num = int(page_match.group(1))
                                    md_contents.append((page_num, content.decode('utf-8')))
                                else:
                                    md_contents.append((0, content.decode('utf-8')))
                    except Exception as e:
                        logger.warning(f"Failed to add derived file {derived_filename}: {e}")


            if md_contents:
                md_contents.sort(key=lambda x: x[0])
                summary_content = ""
                for idx, (page_num, page_md_content) in enumerate(md_contents):
                    if idx > 0:
                        summary_content += "\n\n---\n\n"
                    if page_num > 0:
                        summary_content += f"\n"
                    summary_content += page_md_content

                summary_filename = f"{base_name}.md"
                zip_file.writestr(summary_filename, summary_content.encode('utf-8'))
                logger.info(f"Added markdown merge file to ZIP: {summary_filename} ({len(md_contents)} pages)")

        zip_buffer.seek(0)
        zip_filename = f"{base_name}_with_ocr.zip"
        encoded_zip_filename = quote(zip_filename, safe='')
        content_disposition = f"attachment; filename=\"{zip_filename.encode('ascii', 'ignore').decode('ascii') or 'download.zip'}\"; filename*=UTF-8''{encoded_zip_filename}"

        logger.info(f"ZIP package created successfully: {zip_filename} (original + {len(derived_files)} derived files)")

        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={"Content-Disposition": content_disposition}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download with OCR error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download-with-derivatives/{filename}")
async def download_file_with_derivatives(filename: str):
    """
    Smart download file (package download of original file and all derived files based on processing status)

    Supported processing types:
    1. chunk_processed="chunk_success" + ocr_processed="ocr_success":
       - Original file + OCR derived files + Chunk file
    2. chunk_processed="chunk_success" only:
       - Original file + Chunk file
    3. ocr_processed="ocr_success" only:
       - Original file + OCR derived files
    4. No processing:
       - Original file only

    Returns:
    - If there are derived files: Returns ZIP archive
    - If no derived files: Returns original file
    """
    try:
        file_data = minio_client.download_file(filename)
        if file_data is None:
            raise HTTPException(status_code=404, detail="Original file not found")

        metadata = minio_client.get_file_metadata(filename)
        logger.info(f"File metadata for {filename}: {metadata}")

        ocr_processed = metadata.get("ocr_processed") == "ocr_success" if metadata else False
        chunk_processed = metadata.get("chunk_processed") == "chunk_success" if metadata else False

        logger.info(f"File processing status - OCR: {ocr_processed}, Chunk: {chunk_processed}")

        if not ocr_processed and not chunk_processed:
            logger.info(f"File {filename} has no processing, downloading original file only")
            stat = minio_client.get_file_stat(filename)
            encoded_filename = quote(filename, safe='')
            content_disposition = f"attachment; filename=\"{filename.encode('ascii', 'ignore').decode('ascii') or 'download'}\"; filename*=UTF-8''{encoded_filename}"

            return StreamingResponse(
                file_data,
                media_type=stat.content_type or "application/octet-stream",
                headers={"Content-Disposition": content_disposition}
            )

        logger.info(f"Creating ZIP package with derivatives for file: {filename}")
        sys_bucket = os.getenv("MINIO_BUCKET_SYS", "sysfile")
        base_name = Path(filename).stem

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            file_data.seek(0)
            zip_file.writestr(filename, file_data.read())
            logger.info(f"Added original file to ZIP: {filename}")

            md_contents = []

            if ocr_processed:
                derived_files = minio_client.find_derived_files(filename, sys_bucket)
                logger.info(f"Found {len(derived_files)} OCR derived files for {filename}")

                if derived_files:
                    for derived_filename in derived_files:
                        if derived_filename.endswith('_chunklevel.md'):
                            logger.debug(f"Skipping _chunklevel.md file in OCR derived: {derived_filename}")
                            continue
                        try:
                            derived_data = minio_client.download_file(derived_filename, bucket_name=sys_bucket)
                            if derived_data:
                                zip_path = f"ocr_derived/{derived_filename}"
                                content = derived_data.read()
                                zip_file.writestr(zip_path, content)
                                logger.debug(f"Added OCR derived file to ZIP: {zip_path}")

                                if derived_filename.endswith('.md') and not derived_filename.endswith('_chunklevel.md'):
                                    import re
                                    page_match = re.match(r'page_(\d+)_.*\.md', derived_filename)
                                    if page_match:
                                        page_num = int(page_match.group(1))
                                        md_contents.append((page_num, content.decode('utf-8')))
                                    else:
                                        md_contents.append((0, content.decode('utf-8')))
                        except Exception as e:
                            logger.warning(f"Failed to add OCR derived file {derived_filename}: {e}")

            if chunk_processed:
                chunk_filename = f"{base_name}_chunklevel.md"
                try:
                    chunk_data = minio_client.download_file(chunk_filename, bucket_name=sys_bucket)
                    if chunk_data:
                        zip_path = f"chunk_derived/{chunk_filename}"
                        content = chunk_data.read()
                        zip_file.writestr(zip_path, content)
                        logger.info(f"Added chunk file to ZIP: {zip_path}")
                    else:
                        logger.warning(f"Chunk file {chunk_filename} not found in sys bucket")
                except Exception as e:
                    logger.warning(f"Failed to add chunk file {chunk_filename}: {e}")

            if md_contents:
                md_contents.sort(key=lambda x: x[0])
                summary_content = ""
                for idx, (page_num, page_md_content) in enumerate(md_contents):
                    if idx > 0:
                        summary_content += "\n\n---\n\n"
                    if page_num > 0:
                        summary_content += f"\n"
                    summary_content += page_md_content

                summary_filename = f"{base_name}_ocr_merged.md"
                zip_file.writestr(summary_filename, summary_content.encode('utf-8'))
                logger.info(f"Added OCR markdown merge file to ZIP: {summary_filename} ({len(md_contents)} pages)")

        zip_buffer.seek(0)

        if chunk_processed and ocr_processed:
            zip_filename = f"{base_name}_with_ocr_chunk.zip"
        elif chunk_processed:
            zip_filename = f"{base_name}_with_chunk.zip"
        else:
            zip_filename = f"{base_name}_with_ocr.zip"

        encoded_zip_filename = quote(zip_filename, safe='')
        content_disposition = f"attachment; filename=\"{zip_filename.encode('ascii', 'ignore').decode('ascii') or 'download.zip'}\"; filename*=UTF-8''{encoded_zip_filename}"

        logger.info(f"ZIP package created successfully: {zip_filename}")

        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={"Content-Disposition": content_disposition}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download with derivatives error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download-sys/{filename}")
async def download_sys_file(filename: str):
    """
    Download file from system bucket (MINIO_BUCKET_SYS)

    Used to download derived files, such as:
    - OCR derived .md, .json, _layout files
    - Chunk derived _chunklevel.md files
    """
    try:
        sys_bucket = os.getenv("MINIO_BUCKET_SYS", "sysfile")
        file_data = minio_client.download_file(filename, bucket_name=sys_bucket)

        if file_data is None:
            raise HTTPException(status_code=404, detail=f"File not found in system bucket: {filename}")

        stat = minio_client.client.stat_object(sys_bucket, filename)
        encoded_filename = quote(filename, safe='')
        content_disposition = f"attachment; filename=\"{filename.encode('ascii', 'ignore').decode('ascii') or 'download'}\"; filename*=UTF-8''{encoded_filename}"

        return StreamingResponse(
            file_data,
            media_type=stat.content_type or "application/octet-stream",
            headers={"Content-Disposition": content_disposition}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download sys file error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-sys/{filename}")
async def upload_sys_file(filename: str, request: dict):
    """
    Upload or update file in system bucket (MINIO_BUCKET_SYS)

    Used to save edited derived files, such as:
    - Chunk derived _chunklevel.md files

    Request Body:
        {
            "content": "File content as string"
        }
    """
    try:
        sys_bucket = os.getenv("MINIO_BUCKET_SYS", "sysfile")
        content = request.get("content")

        if content is None:
            raise HTTPException(status_code=400, detail="Missing content in request body")

        existing_metadata = minio_client.get_file_metadata(filename, bucket_name=sys_bucket) or {}

        metadata = {**existing_metadata}
        metadata["last_modified"] = datetime.now(timezone.utc).isoformat()
        metadata["is_manual_edited"] = "true"

        success, _ = minio_client.upload_file(
            file_data=content.encode('utf-8'),
            object_name=filename,
            content_type="text/markdown",
            metadata=metadata,
            bucket_name=sys_bucket
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to upload file to system bucket")

        logger.info(f"Updated file in system bucket: {filename}")

        return {
            "message": "File saved successfully",
            "filename": filename
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload sys file error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ocr-results/{filename}")
async def get_ocr_results(filename: str):
    """
    Get OCR results for an image or PDF (reads derived files from MINIO_BUCKET_SYS)

    For single-page images:
        Returns single page data

    For multi-page PDFs:
        Detects page_1_*, page_2_*, etc. files and returns all pages

    Args:
        filename: Original file name (e.g., test5.png or document.pdf)

    Returns:
        For single page:
        {
            "filename": "test5.png",
            "is_multi_page": false,
            "markdown_text": "...",
            "structured_data": [...],
            "layout_image_url": "/api/files/ocr-results/test5.png/layout",
            "original_image_url": "/api/files/download/test5.png"
        }

        For multi-page:
        {
            "filename": "document.pdf",
            "is_multi_page": true,
            "total_pages": 3,
            "pages": [
                {
                    "page_num": 1,
                    "markdown_text": "...",
                    "structured_data": [...],
                    "layout_image_url": "/api/files/ocr-results/document.pdf/layout/1"
                },
                {...},
                {...}
            ],
            "original_image_url": "/api/files/download/document.pdf"
        }
    """
    try:
        sys_bucket = os.getenv("MINIO_BUCKET_SYS", "sysfile")
        base_name = Path(filename).stem

        # try to detect multi-page documents. Check if page_1_* file exists without generating error logs
        page_1_md = f"page_1_{base_name}.md"
        page_1_exists = False

        try:
            minio_client.client.stat_object(sys_bucket, page_1_md)
            page_1_exists = True
        except Exception:
            pass

        if page_1_exists:
            page_1_data = minio_client.download_file(page_1_md, bucket_name=sys_bucket)
            if not page_1_data:
                raise HTTPException(status_code=404, detail=f"OCR result not found: {page_1_md}")
            logger.info(f"Multi-page document detected for {filename}")
            pages = []
            page_num = 1

            while True:
                page_md_filename = f"page_{page_num}_{base_name}.md"
                page_json_filename = f"page_{page_num}_{base_name}.json"

                try:
                    minio_client.client.stat_object(sys_bucket, page_md_filename)
                except Exception:
                    break

                try:
                    minio_client.client.stat_object(sys_bucket, page_json_filename)
                except Exception:
                    logger.warning(f"JSON file missing for {page_md_filename}")
                    break

                md_data = minio_client.download_file(page_md_filename, bucket_name=sys_bucket)
                json_data = minio_client.download_file(page_json_filename, bucket_name=sys_bucket)

                if not md_data or not json_data:
                    logger.warning(f"Failed to download files for page {page_num}")
                    break

                import json as json_module

                pages.append({
                    "page_num": page_num,
                    "markdown_text": md_data.read().decode('utf-8'),
                    "structured_data": json_module.loads(json_data.read().decode('utf-8')),
                    "layout_image_url": f"/api/files/ocr-results/{filename}/layout/{page_num}"
                })

                page_num += 1

            if not pages:
                raise HTTPException(status_code=404, detail=f"No pages found for {filename}")

            return {
                "filename": filename,
                "is_multi_page": True,
                "total_pages": len(pages),
                "pages": pages,
                "original_image_url": f"/api/files/download/{filename}"
            }

        else:
            logger.info(f"Single-page document for {filename}")

            md_filename = f"{base_name}.md"
            try:
                minio_client.client.stat_object(sys_bucket, md_filename)
            except Exception:
                raise HTTPException(status_code=404, detail=f"OCR result not found: {md_filename}")

            json_filename = f"{base_name}.json"
            try:
                minio_client.client.stat_object(sys_bucket, json_filename)
            except Exception:
                raise HTTPException(status_code=404, detail=f"OCR result not found: {json_filename}")

            md_data = minio_client.download_file(md_filename, bucket_name=sys_bucket)
            if not md_data:
                raise HTTPException(status_code=500, detail=f"Failed to download: {md_filename}")
            markdown_text = md_data.read().decode('utf-8')

            json_data = minio_client.download_file(json_filename, bucket_name=sys_bucket)
            if not json_data:
                raise HTTPException(status_code=500, detail=f"Failed to download: {json_filename}")

            import json as json_module
            structured_data = json_module.loads(json_data.read().decode('utf-8'))

            return {
                "filename": filename,
                "is_multi_page": False,
                "markdown_text": markdown_text,
                "structured_data": structured_data,
                "layout_image_url": f"/api/files/ocr-results/{filename}/layout",
                "original_image_url": f"/api/files/download/{filename}"
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get OCR results error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ocr-results/{filename}/layout")
@router.get("/ocr-results/{filename}/layout/{page_num}")
async def get_ocr_layout_image(filename: str, page_num: int = None):
    """
    Get OCR layout image (annotated image)

    Args:
        filename: Original file name
        page_num: Page number (for multi-page PDFs, e.g., 1, 2, 3...)

    Returns:
        StreamingResponse: Layout image
    """
    try:
        sys_bucket = os.getenv("MINIO_BUCKET_SYS", "sysfile")
        base_name = Path(filename).stem

        supported_formats = [
            ('png', 'image/png'),
            ('jpg', 'image/jpeg'),
            ('bmp', 'image/bmp'),
            ('webp', 'image/webp')
        ]

        file_data = None
        layout_filename = None
        media_type = None

        if page_num is not None:
            for ext, mime_type in supported_formats:
                layout_filename = f"page_{page_num}_{base_name}_layout.{ext}"
                try:
                    minio_client.client.stat_object(sys_bucket, layout_filename)
                    file_data = minio_client.download_file(layout_filename, bucket_name=sys_bucket)
                    if file_data:
                        media_type = mime_type
                        break
                except Exception:
                    continue
        else:
            for ext, mime_type in supported_formats:
                layout_filename = f"{base_name}_layout.{ext}"
                try:
                    minio_client.client.stat_object(sys_bucket, layout_filename)
                    file_data = minio_client.download_file(layout_filename, bucket_name=sys_bucket)
                    if file_data:
                        media_type = mime_type
                        break
                except Exception:
                    continue

        if not file_data:
            raise HTTPException(status_code=404, detail=f"Layout image not found for: {base_name}")

        encoded_filename = quote(layout_filename, safe='')
        content_disposition = f"inline; filename=\"{layout_filename.encode('ascii', 'ignore').decode('ascii') or 'layout.png'}\"; filename*=UTF-8''{encoded_filename}"

        return StreamingResponse(
            file_data,
            media_type=media_type,
            headers={"Content-Disposition": content_disposition}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get layout image error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ocr-results/{filename}/save")
async def save_ocr_edits(filename: str, request: dict):
    """
    Save OCR editing results (overwrite mode)

    Request Body:
        For single page:
        {
            "markdown_text": "Modified markdown text",
            "structured_data": [...]  # Modified JSON structure
        }

        For multi-page:
        {
            "page_num": 1,  # Page number to save
            "markdown_text": "Modified markdown text",
            "structured_data": [...]  # Modified JSON structure
        }

    Returns:
        {
            "message": "OCR results saved",
            "files_updated": ["test5.md", "test5.json"]
        }
    """
    try:
        sys_bucket = os.getenv("MINIO_BUCKET_SYS", "sysfile")
        base_name = Path(filename).stem

        markdown_text = request.get("markdown_text")
        structured_data = request.get("structured_data")
        page_num = request.get("page_num")  
        if not markdown_text or not structured_data:
            raise HTTPException(status_code=400, detail="Missing markdown_text or structured_data")

        updated_files = []

        if page_num is not None:
            md_filename = f"page_{page_num}_{base_name}.md"
            json_filename = f"page_{page_num}_{base_name}.json"
        else:
            md_filename = f"{base_name}.md"
            json_filename = f"{base_name}.json"

        existing_md_metadata = minio_client.get_file_metadata(md_filename, bucket_name=sys_bucket) or {}
        existing_json_metadata = minio_client.get_file_metadata(json_filename, bucket_name=sys_bucket) or {}

        md_metadata = {**existing_md_metadata}
        md_metadata["last_modified"] = datetime.now(timezone.utc).isoformat()
        md_metadata["is_manual_edited"] = "true"

        md_success, _ = minio_client.upload_file(
            file_data=markdown_text.encode('utf-8'),
            object_name=md_filename,
            content_type="text/markdown",
            metadata=md_metadata,
            bucket_name=sys_bucket
        )
        if md_success:
            updated_files.append(md_filename)
            logger.info(f"Updated markdown file: {md_filename}")

        json_metadata = {**existing_json_metadata}
        json_metadata["last_modified"] = datetime.now(timezone.utc).isoformat()
        json_metadata["is_manual_edited"] = "true"

        import json as json_module
        json_success, _ = minio_client.upload_file(
            file_data=json_module.dumps(structured_data, ensure_ascii=False, indent=2).encode('utf-8'),
            object_name=json_filename,
            content_type="application/json",
            metadata=json_metadata,
            bucket_name=sys_bucket
        )
        if json_success:
            updated_files.append(json_filename)
            logger.info(f"Updated JSON file: {json_filename}")

        return {
            "message": "OCR results saved",
            "files_updated": updated_files
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Save OCR edits error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _validate_excel_file(filename: str) -> None:
    """Validate if the file is an Excel file"""
    if not filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(
            status_code=400,
            detail=f"❌ File format error: Only Excel files (.xlsx or .xls) are supported, current file is '{filename}'"
        )


def _read_excel_headers(sheet) -> List[str]:
    """Read and return headers from the first row of Excel sheet"""
    headers = []
    for cell in sheet[1]:
        if cell.value is None:
            break
        headers.append(str(cell.value).strip())
    return headers


def _validate_excel_headers(headers: List[str]) -> None:
    """Validate Excel headers format"""
    if len(headers) < 2:
        raise HTTPException(
            status_code=400,
            detail="❌ Excel format error: Must contain at least 2 columns - 'File Name' (Column A) and 'ETAG' (Column B)"
        )
    
    if headers[0] != 'File Name' or headers[1] != 'ETAG':
        raise HTTPException(
            status_code=400,
            detail=f"❌ Excel column name error: First two columns must be 'File Name' and 'ETAG', currently '{headers[0]}' and '{headers[1]}'. Please use the export function to generate the correct format template."
        )


def _validate_metadata_columns(metadata_columns: List[str]) -> None:
    """Validate metadata column names"""
    import re
    
    MAX_CUSTOM_FIELDS = 50
    if len(metadata_columns) > MAX_CUSTOM_FIELDS:
        raise HTTPException(
            status_code=400,
            detail=f"❌ Too many metadata fields: {len(metadata_columns)} custom fields exceed the maximum limit of {MAX_CUSTOM_FIELDS}. Please reduce the number of columns and try again."
        )
    
    FIELD_NAME_PATTERN = re.compile(r'^[\w\s\u4e00-\u9fff-]{1,100}$')
    for col_name in metadata_columns:
        if not FIELD_NAME_PATTERN.match(col_name):
            raise HTTPException(
                status_code=400,
                detail=f"❌ Invalid field name: '{col_name}'. Field names must be 1-100 characters, supporting only Chinese, English, numbers, spaces, underscores and hyphens."
            )


def _validate_row_data(filename: str, etag_from_excel: str, minio_files: Dict[str, str]) -> tuple[bool, str]:
    """
    Validate row data (filename and ETAG)
    
    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    if not filename or not etag_from_excel:
        return False, "Missing required fields: File name or ETAG cannot be empty"
    
    if filename not in minio_files:
        return False, "File does not exist: File not found in storage system, please check if the file name is correct"
    
    if minio_files[filename] != etag_from_excel:
        return False, f"ETAG mismatch: Expected {etag_from_excel}, actual {minio_files[filename]}. File may have been modified."
    
    return True, ""


def _extract_cell_metadata(row: tuple, metadata_columns: List[str]) -> Dict[str, str]:
    """Extract metadata from Excel row"""
    metadata = {}
    for col_idx, col_name in enumerate(metadata_columns, start=2):
        cell_value = row[col_idx] if col_idx < len(row) else None
        if cell_value is not None:
            metadata[col_name] = str(cell_value).strip()
        else:
            metadata[col_name] = ""
    return metadata


def _preserve_timestamp_fields(metadata: Dict[str, str], existing_metadata: Dict[str, str]) -> None:
    """Preserve timestamp fields from existing metadata"""
    timestamp_fields = [
        'key_timepoints_min_stamp',
        'key_timepoints_max_stamp',
        'publish_date_min_stamp',
        'publish_date_max_stamp'
    ]
    for field in timestamp_fields:
        if field in existing_metadata:
            metadata[field] = existing_metadata[field]


def _process_publish_date_timestamp(metadata: Dict[str, str], filename: str) -> None:
    """Process and add publish_date timestamp to metadata"""
    from ...utils.date_utils import date_to_time_range
    
    if 'publish_date' not in metadata or not metadata['publish_date']:
        return
    
    try:
        min_stamp, max_stamp = date_to_time_range(metadata['publish_date'])
        if min_stamp is not None:
            metadata['publish_date_min_stamp'] = min_stamp
        if max_stamp is not None:
            metadata['publish_date_max_stamp'] = max_stamp
    except Exception as e:
        logger.warning(f"Failed to parse publish_date for {filename}: {e}")


def _process_key_timepoints_timestamp(metadata: Dict[str, str], filename: str) -> None:
    """Process and add key_timepoints timestamp to metadata"""
    from ...utils.date_utils import date_to_time_range
    
    if 'key_timepoints' not in metadata or not metadata['key_timepoints']:
        return
    
    try:
        timepoints = [tp.strip() for tp in metadata['key_timepoints'].split(';') if tp.strip()]
        if not timepoints:
            return
        
        min_stamps, max_stamps = [], []
        for timepoint in timepoints:
            min_stamp, max_stamp = date_to_time_range(timepoint)
            if min_stamp is not None:
                min_stamps.append(min_stamp)
            if max_stamp is not None:
                max_stamps.append(max_stamp)
        
        if min_stamps:
            metadata['key_timepoints_min_stamp'] = min(min_stamps)
        if max_stamps:
            metadata['key_timepoints_max_stamp'] = max(max_stamps)
    except Exception as e:
        logger.warning(f"Failed to parse key_timepoints for {filename}: {e}")


def _update_file_metadata(filename: str, metadata: Dict[str, str]) -> tuple[bool, str]:
    """
    Update file metadata in MinIO
    
    Returns:
        tuple[bool, str]: (is_successful, error_message)
    """
    try:
        success = minio_client.update_metadata(filename, metadata)
        if success:
            logger.info(f"Updated metadata for '{filename}'")
            return True, ""
        else:
            return False, "Failed to update metadata"
    except Exception as e:
        return False, f"Error updating metadata: {str(e)}"


def _process_single_row(
    row_idx: int,
    row: tuple,
    metadata_columns: List[str],
    minio_files: Dict[str, str]
) -> tuple[bool, Dict[str, Any]]:
    """
    Process a single Excel row
    
    Returns:
        tuple[bool, Dict[str, Any]]: (is_successful, error_dict or None)
    """
    filename = str(row[0]).strip() if row[0] else None
    etag_from_excel = str(row[1]).strip() if row[1] else None
    
    # Validate row data
    is_valid, error_msg = _validate_row_data(filename, etag_from_excel, minio_files)
    if not is_valid:
        return False, {
            "row": row_idx,
            "filename": filename or "N/A",
            "error": error_msg
        }
    
    # Get existing metadata
    existing_metadata = minio_client.get_file_metadata(filename) or {}
    
    # Extract metadata from row
    metadata = _extract_cell_metadata(row, metadata_columns)
    
    # Preserve timestamp fields
    _preserve_timestamp_fields(metadata, existing_metadata)
    
    # Process timestamp fields
    _process_publish_date_timestamp(metadata, filename)
    _process_key_timepoints_timestamp(metadata, filename)
    
    # Update metadata
    success, error_msg = _update_file_metadata(filename, metadata)
    if not success:
        return False, {
            "row": row_idx,
            "filename": filename,
            "error": error_msg
        }
    
    return True, None


@router.post("/import-metadata", response_model=MetadataImportResult)
async def import_metadata(file: UploadFile = File(...)):
    """
    Import metadata from Excel file and update MinIO file metadata

    Excel format:
    - Column A: File Name (required)
    - Column B: ETAG (required, for validation)
    - Column C+: Custom metadata fields

    Update strategy: Complete overwrite (replace all metadata with Excel data)
    ETAG validation: Skip files with mismatched ETAG
    """
    try:
        import openpyxl
        import io
        
        # Validate file format
        _validate_excel_file(file.filename)
        
        # Load Excel workbook
        contents = await file.read()
        workbook = openpyxl.load_workbook(io.BytesIO(contents), read_only=True, data_only=True)
        
        sheet = workbook.worksheets[0]
        logger.info(f"Reading sheet: {sheet.title} (total sheets: {len(workbook.worksheets)})")
        
        # Read and validate headers
        headers = _read_excel_headers(sheet)
        _validate_excel_headers(headers)
        
        metadata_columns = headers[2:]
        _validate_metadata_columns(metadata_columns)
        
        # Initialize counters
        total_rows = 0
        successful = 0
        failed = 0
        errors = []
        
        # Get all MinIO files for validation
        minio_files = {f.object_name: f.etag for f in minio_client.list_files()}
        
        # Process each row
        for row_idx, row in enumerate(sheet.iter_rows(min_row=2, values_only=True), start=2):
            if not row or not row[0]:
                continue
            
            total_rows += 1
            
            # Process single row
            is_successful, error_dict = _process_single_row(
                row_idx, row, metadata_columns, minio_files
            )
            
            if is_successful:
                successful += 1
            else:
                failed += 1
                errors.append(error_dict)
        
        return MetadataImportResult(
            total_rows=total_rows,
            successful=successful,
            failed=failed,
            errors=errors
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Import metadata error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
