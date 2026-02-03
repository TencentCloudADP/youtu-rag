"""File toolkit for downloading files from MinIO knowledge base storage.

This toolkit provides tools to download files from MinIO to the Python executor workspace,
enabling Excel QA and other agents to work with files stored in the knowledge base.
"""

import logging
import os
from typing import Optional, List

from ...config import ToolkitConfig
from ...tools.base import AsyncBaseToolkit, register_tool

logger = logging.getLogger(__name__)


class FileToolkit(AsyncBaseToolkit):
    """File toolkit for downloading files from MinIO knowledge base.

    Tools:
        - download_kb_text_content: Download text content (OCR-processed markdown or original files)
        - download_kb_files: Download original files from MinIO to local workspace
        - get_python_workspace: Get the current Python executor workspace directory
    """

    def __init__(self, config: ToolkitConfig | dict | None = None):
        """Initialize File toolkit.

        Args:
            config: Toolkit configuration
        """
        super().__init__(config)

        from ..api.minio_client import minio_client

        self.minio_client = minio_client
        # Store reference to python executor workspace (will be set by agent context)
        self._python_workspace = None
        logger.info("FileToolkit initialized with MinIO client")

    def set_python_workspace(self, workspace_root: str):
        """Set the Python executor workspace directory.

        This is typically called by the agent framework to share the workspace
        between python_executor and file_toolkit.

        Args:
            workspace_root: Path to Python executor workspace
        """
        self._python_workspace = workspace_root
        logger.info(f"Python workspace set to: {workspace_root}")

    @register_tool
    async def get_python_workspace(self) -> str:
        """Get the current Python executor workspace directory path.

        This tool returns the working directory used by the Python executor,
        which is needed for downloading files to the correct location.

        Returns:
            String containing the workspace path, e.g.:
            "/tmp/utu/python_executor/20251224_175334_2e1787ac"

        Example:
            # Get workspace path first
            workspace = get_python_workspace()
            # Then use it to download files
            download_kb_files(file_names="data.xlsx", target_dir=workspace)

        Note:
            If the workspace hasn't been initialized yet, this will return a
            default path. It's recommended to call execute_python_code at least
            once before using this tool to ensure the workspace is properly set up.
        """
        import json

        if self._python_workspace:
            return json.dumps(
                {"workspace_root": self._python_workspace, "status": "initialized"},
                ensure_ascii=False,
            )
        else:
            # Return default path if not yet initialized
            from datetime import datetime
            import uuid

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            default_workspace = f"/tmp/utu/python_executor/{timestamp}_{unique_id}"

            return json.dumps(
                {
                    "workspace_root": default_workspace,
                    "status": "not_initialized",
                    "note": "This is a default path. Execute Python code first to initialize the workspace.",
                },
                ensure_ascii=False,
            )

    @register_tool
    async def download_kb_text_content(
        self,
        file_names: str | List[str],
        target_dir: str,
        bucket_name: Optional[str] = None,
    ) -> str:
        """Download text content of files from MinIO knowledge base to local directory.
        **Full functionality, download all derived text content**

        This tool downloads files and their derived content:
        1. Always downloads the original file from user bucket (MINIO_BUCKET)
        2. If file has been OCR and chunk processed (ocr_processed=true & chunk_processed=true):
           - Downloads all OCR-derived markdown files (e.g., page_1_xxx.md, page_2_xxx.md)
           - Downloads chunk-level markdown file (e.g., xxx_chunklevel.md)
        3. All files are downloaded separately to target directory

        Args:
            file_names: Single file name (str) or list of file names to download.
                       Examples: "document.pdf" or ["file1.pdf", "file2.docx"]
            target_dir: Target directory path where files will be downloaded.
                       Example: "/tmp/utu/python_executor/20251224_175334_2e1787ac"
            bucket_name: Optional source bucket name for original files.
                        If not provided, uses default MINIO_BUCKET from config.

        Returns:
            JSON string with download results, e.g.
            ```
            {
                "success": true,
                "total_files": 2,
                "downloaded": 2,
                "failed": 0,
                "target_dir": "/tmp/utu/python_executor/...",
                "files": [
                    {
                        "file_name": "document.pdf",
                        "status": "success",
                        "original_file": {
                            "local_path": "/tmp/.../document.pdf",
                            "source_bucket": "ufile"
                        },
                        "derived_files": [
                            {
                                "file_name": "page_1_document.md",
                                "local_path": "/tmp/.../page_1_document.md",
                                "file_type": "ocr_markdown"
                            },
                            {
                                "file_name": "page_2_document.md",
                                "local_path": "/tmp/.../page_2_document.md",
                                "file_type": "ocr_markdown"
                            },
                            {
                                "file_name": "document_chunklevel.md",
                                "local_path": "/tmp/.../document_chunklevel.md",
                                "file_type": "chunk_markdown"
                            }
                        ],
                        "ocr_processed": true,
                        "chunk_processed": true,
                        "total_derived_files": 3
                    }
                ],
                "errors": []
            }
            ```

        Example:
            ```
            # Download single file with all derived content
            download_kb_text_content(
                file_names="report.pdf",
                target_dir="/tmp/utu/python_executor/20251224_175334_2e1787ac"
            )

            # Download multiple files
            download_kb_text_content(
                file_names=["doc1.pdf", "doc2.docx"],
                target_dir="/tmp/utu/python_executor/20251224_175334_2e1787ac"
            )
            ```
        """
        import json
        import os
        import re

        try:
            if isinstance(file_names, str):
                files_to_download = [file_names]
            else:
                files_to_download = file_names

            sys_bucket_name = os.getenv("MINIO_BUCKET_SYS", "sysfile")

            os.makedirs(target_dir, exist_ok=True)

            result_data = {
                "success": True,
                "total_files": len(files_to_download),
                "downloaded": 0,
                "failed": 0,
                "target_dir": target_dir,
                "files": [],
                "errors": [],
            }

            for file_name in files_to_download:
                try:
                    logger.info(f"Processing text content for {file_name}...")

                    metadata = self.minio_client.get_file_metadata(
                        file_name, bucket_name=bucket_name
                    )

                    if metadata is None:
                        raise Exception(f"File not found in MinIO: {file_name}")

                    ocr_processed = metadata.get("ocr_processed") == "ocr_success"
                    chunk_processed = metadata.get("chunk_processed") == "chunk_success"

                    file_info = {
                        "file_name": file_name,
                        "status": "success",
                        "ocr_processed": ocr_processed,
                        "chunk_processed": chunk_processed,
                        "original_file": {},
                        "derived_files": [],
                    }

                    # Always download original file
                    logger.info(f"Downloading original file: {file_name}...")
                    file_data = self.minio_client.download_file(
                        file_name, bucket_name=bucket_name
                    )

                    if file_data is None:
                        raise Exception(f"File not found in MinIO: {file_name}")

                    original_local_path = os.path.join(target_dir, file_name)

                    local_dir = os.path.dirname(original_local_path)
                    if local_dir and local_dir != target_dir:
                        os.makedirs(local_dir, exist_ok=True)

                    with open(original_local_path, "wb") as f:
                        f.write(file_data.read())

                    file_info["original_file"] = {
                        "local_path": original_local_path,
                        "source_bucket": bucket_name or self.minio_client.bucket_name,
                    }

                    logger.info(f"✓ Downloaded original file to {original_local_path}")

                    if ocr_processed and chunk_processed:
                        logger.info(
                            f"File {file_name} is processed, downloading derived files from system bucket..."
                        )

                        derived_file_names = self.minio_client.find_derived_files(
                            source_filename=file_name, sys_bucket=sys_bucket_name
                        )

                        if derived_file_names:
                            logger.info(
                                f"Found {len(derived_file_names)} derived files for {file_name}"
                            )

                            md_files = [f for f in derived_file_names if f.endswith(".md")]

                            page_pattern = re.compile(r"^page_\d+_.+\.md$")
                            chunklevel_pattern = re.compile(r"^.+_chunklevel\.md$")

                            for md_file in md_files:
                                try:
                                    if chunklevel_pattern.match(md_file):
                                        file_type = "chunk_markdown"
                                    elif page_pattern.match(md_file):
                                        file_type = "ocr_markdown"
                                    else:
                                        file_type = "other_markdown"

                                    md_data = self.minio_client.download_file(
                                        md_file, bucket_name=sys_bucket_name
                                    )

                                    if md_data:
                                        md_local_path = os.path.join(target_dir, md_file)

                                        try:
                                            content = md_data.read().decode("utf-8")
                                        except UnicodeDecodeError:
                                            logger.warning(
                                                f"Encoding error in {md_file}, using fallback decoding"
                                            )
                                            md_data.seek(0)
                                            content = md_data.read().decode(
                                                "utf-8", errors="replace"
                                            )

                                        with open(md_local_path, "w", encoding="utf-8") as f:
                                            f.write(content)

                                        file_info["derived_files"].append(
                                            {
                                                "file_name": md_file,
                                                "local_path": md_local_path,
                                                "file_type": file_type,
                                            }
                                        )

                                        logger.info(
                                            f"  ✓ Downloaded {file_type}: {md_file}"
                                        )
                                    else:
                                        logger.warning(
                                            f"  ✗ Failed to download derived file: {md_file}"
                                        )

                                except Exception as e:
                                    logger.warning(
                                        f"  ✗ Error downloading {md_file}: {str(e)}"
                                    )

                            file_info["total_derived_files"] = len(
                                file_info["derived_files"]
                            )

                            logger.info(
                                f"✓ Downloaded {len(file_info['derived_files'])} derived files "
                                f"for {file_name}"
                            )
                        else:
                            logger.warning(
                                f"No derived files found for {file_name} in system bucket"
                            )
                            file_info["total_derived_files"] = 0
                    else:
                        logger.info(
                            f"File {file_name} not fully processed "
                            f"(OCR: {ocr_processed}, Chunk: {chunk_processed}), "
                            f"skipping derived files"
                        )
                        file_info["total_derived_files"] = 0

                    result_data["files"].append(file_info)
                    result_data["downloaded"] += 1

                except Exception as e:
                    error_msg = f"Failed to download text content for {file_name}: {str(e)}"
                    logger.error(error_msg)

                    result_data["files"].append(
                        {
                            "file_name": file_name,
                            "status": "failed",
                            "error": str(e),
                        }
                    )
                    result_data["failed"] += 1
                    result_data["errors"].append(error_msg)

            if result_data["failed"] > 0:
                result_data["success"] = False

            logger.info(
                f"Text content download completed: {result_data['downloaded']}/{result_data['total_files']} successful"
            )

            return json.dumps(result_data, ensure_ascii=False, indent=2)

        except Exception as e:
            error_msg = f"Text content download failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return json.dumps(
                {
                    "success": False,
                    "error": str(e),
                    "total_files": len(files_to_download)
                    if "files_to_download" in locals()
                    else 0,
                    "downloaded": 0,
                    "failed": 0,
                },
                ensure_ascii=False,
            )

    @register_tool
    async def download_kb_files(
        self,
        file_names: str | List[str],
        target_dir: str,
        bucket_name: Optional[str] = None,
    ) -> str:
        """Download files from MinIO knowledge base to local directory.
        **Lightweight, fast download of original files**

        This tool downloads specified files from MinIO storage to a local directory,
        typically the Python executor workspace. It's designed to work with files
        stored in the knowledge base.

        Args:
            file_names: Single file name (str) or list of file names to download.
                       Examples: "data.xlsx" or ["file1.xlsx", "file2.csv"]
            target_dir: Target directory path where files will be downloaded.
                       For Python executor, use the workspace_root path.
                       Example: "/tmp/utu/python_executor/20251224_175334_2e1787ac"
            bucket_name: Optional MinIO bucket name. If not provided, uses default bucket
                        from MinIO client configuration.

        Returns:
            JSON string with download results, e.g.
            ```
            {
                "success": true,
                "total_files": 2,
                "downloaded": 2,
                "failed": 0,
                "target_dir": "/tmp/utu/python_executor/...",
                "files": [
                    {
                        "file_name": "data.xlsx",
                        "status": "success",
                        "local_path": "/tmp/utu/python_executor/.../data.xlsx"
                    },
                    {
                        "file_name": "report.csv",
                        "status": "success",
                        "local_path": "/tmp/utu/python_executor/.../report.csv"
                    }
                ],
                "errors": []
            }
            ```

        Example:
            ```
            # Download single file
            download_kb_files(
                file_names="sales_data.xlsx",
                target_dir="/tmp/utu/python_executor/20251224_175334_2e1787ac"
            )

            # Download multiple files
            download_kb_files(
                file_names=["file1.xlsx", "file2.csv"],
                target_dir="/tmp/utu/python_executor/20251224_175334_2e1787ac"
            )
            ```
        """
        import json

        try:
            # Normalize file_names to list
            if isinstance(file_names, str):
                files_to_download = [file_names]
            else:
                files_to_download = file_names

            # Ensure target directory exists
            os.makedirs(target_dir, exist_ok=True)

            result_data = {
                "success": True,
                "total_files": len(files_to_download),
                "downloaded": 0,
                "failed": 0,
                "target_dir": target_dir,
                "files": [],
                "errors": [],
            }

            for file_name in files_to_download:
                try:
                    logger.info(f"Downloading {file_name} from MinIO to {target_dir}...")

                    file_data = self.minio_client.download_file(file_name, bucket_name=bucket_name)

                    if file_data is None:
                        raise Exception(f"File not found in MinIO: {file_name}")

                    local_path = os.path.join(target_dir, file_name)

                    local_dir = os.path.dirname(local_path)
                    if local_dir and local_dir != target_dir:
                        os.makedirs(local_dir, exist_ok=True)

                    with open(local_path, "wb") as f:
                        f.write(file_data.read())

                    result_data["files"].append(
                        {"file_name": file_name, "status": "success", "local_path": local_path}
                    )
                    result_data["downloaded"] += 1

                    logger.info(f"✓ Downloaded {file_name} to {local_path}")

                except Exception as e:
                    error_msg = f"Failed to download {file_name}: {str(e)}"
                    logger.error(error_msg)

                    result_data["files"].append(
                        {"file_name": file_name, "status": "failed", "error": str(e)}
                    )
                    result_data["failed"] += 1
                    result_data["errors"].append(error_msg)

            if result_data["failed"] > 0:
                result_data["success"] = False

            logger.info(
                f"Download completed: {result_data['downloaded']}/{result_data['total_files']} successful"
            )

            return json.dumps(result_data, ensure_ascii=False, indent=2)

        except Exception as e:
            error_msg = f"File download failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return json.dumps(
                {
                    "success": False,
                    "error": str(e),
                    "total_files": len(files_to_download) if "files_to_download" in locals() else 0,
                    "downloaded": 0,
                    "failed": 0,
                },
                ensure_ascii=False,
            )
