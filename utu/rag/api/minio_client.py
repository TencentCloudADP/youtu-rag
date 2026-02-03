from minio import Minio
from minio.error import S3Error
from typing import Optional, Dict, Any, List, Tuple
import io
import logging
import json
import base64
import re
import hashlib
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class MinIOClient:
    """
    MinIO client wrapper for file operations and metadata management
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket_name: str = "rag-documents",
        tmp_dir: str = "/tmp/minio",
        secure: bool = False
    ):
        """
        Initialize MinIO client

        Args:
            endpoint: MinIO server endpoint (e.g., "localhost:9000")
            access_key: MinIO access key
            secret_key: MinIO secret key
            bucket_name: Bucket name for storing documents
            secure: Use HTTPS if True
        """
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        self.bucket_name = bucket_name
        self.tmp_dir = os.path.join(tmp_dir, bucket_name)
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        self._ensure_bucket()

    def _ensure_bucket(self):
        """Create bucket if it doesn't exist"""
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Created bucket: {self.bucket_name}")
            else:
                logger.info(f"Bucket already exists: {self.bucket_name}")
        except S3Error as e:
            logger.error(f"Error ensuring bucket exists: {e}")
            raise

    def check_health(self) -> bool:
        """
        Check if MinIO connection is healthy

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            # Try to list buckets to verify connection
            self.client.list_buckets()
            return True
        except Exception as e:
            logger.error(f"MinIO health check failed: {e}")
            return False

    def upload_file(
        self,
        file_data: bytes,
        object_name: str,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, Any]] = None,
        bucket_name: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Upload a file to MinIO with metadata

        Args:
            file_data: File content as bytes
            object_name: Name of the object in MinIO
            content_type: MIME type of the file
            metadata: User-defined metadata dictionary
            bucket_name: Target bucket name (uses self.bucket_name if not specified)

        Returns:
            Tuple of (success, etag) where success is True if upload successful,
            and etag is the ETag of the uploaded file (or None if failed)
        """
        try:
            # Use default bucket if not specified
            target_bucket = bucket_name or self.bucket_name

            # Ensure target bucket exists
            if not self.client.bucket_exists(target_bucket):
                self.client.make_bucket(target_bucket)
                logger.info(f"Created bucket: {target_bucket}")

            # Encode metadata as Base64 JSON to support non-ASCII characters (e.g., Chinese)
            minio_metadata = {}
            if metadata:
                # Serialize metadata to JSON and encode as Base64
                json_str = json.dumps(metadata, ensure_ascii=False)
                encoded = base64.b64encode(json_str.encode('utf-8')).decode('ascii')
                minio_metadata['custom-metadata'] = encoded

            # Upload file
            result = self.client.put_object(
                bucket_name=target_bucket,
                object_name=object_name,
                data=io.BytesIO(file_data),
                length=len(file_data),
                content_type=content_type,
                metadata=minio_metadata
            )

            # Extract ETag from the result
            etag = result.etag.strip('"') if result and result.etag else None

            logger.info(f"Uploaded file: {object_name} with metadata, ETag: {etag}")
            return True, etag

        except S3Error as e:
            logger.error(f"Error uploading file {object_name}: {e}")
            return False, None

    def list_files(self, bucket_name: Optional[str] = None) -> List[Any]:
        """
        List all files in the bucket

        Args:
            bucket_name: Target bucket name (uses self.bucket_name if not specified)

        Returns:
            List of object information
        """
        try:
            target_bucket = bucket_name or self.bucket_name
            objects = self.client.list_objects(target_bucket, recursive=True)
            return list(objects)
        except S3Error as e:
            logger.error(f"Error listing files: {e}")
            return []

    def get_file_metadata(self, object_name: str, bucket_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific file

        Args:
            object_name: Name of the object
            bucket_name: Target bucket name (uses self.bucket_name if not specified)

        Returns:
            Dictionary of metadata, or None if file not found
        """
        try:
            target_bucket = bucket_name or self.bucket_name
            stat = self.client.stat_object(target_bucket, object_name)

            # Extract user metadata
            metadata = {}
            if stat.metadata:
                # Look for our custom-metadata field (Base64 encoded JSON)
                custom_metadata_key = None
                for key in stat.metadata.keys():
                    if 'custom-metadata' in key.lower():
                        custom_metadata_key = key
                        break

                if custom_metadata_key and stat.metadata[custom_metadata_key]:
                    # Decode Base64 and parse JSON
                    try:
                        encoded = stat.metadata[custom_metadata_key]
                        decoded = base64.b64decode(encoded).decode('utf-8')
                        metadata = json.loads(decoded)
                    except Exception as e:
                        logger.error(f"Error decoding custom metadata: {e}")
                        metadata = {}
                else:
                    # Fallback: try old format (for backward compatibility)
                    for key, value in stat.metadata.items():
                        clean_key = key.replace('x-amz-meta-', '')
                        if ';' in value:
                            metadata[clean_key] = value.split(';')
                        else:
                            metadata[clean_key] = value

            return metadata

        except S3Error as e:
            logger.error(f"Error getting metadata for {object_name}: {e}")
            return None

    def get_file_stat(self, object_name: str, bucket_name: Optional[str] = None) -> Any:
        """
        Get file statistics

        Args:
            object_name: Name of the object
            bucket_name: Target bucket name (uses self.bucket_name if not specified)

        Returns:
            Object stat information
        """
        try:
            target_bucket = bucket_name or self.bucket_name
            return self.client.stat_object(target_bucket, object_name)
        except S3Error as e:
            logger.error(f"Error getting file stat for {object_name}: {e}")
            raise

    def file_exists(self, object_name: str, bucket_name: Optional[str] = None) -> bool:
        """
        Check if a file exists in MinIO (without logging errors for non-existent files)

        Args:
            object_name: Name of the object
            bucket_name: Target bucket name (uses self.bucket_name if not specified)

        Returns:
            True if file exists, False otherwise
        """
        try:
            target_bucket = bucket_name or self.bucket_name
            self.client.stat_object(target_bucket, object_name)
            return True
        except S3Error as e:
            # Don't log error for "not found" - it's expected behavior
            if e.code == 'NoSuchKey':
                return False
            # Log other errors
            logger.warning(f"Unexpected error checking if file exists {object_name}: {e}")
            return False

    def check_file_is_local(self, object_name: str) -> bool:
        """
        Check if a file is local
        Args:
            object_name: Name of the object
        
        Returns:
            True if file is local, False otherwise
        """
        local_file_path = os.path.join(self.tmp_dir, object_name)
        return os.path.exists(local_file_path)

    def download_file(self, object_name: str, bucket_name: Optional[str] = None) -> Optional[io.BytesIO]:
        """
        Download a file from MinIO

        Args:
            object_name: Name of the object
            bucket_name: Target bucket name (uses self.bucket_name if not specified)

        Returns:
            BytesIO object containing file data, or None if file not found
        """
        try:
            target_bucket = bucket_name or self.bucket_name
            response = self.client.get_object(target_bucket, object_name)
            data = io.BytesIO(response.read())
            response.close()
            response.release_conn()
            return data

        except S3Error as e:
            logger.error(f"Error downloading file {object_name}: {e}")
            return None

    def download_file_to_local(self, object_name: str) -> Optional[str]:
        """
        Download a file from MinIO to local tmp_dir

        Args:
            object_name: Name of the object

        Returns:
            Local file path if successful, None if file not found
        """
        try:
            local_file_path = os.path.join(self.tmp_dir, object_name)
 
            local_dir = os.path.dirname(local_file_path)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            
            self.client.fget_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                file_path=local_file_path
            )
            
            logger.info(f"Downloaded file to local: {local_file_path}")
            return local_file_path

        except S3Error as e:
            logger.error(f"Error downloading file {object_name} to local: {e}")
            return None

    def delete_file(self, object_name: str, bucket_name: Optional[str] = None) -> bool:
        """
        Delete a file from MinIO

        Args:
            object_name: Name of the object
            bucket_name: Target bucket name (uses self.bucket_name if not specified)

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            target_bucket = bucket_name or self.bucket_name
            self.client.remove_object(target_bucket, object_name)
            logger.info(f"Deleted file: {object_name} from bucket: {target_bucket}")
            return True

        except S3Error as e:
            logger.error(f"Error deleting file {object_name} from bucket {bucket_name}: {e}")
            return False

    def update_metadata(
        self,
        object_name: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Update metadata for an existing file

        Note: MinIO doesn't support direct metadata updates.
        This method downloads the file, then re-uploads with new metadata.

        Args:
            object_name: Name of the object
            metadata: New metadata dictionary

        Returns:
            True if update successful, False otherwise
        """
        try:
            file_data = self.download_file(object_name)
            if file_data is None:
                return False

            stat = self.get_file_stat(object_name)

            # Re-upload with new metadata
            file_data.seek(0)
            success, _ = self.upload_file(
                file_data=file_data.read(),
                object_name=object_name,
                content_type=stat.content_type,
                metadata=metadata
            )
            return success

        except Exception as e:
            logger.error(f"Error updating metadata for {object_name}: {e}")
            return False

    def search_by_metadata(
        self,
        key: str,
        value: str,
        bucket_name: Optional[str] = None
    ) -> List[str]:
        """
        Search files by metadata key-value pair

        Args:
            key: Metadata key to search
            value: Metadata value to match
            bucket_name: Target bucket name (uses self.bucket_name if not specified)

        Returns:
            List of object names matching the criteria
        """
        try:
            matching_files = []
            target_bucket = bucket_name or self.bucket_name

            for obj in self.list_files(bucket_name=target_bucket):
                metadata = self.get_file_metadata(obj.object_name, bucket_name=target_bucket)
                if metadata and key in metadata:
                    meta_value = metadata[key]
                    # Handle both string and list values
                    if isinstance(meta_value, list):
                        if value in meta_value:
                            matching_files.append(obj.object_name)
                    elif str(meta_value) == str(value):
                        matching_files.append(obj.object_name)

            return matching_files

        except Exception as e:
            logger.error(f"Error searching by metadata: {e}")
            return []

    def find_derived_files(self, source_filename: str, sys_bucket: str) -> List[str]:
        """
        Find all derived files associated with a source file

        Args:
            source_filename: Original source file name
            sys_bucket: System bucket name where derived files are stored

        Returns:
            List of derived file names
        """
        try:
            return self.search_by_metadata(
                key="source_image",
                value=source_filename,
                bucket_name=sys_bucket
            )
        except Exception as e:
            logger.error(f"Error finding derived files for {source_filename}: {e}")
            return []

    def load_derived_markdown_files(
        self,
        source_filename: str,
        sys_bucket: str
    ) -> Tuple[Optional[str], List[str]]:
        """
        Load and combine all derived markdown files for a source file

        For single-page images: {filename}.md
        For multi-page PDFs: page_1_{filename}.md, page_2_{filename}.md, ...

        Args:
            source_filename: Original source file name
            sys_bucket: System bucket name where derived files are stored

        Returns:
            Tuple of (combined_markdown_text, etag_list)
            - combined_markdown_text: Combined markdown content, None if failed
            - etag_list: List of ETags from all markdown files
        """
        try:
            derived_files = self.find_derived_files(source_filename, sys_bucket)
            if not derived_files:
                logger.warning(f"No derived files found for {source_filename}")
                return None, []

            # Filter markdown files only
            md_files = [f for f in derived_files if f.endswith('.md')]
            if not md_files:
                logger.warning(f"No markdown files found in derived files for {source_filename}")
                return None, []

            # Detect if multi-page PDF or single-page image
            page_pattern = re.compile(r'^page_(\d+)_(.+)\.md$')
            pages = []
            single_file = None

            for md_file in md_files:
                match = page_pattern.match(md_file)
                if match:  # Multi-page PDF
                    page_num = int(match.group(1))
                    pages.append((page_num, md_file))
                else:  # Single-page image
                    single_file = md_file

            # Sort pages by page number
            if pages:
                pages.sort(key=lambda x: x[0])
                md_files_sorted = [f for _, f in pages]
            elif single_file:
                md_files_sorted = [single_file]
            else:
                logger.error(f"Unexpected markdown file naming pattern for {source_filename}")
                return None, []

            # Verify all markdown files exist before downloading
            for md_file in md_files_sorted:
                try:
                    # Try to get file stat using the sys_bucket
                    # Note: get_file_stat uses self.bucket_name by default, so we need to use client directly
                    self.client.stat_object(sys_bucket, md_file)
                except S3Error as e:
                    if e.code == 'NoSuchKey':
                        logger.warning(
                            f"Derived markdown file not found: {md_file} in bucket {sys_bucket}. "
                            f"This is expected if OCR processing was incomplete or files were deleted. "
                            f"Falling back to original file loader for {source_filename}"
                        )
                    else:
                        logger.warning(
                            f"Cannot access derived file {md_file}: {e}. "
                            f"Falling back to original file loader for {source_filename}"
                        )
                    return None, []
                except Exception as e:
                    logger.warning(
                        f"Unexpected error checking derived file {md_file}: {e}. "
                        f"Falling back to original file loader"
                    )
                    return None, []

            # Download and combine markdown content
            all_content = []
            etag_list = []

            for md_file in md_files_sorted:
                try:
                    md_stream = self.download_file(md_file, bucket_name=sys_bucket)
                    if not md_stream:
                        logger.warning(f"Failed to download {md_file}, falling back to original loader")
                        return None, []

                    try:
                        content = md_stream.read().decode('utf-8')
                    except UnicodeDecodeError:
                        logger.warning(f"Encoding error in {md_file}, using fallback decoding")
                        md_stream.seek(0)
                        content = md_stream.read().decode('utf-8', errors='replace')

                    all_content.append(content)

                    try:
                        stat = self.client.stat_object(sys_bucket, md_file)
                        if stat and stat.etag:
                            etag_list.append(stat.etag.strip('"'))
                    except Exception as e:
                        logger.warning(f"Failed to get ETag for {md_file}: {e}")

                except Exception as e:
                    logger.warning(f"Error loading derived markdown file {md_file}: {e}. Falling back to original loader")
                    return None, []

            if not all_content:
                logger.warning(f"No content loaded from derived markdown files for {source_filename}. Falling back to original loader")
                return None, []

            combined_markdown = "\n\n---\n\n".join(all_content)  # Combine content with separator

            logger.info(
                f"Successfully loaded {len(md_files_sorted)} derived markdown file(s) "
                f"for {source_filename} ({len(combined_markdown)} chars)"
            )

            return combined_markdown, etag_list

        except Exception as e:
            logger.warning(f"Error in load_derived_markdown_files for {source_filename}: {e}. Falling back to original loader")
            return None, []

    def calculate_derived_files_hash(self, etag_list: List[str]) -> str:
        """
        Calculate MD5 hash from list of derived file ETags

        Args:
            etag_list: List of ETag strings

        Returns:
            MD5 hash of sorted ETags (hexadecimal string)
        """
        if not etag_list:
            return ""

        sorted_etags = sorted(etag_list)  # Sort for consistency

        combined = ";".join(sorted_etags)  # Join with separator

        hash_obj = hashlib.md5(combined.encode('utf-8'))

        return hash_obj.hexdigest()

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics for the bucket

        Returns:
            Dictionary containing storage statistics
        """
        try:
            objects = self.list_files()

            total_size = sum(obj.size for obj in objects)
            file_count = len(objects)

            # Group by file extension
            extensions = {}
            for obj in objects:
                ext = obj.object_name.split('.')[-1] if '.' in obj.object_name else 'unknown'
                extensions[ext] = extensions.get(ext, 0) + 1

            return {
                "total_files": file_count,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_types": extensions,
                "bucket_name": self.bucket_name
            }

        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {}


import os
from dotenv import load_dotenv
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
env_path = project_root / ".env"
load_dotenv(env_path)

minio_client = MinIOClient(
    endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
    access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    bucket_name=os.getenv("MINIO_BUCKET", "rag-documents"),
    tmp_dir=os.getenv("MINIO_LOCAL_TMP_DIR", "/tmp/minio"),
    secure=os.getenv("MINIO_SECURE", "false").lower() == "true"
)

logger.info(f"✓ Global MinIO client initialized (bucket: {minio_client.bucket_name})")
