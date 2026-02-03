"""
- [ ] unified OS environment
- [ ] support advanecd tools for file viewing & editing
    https://github.com/Intelligent-Internet/ii-agent/blob/main/src/ii_agent/tools/str_replace_tool.py
- [ ] context management (for loong files)
"""

import re
import shutil
from datetime import datetime
from pathlib import Path

from ..config import ToolkitConfig
from ..utils import get_logger
from .base import AsyncBaseToolkit, register_tool

logger = get_logger(__name__)


class FileEditToolkit(AsyncBaseToolkit):
    def __init__(self, config: ToolkitConfig = None) -> None:
        super().__init__(config)
        workspace_root = self.config.config.get("workspace_root", "/tmp/")
        self.setup_workspace(workspace_root)

        self.default_encoding = self.config.config.get("default_encoding", "utf-8")
        self.backup_enabled = self.config.config.get("backup_enabled", False)
        logger.info(
            f"FileEditToolkit initialized with output directory: {self.work_dir}, encoding: {self.default_encoding}"
        )

    def setup_workspace(self, workspace_root: str):
        self.work_dir = Path(workspace_root).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_filename(self, filename: str) -> str:
        safe = re.sub(r"[^\w\-.]", "_", filename)
        return safe

    def _resolve_filepath(self, file_path: str) -> Path:
        path_obj = Path(file_path)
        if not path_obj.is_absolute():
            path_obj = self.work_dir / path_obj

        sanitized_filename = self._sanitize_filename(path_obj.name)
        path_obj = path_obj.parent / sanitized_filename
        resolved_path = path_obj.resolve()

        if not resolved_path.is_relative_to(self.work_dir):
            raise ValueError(
                f"Security error: Path '{file_path}' resolves to '{resolved_path}' "
                f"which is outside the workspace directory '{self.work_dir}'"
            )

        self._create_backup(resolved_path)
        return resolved_path

    def _create_backup(self, file_path: Path) -> None:
        if not self.backup_enabled or not file_path.exists():
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.parent / f"{file_path.name}.{timestamp}.bak"
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup at {backup_path}")

    @register_tool
    async def edit_file(self, path: str, diff: str) -> str:
        r"""Edit a file by applying the provided diff.

        Args:
            file_name (str): The name of the file to edit.
            diff (str): (required) One or more SEARCH/REPLACE blocks following this exact format:
                ```
                <<<<<<< SEARCH
                [exact content to find]
                =======
                [new content to replace with]
                >>>>>>> REPLACE
                ```
        """
        resolved_path = self._resolve_filepath(path)

        with open(resolved_path, encoding=self.default_encoding) as f:
            content = f.read()
        modified_content = content
        pattern = r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE"
        matches = re.findall(pattern, diff, re.DOTALL)
        if not matches:
            return "Error! No valid diff blocks found in the provided diff"

        # Apply each search/replace pair
        for search_text, replace_text in matches:
            if search_text in modified_content:
                modified_content = modified_content.replace(search_text, replace_text)
            else:
                logger.warning(f"Search text not found in file: {search_text[:50]}...")

        with open(resolved_path, "w", encoding=self.default_encoding) as f:
            f.write(modified_content)
        return f"Successfully edited file: {resolved_path}"

    @register_tool
    async def write_file(self, path: str, file_text: str) -> str:
        """Write text content to a file.

        Args:
            path (str): The path of the file to write.
            file_text (str): The full text content to write.
        """
        path_obj = self._resolve_filepath(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(file_text)
        return f"Successfully wrote file: {path_obj}"

    @register_tool
    async def read_file(self, path: str) -> str:
        """Read and return the contents of a file.

        Args:
            path (str): The path of the file to read.
        """
        path_obj = self._resolve_filepath(path)
        return path_obj.read_text()
