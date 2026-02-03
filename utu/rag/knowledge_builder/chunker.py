"""Text chunking strategies."""

import re
from typing import Any

from ..base import BaseTextSplitter
from ..config import ChunkingConfig


class RecursiveTextSplitter(BaseTextSplitter):
    """Recursively split text using multiple separators."""

    def __init__(self, config: ChunkingConfig | None = None):
        """Initialize recursive text splitter.

        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig(strategy="recursive")
        self.separators = self.config.separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str, metadata: dict[str, Any] | None = None) -> list[str]:
        """Split text recursively using separators.

        Args:
            text: Text to split
            metadata: Optional metadata (not used in this implementation)

        Returns:
            List of text chunks
        """
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using separators.

        Args:
            text: Text to split
            separators: List of separators to try

        Returns:
            List of text chunks
        """
        if not separators:
            return self._split_by_length(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            return self._split_by_length(text)

        splits = text.split(separator)
        chunks = []
        current_chunk = ""

        for i, split in enumerate(splits):
            test_chunk = current_chunk + split
            if self.config.keep_separator and i < len(splits) - 1:
                test_chunk += separator

            if len(test_chunk) <= self.config.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # If split is still too large, recursively split it
                if len(split) > self.config.chunk_size:
                    chunks.extend(self._recursive_split(split, remaining_separators))
                    current_chunk = ""
                else:
                    current_chunk = split
                    if self.config.keep_separator and i < len(splits) - 1:
                        current_chunk += separator

        if current_chunk:
            chunks.append(current_chunk)

        # Handle overlap
        if self.config.chunk_overlap > 0:
            chunks = self._add_overlap(chunks)

        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _split_by_length(self, text: str) -> list[str]:
        """Split text by fixed length.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        chunks = []
        for i in range(0, len(text), self.config.chunk_size - self.config.chunk_overlap):
            chunks.append(text[i : i + self.config.chunk_size])
        return chunks

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlap between chunks.

        Args:
            chunks: List of chunks

        Returns:
            List of chunks with overlap
        """
        if len(chunks) <= 1:
            return chunks

        overlapped_chunks = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]

            # Add overlap from previous chunk
            overlap_text = prev_chunk[-self.config.chunk_overlap :]
            overlapped_chunk = overlap_text + current_chunk
            overlapped_chunks.append(overlapped_chunk)

        return overlapped_chunks


class HierarchicalMarkdownSplitter(BaseTextSplitter):
    """Split markdown text by hierarchical structure (H1/H2 headers).

    This splitter is designed for chunklevel.md files that have hierarchical structure:
    - Respects markdown heading hierarchy (# and ##)
    - Keeps each line (sentence) intact without truncation
    - Groups content under headers into logical chunks
    - Respects chunk_size while preserving sentence boundaries
    """

    def __init__(self, config: ChunkingConfig | None = None):
        """Initialize hierarchical markdown splitter.

        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig(strategy="hierarchical")
        # Pattern to match markdown headers (# H1 and ## H2)
        self.h1_pattern = re.compile(r'^#\s+(.+)$', re.MULTILINE)
        self.h2_pattern = re.compile(r'^##\s+(.+)$', re.MULTILINE)

    def split_text(self, text: str, metadata: dict[str, Any] | None = None) -> list[str]:
        """Split text by hierarchical markdown structure.

        Strategy:
        1. Parse markdown into hierarchical sections (H1 → H2 → content)
        2. Split sections based on chunk_size while keeping lines intact
        3. Add header context to each chunk for better retrieval

        Args:
            text: Markdown text to split
            metadata: Optional metadata (not used)

        Returns:
            List of text chunks with header context preserved
        """
        if not text or not text.strip():
            return []

        # Parse text into hierarchical sections
        sections = self._parse_hierarchical_sections(text)

        # Split sections into chunks respecting size limits
        chunks = []
        for section in sections:
            section_chunks = self._split_section(section)
            chunks.extend(section_chunks)

        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _parse_hierarchical_sections(self, text: str) -> list[dict[str, Any]]:
        """Parse markdown text into hierarchical sections.

        Args:
            text: Markdown text

        Returns:
            List of section dictionaries with structure:
            {
                'h1': str,           # H1 header text (if exists)
                'h2': str,           # H2 header text (if exists)
                'content': list[str], # Content lines
                'level': int         # 1 for H1 section, 2 for H2 section, 0 for no header
            }
        """
        lines = text.split('\n')
        sections = []
        current_h1 = None
        current_h2 = None
        current_content = []

        def save_section():
            """Save current section if it has content."""
            if current_content:
                # Determine section level
                level = 0
                if current_h2:
                    level = 2
                elif current_h1:
                    level = 1

                sections.append({
                    'h1': current_h1,
                    'h2': current_h2,
                    'content': current_content.copy(),
                    'level': level
                })
                current_content.clear()

        for line in lines:
            # Check for H1 header
            h1_match = re.match(r'^#\s+(.+)$', line)
            if h1_match:
                save_section()
                current_h1 = h1_match.group(1).strip()
                current_h2 = None  # Reset H2 when new H1 starts
                continue

            # Check for H2 header
            h2_match = re.match(r'^##\s+(.+)$', line)
            if h2_match:
                save_section()
                current_h2 = h2_match.group(1).strip()
                continue

            # Regular content line
            if line.strip():  # Only add non-empty lines
                current_content.append(line)

        # Save the last section
        save_section()

        return sections

    def _split_section(self, section: dict[str, Any]) -> list[str]:
        """Split a section into chunks while preserving line boundaries.

        Args:
            section: Section dictionary from _parse_hierarchical_sections

        Returns:
            List of chunk strings
        """
        chunks = []

        # Build header context
        header_parts = []
        if section['h1']:
            header_parts.append(f"# {section['h1']}")
        if section['h2']:
            header_parts.append(f"## {section['h2']}")

        header_text = '\n'.join(header_parts)
        header_length = len(header_text)

        # Process content lines
        content_lines = section['content']
        if not content_lines:
            # If section has only header, return header as a chunk
            if header_text:
                chunks.append(header_text)
            return chunks

        current_chunk_lines = []
        current_length = header_length

        for line in content_lines:
            line_length = len(line) + 1  # +1 for newline

            # Check if adding this line would exceed chunk_size
            if current_length + line_length > self.config.chunk_size and current_chunk_lines:
                # Save current chunk with header
                chunk_content = '\n'.join(current_chunk_lines)
                if header_text:
                    chunk = f"{header_text}\n\n{chunk_content}"
                else:
                    chunk = chunk_content
                chunks.append(chunk)

                # Start new chunk
                current_chunk_lines = [line]
                current_length = header_length + line_length
            else:
                # Add line to current chunk
                current_chunk_lines.append(line)
                current_length += line_length

        # Save remaining content
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            if header_text:
                chunk = f"{header_text}\n\n{chunk_content}"
            else:
                chunk = chunk_content
            chunks.append(chunk)

        # Handle overlap if configured
        if self.config.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_header_aware_overlap(chunks, header_text)

        return chunks

    def _add_header_aware_overlap(self, chunks: list[str], header_text: str) -> list[str]:
        """Add overlap between chunks while preserving header structure.

        Args:
            chunks: List of chunks
            header_text: Header text to preserve

        Returns:
            List of chunks with overlap
        """
        if len(chunks) <= 1:
            return chunks

        overlapped_chunks = [chunks[0]]
        header_length = len(header_text) if header_text else 0

        for i in range(1, len(chunks)):
            current_chunk = chunks[i]

            # Extract content (remove header if present)
            if header_text and current_chunk.startswith(header_text):
                content = current_chunk[len(header_text):].lstrip('\n')
            else:
                content = current_chunk

            # Get overlap from previous chunk
            prev_chunk = chunks[i - 1]
            if header_text and prev_chunk.startswith(header_text):
                prev_content = prev_chunk[len(header_text):].lstrip('\n')
            else:
                prev_content = prev_chunk

            # Take last N characters from previous content as overlap
            overlap_text = prev_content[-self.config.chunk_overlap:].lstrip()

            # Combine header + overlap + current content
            if header_text:
                overlapped_chunk = f"{header_text}\n\n{overlap_text}\n{content}"
            else:
                overlapped_chunk = f"{overlap_text}\n{content}"

            overlapped_chunks.append(overlapped_chunk)

        return overlapped_chunks
