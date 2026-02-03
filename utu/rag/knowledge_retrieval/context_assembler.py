"""Assemble retrieved context for LLM consumption."""

import logging
from typing import Any

from ..base import RetrievalResult

logger = logging.getLogger(__name__)


class ContextAssembler:
    """Assemble retrieved chunks into formatted context."""

    def __init__(self, max_context_length: int = 4000):
        """Initialize context assembler.

        Args:
            max_context_length: Maximum length of assembled context in characters
        """
        self.max_context_length = max_context_length

    def assemble(
        self, results: list[RetrievalResult], include_metadata: bool = True, format_style: str = "markdown"
    ) -> str:
        """Assemble retrieval results into formatted context.

        Args:
            results: List of retrieval results
            include_metadata: Whether to include metadata in context
            format_style: Format style ("markdown", "plain", "json")

        Returns:
            Formatted context string
        """
        if not results:
            return ""

        if format_style == "markdown":
            return self._assemble_markdown(results, include_metadata)
        elif format_style == "plain":
            return self._assemble_plain(results, include_metadata)
        elif format_style == "json":
            return self._assemble_json(results, include_metadata)
        else:
            msg = f"Unknown format style: {format_style}"
            raise ValueError(msg)

    def _assemble_markdown(self, results: list[RetrievalResult], include_metadata: bool) -> str:
        """Assemble in markdown format.

        Args:
            results: List of retrieval results
            include_metadata: Include metadata

        Returns:
            Markdown formatted context
        """
        sections = []
        current_length = 0

        for i, result in enumerate(results, 1):
            # Build section
            section_parts = [f"## Context {i} (Relevance: {result.score:.2f})"]

            if include_metadata and result.chunk.metadata:
                metadata_str = self._format_metadata(result.chunk.metadata)
                section_parts.append(f"**Metadata:** {metadata_str}")

            section_parts.append(result.chunk.content)
            section = "\n\n".join(section_parts)

            # Check length
            if current_length + len(section) > self.max_context_length:
                break

            sections.append(section)
            current_length += len(section)

        return "\n\n---\n\n".join(sections)

    def _assemble_plain(self, results: list[RetrievalResult], include_metadata: bool) -> str:
        """Assemble in plain text format.

        Args:
            results: List of retrieval results
            include_metadata: Include metadata

        Returns:
            Plain text formatted context
        """
        sections = []
        current_length = 0

        for i, result in enumerate(results, 1):
            section_parts = [f"Context {i}:"]

            if include_metadata and result.chunk.metadata:
                metadata_str = self._format_metadata(result.chunk.metadata)
                section_parts.append(f"Metadata: {metadata_str}")

            section_parts.append(result.chunk.content)
            section = "\n".join(section_parts)

            if current_length + len(section) > self.max_context_length:
                break

            sections.append(section)
            current_length += len(section)

        return "\n\n".join(sections)

    def _assemble_json(self, results: list[RetrievalResult], include_metadata: bool) -> str:
        """Assemble in JSON format.

        Args:
            results: List of retrieval results
            include_metadata: Include metadata

        Returns:
            JSON formatted context
        """
        import json

        contexts = []
        current_length = 0

        for result in results:
            context_item = {"content": result.chunk.content, "score": result.score, "rank": result.rank}

            if include_metadata and result.chunk.metadata:
                context_item["metadata"] = result.chunk.metadata

            json_str = json.dumps(context_item, ensure_ascii=False)

            if current_length + len(json_str) > self.max_context_length:
                break

            contexts.append(context_item)
            current_length += len(json_str)

        return json.dumps(contexts, ensure_ascii=False, indent=2)

    def _format_metadata(self, metadata: dict[str, Any]) -> str:
        """Format metadata as string.

        Args:
            metadata: Metadata dictionary

        Returns:
            Formatted metadata string
        """
        items = [f"{k}={v}" for k, v in metadata.items() if k not in ("chunk_index", "total_chunks")]
        return ", ".join(items)
