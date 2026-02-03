"""Knowledge Base Search Toolkit (kb_search) - Embedding search and reranking as separate tools."""

import json
import logging
import os
from typing import Any, Optional

from ...config import ToolkitConfig
from ...tools.base import register_tool
from ..rerankers.factory import RerankerFactory
from ..base import RetrievalResult, Chunk
from .base_toolkit import BaseRAGToolkit

logger = logging.getLogger(__name__)


class KBSearchToolkit(BaseRAGToolkit):
    """Knowledge Base Search Toolkit with separate embedding and reranking tools.

    Tools:
        - kb_file_search: File-level search based on document summaries
        - kb_embedding_search: Vector similarity search in knowledge base (chunk-level)
        - kb_rerank: Rerank search candidates using reranker model
    """

    def __init__(self, config: ToolkitConfig = None):
        """Initialize KB Search toolkit.

        Args:
            config: Toolkit configuration with default retrieval settings
        """
        super().__init__(config)

        self.default_top_k = self.config.config.get("top_k", 3)
        self.file_search_top_k = self.config.config.get("top_k", 15)

        # Recall multiplier for two-stage retrieval (only used when auto_rerank=True):
        # - Stage 1 (Recall): Retrieve (top_k × recall_multiplier) candidates from vector DB
        # - Stage 2 (Rerank): Use reranker to select top_k most relevant results
        # Example: top_k=3, recall_multiplier=3 → retrieve 9 candidates → rerank to top 3
        # Higher multiplier = better recall (more candidates to choose from) but higher costs
        # When auto_rerank=False, this parameter is ignored (direct retrieval of top_k)
        self.recall_multiplier = self.config.config.get("recall_multiplier", 3)

        self.reranker_config = self.config.config.get("reranker", {})
        self.reranker = self._init_reranker()

        logger.info(
            f"KBSearchToolkit initialized - "
            f"Content search (top_k={self.default_top_k}), "
            f"File search (top_k={self.file_search_top_k}), "
            f"Recall multiplier={self.recall_multiplier}"
        )
    
    def _init_reranker(self):
        try:
            reranker = RerankerFactory.create(backend=self.reranker_config.get("backend", "jina"))
        except Exception as e:
            logger.error(f"Failed to initialize reranker: {e}")
            reranker = None
        return reranker

    def _build_metadata_filters(
        self, metadata_filters: Optional[dict] = None
    ) -> Optional[dict]:
        """Build ChromaDB metadata filters.

        Args:
            metadata_filters: Metadata filters dict. Supports:
                - Simple equality: {"key": "value"} → {"key": {"$eq": "value"}}
                - ChromaDB operators: {"key": {"$op": "value"}} (e.g., $eq, $ne, $in, $gte, $regex)
                - Multiple conditions are combined with $and
                - Special case for source filtering: {"source": {"$in": ["file1.pdf", "file2.pdf"]}}

        Returns:
            ChromaDB where clause or None
        """
        if not metadata_filters:
            return None

        filters = []

        for key, value in metadata_filters.items():
            if isinstance(value, dict) and any(k.startswith("$") for k in value.keys()):
                # Already has operator (e.g., {"source": {"$in": ["file1.pdf"]}})
                filters.append({key: value})
            else:
                # Simple equality (e.g., {"source": "file1.pdf"})
                filters.append({key: {"$eq": value}})

        if not filters:
            return None
        elif len(filters) == 1:
            return filters[0]
        else:
            return {"$and": filters}  # Combine multiple filters with $and

    @register_tool
    async def kb_embedding_search(
        self,
        kb_id: int,
        query: str,
        top_k: Optional[int] = None,
        metadata_filters: Optional[dict] = None,
        auto_rerank: bool = True,
    ) -> str:
        """Search knowledge base using vector embedding similarity with optional automatic reranking.

        This tool performs semantic search in ChromaDB:
        1. Generates query embedding vector
        2. Searches for similar chunks in the specified knowledge base
        3. Filters by metadata if provided (including file names via 'source' field)
        4. (Optional) Automatically reranks results using reranker model
        5. Returns top_k final results with relevance scores

        Args:
            kb_id: Knowledge base ID (required)
            query: Search query text
            top_k: Number of final results to return (default from config, typically 3-5)
                   Note: When auto_rerank=True, internally retrieves (top_k * recall_multiplier) candidates
                   for better recall, then reranks and returns top_k results
            metadata_filters: Optional dict of metadata filters. Supports:
                - File filtering: {"source": {"$in": ["file1.pdf", "file2.pdf"]}} for multiple files
                                 {"source": "file1.pdf"} for single file
                - Custom metadata: {"author": "John", "year": {"$gte": 2020}}
                - ChromaDB operators: $eq, $ne, $in, $nin, $gt, $gte, $lt, $lte, $regex
                - Multiple conditions are combined with $and
            auto_rerank: Whether to automatically rerank results (default True, recommended for better quality)
                        - True: Retrieve more candidates and rerank to get top_k results (higher quality)
                        - False: Directly return top_k embedding results (faster, lower quality)

        Returns:
            JSON string with search results, e.g.
            ```
            {
                "kb_id": 1,
                "query": "...",
                "total_results": 3,
                "top_k": 3,
                "reranked": true,
                "filters_applied": {...},
                "results": [
                    {
                        "rank": 1,
                        "rerank_score": 0.95,       # Relevance score from reranker (only if reranked=true)
                        "embedding_score": 0.78,    # Original embedding similarity score
                        "content": "chunk text...",
                        "chunk_id": "doc1_chunk_0",
                        "document_id": "doc1",
                        "source": "doc1.pdf",
                        "metadata": {"author": "...", ...}
                    },
                    ...
                ]
            }
            ```

            NOTE When reranked=false, only "embedding_score" is present.
                 When reranked=true, both "rerank_score" and "embedding_score" are present.

        Example:
            ```
            # Basic search with auto-rerank (default, recommended)
            kb_embedding_search(kb_id=1, query="What is machine learning?")

            # Search without auto-rerank - directly return 10 embedding results
            kb_embedding_search(kb_id=1, query="neural networks", auto_rerank=False, top_k=10)

            # Filter by specific files (single file)
            kb_embedding_search(
                kb_id=1,
                query="neural networks",
                metadata_filters={"source": "ml_book.pdf"}
            )

            # Filter by multiple files
            kb_embedding_search(
                kb_id=1,
                query="neural networks",
                top_k=5,
                metadata_filters={"source": {"$in": ["ml_book.pdf", "dl_paper.pdf"]}}
            )

            # Combine file filtering with custom metadata
            kb_embedding_search(
                kb_id=1,
                query="recent research",
                top_k=3,
                metadata_filters={
                    "source": {"$in": ["paper1.pdf", "paper2.pdf"]},
                    "year": {"$gte": 2023},
                    "category": "AI"
                }
            )
            ```
        """
        if auto_rerank and not self.reranker:
            auto_rerank = False

        try:
            top_k = top_k if top_k is not None else self.default_top_k

            # Two-stage retrieval strategy:
            # - If auto_rerank=True: Retrieve (top_k × recall_multiplier) candidates,
            #   then reranker selects top_k most relevant ones
            # - If auto_rerank=False: Directly retrieve top_k results (no reranking)
            retrieval_top_k = top_k * self.recall_multiplier if auto_rerank else top_k

            logger.info(
                f"[kb_embedding_search] KB={kb_id}, query='{query[:50]}...', "
                f"top_k={top_k}, auto_rerank={auto_rerank}, retrieval_top_k={retrieval_top_k}"
            )

            filters = self._build_metadata_filters(metadata_filters)
            if filters:
                logger.info(f"Applying filters: {filters}")

            retriever = await self._create_retriever(kb_id, retrieval_top_k)

            results = await retriever.retrieve(query=query, filters=filters)

            result_data = {
                "kb_id": kb_id,
                "query": query,
                "total_results": len(results),
                "top_k": top_k,
                "filters_applied": metadata_filters if metadata_filters else None,
                "results": [],
            }

            for result in results:
                result_data["results"].append(
                    {
                        "rank": result.rank,
                        "embedding_score": round(result.score, 4),  # Embedding similarity score
                        "content": result.chunk.content,
                        "chunk_id": result.chunk.id,
                        "document_id": result.chunk.document_id,
                        "source": result.chunk.metadata.get("source", ""),
                        "metadata": result.chunk.metadata,
                    }
                )

            logger.info(f"✓ Found {len(results)} results from embedding search")

            # Stage 2: Auto-rerank if enabled (precision optimization)
            # This is the second stage of two-stage retrieval:
            # - We already retrieved (top_k × recall_multiplier) candidates in Stage 1
            # - Now reranker semantically evaluates all candidates and selects top_k best ones
            if auto_rerank and results:
                logger.info(f"🔄 Auto-reranking {len(results)} candidates to top {top_k}...")

                try:
                    # Save original embedding scores before reranking
                    # Create mapping: chunk_id -> embedding_score
                    embedding_scores = {result.chunk.id: result.score for result in results}

                    reranked_results = await self.reranker.rerank(query=query, results=results, top_k=top_k)

                    result_data["reranked"] = True
                    result_data["total_results"] = len(reranked_results)
                    result_data["results"] = []

                    for result in reranked_results:
                        chunk_id = result.chunk.id
                        result_data["results"].append(
                            {
                                "rank": result.rank,
                                "rerank_score": round(result.score, 4),  # Rerank relevance score
                                "embedding_score": round(embedding_scores.get(chunk_id, 0.0), 4),  # Original embedding score
                                "content": result.chunk.content,
                                "chunk_id": chunk_id,
                                "document_id": result.chunk.document_id,
                                "source": result.chunk.metadata.get("source", ""),
                                "metadata": result.chunk.metadata,
                            }
                        )

                    logger.info(f"✅ Auto-reranked from {len(results)} candidates to {len(reranked_results)} results")

                except Exception as rerank_error:
                    logger.warning(f"Auto-rerank failed: {str(rerank_error)}, returning embedding results")
                    result_data["reranked"] = False
                    result_data["rerank_error"] = str(rerank_error)
            else:
                result_data["reranked"] = False

            return json.dumps(result_data, ensure_ascii=False, indent=2)

        except ValueError as e:
            error_msg = f"KB search error: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"error": str(e), "kb_id": kb_id, "query": query}, ensure_ascii=False)

        except Exception as e:
            error_msg = f"KB search failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return json.dumps(
                {"error": str(e), "kb_id": kb_id, "query": query}, ensure_ascii=False
            )

    @register_tool
    async def kb_rerank(
        self,
        query: str,
        candidates: str,
        top_k: Optional[int] = None,
        model: Optional[str] = None,
        boost_metadata: Optional[dict] = None,
    ) -> str:
        """Rerank search candidates using a reranker model for better relevance.

        This tool performs semantic reranking:
        1. Parses candidate results from embedding search (JSON format)
        2. Sends query + candidates to reranker API (e.g., Jina AI)
        3. Returns top-k results sorted by relevance score
        4. Optionally applies metadata-based boosting (future feature)

        Args:
            query: Original search query
            candidates: JSON string from kb_embedding_search containing candidate results
            top_k: Number of top results to return after reranking (default from config, typically 3-5)
            model: Reranker model name (default from config, e.g., "jina-reranker-v2-base-multilingual")
            boost_metadata: Optional metadata-based boosting strategy (reserved for future enhancement)
                           Example: {"source_type": {"pdf": 1.2, "markdown": 1.0}, "recency_boost": True}

        Returns:
            JSON string with reranked results (same format as input but with updated scores and ranks)

        Example:
            ```
            # Basic reranking
            kb_rerank(query="machine learning", candidates=embedding_results)

            # Rerank with custom top_k
            kb_rerank(query="neural networks", candidates=embedding_results, top_k=5)

            # Future: metadata boosting
            kb_rerank(
                query="recent AI research",
                candidates=embedding_results,
                boost_metadata={"year": {"2024": 1.5, "2023": 1.2}}
            )
            ```
        """
        try:
            top_k = top_k if top_k is not None else self.default_top_k
            model = model or self.reranker_config.get(
                "model", "jina-reranker-v2-base-multilingual"
            )

            logger.info(f"[kb_rerank] query='{query[:50]}...', top_k={top_k}, model={model}")

            try:
                candidates_data = json.loads(candidates)
            except json.JSONDecodeError as e:
                return json.dumps(
                    {"error": f"Invalid JSON format in candidates: {str(e)}"}, ensure_ascii=False
                )

            if "results" not in candidates_data:
                return json.dumps(
                    {"error": "Missing 'results' field in candidates JSON"}, ensure_ascii=False
                )

            candidate_results = candidates_data["results"]
            if not candidate_results:
                logger.info("No candidates to rerank")
                return candidates  # Return original if empty

            if len(candidate_results) == 1:
                logger.info("Only 1 candidate, skipping rerank")
                return candidates  # No need to rerank single result

            logger.info(f"Reranking {len(candidate_results)} candidates...")

            embedding_scores = {}
            retrieval_results = []
            for item in candidate_results:
                chunk_id = item["chunk_id"]

                # Extract score: prioritize embedding_score, fallback to score for backward compatibility
                score = item.get("embedding_score", item.get("score", 0.0))
                embedding_scores[chunk_id] = score

                chunk = Chunk(
                    id=chunk_id,
                    document_id=item["document_id"],
                    content=item["content"],
                    chunk_index=0,  # Not used in reranking
                    metadata=item.get("metadata", {}),
                )
                result = RetrievalResult(chunk=chunk, score=score, rank=item["rank"])
                retrieval_results.append(result)

            reranked_results = await self.reranker.rerank(
                query=query, results=retrieval_results, top_n=top_k
            )

            # Apply metadata boosting if provided (future feature)
            if boost_metadata:
                logger.warning(
                    "boost_metadata is not yet implemented, ignoring boosting strategy"
                )
                # TODO: Implement metadata-based score boosting
                # for result in reranked_results:
                #     boost_factor = calculate_boost(result.chunk.metadata, boost_metadata)
                #     result.score *= boost_factor

            reranked_data = {
                "kb_id": candidates_data.get("kb_id"),
                "query": query,
                "total_results": len(reranked_results),
                "original_count": len(candidate_results),
                "rerank_top_k": top_k,
                "rerank_model": model,
                "reranked": True,
                "results": [],
            }

            for result in reranked_results:
                chunk_id = result.chunk.id
                reranked_data["results"].append(
                    {
                        "rank": result.rank,
                        "rerank_score": round(result.score, 4),  # Rerank relevance score
                        "embedding_score": round(embedding_scores.get(chunk_id, 0.0), 4),  # Original embedding score
                        "content": result.chunk.content,
                        "chunk_id": chunk_id,
                        "document_id": result.chunk.document_id,
                        "source": result.chunk.metadata.get("source", ""),
                        "metadata": result.chunk.metadata,
                    }
                )

            logger.info(f"✓ Reranked from {len(candidate_results)} to {len(reranked_results)} results")

            return json.dumps(reranked_data, ensure_ascii=False, indent=2)

        except Exception as e:
            error_msg = f"Reranking failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return json.dumps({"error": str(e), "query": query}, ensure_ascii=False)

    @register_tool
    async def kb_file_search(
        self,
        kb_id: int,
        query: str,
        top_k: Optional[int] = None,
        metadata_filters: Optional[dict] = None,
        auto_rerank: bool = True,
        include_summary: bool = True,
    ) -> str:
        """File-level semantic search based on document summaries (index_summary).

        This tool performs file discovery using document summaries:
        1. Searches index_summary vectors (filename + summary concatenation)
        2. Returns a list of relevant files with their summaries
        3. Useful as first-stage retrieval before detailed chunk-level search
        4. Automatically deduplicates by file and aggregates metadata

        Args:
            kb_id: Knowledge base ID (required)
            query: Search query (matched against file summaries)
            top_k: Number of files to return (default 15, file search typically needs more results)
            metadata_filters: Optional metadata filters (e.g., {"authors": "John", "publish_date": {"$gte": "2024-01-01"}})
            auto_rerank: Whether to automatically rerank results (default True)
            include_summary: Whether to include full summary text in results (default True)

        Returns:
            JSON string with file-level results, e.g.
            ```
            {
                "kb_id": 1,
                "query": "machine learning",
                "total_files": 8,
                "search_type": "file_level",
                "index_type": "index_summary",
                "reranked": true,
                "files": [
                    {
                        "rank": 1,
                        "file_name": "ML_Book.pdf",
                        "relevance_score": 0.92,
                        "summary": "This book introduces basic concepts of machine learning...",
                        "metadata": {
                            "authors": "Li Ming",
                            "publish_date": "2024-01-15",
                            "file_type": "pdf",
                            ...
                        }
                    },
                    ...
                ]
            }
            ```

        Example:
            ```
            # Basic file search
            kb_file_search(kb_id=1, query="machine learning basics")

            # Search files by specific author
            kb_file_search(
                kb_id=1,
                query="deep learning",
                metadata_filters={"authors": "Andrew Ng"}
            )

            # Two-stage retrieval: find files first, then search content
            files_result = kb_file_search(kb_id=1, query="natural language processing")
            files = [f["file_name"] for f in json.loads(files_result)["files"][:5]]
            content_result = kb_embedding_search(
                kb_id=1,
                query="transformer architecture",
                metadata_filters={"source": {"$in": files}}
            )
            ```
        """
        try:
            top_k = top_k if top_k is not None else self.file_search_top_k
            retrieval_top_k = top_k * self.recall_multiplier if auto_rerank else top_k
            logger.info(
                f"[kb_file_search] KB={kb_id}, query='{query[:50]}...', "
                f"top_k={top_k}, auto_rerank={auto_rerank}, retrieval_top_k={retrieval_top_k}"
            )
            base_filters = self._build_metadata_filters(metadata_filters)

            index_type_filter = {"index_type": {"$eq": "index_summary"}}

            if base_filters:
                filters = {"$and": [base_filters, index_type_filter]}
            else:
                filters = index_type_filter

            logger.info(f"Applying filters (including index_type=index_summary): {filters}")

            retriever = await self._create_retriever(kb_id, retrieval_top_k)
            results = await retriever.retrieve(query=query, filters=filters)

            logger.info(f"✓ Retrieved {len(results)} summary vectors")

            files_dict = {}
            for result in results:
                file_name = result.chunk.metadata.get("source", result.chunk.document_id)
                if file_name in files_dict:  # Deduplicate by file
                    continue
                file_entry = {
                    "file_name": file_name,
                    "relevance_score": result.score,
                    "chunk_id": result.chunk.id, 
                    "content": result.chunk.content,
                }

                if include_summary:
                    file_entry["summary"] = result.chunk.metadata.get("summary", "")

                excluded_fields = {"index_type", "chunk_index", "_derived_files_etags"}
                file_entry["metadata"] = {
                    k: v
                    for k, v in result.chunk.metadata.items()
                    if k not in excluded_fields
                }

                files_dict[file_name] = file_entry

            files_list = sorted(files_dict.values(), key=lambda x: x["relevance_score"], reverse=True)

            result_data = {
                "kb_id": kb_id,
                "query": query,
                "total_files": len(files_list),
                "search_type": "file_level",
                "index_type": "index_summary",
                "top_k": top_k,
                "filters_applied": metadata_filters if metadata_filters else None,
                "reranked": False,
                "files": [],
            }

            if auto_rerank and len(files_list) > 1:
                logger.info(f"🔄 Auto-reranking {len(files_list)} files to top {top_k}...")

                try:
                    retrieval_results = []
                    embedding_scores = {}

                    for file_entry in files_list:
                        chunk_id = file_entry["chunk_id"]
                        score = file_entry["relevance_score"]
                        embedding_scores[chunk_id] = score

                        chunk = Chunk(
                            id=chunk_id,
                            document_id=file_entry["file_name"],
                            content=file_entry["content"], 
                            chunk_index=-1,
                            metadata=file_entry["metadata"],
                        )
                        result = RetrievalResult(chunk=chunk, score=score, rank=0)
                        retrieval_results.append(result)

                    reranker = RerankerFactory.create(backend=self.reranker_config.get("backend", "jina"))

                    reranked_results = await reranker.rerank(
                        query=query, results=retrieval_results, top_k=top_k
                    )

                    result_data["reranked"] = True
                    result_data["total_files"] = len(reranked_results)

                    for idx, result in enumerate(reranked_results, 1):
                        file_entry = {
                            "rank": idx,
                            "file_name": result.chunk.document_id,
                            "rerank_score": round(result.score, 4),
                            "embedding_score": round(embedding_scores.get(result.chunk.id, 0.0), 4),
                            "metadata": result.chunk.metadata,
                        }

                        if include_summary:
                            file_entry["summary"] = result.chunk.metadata.get("summary", "")
                        result_data["files"].append(file_entry)

                    logger.info(f"✅ Auto-reranked from {len(files_list)} files to {len(reranked_results)} results")

                except Exception as rerank_error:
                    logger.warning(f"Auto-rerank failed: {str(rerank_error)}, returning embedding results")
                    result_data["reranked"] = False
                    result_data["rerank_error"] = str(rerank_error)

                    # Fallback to embedding results
                    for idx, file_entry in enumerate(files_list[:top_k], 1):
                        result_entry = {
                            "rank": idx,
                            "file_name": file_entry["file_name"],
                            "embedding_score": round(file_entry["relevance_score"], 4),
                            "metadata": file_entry["metadata"],
                        }
                        if include_summary:
                            result_entry["summary"] = file_entry.get("summary", "")

                        result_data["files"].append(result_entry)
            else:
                # No reranking - return top_k embedding results
                for idx, file_entry in enumerate(files_list[:top_k], 1):
                    result_entry = {
                        "rank": idx,
                        "file_name": file_entry["file_name"],
                        "embedding_score": round(file_entry["relevance_score"], 4),
                        "metadata": file_entry["metadata"],
                    }
                    if include_summary:
                        result_entry["summary"] = file_entry.get("summary", "")

                    result_data["files"].append(result_entry)

            logger.info(
                f"✓ File search completed: {len(result_data['files'])} files "
                f"(reranked={result_data['reranked']})"
            )

            return json.dumps(result_data, ensure_ascii=False, indent=2)

        except ValueError as e:
            error_msg = f"File search error: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"error": str(e), "kb_id": kb_id, "query": query}, ensure_ascii=False)

        except Exception as e:
            error_msg = f"File search failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return json.dumps(
                {"error": str(e), "kb_id": kb_id, "query": query}, ensure_ascii=False
            )
