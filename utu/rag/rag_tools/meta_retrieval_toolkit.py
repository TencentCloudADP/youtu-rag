"""Meta Retrieval Toolkit."""

import json
import os
import logging
from datetime import datetime
from typing import Any, Optional

from pathlib import Path

from ...config import ToolkitConfig
from ...tools.base import register_tool
from ..embeddings.factory import EmbedderFactory
from ..rerankers.factory import RerankerFactory
from ..base import RetrievalResult, Chunk
from ..utils import date_to_time_range, strf_to_timestamp
from ...utils import SimplifiedAsyncOpenAI
from .parser_timeliness import TimeParser
from .base_toolkit import BaseRAGToolkit

logger = logging.getLogger(__name__)


class MetaRetrievalToolkit(BaseRAGToolkit):
    """Meta Retrieval Toolkit with separate embedding and reranking tools."""

    def __init__(self, config: ToolkitConfig = None):
        """Initialize the toolkit.

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

        self.embedder = self._init_embedder()

        self.reranker_config = self.config.config.get("reranker", {})
        self.reranker = self._init_reranker()

        self.time_parser = TimeParser()

        self.filters = None

        logger.info(
            f"MetaRetrievalToolkit initialized - "
            f"Content search (top_k={self.default_top_k}), "
            f"File search (top_k={self.file_search_top_k}), "
            f"Recall multiplier={self.recall_multiplier}"
            f"Embedder={self.embedder}, Reranker={self.reranker}"
        )

        self.query = None
        self.kb_id = None
        self.top_k = None
        self.query_intent = None
        self.query_time_ranges = None
        self.query_match_strategy = None
        self.valid_results = None
    
    def _reset_query_state(self):
        """Reset query-specific state variables for new query session.
        
        This method should be called at the beginning of each new query to ensure
        clean state and avoid interference from previous queries.
        """
        self.query = None
        self.kb_id = None
        self.top_k = None
        self.query_intent = None
        self.query_time_ranges = None
        self.query_match_strategy = None
        self.valid_results = None
        self.filters = None
        logger.info("Query state reset for new session")
    
    def _init_embedder(self):
        backend = self.embedding_config.get("backend", "openai")
        embedder_params = self._build_embedder_params()
        embedder = EmbedderFactory.create(backend=backend, **embedder_params)
        return embedder

    def _init_reranker(self):
        try:
            reranker = RerankerFactory.create(backend=self.reranker_config.get("backend", "jina"))
        except Exception as e:
            logger.error(f"Failed to initialize reranker: {e}")
            reranker = None
        return reranker

    def _build_metadata_filters(self, file_ids: Optional[list[str]] = None, metadata_filters: Optional[dict] = None) -> Optional[dict]:
        """Build ChromaDB metadata filters with flexible filtering strategies.

        Args:
            file_ids: List of file identifiers to filter by
            metadata_filters: Flexible metadata filters supporting multiple strategies:
                - Simple equality: {"author": "John", "category": "AI"}
                - Operator syntax: {"year": {"$gte": 2020}}
                - Time ranges (special handling): {
                    "time_ranges": {
                        "field": "创建时间",  # field name (default: "创建时间")
                        "ranges": [["2025-01-01", "2025-06-30"], ["2025-07-01", "2025-09-30"]]
                    }
                  }

        Returns:
            ChromaDB where clause or None

        Examples:
            ```
            # Simple filters
            metadata_filters = {"author": "张三", "category": "AI"}
            # Result: author == "张三" AND category == "AI"

            # Operator filters
            metadata_filters = {"year": {"$gte": 2020}}
            # Result: year >= 2020

            # Time range filters (single range)
            metadata_filters = {
                "time_ranges": {
                    "field": "创建时间",
                    "ranges": [["2025-01-01", "2025-06-30"]]
                }
            }
            # Result: 创建时间 >= "2025-01-01" AND 创建时间 <= "2025-06-30"

            # Time range filters (multiple ranges - union)
            metadata_filters = {
                "time_ranges": {
                    "field": "创建时间",
                    "ranges": [
                        ["2025-01-01", "2025-03-31"],  # Q1
                        ["2025-07-01", "2025-09-30"]   # Q3
                    ]
                }
            }
            # Result: (Q1 range) OR (Q3 range)

            # Combined filters
            metadata_filters = {
                "author": "张三",
                "time_ranges": {
                    "field": "创建时间",
                    "ranges": [["2025-01-01", "2025-06-30"]]
                }
            }
            # Result: author == "张三" AND (time range condition)
            ```
        """
        filters = []

        if file_ids:
            if len(file_ids) == 1:
                filters.append({"source": {"$eq": file_ids[0]}})
            else:
                filters.append({"$or": [{"source": {"$eq": fid}} for fid in file_ids]})

        if metadata_filters:
            for key, value in metadata_filters.items():
                if key == "time_ranges":  # Special handling for time_ranges
                    time_filter = self._build_time_range_filter(value)
                    if time_filter:
                        filters.append(time_filter)
                elif isinstance(value, dict) and any(k.startswith("$") for k in value.keys()):  # Operator syntax (e.g., {"$gte": 2020})
                    filters.append({key: value})
                else:  # Simple equality
                    filters.append({key: {"$eq": value}})

        if not filters:
            return None
        elif len(filters) == 1:
            return filters[0]
        else:
            return {"$and": filters}  # Combine filters with AND

    def _build_time_range_filter(self, time_range_configs: list) -> Optional[dict]:
        """Build time range filter from configuration.

        Args:
            time_range_config: Time range configuration list with keys:
                - "field": field name (example: "创建时间")
                - "ranges": list of [start_date, end_date] pairs

        Returns:
            ChromaDB filter dict or None

        Examples:
            ```
            # Single range
            config = {
                "field": "创建时间",
                "ranges": [["2025-01-01", "2025-06-30"]]
            }
            # Returns: {"$and": [{"创建时间": {"$gte": "2025-01-01"}}, {"创建时间": {"$lte": "2025-06-30"}}]}

            # Multiple ranges (union)
            config = {
                "field": "创建时间",
                "ranges": [["2025-01-01", "2025-03-31"], ["2025-07-01", "2025-09-30"]]
            }
            # Returns: {"$or": [range1_filter, range2_filter]}

            # Simplified format (auto-use default field)
            config = {
                "ranges": [["2025-01-01", "2025-06-30"]]
            }
            # Uses default field "创建时间"
            ```
        """
        time_range_filters = []

        if isinstance(time_range_configs, list):
            for time_range_config in time_range_configs:
                ranges = time_range_config.get("ranges", [])
                field = time_range_config.get("field")
                if not ranges:
                    continue

                if len(ranges) != 2:
                    logger.warning(f"Invalid time range format: {time_range_config}")
                    continue

                start_date, end_date = ranges

                range_filter = {
                    "$and": [
                        {f"{field}_min_stamp": {"$lte": end_date}},
                        {f"{field}_max_stamp": {"$gte": start_date}},
                    ]
                }

                time_range_filters.append(range_filter)

        else:
            logger.warning(f"Invalid time_range_configs format: {time_range_configs}")
            return None

        if not time_range_filters:
            return None
        elif len(time_range_filters) == 1:
            return time_range_filters[0]
        else:
            return {"$or": time_range_filters}  # Combine all time ranges with OR (union)

    def _normalize_date_for_filter(self, date_str: str, is_end_date: bool = False) -> str:
        """Normalize query date to match database format.
        
        Args:
            date_str: Date string (e.g., "2025-01-01")
            is_end_date: Whether it's an end date (for adding time boundary)
        
        Returns:
            Formatted date string (e.g., "2025/01/01 00:00:00")
        """
        from datetime import datetime
        
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        
        if is_end_date:
            # End date set to 23:59:59
            dt = dt.strftime("%Y/%m/%d 23:59:59")
        else:
            # Start date set to 00:00:00
            dt = dt.strftime("%Y/%m/%d 00:00:00")
        return int(datetime.strptime(dt, "%Y/%m/%d %H:%M:%S").timestamp())


    @register_tool
    async def query_analysis(self, query: str) -> dict:
        """Analyze user query to extract temporal information and time-related intent.

        This tool uses LLM to analyze the query and identify:
        1. Whether the query involves specific time points or time ranges
        2. Time orientation (past/present/future/latest)
        3. Standardized time tags (e.g., "2025-Q3", "2025-H1", "2025")
        4. Matching strategy (publish_date/key_timepoints/both)

        Args:
            query: User query text to analyze

        Returns:
            JSON string with temporal analysis results, e.g.
            ```
            {
                "is_temporal": true,                    # Whether query involves time-related information
                "time_orientation": "past",             # One of: past/present/future/range/latest/none
                "standard_tags": ["2025-H1"],          # Normalized time tags (YYYY, YYYY-MM, YYYY-QX, YYYY-HX, YYYY-MM-DD)
                "match_strategy": "both",               # One of: publish_date/key_timepoints/both
                "reasoning": "用户明确要求25年中报，对应2025年上半年。"  # Brief explanation of the analysis
            }
            ```

        Match Strategy:
            - "publish_date": Filter by document publish date (e.g., "上个月发布的报告")
            - "key_timepoints": Filter by time mentioned in content (e.g., "2026年预测", "25年规划")
            - "both": Consider both publish date and content time (default, most flexible)

        Example Queries:
            1. Query: "招金黄金25年中报披露了哪些金矿信息？"
               Result: {
                   "is_temporal": true,
                   "time_orientation": "past",
                   "standard_tags": ["2025-H1"],
                   "match_strategy": "both",
                   "reasoning": "用户明确要求25年中报，对应2025年上半年。"
               }

            2. Query: "字节最新的2026年资本支出预测"
               Result: {
                   "is_temporal": true,
                   "time_orientation": "future",
                   "standard_tags": ["2026"],
                   "match_strategy": "key_timepoints",
                   "reasoning": "2026年为未来时间点，通常存在于研报的内容预测中，而非发布日期。"
               }
        """
        # Reset state variables for new query session
        self._reset_query_state()
        self.query = query

        res = await self.time_parser.parse(query)

        self.query_intent = res

        if res['is_temporal']:

            match_strategy = res['match_strategy']
            if match_strategy == "both":
                match_strategy = ["publish_date", "key_timepoints"]
            else:
                match_strategy = [match_strategy]

            standard_tags = res['standard_tags']
            time_ranges = []
            if standard_tags:
                for tag in standard_tags:
                    start_date, end_date = date_to_time_range(tag)
                    if start_date and end_date:
                        time_ranges.append([start_date, end_date])
            
            self.query_time_ranges = time_ranges
            self.query_match_strategy = match_strategy
            metadata_filters = {"time_ranges": []}
            for strategy in match_strategy:
                for time_range in time_ranges:
                    metadata_filters["time_ranges"].append({
                        "field": strategy,
                        "ranges": time_range
                    })

            self.filters = self._build_metadata_filters(metadata_filters=metadata_filters)
        else:
            self.filters = None

        return json.dumps(res, ensure_ascii=False, indent=4)

    @register_tool
    async def expand_filter_scope(self) -> str:
        """Expand filter scope based on standard tags.

        The tool expands the filter scope based on original time ranges parsed from the query.
            
        Returns:
            Original time ranges and expanded time ranges
        """
        if not self.query_time_ranges or not self.query_match_strategy:
            return "No time ranges found, you need to use **query_analysis** tool first."

        if self.query_intent['time_orientation'] in ["past", "present", "latest", "none"]:
            method = "expand_past"
        elif self.query_intent['time_orientation'] == "future":
            method = "expand_future"
        elif self.query_intent['time_orientation'] == "range":
            method = "expand_range"

        new_time_ranges = []
        for time_range in self.query_time_ranges:
            start_date, end_date = time_range
            window = end_date - start_date
            if method == "expand_past":
                new_time_ranges.append([start_date - 2 * window, end_date])
            elif method == "expand_future":
                new_time_ranges.append([start_date, end_date + 2 * window])
            elif method == "expand_range":
                new_time_ranges.append([start_date - window, end_date + window])

        metadata_filters = {"time_ranges": []}
        for strategy in self.query_match_strategy:
            for time_range in new_time_ranges:
                metadata_filters["time_ranges"].append({
                    "field": strategy,
                    "ranges": time_range
                })

        self.filters = self._build_metadata_filters(metadata_filters=metadata_filters)

        tool_response = f"Raw time ranges:\n"
        for id_, time_range in enumerate(self.query_time_ranges, 1):
            start_date, end_date = time_range
            start_date_str = datetime.fromtimestamp(start_date).strftime("%Y-%m-%d")
            end_date_str = datetime.fromtimestamp(end_date).strftime("%Y-%m-%d")
            tool_response += f"[{id_}]. {start_date_str} -- {end_date_str}\n"
        tool_response += f"\nExpanded time ranges:\n"
        for id_, time_range in enumerate(new_time_ranges, 1):
            start_date, end_date = time_range
            start_date_str = datetime.fromtimestamp(start_date).strftime("%Y-%m-%d")
            end_date_str = datetime.fromtimestamp(end_date).strftime("%Y-%m-%d")
            tool_response += f"[{id_}]. {start_date_str} -- {end_date_str}\n"
        
        tool_response += f"\nFilters already expanded and updated!"

        self.query_time_ranges = new_time_ranges
        return tool_response
        
        
    @register_tool
    async def kb_embedding_search(
        self,
        kb_id: int,
        query: str,
        top_k: Optional[int] = None,
        auto_rerank: bool = True,
    ) -> str:
        """Search knowledge base using vector embedding similarity with optional automatic reranking.

        This tool performs semantic search in ChromaDB:
        1. Generates query embedding vector
        2. Searches for similar chunks in the specified knowledge base
        3. Filters by file IDs and/or custom metadata if provided
        4. (Optional) Automatically reranks results using reranker model
        5. Returns top_k final results with relevance scores

        Args:
            kb_id: Knowledge base ID (required)
            query: Search query text
            top_k: Number of final results to return (default from config, typically 3-5)
                   Note: When auto_rerank=True, internally retrieves (top_k * recall_multiplier) candidates
                   for better recall, then reranks and returns top_k results
            auto_rerank: Whether to automatically rerank results (default True, recommended for better quality)
                        - True: Retrieve more candidates and rerank to get top_k results (higher quality)
                        - False: Directly return top_k embedding results (faster, lower quality)

        Returns:
            Tool Response about the number of retrieved chunks

        Example:
            ```
            # Search with auto-rerank (default, recommended) - returns 3 results
            kb_embedding_search(kb_id=1, query="What is machine learning?")

            # Search without auto-rerank - directly return 10 embedding results
            kb_embedding_search(kb_id=1, query="neural networks", auto_rerank=False, top_k=10)
            ```
        """
        tool_response = ""

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

            filters = self.filters
            if filters:
                logger.info(f"Applying filters: {filters}")

            retriever = await self._create_retriever(
                kb_id=kb_id,
                top_k=retrieval_top_k,
                embedder=self.embedder,
                persist_directory=self.vector_store_base_config.get("persist_directory", "./data/chroma"),
            )

            results = await retriever.retrieve(query=query, filters=filters)

            result_data = {
                "kb_id": kb_id,
                "query": query,
                "total_results": 0,
                "top_k": top_k,
                "filters_applied": {
                    "metadata_filters": filters if filters else None,
                },
                "results": [],
            }

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
                    reranked_results = [r for r in reranked_results if r.score > self.reranker_config.get("threshold", 0.5)]

                    tool_response += f"Reranked from {len(results)} candidates to {len(reranked_results)} chunks\n"
                    counts = {}
                    for doc_id in [r.chunk.document_id for r in reranked_results]:
                        counts[doc_id] = counts.get(doc_id, 0) + 1
                    for doc_id, count in counts.items():
                        tool_response += f"  - {doc_id}: {count} chunks\n"
                    tool_response += f"Total: {len(reranked_results)} chunks of {len(counts)} documents\n"

                    result_data["reranked"] = True
                    result_data["total_results"] = len(reranked_results)

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

                    if self.valid_results:
                        exist_results_id = [r.chunk.id for r in self.valid_results]
                        for result in reranked_results:
                            if result.chunk.id in exist_results_id:
                                continue
                            self.valid_results.append(result)
                    else:
                        self.valid_results = reranked_results

                    tool_response += f"Total accumulated {len(self.valid_results)} valid chunks\n"

                    logger.info(f"✅ Auto-reranked from {len(results)} candidates to {len(reranked_results)} results")

                except Exception as rerank_error:
                    logger.warning(f"Auto-rerank failed: {str(rerank_error)}, returning embedding results")
                    result_data["reranked"] = False
                    result_data["rerank_error"] = str(rerank_error)
            else:
                result_data["reranked"] = False
                results = [r for r in results if r.score > self.embedding_config.get("threshold", 0.0)]
                logger.info(f"✅ Returning {len(results)} embedding results without reranking")
                result_data["total_results"] = len(results)
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

                tool_response = f"Retrieved {len(results)} chunks from embedding search\n"
                counts = {}
                for doc_id in [r.chunk.document_id for r in results]:
                    counts[doc_id] = counts.get(doc_id, 0) + 1
                for doc_id, count in counts.items():
                    tool_response += f"  - {doc_id}: {count} chunks\n"
                tool_response += f"Total: {len(results)} chunks of {len(counts)} documents\n"

                if self.valid_results:
                    exist_results_id = [r.chunk.id for r in self.valid_results]
                    for result in results:
                        if result.chunk.id in exist_results_id:
                            continue
                        self.valid_results.append(result)
                else:
                    self.valid_results = results

                tool_response += f"Total accumulated {len(self.valid_results)} valid chunks\n"

            return tool_response

        except ValueError as e:
            error_msg = f"KB search error: {str(e)}"
            logger.error(error_msg)
            return f"{error_msg} when searching from Knowledge Base"

        except Exception as e:
            error_msg = f"KB search failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"{error_msg} when searching from Knowledge Base"

    @register_tool
    async def merge_retrieval_results(self) -> str:
        """Merge retrieval results from multiple sources."""
        results = self.valid_results

        # Deduplicate with chunk id
        results = list({r.chunk.id: r for r in results}.values())

        results = sorted(results, key=lambda x: x.score, reverse=True)

        result_data = {
            "kb_id": self.kb_id,
            "query": self.query,
            "total_results": len(results),
            "top_k": self.top_k,
            "filters_applied": {
                "metadata_filters": self.filters if self.filters else None,
            },
            "results": [],
        }

        for result in results:
            result_data["results"].append(
                {
                    "rank": result.rank,
                    "similarity_score": round(result.score, 4),  # Embedding similarity score
                    "content": result.chunk.content,
                    "chunk_id": result.chunk.id,
                    "document_id": result.chunk.document_id,
                    "source": result.chunk.metadata.get("source", ""),
                    "metadata": result.chunk.metadata,
                }
            )
        
        return json.dumps(result_data, ensure_ascii=False, indent=2)

async def main():

    config_file = Path("configs/rag/rag_tools/meta_retrieval.yaml")
    from utu.config import ConfigLoader
    config = ConfigLoader.load_agent_config("ragref/meta_retrieval/meta_retrieval")

    tool = MetaRetrievalToolkit(config=config.toolkits['meta_retrieval'])
    
    query = "请介绍一些2024年上架的关于Apple PRO头显设备AI开发的课程"
    query = "2022年第三季度小米盈利多少"
    # query = "2025年下半年小米盈利多少"
    res = await tool.query_analysis(query=query)
    # results = await tool.kb_embedding_search(kb_id=2, query=query, top_k=10)
    res = await tool.expand_filter_scope()
    pass


if __name__ == "__main__":

    import asyncio
    asyncio.run(main())