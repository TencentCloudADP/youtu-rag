"""Memory toolkit with integrated ChromaDB vector storage and skill learning.

Provides:
1. Vector-based memory storage and semantic retrieval
2. Support for episodic, procedural, semantic, and working memory
3. Skill extraction from trajectory for procedural learning
4. Collections organized by user_id and memory_type
5. Persistent storage capability

Reference: @ii-agent/src/ii_agent/tools/memory/
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from ..config import ToolkitConfig
from ..utils import get_logger
from .base import AsyncBaseToolkit, register_tool

if TYPE_CHECKING:
    from ..db import TrajectoryModel

# Optional ChromaDB import
try:
    import chromadb
    from chromadb.config import Settings

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    chromadb = None  # type: ignore
    Settings = None  # type: ignore

# Optional LLM import for skill extraction
try:
    from agents import Agent, Runner

    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False
    Agent = None  # type: ignore
    Runner = None  # type: ignore

from utu.rag.storage.implementations.memory_store import EmbeddingService

logger = logging.getLogger(__name__)


# ============== Simple Memory Toolkits ==============


class SimpleMemoryToolkit(AsyncBaseToolkit):
    """String-based memory tool for storing and modifying persistent text.

    This tool maintains a single in-memory string that can be read,
    replaced, or selectively edited using string replacement. It provides safety
    warnings when overwriting content or when edit operations would affect
    multiple occurrences.
    """

    def __init__(self, config: ToolkitConfig = None) -> None:
        super().__init__(config)
        self.full_memory = ""

    def _read_memory(self) -> str:
        """Read the current memory contents."""
        return self.full_memory

    def _write_memory(self, content: str) -> str:
        """Replace the entire memory with new content."""
        if self.full_memory:
            previous = self.full_memory
            self.full_memory = content
            return (
                f"Warning: Overwriting existing content. Previous content was:\n{previous}\n\n"
                "Memory has been updated successfully."
            )
        self.full_memory = content
        return "Memory updated successfully."

    def _edit_memory(self, old_string: str, new_string: str) -> str:
        """Replace occurrences of old string with new string."""
        if old_string not in self.full_memory:
            return f"Error: '{old_string}' not found in memory."

        old_memory = self.full_memory
        count = old_memory.count(old_string)

        if count > 1:
            return (
                f"Warning: Found {count} occurrences of '{old_string}'. "
                "Please confirm which occurrence to replace or use more specific context."
            )

        self.full_memory = self.full_memory.replace(old_string, new_string)
        return "Edited memory: 1 occurrence replaced."

    @register_tool
    async def simple_memory(
        self,
        action: Literal["read", "write", "edit"],
        content: str = "",
        old_string: str = "",
        new_string: str = "",
    ) -> str:
        """Tool for managing persistent text memory with read, write and edit operations.

        MEMORY STORAGE GUIDANCE:
        Store information that needs to persist across agent interactions, including:
        - User context: Requirements, goals, preferences, and clarifications
        - Task state: Completed tasks, pending items, current progress
        - Code context: File paths, function signatures, data structures, dependencies
        - Research findings: Key facts, sources, URLs, and reference materials
        - Configuration: Settings, parameters, and environment details
        - Cross-session continuity: Information needed for future interactions

        OPERATIONS:
        - Read: Retrieves full memory contents as a string
        - Write: Replaces entire memory (warns when overwriting existing data)
        - Edit: Performs targeted string replacement (warns on multiple matches)

        Use structured formats (JSON, YAML, or clear sections) for complex data.
        Prioritize information that would be expensive to regenerate or re-research.

        Args:
            action: The action to perform on the memory.
            content: The content to write to the memory. Defaults to "".
            old_string: The string to replace in the memory. Defaults to "".
            new_string: The string to replace the old string with. Defaults to "".
        """
        if action == "read":
            result = self._read_memory()
        elif action == "write":
            result = self._write_memory(content)
        elif action == "edit":
            result = self._edit_memory(old_string, new_string)
        else:
            result = f"Error: Unknown action '{action}'. Valid actions are read, write, edit."
        return result


class CompactifyMemoryToolkit(AsyncBaseToolkit):
    """Memory compactification tool that works with any context manager type.

    Applies the context manager's truncation strategy to compress the conversation history.
    This tool adapts to different context management approaches (summarization, simple truncation, etc.).
    """

    async def compactify_memory(self) -> str:
        """Compactifies the conversation memory using the configured context management strategy.

        Use this tool when the conversation is long and you need to free up context space.
        Helps maintain conversation continuity while staying within token limits.
        """
        raise NotImplementedError
        return "Memory compactified."


# ============== Data Models ==============

MemoryType = Literal["episodic", "procedural", "semantic", "working"]


class ToolCall(BaseModel):
    """Represents a single tool call in a tool sequence."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: str | None = None
    latency_ms: float | None = None
    success: bool = True


class SkillMemory(BaseModel):
    """Procedural memory for learned skills - how to solve specific types of problems.

    Skills are abstracted from successful trajectories and can be reused for similar problems.

    Attributes:
        id: Unique identifier for the skill.
        skill_name: Short descriptive name for the skill.
        description: Detailed description of what the skill does.
        tool_sequence: Ordered sequence of tools used in this skill.
        trigger_patterns: Question patterns that trigger this skill.
        example_qa: Example question-answer pair demonstrating the skill.
        tags: Tool names and categories for filtering.
        success_count: Number of successful applications.
        failure_count: Number of failed applications.
        importance_score: Learned importance score.
        created_at: When the skill was first learned.
        last_used_at: When the skill was last used.
        source_trajectory_id: ID of the trajectory this skill was extracted from.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    skill_name: str
    description: str
    tool_sequence: list[ToolCall] = Field(default_factory=list)
    trigger_patterns: list[str] = Field(default_factory=list)
    example_qa: dict[str, str] = Field(default_factory=dict)  # {"question": ..., "answer": ...}
    tags: list[str] = Field(default_factory=list)  # Tool names for filtering
    success_count: int = Field(default=1)
    failure_count: int = Field(default=0)
    importance_score: float = Field(default=0.7, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.now)
    last_used_at: datetime | None = None
    source_trajectory_id: str | None = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 1.0

    def record_usage(self, success: bool) -> None:
        """Record a usage of this skill."""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.last_used_at = datetime.now()
        # Update importance based on usage
        self.importance_score = min(1.0, self.importance_score + (0.05 if success else -0.1))

    def to_prompt_format(self) -> str:
        """Format skill for inclusion in prompts."""
        tools_str = " → ".join(tc.tool for tc in self.tool_sequence)
        return (
            f"**{self.skill_name}** (成功率: {self.success_rate:.0%})\n"
            f"  描述: {self.description}\n"
            f"  工具链: {tools_str}\n"
            f"  触发模式: {', '.join(self.trigger_patterns[:3])}"
        )

    def to_chroma_document(self) -> dict[str, Any]:
        """Convert to ChromaDB document format."""
        tool_sequence_json = (
            json.dumps([tc.model_dump() for tc in self.tool_sequence], ensure_ascii=False)
            if self.tool_sequence
            else "[]"
        )

        metadata: dict[str, Any] = {
            "skill_name": self.skill_name,
            "memory_type": "procedural",
            "tags": json.dumps(self.tags, ensure_ascii=False),
            "trigger_patterns": json.dumps(self.trigger_patterns, ensure_ascii=False),
            "example_qa": json.dumps(self.example_qa, ensure_ascii=False),
            "tool_sequence": tool_sequence_json,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "importance_score": self.importance_score,
            "created_at": self.created_at.isoformat(),
            "source_trajectory_id": self.source_trajectory_id or "",
        }

        if self.last_used_at:
            metadata["last_used_at"] = self.last_used_at.isoformat()

        # Content combines description and example for better semantic search
        content = f"{self.skill_name}: {self.description}"
        if self.example_qa.get("question"):
            content += f"\n示例问题: {self.example_qa['question']}"

        return {
            "id": self.id,
            "document": content,
            "metadata": metadata,
        }

    @classmethod
    def from_chroma_result(cls, doc_id: str, document: str, metadata: dict[str, Any]) -> "SkillMemory":
        """Create SkillMemory from ChromaDB query result."""
        metadata = metadata.copy()

        # Parse JSON fields
        tool_sequence_raw = metadata.pop("tool_sequence", "[]")
        tool_sequence_data = json.loads(tool_sequence_raw) if isinstance(tool_sequence_raw, str) else []
        tool_sequence = [ToolCall(**tc) for tc in tool_sequence_data]

        tags_raw = metadata.pop("tags", "[]")
        tags = json.loads(tags_raw) if isinstance(tags_raw, str) else tags_raw

        trigger_patterns_raw = metadata.pop("trigger_patterns", "[]")
        trigger_patterns = json.loads(trigger_patterns_raw) if isinstance(trigger_patterns_raw, str) else trigger_patterns_raw

        example_qa_raw = metadata.pop("example_qa", "{}")
        example_qa = json.loads(example_qa_raw) if isinstance(example_qa_raw, str) else example_qa_raw

        created_at = metadata.pop("created_at", None)
        last_used_at = metadata.pop("last_used_at", None)

        return cls(
            id=doc_id,
            skill_name=metadata.pop("skill_name", "Unknown Skill"),
            description=document.split("\n")[0].split(": ", 1)[-1] if ": " in document else document,
            tool_sequence=tool_sequence,
            trigger_patterns=trigger_patterns,
            example_qa=example_qa,
            tags=tags,
            success_count=metadata.pop("success_count", 1),
            failure_count=metadata.pop("failure_count", 0),
            importance_score=metadata.pop("importance_score", 0.7),
            created_at=datetime.fromisoformat(created_at) if created_at else datetime.now(),
            last_used_at=datetime.fromisoformat(last_used_at) if last_used_at else None,
            source_trajectory_id=metadata.pop("source_trajectory_id", None) or None,
        )


class MemoryNode(BaseModel):
    """Memory node for storing conversation and procedural memories.

    Attributes:
        id: Unique identifier for the memory node.
        user_id: User ID for personalized memory.
        session_id: Session ID for context tracing.
        memory_type: Type of memory (episodic, procedural, semantic, working).
        content: Original text content of the memory.
        embedding: Vector representation of the content (set by vector store).
        importance_score: Importance score (0.0-1.0) for retrieval weighting.
        metadata: Flexible metadata field for additional information.
        created_at: Creation timestamp.
        last_accessed_at: Last access timestamp for recency weighting.
        tool_sequence: Procedural tool sequence for agent tool chains.
        success_rate: Success rate (0.0-1.0) based on historical execution.
        avg_latency: Average latency in milliseconds for efficiency evaluation.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = Field(default="default")
    session_id: str = Field(default="default")
    memory_type: MemoryType = Field(default="episodic")
    content: str
    embedding: list[float] | None = Field(default=None, exclude=True)
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed_at: datetime | None = None
    tool_sequence: list[ToolCall] = Field(default_factory=list)
    success_rate: float = Field(default=1.0, ge=0.0, le=1.0)
    avg_latency: float = Field(default=0.0, ge=0.0)

    def is_outdated(self, threshold: float = 0.2) -> bool:
        """Check if this procedural memory is outdated based on success rate."""
        return self.memory_type == "procedural" and self.success_rate < threshold

    def update_stats(self, success: bool, latency_ms: float) -> None:
        """Update success rate and average latency with new execution result."""
        alpha = 0.3
        self.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * self.success_rate

        if self.avg_latency == 0.0:
            self.avg_latency = latency_ms
        else:
            self.avg_latency = alpha * latency_ms + (1 - alpha) * self.avg_latency

        self.last_accessed_at = datetime.now()

    def to_chroma_document(self) -> dict[str, Any]:
        """Convert to ChromaDB document format."""
        tool_sequence_json = (
            json.dumps([tc.model_dump() for tc in self.tool_sequence], ensure_ascii=False)
            if self.tool_sequence
            else "[]"
        )

        metadata: dict[str, Any] = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "memory_type": self.memory_type,
            "importance_score": self.importance_score,
            "created_at": self.created_at.isoformat(),
            "success_rate": self.success_rate,
            "avg_latency": self.avg_latency,
            "tool_sequence": tool_sequence_json,
        }

        if self.last_accessed_at:
            metadata["last_accessed_at"] = self.last_accessed_at.isoformat()

        for key, value in self.metadata.items():
            if value is None:
                continue
            elif isinstance(value, (list, dict)):
                metadata[key] = json.dumps(value, ensure_ascii=False)
            else:
                metadata[key] = value

        return {
            "id": self.id,
            "document": self.content,
            "metadata": metadata,
        }

    @classmethod
    def from_chroma_result(cls, doc_id: str, document: str, metadata: dict[str, Any]) -> "MemoryNode":
        """Create MemoryNode from ChromaDB query result."""
        metadata = metadata.copy()

        tool_sequence_raw = metadata.pop("tool_sequence", "[]")
        if isinstance(tool_sequence_raw, str):
            try:
                tool_sequence_data = json.loads(tool_sequence_raw)
            except json.JSONDecodeError:
                tool_sequence_data = []
        else:
            tool_sequence_data = tool_sequence_raw if tool_sequence_raw else []

        tool_sequence = [ToolCall(**tc) for tc in tool_sequence_data] if tool_sequence_data else []

        created_at = metadata.pop("created_at", None)
        last_accessed_at = metadata.pop("last_accessed_at", None)

        if "entities" in metadata and isinstance(metadata["entities"], str):
            try:
                metadata["entities"] = json.loads(metadata["entities"])
            except json.JSONDecodeError:
                pass

        if "relations" in metadata and isinstance(metadata["relations"], str):
            try:
                metadata["relations"] = json.loads(metadata["relations"])
            except json.JSONDecodeError:
                pass

        return cls(
            id=doc_id,
            user_id=metadata.pop("user_id", "default"),
            session_id=metadata.pop("session_id", "default"),
            memory_type=metadata.pop("memory_type", "episodic"),
            content=document,
            importance_score=metadata.pop("importance_score", 0.5),
            created_at=datetime.fromisoformat(created_at) if created_at else datetime.now(),
            last_accessed_at=datetime.fromisoformat(last_accessed_at) if last_accessed_at else None,
            tool_sequence=tool_sequence,
            success_rate=metadata.pop("success_rate", 1.0),
            avg_latency=metadata.pop("avg_latency", 0.0),
            metadata=metadata,
        )


class MemorySearchResult(BaseModel):
    """Result of a memory search operation."""

    memory: MemoryNode
    score: float = Field(description="Similarity score (lower is more similar for distance)")
    relevance_score: float = Field(
        default=0.0, description="Combined relevance score considering importance and recency"
    )


class SkillSearchResult(BaseModel):
    """Result of a skill search operation."""

    skill: SkillMemory
    score: float = Field(description="Similarity distance score")
    relevance_score: float = Field(default=0.0, description="Combined relevance score")


# ============== Skill Extractor ==============


class SkillExtractor:
    """Extract skills from trajectory using LLM analysis.

    This class analyzes agent execution trajectories and abstracts them into
    reusable skills that can be applied to similar problems.
    """

    EXTRACTION_PROMPT = """你是一个技能抽取专家。分析以下 Agent 执行轨迹，提取可复用的技能。

## 执行轨迹
问题: {question}
答案: {answer}

工具调用序列:
{tool_calls}

## 轨迹摘要
{trajectory_summary}

## 任务
请分析这个执行轨迹，抽取一个可复用的技能。返回 JSON 格式:

```json
{{
    "skill_name": "简短的技能名称（如：搜索并总结、数据库查询、文件分析）",
    "description": "详细描述这个技能做什么，适用于什么场景",
    "trigger_patterns": ["触发这个技能的问题模式1", "问题模式2", "问题模式3"],
    "tags": ["工具标签1", "工具标签2"],
    "importance_score": 0.7
}}
```

要求:
1. skill_name 简洁明了，3-8个字
2. description 说明技能用途和适用场景
3. trigger_patterns 列出 3-5 个会触发这个技能的问题类型
4. tags 包含所有使用的工具名称
5. importance_score 基于技能的通用性和复杂度评分 (0.5-0.9)

只返回 JSON，不要其他内容。
"""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        """Initialize the skill extractor.

        Args:
            model: LLM model name for extraction.
            base_url: API base URL.
            api_key: API key.
        """
        self.model = model or "deepseek-v3"
        self.base_url = base_url
        self.api_key = api_key
        self._agent: Agent | None = None

    def _get_agent(self) -> Agent:
        """Lazy initialization of extraction agent."""
        if self._agent is None and AGENTS_AVAILABLE:
            from utu.utils import AgentsUtils

            model = AgentsUtils.get_agents_model(
                model=self.model,
                base_url=self.base_url,
                api_key=self.api_key,
            )
            self._agent = Agent(
                name="SkillExtractor",
                instructions="你是一个技能抽取专家，负责从 Agent 执行轨迹中提取可复用的技能。",
                model=model,
            )
        return self._agent

    async def extract_skill_from_trajectory(
        self,
        question: str,
        answer: str,
        tool_calls: list[ToolCall],
        trajectory_summary: str | None = None,
        source_trajectory_id: str | None = None,
    ) -> SkillMemory | None:
        """Extract a skill from trajectory data.

        Args:
            question: User's original question.
            answer: Agent's final answer.
            tool_calls: Sequence of tool calls made.
            trajectory_summary: Optional summary of the trajectory.
            source_trajectory_id: ID of the source trajectory.

        Returns:
            Extracted SkillMemory or None if extraction fails.
        """
        if not tool_calls:
            logger.debug("No tool calls to extract skill from")
            return None

        # Format tool calls for prompt
        tool_calls_str = "\n".join(
            f"{i + 1}. {tc.tool}({json.dumps(tc.args, ensure_ascii=False)[:100]})"
            + (f" -> {tc.result[:50]}..." if tc.result else "")
            for i, tc in enumerate(tool_calls)
        )

        prompt = self.EXTRACTION_PROMPT.format(
            question=question,
            answer=answer[:500],
            tool_calls=tool_calls_str,
            trajectory_summary=trajectory_summary or "无",
        )

        try:
            # Try using agent if available
            agent = self._get_agent()
            if agent and AGENTS_AVAILABLE:
                result = await Runner.run(agent, prompt)
                response_text = str(result.final_output) if result.final_output else ""
            else:
                # Fallback: create skill from tool calls directly
                return self._create_skill_from_tools(
                    question, answer, tool_calls, source_trajectory_id
                )

            # Parse JSON response
            json_match = response_text
            if "```json" in response_text:
                json_match = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                json_match = response_text.split("```")[1].split("```")[0]

            skill_data = json.loads(json_match.strip())

            # Create SkillMemory
            skill = SkillMemory(
                skill_name=skill_data.get("skill_name", "未命名技能"),
                description=skill_data.get("description", "从执行轨迹中学习的技能"),
                tool_sequence=tool_calls,
                trigger_patterns=skill_data.get("trigger_patterns", [question[:50]]),
                example_qa={"question": question, "answer": answer[:500]},
                tags=skill_data.get("tags", [tc.tool for tc in tool_calls]),
                importance_score=skill_data.get("importance_score", 0.7),
                source_trajectory_id=source_trajectory_id,
            )

            logger.info(f"Extracted skill: {skill.skill_name}")
            return skill

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse skill extraction response: {e}")
            return self._create_skill_from_tools(question, answer, tool_calls, source_trajectory_id)
        except Exception as e:
            logger.error(f"Skill extraction failed: {e}")
            return self._create_skill_from_tools(question, answer, tool_calls, source_trajectory_id)

    def _create_skill_from_tools(
        self,
        question: str,
        answer: str,
        tool_calls: list[ToolCall],
        source_trajectory_id: str | None = None,
    ) -> SkillMemory:
        """Create a basic skill from tool calls without LLM analysis."""
        tool_names = [tc.tool for tc in tool_calls]
        unique_tools = list(dict.fromkeys(tool_names))

        skill_name = f"{'_'.join(unique_tools[:2])}技能"
        description = f"使用 {', '.join(unique_tools)} 工具解决问题"

        return SkillMemory(
            skill_name=skill_name,
            description=description,
            tool_sequence=tool_calls,
            trigger_patterns=[question[:100]],
            example_qa={"question": question, "answer": answer[:500]},
            tags=unique_tools,
            importance_score=0.6,
            source_trajectory_id=source_trajectory_id,
        )

    def extract_tool_calls_from_trajectory(
        self,
        trajectory: "TrajectoryModel",
    ) -> list[ToolCall]:
        """Extract tool calls from a TrajectoryModel.

        Args:
            trajectory: TrajectoryModel instance.

        Returns:
            List of ToolCall objects.
        """
        tool_calls = []

        # Parse trajectory data
        if hasattr(trajectory, "trajectory") and trajectory.trajectory:
            traj_data = trajectory.trajectory
            if isinstance(traj_data, str):
                try:
                    traj_data = json.loads(traj_data)
                except json.JSONDecodeError:
                    traj_data = {}

            # Extract tool calls from trajectory items
            items = traj_data.get("items", []) if isinstance(traj_data, dict) else []
            for item in items:
                if isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type in ["tool_call", "function_call", "tool_call_item"]:
                        tool_calls.append(
                            ToolCall(
                                tool=item.get("name", item.get("tool", "unknown")),
                                args=item.get("arguments", item.get("args", {})),
                                result=item.get("output", item.get("result")),
                                success=item.get("success", True),
                            )
                        )

        return tool_calls


# ============== Vector Memory Toolkit ==============


class VectorMemoryToolkit:
    """Toolkit for managing agent memories with ChromaDB vector storage.

    Memory Types:
        - episodic: "What happened" - conversation history, events
        - procedural: "How to do" - tool sequences, workflows, skills
        - semantic: "What is" - facts, definitions, domain knowledge
        - working: "Right now" - current session context

    Skill Learning:
        - Automatically extract skills from successful trajectories
        - Store skills as procedural memories with tags for filtering
        - Retrieve relevant skills based on query similarity and tool tags

    Args:
        persist_directory: Directory for persistent storage. None for in-memory.
        collection_prefix: Prefix for collection names.
        default_user_id: Default user ID for operations.
        max_working_memory_turns: Maximum turns in working memory.
        embedding_service: Optional embedding service instance.
        enable_skill_learning: Whether to enable automatic skill extraction.
        skill_extraction_model: Model to use for skill extraction.
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_prefix: str = "memory",
        default_user_id: str = "default",
        max_working_memory_turns: int = 10,
        embedding_service: EmbeddingService | None = None,
        enable_skill_learning: bool = True,
        skill_extraction_model: str | None = None,
    ):
        self.collection_prefix = collection_prefix
        self.default_user_id = default_user_id
        self.max_working_memory_turns = max_working_memory_turns
        self._embedding_service = embedding_service
        self._current_session_id: str | None = None
        self._current_tool_sequence: list[ToolCall] = []
        self.enable_skill_learning = enable_skill_learning

        # Initialize skill extractor
        self._skill_extractor = SkillExtractor(model=skill_extraction_model) if enable_skill_learning else None

        # Initialize ChromaDB
        if persist_directory:
            self._client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
            logger.info(f"Initialized persistent ChromaDB at {persist_directory}")
        else:
            self._client = chromadb.Client(settings=Settings(anonymized_telemetry=False))
            logger.info("Initialized in-memory ChromaDB")

        self._collections: dict[str, chromadb.Collection] = {}

    # ==================== Properties ====================

    @property
    def current_session_id(self) -> str | None:
        """Get current session ID."""
        return self._current_session_id

    @current_session_id.setter
    def current_session_id(self, value: str) -> None:
        """Set current session ID."""
        self._current_session_id = value

    # ==================== Session Management ====================

    def start_session(self, session_id: str | None = None) -> str:
        """Start a new conversation session."""
        self._current_session_id = session_id or str(uuid.uuid4())
        self._current_tool_sequence = []
        return self._current_session_id

    def get_session_id(self) -> str:
        """Get current session ID, creating one if needed."""
        if self._current_session_id is None:
            self._current_session_id = str(uuid.uuid4())
        return self._current_session_id

    def record_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: str | None = None,
        latency_ms: float | None = None,
        success: bool = True,
    ) -> None:
        """Record a tool call in the current sequence."""
        self._current_tool_sequence.append(
            ToolCall(tool=tool_name, args=args, result=result, latency_ms=latency_ms, success=success)
        )

    def get_current_tool_sequence(self) -> list[ToolCall]:
        """Get the current tool call sequence."""
        return self._current_tool_sequence.copy()

    def clear_tool_sequence(self) -> None:
        """Clear the current tool sequence."""
        self._current_tool_sequence = []

    # ==================== Embedding & Collection ====================

    @property
    def embedding_service(self) -> EmbeddingService:
        """Lazy initialization of embedding service."""
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    def _get_collection_name(self, user_id: str, memory_type: MemoryType | None = None) -> str:
        """Generate collection name based on user_id and optional memory_type."""
        if memory_type:
            return f"{self.collection_prefix}_{user_id}_{memory_type}"
        return f"{self.collection_prefix}_{user_id}"

    def _get_skill_collection_name(self, user_id: str) -> str:
        """Get collection name for skills."""
        return f"{self.collection_prefix}_{user_id}_skills"

    def _get_or_create_collection(self, collection_name: str) -> chromadb.Collection:
        """Get or create a ChromaDB collection."""
        if collection_name not in self._collections:
            self._collections[collection_name] = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[collection_name]

    async def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text content."""
        embeddings = await self.embedding_service.embed([text])
        return embeddings[0]

    async def _get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts."""
        return await self.embedding_service.embed(texts)

    # ==================== Core Storage Operations ====================

    async def add_memory(
        self,
        memory: MemoryNode,
        use_type_collection: bool = False,
    ) -> str:
        """Add a memory node to the vector store."""
        if memory.embedding is None:
            memory.embedding = await self._get_embedding(memory.content)

        collection_name = self._get_collection_name(
            memory.user_id,
            memory.memory_type if use_type_collection else None,
        )
        collection = self._get_or_create_collection(collection_name)
        doc_data = memory.to_chroma_document()

        collection.upsert(
            ids=[doc_data["id"]],
            documents=[doc_data["document"]],
            metadatas=[doc_data["metadata"]],
            embeddings=[memory.embedding],
        )
        logger.debug(f"Added memory {memory.id} to {collection_name}")
        return memory.id

    async def search_memories(
        self,
        query: str,
        user_id: str | None = None,
        memory_type: MemoryType | None = None,
        session_id: str | None = None,
        top_k: int = 10,
        min_importance: float = 0.0,
        include_outdated: bool = False,
    ) -> list[MemorySearchResult]:
        """Search memories by semantic similarity."""
        user_id = user_id or self.default_user_id
        query_embedding = await self._get_embedding(query)

        # Build where filter
        where_conditions = []
        if memory_type:
            where_conditions.append({"memory_type": memory_type})
        if session_id:
            where_conditions.append({"session_id": session_id})
        if min_importance > 0:
            where_conditions.append({"importance_score": {"$gte": min_importance}})
        if not include_outdated:
            where_conditions.append({"success_rate": {"$gte": 0.2}})

        where_filter = None
        if len(where_conditions) == 1:
            where_filter = where_conditions[0]
        elif len(where_conditions) > 1:
            where_filter = {"$and": where_conditions}

        collection_name = self._get_collection_name(user_id, memory_type)
        try:
            collection = self._get_or_create_collection(collection_name)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning(f"Query failed: {e}")
            return []

        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                document = results["documents"][0][i] if results["documents"] else ""
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 1.0

                memory = MemoryNode.from_chroma_result(doc_id, document, metadata.copy())
                similarity_score = 1.0 - distance
                recency_score = self._calculate_recency_score(memory.created_at)
                relevance_score = 0.5 * similarity_score + 0.3 * memory.importance_score + 0.2 * recency_score

                search_results.append(MemorySearchResult(memory=memory, score=distance, relevance_score=relevance_score))

        search_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return search_results

    def _calculate_recency_score(self, created_at: datetime) -> float:
        """Calculate recency score (exponential decay, half-life 24h)."""
        age_hours = (datetime.now() - created_at).total_seconds() / 3600
        return 0.5 ** (age_hours / 24)

    # ==================== Skill Storage & Retrieval ====================

    async def store_skill(
        self,
        skill: SkillMemory,
        user_id: str | None = None,
    ) -> str:
        """Store a skill to the skills collection."""
        user_id = user_id or self.default_user_id
        collection_name = self._get_skill_collection_name(user_id)
        collection = self._get_or_create_collection(collection_name)

        doc_data = skill.to_chroma_document()
        embedding = await self._get_embedding(doc_data["document"])

        collection.upsert(
            ids=[doc_data["id"]],
            documents=[doc_data["document"]],
            metadatas=[doc_data["metadata"]],
            embeddings=[embedding],
        )
        logger.info(f"Stored skill: {skill.skill_name} (id={skill.id})")
        return skill.id

    async def search_skills(
        self,
        query: str,
        user_id: str | None = None,
        tool_filter: str | list[str] | None = None,
        top_k: int = 5,
        min_success_rate: float = 0.3,
    ) -> list[SkillSearchResult]:
        """Search skills by semantic similarity and optional tool filter.

        Args:
            query: Query string for semantic search.
            user_id: User ID.
            tool_filter: Filter by tool name(s) in tags.
            top_k: Number of results to return.
            min_success_rate: Minimum success rate threshold.

        Returns:
            List of SkillSearchResult sorted by relevance.
        """
        user_id = user_id or self.default_user_id
        collection_name = self._get_skill_collection_name(user_id)

        try:
            collection = self._get_or_create_collection(collection_name)
        except Exception as e:
            logger.warning(f"Failed to get skill collection: {e}")
            return []

        query_embedding = await self._get_embedding(query)

        # Build where filter for tags if tool_filter is provided
        where_filter = None
        if tool_filter:
            if isinstance(tool_filter, str):
                tool_filter = [tool_filter]
            # ChromaDB doesn't support direct list contains, so we'll filter post-query
            # We can use $contains for string matching in tags JSON
            pass  # Filter will be applied post-query

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2,  # Fetch extra for post-filtering
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.warning(f"Skill query failed: {e}")
            return []

        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                document = results["documents"][0][i] if results["documents"] else ""
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 1.0

                skill = SkillMemory.from_chroma_result(doc_id, document, metadata.copy())

                # Filter by success rate
                if skill.success_rate < min_success_rate:
                    continue

                # Filter by tool tags if specified
                if tool_filter:
                    if not any(tool in skill.tags for tool in tool_filter):
                        continue

                similarity_score = 1.0 - distance
                recency_score = self._calculate_recency_score(skill.created_at)
                relevance_score = (
                    0.4 * similarity_score
                    + 0.3 * skill.importance_score
                    + 0.2 * skill.success_rate
                    + 0.1 * recency_score
                )

                search_results.append(SkillSearchResult(skill=skill, score=distance, relevance_score=relevance_score))

        search_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return search_results[:top_k]

    async def search_skills_by_tool(
        self,
        tool_name: str,
        user_id: str | None = None,
        top_k: int = 10,
    ) -> list[SkillSearchResult]:
        """Search skills by tool name tag.

        Args:
            tool_name: Tool name to filter by.
            user_id: User ID.
            top_k: Number of results.

        Returns:
            List of skills that use the specified tool.
        """
        user_id = user_id or self.default_user_id
        collection_name = self._get_skill_collection_name(user_id)

        try:
            collection = self._get_or_create_collection(collection_name)
            # Get all skills and filter
            results = collection.get(include=["documents", "metadatas"])
        except Exception as e:
            logger.warning(f"Failed to search skills by tool: {e}")
            return []

        skills = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                document = results["documents"][i] if results["documents"] else ""
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                skill = SkillMemory.from_chroma_result(doc_id, document, metadata.copy())

                if tool_name in skill.tags:
                    skills.append(
                        SkillSearchResult(
                            skill=skill,
                            score=0.0,
                            relevance_score=skill.importance_score * skill.success_rate,
                        )
                    )

        skills.sort(key=lambda x: x.relevance_score, reverse=True)
        return skills[:top_k]

    async def update_skill_usage(
        self,
        skill_id: str,
        success: bool,
        user_id: str | None = None,
    ) -> bool:
        """Update skill usage statistics.

        Args:
            skill_id: Skill ID to update.
            success: Whether the skill application was successful.
            user_id: User ID.

        Returns:
            True if updated successfully.
        """
        user_id = user_id or self.default_user_id
        collection_name = self._get_skill_collection_name(user_id)

        try:
            collection = self._get_or_create_collection(collection_name)
            results = collection.get(ids=[skill_id], include=["documents", "metadatas"])

            if not results["ids"]:
                return False

            skill = SkillMemory.from_chroma_result(
                results["ids"][0],
                results["documents"][0] if results["documents"] else "",
                results["metadatas"][0].copy() if results["metadatas"] else {},
            )
            skill.record_usage(success)
            await self.store_skill(skill, user_id)
            return True
        except Exception as e:
            logger.error(f"Failed to update skill usage: {e}")
            return False

    def format_skills_for_prompt(self, skills: list[SkillSearchResult]) -> str:
        """Format skills for inclusion in agent prompts.

        Args:
            skills: List of skill search results.

        Returns:
            Formatted string for prompt injection.
        """
        if not skills:
            return ""

        lines = ["## 可用技能参考"]
        for result in skills[:5]:  # Limit to top 5
            lines.append(result.skill.to_prompt_format())

        return "\n\n".join(lines)

    # ==================== Memory Retrieval Helper ====================

    async def retrieve_all_context(
        self,
        query: str,
        working_memory_max_chars: int = 500,
        episodic_top_k: int = 5,
        episodic_min_score: float = 0.3,
        episodic_max_chars: int = 300,
        semantic_top_k: int = 3,
        semantic_min_score: float = 0.3,
        semantic_max_chars: int = 300,
        skill_top_k: int = 3,
        skill_min_success_rate: float = 0.3,
        include_skills: bool = True,
    ) -> dict[str, str]:
        """检索所有类型的记忆上下文（通用方法）
        
        这个方法封装了常见的记忆检索模式，减少代码重复。
        
        Args:
            query: 查询文本（用于检索 episodic/semantic/skills）
            working_memory_max_chars: working memory 每条的最大字符数
            episodic_top_k: episodic memory 检索数量
            episodic_min_score: episodic memory 最小相关度
            episodic_max_chars: episodic memory 每条的最大字符数
            semantic_top_k: semantic memory 检索数量
            semantic_min_score: semantic memory 最小相关度
            semantic_max_chars: semantic memory 每条的最大字符数
            skill_top_k: skill 检索数量
            skill_min_success_rate: skill 最小成功率
            include_skills: 是否包含 skills 检索
            
        Returns:
            包含各类上下文的字典:
            {
                "working_context": str,
                "episodic_context": str,
                "semantic_context": str,
                "skills_context": str | None,
                "memory_context": str  # 合并后的完整上下文
            }
        """
        # 1. 存储用户问题到 working memory
        await self.store_working_memory(query, role="user")
        logger.debug("Stored user question to working memory")

        # 2. 获取当前 session 的 working memory
        working_memories = await self.get_working_memory()
        working_lines = []
        for mem in working_memories[:-1]:  # 排除刚存入的当前问题
            role = mem.metadata.get("role", "unknown")
            content = mem.content[:working_memory_max_chars]
            working_lines.append(f"[{role}]: {content}")
        working_context = "\n".join(working_lines) if working_lines else ""

        # 3. 检索跨 session 的 episodic memory
        episodic_context = ""
        try:
            episodic_results = await self.search_memories(
                query=query,
                memory_type="episodic",
                top_k=episodic_top_k,
            )
            episodic_lines = []
            for result in episodic_results:
                if result.relevance_score < episodic_min_score:
                    continue
                content = result.memory.content[:episodic_max_chars]
                score = result.relevance_score
                episodic_lines.append(f"- [相关度: {score:.2f}] {content}")
            episodic_context = "\n".join(episodic_lines) if episodic_lines else ""
        except Exception as e:
            logger.debug(f"Episodic memory search failed: {e}")

        # 4. 检索 semantic memory（知识库）
        semantic_context = ""
        try:
            semantic_results = await self.search_memories(
                query=query,
                memory_type="semantic",
                top_k=semantic_top_k,
            )
            semantic_lines = []
            for result in semantic_results:
                if result.relevance_score < semantic_min_score:
                    continue
                content = result.memory.content[:semantic_max_chars]
                semantic_lines.append(f"- {content}")
            semantic_context = "\n".join(semantic_lines) if semantic_lines else ""
        except Exception as e:
            logger.debug(f"Semantic memory search failed: {e}")

        # 5. 检索相关 Skills（程序性记忆）
        skills_context = None
        if include_skills:
            try:
                skill_results = await self.search_skills(
                    query=query,
                    top_k=skill_top_k,
                    min_success_rate=skill_min_success_rate,
                )
                if skill_results:
                    skills_context = self.format_skills_for_prompt(skill_results)
                    logger.info(f"Retrieved {len(skill_results)} relevant skills")
            except Exception as e:
                logger.debug(f"Skill search failed: {e}")

        # 6. 合并所有上下文
        memory_parts = []
        if skills_context:
            memory_parts.append(skills_context)  # Skills 放最前面，最重要
        if working_context:
            memory_parts.append(f"## 当前对话历史\n{working_context}")
        if episodic_context:
            memory_parts.append(f"## 相关历史记忆\n{episodic_context}")
        if semantic_context:
            memory_parts.append(f"## 相关知识\n{semantic_context}")

        memory_context = "\n\n".join(memory_parts) if memory_parts else ""

        return {
            "working_context": working_context,
            "episodic_context": episodic_context,
            "semantic_context": semantic_context,
            "skills_context": skills_context,
            "memory_context": memory_context,
        }

    # ==================== Skill Extraction from Trajectory ====================

    async def extract_and_store_skill(
        self,
        question: str,
        answer: str,
        tool_calls: list[ToolCall] | None = None,
        trajectory_summary: str | None = None,
        source_trajectory_id: str | None = None,
        user_id: str | None = None,
    ) -> str | None:
        """Extract a skill from execution data and store it.

        Args:
            question: User's original question.
            answer: Agent's final answer.
            tool_calls: Tool calls made during execution.
            trajectory_summary: Optional trajectory summary.
            source_trajectory_id: ID of the source trajectory.
            user_id: User ID.

        Returns:
            Skill ID if stored successfully, None otherwise.
        """
        if not self.enable_skill_learning or not self._skill_extractor:
            return None

        # Use provided tool_calls or current sequence
        tools = tool_calls if tool_calls is not None else self._current_tool_sequence

        if len(tools) < 1:
            logger.debug("Not enough tool calls to extract skill")
            return None

        try:
            skill = await self._skill_extractor.extract_skill_from_trajectory(
                question=question,
                answer=answer,
                tool_calls=tools,
                trajectory_summary=trajectory_summary,
                source_trajectory_id=source_trajectory_id,
            )

            if skill:
                skill_id = await self.store_skill(skill, user_id)
                # Clear current sequence after successful extraction
                self._current_tool_sequence = []
                return skill_id
        except Exception as e:
            logger.error(f"Failed to extract and store skill: {e}")

        return None

    async def extract_skill_from_trajectory_model(
        self,
        trajectory: "TrajectoryModel",
        user_id: str | None = None,
    ) -> str | None:
        """Extract skill from a TrajectoryModel instance.

        Args:
            trajectory: TrajectoryModel instance.
            user_id: User ID.

        Returns:
            Skill ID if stored successfully.
        """
        if not self.enable_skill_learning or not self._skill_extractor:
            return None

        tool_calls = self._skill_extractor.extract_tool_calls_from_trajectory(trajectory)

        if len(tool_calls) < 1:
            return None

        return await self.extract_and_store_skill(
            question=trajectory.task or "",
            answer=trajectory.final_output or "",
            tool_calls=tool_calls,
            trajectory_summary=str(trajectory.trajectory) if trajectory.trajectory else None,
            source_trajectory_id=trajectory.trace_id,
            user_id=user_id,
        )

    # ==================== Working Memory ====================

    async def get_working_memory(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
        max_turns: int | None = None,
    ) -> list[MemoryNode]:
        """Get working memory for a specific session."""
        user_id = user_id or self.default_user_id
        session_id = session_id or self.get_session_id()
        max_turns = max_turns or self.max_working_memory_turns

        collection_name = self._get_collection_name(user_id)
        try:
            collection = self._get_or_create_collection(collection_name)
            results = collection.get(
                where={"$and": [{"session_id": session_id}, {"memory_type": "working"}]},
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.warning(f"Failed to get working memory: {e}")
            return []

        memories = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                document = results["documents"][i] if results["documents"] else ""
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                memories.append(MemoryNode.from_chroma_result(doc_id, document, metadata.copy()))

        memories.sort(key=lambda x: x.created_at)
        return memories[-max_turns:]

    async def store_working_memory(
        self,
        content: str,
        role: str = "user",
        user_id: str | None = None,
    ) -> str:
        """Store a working memory entry."""
        memory = MemoryNode(
            user_id=user_id or self.default_user_id,
            session_id=self.get_session_id(),
            memory_type="working",
            content=content,
            importance_score=0.3,
            metadata={"role": role},
        )
        return await self.add_memory(memory)

    # ==================== Episodic Memory ====================

    async def store_episodic_memory(
        self,
        content: str,
        importance_score: float = 0.5,
        metadata: dict[str, Any] | None = None,
        user_id: str | None = None,
    ) -> str:
        """Store an episodic memory (conversation or event)."""
        memory = MemoryNode(
            user_id=user_id or self.default_user_id,
            session_id=self.get_session_id(),
            memory_type="episodic",
            content=content,
            importance_score=importance_score,
            metadata=metadata or {},
        )
        memory_id = await self.add_memory(memory, use_type_collection=True)
        logger.info(f"Stored episodic memory: {memory_id}")
        return memory_id

    async def save_conversation_to_episodic(
        self,
        question: str,
        answer: str,
        importance_score: float = 0.5,
        user_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save a complete Q&A pair to episodic memory."""
        content = f"Question: {question}\nAnswer: {answer}"
        metadata = {"question": question, "answer_preview": answer[:200]}
        if extra_metadata:
            metadata.update(extra_metadata)

        memory = MemoryNode(
            user_id=user_id or self.default_user_id,
            session_id=self.get_session_id(),
            memory_type="episodic",
            content=content,
            importance_score=importance_score,
            metadata=metadata,
        )

        return await self.add_memory(memory, use_type_collection=True)

    # ==================== Procedural Memory ====================

    async def store_procedural_memory(
        self,
        content: str,
        tool_sequence: list[ToolCall] | None = None,
        importance_score: float = 0.7,
        user_id: str | None = None,
    ) -> str:
        """Store a procedural memory (tool sequence)."""
        tools = tool_sequence if tool_sequence is not None else self._current_tool_sequence
        total_latency = sum(tc.latency_ms or 0 for tc in tools)

        memory = MemoryNode(
            user_id=user_id or self.default_user_id,
            session_id=self.get_session_id(),
            memory_type="procedural",
            content=content,
            importance_score=importance_score,
            tool_sequence=tools,
            avg_latency=total_latency,
            success_rate=1.0,
        )
        memory_id = await self.add_memory(memory, use_type_collection=True)
        logger.info(f"Stored procedural memory: {memory_id} with {len(tools)} tools")
        self._current_tool_sequence = []
        return memory_id

    async def update_procedural_stats(
        self,
        memory_id: str,
        success: bool,
        latency_ms: float,
        user_id: str | None = None,
    ) -> bool:
        """Update statistics for a procedural memory."""
        user_id = user_id or self.default_user_id
        collection_name = self._get_collection_name(user_id, "procedural")

        try:
            collection = self._get_or_create_collection(collection_name)
            results = collection.get(ids=[memory_id], include=["documents", "metadatas"])

            if not results["ids"]:
                return False

            memory = MemoryNode.from_chroma_result(
                results["ids"][0],
                results["documents"][0] if results["documents"] else "",
                results["metadatas"][0].copy() if results["metadatas"] else {},
            )
            memory.update_stats(success, latency_ms)
            await self.add_memory(memory, use_type_collection=True)
            return True
        except Exception as e:
            logger.error(f"Failed to update procedural stats: {e}")
            return False

    # ==================== Semantic Memory ====================

    async def store_semantic_memory(
        self,
        content: str,
        category: str | None = None,
        source: str | None = None,
        importance_score: float = 0.6,
        entities: list[str] | None = None,
        user_id: str | None = None,
    ) -> str:
        """Store semantic memory (facts, concepts, domain knowledge)."""
        metadata: dict[str, Any] = {}
        if category:
            metadata["category"] = category
        if source:
            metadata["source"] = source
        if entities:
            metadata["entities"] = entities

        memory = MemoryNode(
            user_id=user_id or self.default_user_id,
            session_id=self.get_session_id(),
            memory_type="semantic",
            content=content,
            importance_score=importance_score,
            metadata=metadata,
        )
        memory_id = await self.add_memory(memory, use_type_collection=True)
        logger.info(f"Stored semantic memory: {memory_id}, category={category}")
        return memory_id

    async def store_schema_knowledge(
        self,
        table_name: str,
        columns: list[dict[str, str]],
        description: str | None = None,
        relationships: list[str] | None = None,
        user_id: str | None = None,
    ) -> str:
        """Store database schema as semantic memory."""
        column_strs = [
            f"  - {col['name']}: {col.get('type', 'unknown')} ({col.get('description', '')})" for col in columns
        ]
        content = f"Table: {table_name}\n"
        if description:
            content += f"Description: {description}\n"
        content += "Columns:\n" + "\n".join(column_strs)
        if relationships:
            content += "\nRelationships:\n" + "\n".join(f"  - {r}" for r in relationships)

        return await self.store_semantic_memory(
            content=content,
            category="schema",
            source="schema_definition",
            importance_score=0.8,
            entities=[table_name] + [col["name"] for col in columns],
            user_id=user_id,
        )

    async def store_sql_pattern(
        self,
        pattern_name: str,
        sql_pattern: str,
        description: str,
        examples: list[str] | None = None,
        user_id: str | None = None,
    ) -> str:
        """Store a SQL pattern as semantic memory."""
        content = f"SQL Pattern: {pattern_name}\nDescription: {description}\nPattern: {sql_pattern}"
        if examples:
            content += "\nExamples:\n" + "\n".join(f"  - {ex}" for ex in examples)

        return await self.store_semantic_memory(
            content=content,
            category="sql_pattern",
            source="learned",
            importance_score=0.7,
            user_id=user_id,
        )

    # ==================== Trajectory Summary Storage ====================

    async def store_trajectory_summary(
        self,
        question: str,
        answer: str,
        tool_calls: list[ToolCall],
        trajectory_id: str | None = None,
        user_id: str | None = None,
        auto_extract_skill: bool = True,
    ) -> dict[str, str | None]:
        """Store trajectory summary and optionally extract skill.

        This method stores the trajectory as both:
        1. Episodic memory (what happened)
        2. Procedural memory (how it was done)
        3. Optionally extracts and stores a skill

        Args:
            question: User's original question.
            answer: Agent's final answer.
            tool_calls: Sequence of tool calls.
            trajectory_id: Optional trajectory ID.
            user_id: User ID.
            auto_extract_skill: Whether to auto-extract skill.

        Returns:
            Dictionary with memory IDs: {"episodic_id", "procedural_id", "skill_id"}
        """
        user_id = user_id or self.default_user_id
        result = {"episodic_id": None, "procedural_id": None, "skill_id": None}

        # 1. Store to episodic memory
        episodic_id = await self.save_conversation_to_episodic(
            question=question,
            answer=answer,
            importance_score=0.6,
            user_id=user_id,
            extra_metadata={
                "trajectory_id": trajectory_id,
                "tool_count": len(tool_calls),
                "tools_used": [tc.tool for tc in tool_calls],
            },
        )
        result["episodic_id"] = episodic_id

        # 2. Store to procedural memory if there are tool calls
        if tool_calls:
            procedural_content = f"Task: {question}\nSolution: Used {len(tool_calls)} tools"
            procedural_id = await self.store_procedural_memory(
                content=procedural_content,
                tool_sequence=tool_calls,
                importance_score=0.7,
                user_id=user_id,
            )
            result["procedural_id"] = procedural_id

        # 3. Extract and store skill if enabled
        if auto_extract_skill and self.enable_skill_learning and len(tool_calls) >= 1:
            skill_id = await self.extract_and_store_skill(
                question=question,
                answer=answer,
                tool_calls=tool_calls,
                source_trajectory_id=trajectory_id,
                user_id=user_id,
            )
            result["skill_id"] = skill_id

        return result

    
        # ==================== Context Retrieval ====================

    async def retrieve_relevant_context(
        self,
        query: str,
        user_id: str | None = None,
        include_episodic: bool = True,
        include_procedural: bool = True,
        include_semantic: bool = True,
        include_skills: bool = True,
        top_k: int = 5,
    ) -> dict[str, list]:
        """Retrieve relevant context from all memory types including skills.

        Args:
            query: Query string for semantic search.
            user_id: User ID for personalized retrieval.
            include_episodic: Whether to include episodic memories.
            include_procedural: Whether to include procedural memories.
            include_semantic: Whether to include semantic memories.
            include_skills: Whether to include learned skills.
            top_k: Number of results per memory type.

        Returns:
            Dictionary with lists of search results for each memory type.
        """
        user_id = user_id or self.default_user_id
        context: dict[str, list] = {
            "episodic": [],
            "procedural": [],
            "semantic": [],
            "skills": [],
        }

        if include_episodic:
            context["episodic"] = await self.search_memories(query, user_id, "episodic", top_k=top_k)
        if include_procedural:
            context["procedural"] = await self.search_memories(query, user_id, "procedural", top_k=top_k)
        if include_semantic:
            context["semantic"] = await self.search_memories(query, user_id, "semantic", top_k=top_k)
        if include_skills:
            context["skills"] = await self.search_skills(query, user_id, top_k=top_k)

        return context

    def format_context_for_prompt(self, context: dict[str, list]) -> str:
        """Format retrieved context including skills for inclusion in prompts.

        Args:
            context: Dictionary of search results from retrieve_relevant_context.

        Returns:
            Formatted string suitable for prompt injection.
        """
        parts = []

        # Format skills first (most actionable - helps agent know HOW to solve)
        if context.get("skills"):
            skill_lines = ["## 🛠️ 可用技能参考"]
            for result in context["skills"][:5]:
                if isinstance(result, SkillSearchResult):
                    skill = result.skill
                    tools_str = " → ".join(tc.tool for tc in skill.tool_sequence[:5])
                    skill_lines.append(
                        f"- **{skill.skill_name}** (成功率: {skill.success_rate:.0%})\n"
                        f"  描述: {skill.description}\n"
                        f"  工具链: {tools_str}\n"
                        f"  触发模式: {', '.join(skill.trigger_patterns[:3])}"
                    )
            if len(skill_lines) > 1:
                parts.append("\n".join(skill_lines))

        # Format semantic memories (domain knowledge)
        if context.get("semantic"):
            semantic_lines = ["## 📚 相关知识"]
            for result in context["semantic"][:3]:
                if isinstance(result, MemorySearchResult):
                    memory = result.memory
                    category = memory.metadata.get("category", "general")
                    content_preview = memory.content[:200].replace("\n", " ")
                    semantic_lines.append(f"- [{category}] {content_preview}...")
            if len(semantic_lines) > 1:
                parts.append("\n".join(semantic_lines))

        # Format episodic memories (conversation history)
        if context.get("episodic"):
            episodic_lines = ["## 💬 相关历史对话"]
            for result in context["episodic"][:3]:
                if isinstance(result, MemorySearchResult):
                    memory = result.memory
                    date_str = memory.created_at.strftime("%Y-%m-%d %H:%M")
                    content_preview = memory.content[:150].replace("\n", " ")
                    episodic_lines.append(f"- [{date_str}] {content_preview}...")
            if len(episodic_lines) > 1:
                parts.append("\n".join(episodic_lines))

        # Format procedural memories (past solutions)
        if context.get("procedural"):
            procedural_lines = ["## 🔧 历史解决方案"]
            for result in context["procedural"][:2]:
                if isinstance(result, MemorySearchResult):
                    memory = result.memory
                    tools = [tc.tool for tc in memory.tool_sequence]
                    tools_str = " → ".join(tools) if tools else "无工具调用"
                    content_preview = memory.content[:80].replace("\n", " ")
                    procedural_lines.append(
                        f"- 任务: {content_preview}\n"
                        f"  工具序列: {tools_str}\n"
                        f"  成功率: {memory.success_rate:.0%}"
                    )
            if len(procedural_lines) > 1:
                parts.append("\n".join(procedural_lines))

        return "\n\n".join(parts) if parts else ""

    # ==================== Cleanup ====================

    async def cleanup_outdated_memories(
        self,
        user_id: str | None = None,
        success_rate_threshold: float = 0.2,
    ) -> int:
        """Clean up outdated procedural memories with low success rate.

        Args:
            user_id: User ID for cleanup.
            success_rate_threshold: Memories below this threshold are deleted.

        Returns:
            Number of memories deleted.
        """
        user_id = user_id or self.default_user_id
        collection_name = self._get_collection_name(user_id, "procedural")

        try:
            collection = self._get_or_create_collection(collection_name)
            results = collection.get(
                where={"success_rate": {"$lt": success_rate_threshold}},
                include=["metadatas"],
            )

            if results["ids"]:
                collection.delete(ids=results["ids"])
                logger.info(f"Cleaned up {len(results['ids'])} outdated memories")
                return len(results["ids"])
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

        return 0

    async def cleanup_outdated_skills(
        self,
        user_id: str | None = None,
        success_rate_threshold: float = 0.2,
    ) -> int:
        """Clean up skills with low success rate.

        Args:
            user_id: User ID for cleanup.
            success_rate_threshold: Skills below this threshold are deleted.

        Returns:
            Number of skills deleted.
        """
        user_id = user_id or self.default_user_id
        collection_name = self._get_skill_collection_name(user_id)

        try:
            collection = self._get_or_create_collection(collection_name)
            results = collection.get(include=["metadatas"])

            ids_to_delete = []
            if results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i] if results["metadatas"] else {}
                    success_count = metadata.get("success_count", 1)
                    failure_count = metadata.get("failure_count", 0)
                    total = success_count + failure_count
                    success_rate = success_count / total if total > 0 else 1.0
                    if success_rate < success_rate_threshold:
                        ids_to_delete.append(doc_id)

            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                logger.info(f"Cleaned up {len(ids_to_delete)} outdated skills")
                return len(ids_to_delete)
        except Exception as e:
            logger.warning(f"Skill cleanup failed: {e}")

        return 0

    async def delete_memory(
        self,
        memory_id: str,
        user_id: str | None = None,
        memory_type: MemoryType | None = None,
    ) -> bool:
        """Delete a memory by ID.

        Args:
            memory_id: Memory ID to delete.
            user_id: User ID.
            memory_type: Memory type for collection lookup.

        Returns:
            True if deleted successfully.
        """
        user_id = user_id or self.default_user_id
        collection_name = self._get_collection_name(user_id, memory_type)

        try:
            collection = self._get_or_create_collection(collection_name)
            collection.delete(ids=[memory_id])
            return True
        except Exception as e:
            logger.warning(f"Failed to delete memory {memory_id}: {e}")
            return False

    async def delete_skill(
        self,
        skill_id: str,
        user_id: str | None = None,
    ) -> bool:
        """Delete a skill by ID.

        Args:
            skill_id: Skill ID to delete.
            user_id: User ID.

        Returns:
            True if deleted successfully.
        """
        user_id = user_id or self.default_user_id
        collection_name = self._get_skill_collection_name(user_id)

        try:
            collection = self._get_or_create_collection(collection_name)
            collection.delete(ids=[skill_id])
            return True
        except Exception as e:
            logger.warning(f"Failed to delete skill {skill_id}: {e}")
            return False

    def list_collections(self) -> list[str]:
        """List all collection names.

        Returns:
            List of collection names.
        """
        return [c.name for c in self._client.list_collections()]

    async def get_all_skills(
        self,
        user_id: str | None = None,
    ) -> list[SkillMemory]:
        """Get all skills for a user.

        Args:
            user_id: User ID.

        Returns:
            List of all SkillMemory objects.
        """
        user_id = user_id or self.default_user_id
        collection_name = self._get_skill_collection_name(user_id)

        try:
            collection = self._get_or_create_collection(collection_name)
            results = collection.get(include=["documents", "metadatas"])

            skills = []
            if results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    document = results["documents"][i] if results["documents"] else ""
                    metadata = results["metadatas"][i] if results["metadatas"] else {}
                    skills.append(SkillMemory.from_chroma_result(doc_id, document, metadata.copy()))

            return skills
        except Exception as e:
            logger.warning(f"Failed to get all skills: {e}")
            return []

    async def get_skill_by_id(
        self,
        skill_id: str,
        user_id: str | None = None,
    ) -> SkillMemory | None:
        """Get a skill by ID.

        Args:
            skill_id: Skill ID.
            user_id: User ID.

        Returns:
            SkillMemory if found, None otherwise.
        """
        user_id = user_id or self.default_user_id
        collection_name = self._get_skill_collection_name(user_id)

        try:
            collection = self._get_or_create_collection(collection_name)
            results = collection.get(ids=[skill_id], include=["documents", "metadatas"])

            if results["ids"]:
                return SkillMemory.from_chroma_result(
                    results["ids"][0],
                    results["documents"][0] if results["documents"] else "",
                    results["metadatas"][0].copy() if results["metadatas"] else {},
                )
        except Exception as e:
            logger.warning(f"Failed to get skill {skill_id}: {e}")

        return None

    async def get_skills_stats(
        self,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Get statistics about stored skills.

        Args:
            user_id: User ID.

        Returns:
            Dictionary with skill statistics.
        """
        user_id = user_id or self.default_user_id
        skills = await self.get_all_skills(user_id)

        if not skills:
            return {
                "total_skills": 0,
                "avg_success_rate": 0.0,
                "total_usage": 0,
                "tools_distribution": {},
                "top_skills": [],
            }

        total_usage = sum(s.success_count + s.failure_count for s in skills)
        avg_success_rate = sum(s.success_rate for s in skills) / len(skills) if skills else 0.0

        # Tool distribution
        tools_count: dict[str, int] = {}
        for skill in skills:
            for tag in skill.tags:
                tools_count[tag] = tools_count.get(tag, 0) + 1

        # Top skills by usage
        top_skills = sorted(skills, key=lambda s: s.success_count + s.failure_count, reverse=True)[:5]

        return {
            "total_skills": len(skills),
            "avg_success_rate": avg_success_rate,
            "total_usage": total_usage,
            "tools_distribution": tools_count,
            "top_skills": [
                {
                    "id": s.id,
                    "name": s.skill_name,
                    "success_rate": s.success_rate,
                    "usage_count": s.success_count + s.failure_count,
                    "tags": s.tags,
                }
                for s in top_skills
            ],
        }