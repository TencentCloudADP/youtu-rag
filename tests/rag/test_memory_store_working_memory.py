import uuid

import pytest

from utu.rag.base import Chunk
from utu.rag.config import VectorStoreConfig
from utu.rag.storage.implementations.memory_store import MemoryVectorStore


@pytest.fixture(scope="module")
def working_memory_store(tmp_path_factory):
    return MemoryVectorStore(config=VectorStoreConfig(persist_directory=str(tmp_path_factory.mktemp("working_memory"))))


@pytest.mark.parametrize("embedding", [[0.25, 0.5], [0.0, 0.0], [0.0], [0.75]])
async def test_get_working_memory_preserves_embedding(working_memory_store, embedding):
    user_id = uuid.uuid4().hex
    chunk = Chunk(
        id="working-1",
        document_id="conversation-1",
        content="A working memory",
        chunk_index=3,
        metadata={"session_id": "session-1", "memory_type": "working", "created_at": "2026-01-01T12:00:00"},
        embedding=embedding,
    )
    await working_memory_store.add_chunks([chunk], working_memory_store.get_collection_name(user_id))

    results = await working_memory_store.get_working_memory(user_id, "session-1")

    assert len(results) == 1
    result = results[0]
    assert result.id == chunk.id
    assert result.document_id == chunk.document_id
    assert result.content == chunk.content
    assert result.chunk_index == chunk.chunk_index
    assert result.metadata.items() >= chunk.metadata.items()
    assert result.embedding is not None
    assert list(result.embedding) == pytest.approx(embedding)


async def test_get_working_memory_filters_and_sorts_latest_turns(working_memory_store):
    user_id = uuid.uuid4().hex
    chunks = []
    for chunk_id, session_id, memory_type, created_at in (
        ("newest", "session-1", "working", "2026-01-03"),
        ("oldest", "session-1", "working", "2026-01-01"),
        ("middle", "session-1", "working", "2026-01-02"),
        ("other-session", "session-2", "working", "2026-01-04"),
        ("other-type", "session-1", "episodic", "2026-01-05"),
    ):
        chunks.append(
            Chunk(
                id=chunk_id,
                document_id="conversation-1",
                content=chunk_id,
                chunk_index=len(chunks),
                metadata={"session_id": session_id, "memory_type": memory_type, "created_at": created_at},
                embedding=[0.25, 0.5],
            )
        )
    await working_memory_store.add_chunks(chunks, working_memory_store.get_collection_name(user_id))

    results = await working_memory_store.get_working_memory(user_id, "session-1", max_turns=2)

    assert [chunk.id for chunk in results] == ["middle", "newest"]
    for chunk in results:
        assert list(chunk.embedding) == pytest.approx([0.25, 0.5])


async def test_get_working_memory_returns_empty_for_missing_session(working_memory_store):
    results = await working_memory_store.get_working_memory(uuid.uuid4().hex, "missing-session")

    assert results == []
