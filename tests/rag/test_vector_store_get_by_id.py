import pytest

from utu.rag.base import Chunk
from utu.rag.config import VectorStoreConfig
from utu.rag.storage.implementations.chroma_store import ChromaVectorStore
from utu.rag.storage.implementations.memory_store import MemoryVectorStore


@pytest.fixture(scope="module")
def chroma_directory(tmp_path_factory):
    return tmp_path_factory.mktemp("chroma")


@pytest.fixture(scope="module", params=[ChromaVectorStore, MemoryVectorStore], ids=["chroma", "memory"])
def vector_store(request, chroma_directory):
    config = VectorStoreConfig(
        collection_name=f"lookup_{request.param.__name__.lower()}",
        persist_directory=str(chroma_directory),
    )
    store = request.param(config=config)
    if isinstance(store, MemoryVectorStore):
        store.get_or_create_collection()
    return store


@pytest.mark.parametrize("embedding", [[0.25, 0.5], [0.0, 0.0], [0.0], [0.75]])
async def test_get_by_id_preserves_stored_embedding(vector_store, embedding):
    await vector_store.clear()
    chunk = Chunk(
        id="chunk-1",
        document_id="document-1",
        content="A stored document chunk",
        chunk_index=3,
        metadata={"source": "example.txt"},
        embedding=embedding,
    )
    await vector_store.add_chunks([chunk])

    result = await vector_store.get_by_id(chunk.id)

    assert result is not None
    assert result.id == chunk.id
    assert result.document_id == chunk.document_id
    assert result.content == chunk.content
    assert result.chunk_index == chunk.chunk_index
    assert result.metadata["source"] == "example.txt"
    assert result.embedding is not None
    assert list(result.embedding) == pytest.approx(embedding)


async def test_get_by_id_returns_none_for_missing_chunk(vector_store):
    assert await vector_store.get_by_id("missing-chunk") is None
