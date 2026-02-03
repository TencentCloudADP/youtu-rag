import pytest

from utu.tools.serper_toolkit import SerperToolkit


@pytest.fixture
def serper_toolkit():
    return SerperToolkit()


@pytest.mark.asyncio
async def test_google_search(serper_toolkit: SerperToolkit):
    query = "test"
    result = await serper_toolkit.google_search(query)
    assert result["status"] == "success"
    assert result["query"] == query
    assert "results" in result


@pytest.mark.asyncio
async def test_autocomplete(serper_toolkit: SerperToolkit):
    query = "te"
    result = await serper_toolkit.autocomplete(query)
    assert result["status"] == "success"
    assert result["query"] == query
    assert "suggestions" in result


@pytest.mark.asyncio
async def test_google_lens(serper_toolkit: SerperToolkit):
    url = "https://example.com/image.jpg"
    result = await serper_toolkit.google_lens(url)
    assert result["status"] == "success"
    assert result["url"] == url
    assert "results" in result


@pytest.mark.asyncio
async def test_image_search(serper_toolkit: SerperToolkit):
    query = "test image"
    result = await serper_toolkit.image_search(query)
    assert result["status"] == "success"
    assert result["query"] == query
    assert "results" in result


@pytest.mark.asyncio
async def test_map_search(serper_toolkit: SerperToolkit):
    query = "test location"
    result = await serper_toolkit.map_search(query)
    assert result["status"] == "success"
    assert result["query"] == query
    assert "results" in result


@pytest.mark.asyncio
async def test_news_search(serper_toolkit: SerperToolkit):
    query = "test news"
    result = await serper_toolkit.news_search(query)
    assert result["status"] == "success"
    assert result["query"] == query
    assert "results" in result


@pytest.mark.asyncio
async def test_place_search(serper_toolkit: SerperToolkit):
    query = "test place"
    result = await serper_toolkit.place_search(query)
    assert result["status"] == "success"
    assert result["query"] == query
    assert "results" in result


@pytest.mark.asyncio
async def test_scholar_search(serper_toolkit: SerperToolkit):
    query = "test scholar"
    result = await serper_toolkit.scholar_search(query)
    assert result["status"] == "success"
    assert result["query"] == query
    assert "results" in result


@pytest.mark.asyncio
async def test_video_search(serper_toolkit: SerperToolkit):
    query = "test video"
    result = await serper_toolkit.video_search(query)
    assert result["status"] == "success"
    assert result["query"] == query
    assert "results" in result
