import pytest

from utu.config import ConfigLoader
from utu.tools import ArxivToolkit


@pytest.fixture
def arxiv_toolkit():
    config = ConfigLoader.load_toolkit_config("arxiv")
    return ArxivToolkit(config=config)


async def test_search_papers(arxiv_toolkit: ArxivToolkit):
    result = await arxiv_toolkit.search_papers("tool maker")
    print(result)
    result = await arxiv_toolkit.search_papers("au:del_maestro AND ti:checkerboard")
    print(result)
    result = await arxiv_toolkit.search_papers("au:del_maestro AND submittedDate:[202301010600 TO 202401010600]")
    print(result)
