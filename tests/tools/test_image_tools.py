import pytest

from utu.config import ConfigLoader
from utu.tools import ImageToolkit


@pytest.fixture
def image_toolkit() -> ImageToolkit:
    config = ConfigLoader.load_toolkit_config("image")
    return ImageToolkit(config=config)


IMAGE_URL = "https://github.com/TencentCloudADP/youtu-agent/raw/main/docs/assets/mascot.png"
tasks = ((IMAGE_URL, "What is main colors of this image?"), (IMAGE_URL,))


async def test_image_toolkit(image_toolkit: ImageToolkit):
    for task in tasks:
        result = await image_toolkit.image_qa(*task)
        print(f"{task}: {result}")
