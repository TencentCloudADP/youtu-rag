import pytest

from utu.config import ConfigLoader
from utu.tools import VideoToolkit


@pytest.fixture
def video_toolkit() -> VideoToolkit:
    config = ConfigLoader.load_toolkit_config("video")
    return VideoToolkit(config=config)


async def test_video_qa(video_toolkit: VideoToolkit):
    video_url = "https://www.youtube.com/watch?v=d7-WQa2_mX8"
    question = "explain the main content of the video"
    result = await video_toolkit.video_qa(video_url, question)
    print(result)
