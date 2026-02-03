import pytest

from utu.config import ConfigLoader
from utu.tools import AudioToolkit


@pytest.fixture
def audio_toolkit() -> AudioToolkit:
    config = ConfigLoader.load_toolkit_config("audio")
    return AudioToolkit(config=config)


async def test_audio_qa(audio_toolkit: AudioToolkit):
    audio_url = "http://www.pthxx.com/b_audio/pthxx_com_mp3/01_langdu/02.mp3"
    question = "explain the main content of the audio"
    result = await audio_toolkit.audio_qa(audio_url, question)
    print(result)
