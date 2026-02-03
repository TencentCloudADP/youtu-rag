import base64

import requests

from utu.config import ConfigLoader
from utu.utils import SimplifiedAsyncOpenAI

config = ConfigLoader.load_model_config("base")
client = SimplifiedAsyncOpenAI(**config.model_provider.model_dump())


# https://platform.openai.com/docs/guides/audio?example=audio-in#add-audio-to-your-existing-application
async def test_audio_input():
    # Fetch the audio file and convert it to a base64 encoded string
    url = "https://cdn.openai.com/API/docs/audio/alloy.wav"
    response = requests.get(url)
    response.raise_for_status()
    wav_data = response.content
    encoded_string = base64.b64encode(wav_data).decode("utf-8")

    completion = await client.chat_completions_create(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        # voice: alloy, ash, ballad, coral, echo, fable, nova, onyx, sage, and shimmer
        # format: wav, mp3, flac, opus, or pcm16
        audio={"voice": "alloy", "format": "wav"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this recording?"},
                    {"type": "input_audio", "input_audio": {"data": encoded_string, "format": "wav"}},
                ],
            },
        ],
    )

    print(completion.choices[0].message)


# https://platform.openai.com/docs/api-reference/audio/createTranscription
async def test_transcription():
    # flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm
    url = "https://cdn.openai.com/API/docs/audio/alloy.wav"
    response = requests.get(url)
    response.raise_for_status()
    with open("alloy.wav", "wb") as f:
        f.write(response.content)
    audio_file = open("alloy.wav", "rb")

    # model: gpt-4o-transcribe, gpt-4o-mini-transcribe, and whisper-1
    transcript = await client.audio.transcriptions.create(
        model="whisper-1", file=audio_file, response_format="verbose_json", timestamp_granularities=["segment"]
    )
    print(transcript)
