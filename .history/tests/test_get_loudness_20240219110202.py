import pytest
from audio_analysis import get_loudness, load_audio 


def test_get_loudness(example_audio_file):

    audio, _ = load_audio(example_audio_file)
    loudness = get_loudness(audio)

    # assertions
    assert isinstance(loudness, float)