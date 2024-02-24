import pytest
from audio_analysis import get_tempo, load_audio 


# Test for get_tempo function
def test_get_tempo(example_audio_file):
    # Load example audio file
    _, audio = load_audio(example_audio_file)

    # Call the function under test
    tempo = get_tempo(audio)

    # Assertions
    assert isinstance(tempo, float)
    assert tempo > 0

