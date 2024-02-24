import pytest
from audio_analysis import get_key, load_audio 


# Test for get_tempo function
def test_get_key(example_audio_file):
    # Load example audio file
    _, audio = load_audio(example_audio_file)

    # Call the function under test
    keys_and_scales = get_key(audio)

    # Assertions
    assert isinstance(tempo, float)
    assert tempo > 0