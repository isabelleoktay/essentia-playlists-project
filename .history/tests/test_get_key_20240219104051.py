import pytest
from audio_analysis import get_key, load_audio 


# Test for get_tempo function
def test_get_key(example_audio_file):
    # Load example audio file
    _, audio = load_audio(example_audio_file)

    # Call the function under test
    keys_and_scales = get_key(audio)

    # Assertions
    assert isinstance(keys_and_scales, dict)
    assert len(keys_and_scales) == 3  # Expecting three profiles
    assert all(profile in keys_and_scales for profile in ['temperley', 'krumhans', 'edma'])
    for key, scale in keys_and_scales.values():
        assert isinstance(key, str)
        assert isinstance(scale, str)