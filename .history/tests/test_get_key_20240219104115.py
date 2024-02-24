import pytest
from audio_analysis import get_key, load_audio 


def test_get_key(example_audio_file):

    _, audio = load_audio(example_audio_file)
    keys_and_scales = get_key(audio)

    # assertions
    assert isinstance(keys_and_scales, dict)
    assert len(keys_and_scales) == 3  # expecting three profiles
    assert all(profile in keys_and_scales for profile in ['temperley', 'krumhans', 'edma'])
    for key, scale in keys_and_scales.values():
        assert isinstance(key, str)
        assert isinstance(scale, str)