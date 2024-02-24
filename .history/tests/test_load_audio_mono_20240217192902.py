import pytest
import essentia.standard as es
from audio_analysis import load_audio_mono

def test_load_audio_mono(example_audio_file):
    stereo_audio, mono_audio = load_audio_mono(example_audio_file)

    # Assert that stereo_audio is not None
    assert stereo_audio is not None

    # Assert that mono_audio is not None
    assert mono_audio is not None

    # Assert that mono_audio has only one channel
    assert len(mono_audio.shape) == 1

    # Assert that mono_audio has the same length as stereo_audio
    assert len(mono_audio) == len(stereo_audio)



