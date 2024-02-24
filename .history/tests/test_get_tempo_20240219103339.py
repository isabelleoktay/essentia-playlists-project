import pytest
import numpy as np
import essentia.standard as es
from audio_analysis import get_tempo, load_audio 

# Mocking the AudioLoader function
@pytest.fixture
def mock_audio_loader(mocker, example_audio_file):
    with mocker.patch.object(es, 'AudioLoader') as mock_loader:
        # Provide mocked stereo audio and num_channels
        stereo_audio = np.random.rand(10000)  # Assuming 10000 samples for the audio
        num_channels = 2  # Stereo audio
        mock_loader.return_value = MagicMock(return_value=(stereo_audio, None, num_channels, None, None, None))
        yield mock_loader

# Test for get_tempo function
def test_get_tempo(mock_audio_loader, example_audio_file):
    # Load example audio file
    _, audio = load_audio(example_audio_file)

    # Call the function under test
    tempo = get_tempo(audio)

    # Assertions
    assert isinstance(tempo, float)
    assert tempo > 0

