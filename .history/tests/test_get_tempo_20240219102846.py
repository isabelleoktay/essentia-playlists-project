import pytest
from unittest.mock import MagicMock
from audio_analysis import get_tempo  # Replace 'your_module' with the actual module containing get_tempo
import essentia.standard as es

# Mocking the RhythmExtractor2013 function
@pytest.fixture
def mock_rhythm_extractor(mocker):
    with mocker.patch.object(es, 'RhythmExtractor2013') as mock_rhythm:
        yield mock_rhythm

# Test for get_tempo function
def test_get_tempo(mock_rhythm_extractor):
    # Mocking the return value of RhythmExtractor2013
    mock_bpm = 120.0
    mock_rhythm_extractor.return_value = MagicMock(return_value=(mock_bpm, None, None, None, None))

    # Call the function under test
    audio_mock = MagicMock()
    tempo = get_tempo(audio_mock)

    # Assertions
    assert tempo == mock_bpm
    mock_rhythm_extractor.assert_called_once_with(audio_mock)

