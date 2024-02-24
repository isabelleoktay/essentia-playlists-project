import pytest
import essentia.standard as es


# Mocking the RhythmExtractor2013 function
@pytest.fixture
def mock_rhythm_extractor():
    with pytest.mock.patch.object(es, 'RhythmExtractor2013') as mock_rhythm:
        yield mock_rhythm