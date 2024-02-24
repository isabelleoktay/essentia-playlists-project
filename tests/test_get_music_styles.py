import pytest
import numpy as np
from audio_analysis import get_music_styles, get_embeddings, load_audio 


def test_get_loudness(example_audio_file):

    _, audio = load_audio(example_audio_file)
    discogs_embeddings, _ = get_embeddings(audio)

    predictions = get_music_styles(discogs_embeddings)

    # Assertions
    assert isinstance(predictions, np.ndarray)