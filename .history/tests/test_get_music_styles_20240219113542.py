import pytest
import numpy as np
from audio_analysis import get_music_styles, get_embeddings, load_audio 


def test_get_loudness(example_audio_file):

    _, audio = load_audio(example_audio_file)
    discogs_embeddings, musicnn_embeddings = get_embeddings(audio)

    # assertions
    assert isinstance(discogs_embeddings, np.ndarray)
    assert isinstance(musicnn_embeddings, np.ndarray)