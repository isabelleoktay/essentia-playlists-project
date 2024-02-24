import pytest
import numpy as np
from audio_analysis import get_arousal_and_valence, get_embeddings, load_audio 


def test_get_arousal_and_valence(example_audio_file):

    _, mono_audio = load_audio(example_audio_file)
    _, musiCNN_embeddings = get_embeddings(mono_audio)

    predictions = get_arousal_and_valence(musiCNN_embeddings)

    # assertions
    assert isinstance(predictions, np.ndarray)