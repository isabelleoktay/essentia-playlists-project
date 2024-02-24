import pytest
import numpy as np
from audio_analysis import classify_voice_or_instrument, get_embeddings, load_audio 


def test_classify_voice_or_instrument(example_audio_file):

    _, audio = load_audio(example_audio_file)
    discogs_embeddings, _ = get_embeddings(audio)

    predictions = classify_voice_or_instrument(discogs_embeddings)

    # Assertions
    assert isinstance(predictions, np.ndarray)