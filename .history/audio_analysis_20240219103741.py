import essentia.standard as es
import IPython.display as ipd


def get_tempo(audio):
    bpm, _, _, _, _ = es.RhythmExtractor2013()(audio)

    return bpm


def get_key(audio):
    keys_and_scales = {'temperley': None,
                       'krumhans': None,
                       'edma': None}

    for profile in keys_and_scales:
        key, scale = es.KeyExtractor(audio, profileType=profile)
        keys_and_scales[profile] = [key, scale]

    return keys_and_scales


def get_loudness(audio):
    _, _, integrated_loudness, _ = es.LoudnessEBUR128(audio)

    return integrated_loudness


def get_embeddings(audio):
    discogs_model = es.TensorflowPredictEffnetDiscogs(graphFilename="discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
    discogs_embeddings = discogs_model(audio)

    musiCNN_model = es.TensorflowPredictMusiCNN(graphFilename="msd-musicnn-1.pb", output="model/dense/BiasAdd")
    musiCNN_embeddings = musiCNN_model(audio)

    return discogs_embeddings, musiCNN_embeddings


def get_music_styles(embeddings):
    model = es.TensorflowPredict2D(graphFilename="genre_discogs400-discogs-effnet-1.pb", input="serving_default_model_Placeholder", output="PartitionedCall:0")
    predictions = model(embeddings)

    return predictions


def classify_voice_or_instrument(embeddings):
    # need to average discogs and musiCNN embeddings predictions
    discogs_model = es.TensorflowPredict2D(graphFilename="voice_instrumental-discogs-effnet-1.pb", output="model/Softmax")
    discogs_predictions = discogs_model(embeddings)

    musiCNN_model = es.TensorflowPredict2D(graphFilename="voice_instrumental-msd-musicnn-1.pb", output="model/Softmax")
    musiCNN_predictions = musiCNN_model(embeddings)

    predictions_mean = (discogs_predictions + musiCNN_predictions) / 2

    return predictions_mean

 
def get_danceability(embeddings):
    model = es.TensorflowPredict2D(graphFilename="danceability-discogs-effnet-1.pb", output="model/Softmax")
    predictions = model(embeddings)

    return predictions


def get_arousal_and_valence(embeddings):
    model = es.TensorflowPredict2D(graphFilename="emomusic-msd-musicnn-2.pb", output="model/Identity")
    predictions = model(embeddings)

    return predictions


def load_audio(filename):
    stereo_audio, _, num_channels, _, _, _ = es.AudioLoader(filename=filename)()
    mono_audio = es.MonoMixer()(stereo_audio, num_channels)

    return stereo_audio, mono_audio