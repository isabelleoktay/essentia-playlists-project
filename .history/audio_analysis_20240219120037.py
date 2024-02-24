import essentia.standard as es
import IPython.display as ipd


def get_tempo(audio):
    bpm, _, _, _, _ = es.RhythmExtractor2013()(audio)

    return bpm


def get_key(audio):
    keys_and_scales = {'temperley': None,
                       'krumhansl': None,
                       'edma': None}

    for profile in keys_and_scales:
        key, scale, _ = es.KeyExtractor(profileType=profile)(audio)
        keys_and_scales[profile] = [key, scale]

    return keys_and_scales


def get_loudness(audio):
    _, _, integrated_loudness, _ = es.LoudnessEBUR128()(audio)

    return integrated_loudness


def get_embeddings(audio):
    discogs_model = es.TensorflowPredictEffnetDiscogs(graphFilename="models/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
    discogs_embeddings = discogs_model(audio)

    musiCNN_model = es.TensorflowPredictMusiCNN(graphFilename="models/msd-musicnn-1.pb", output="model/dense/BiasAdd")
    musiCNN_embeddings = musiCNN_model(audio)

    return discogs_embeddings, musiCNN_embeddings


def get_music_styles(discogs_embeddings):
    discogs_model = es.TensorflowPredict2D(graphFilename="models/genre_discogs400-discogs-effnet-1.pb", input="serving_default_model_Placeholder", output="PartitionedCall:0")
    discogs_predictions = discogs_model(discogs_embeddings)

    return discogs_predictions


def classify_voice_or_instrument(discogs_embeddings):
    discogs_model = es.TensorflowPredict2D(graphFilename="models/voice_instrumental-discogs-effnet-1.pb", output="model/Softmax")
    discogs_predictions = discogs_model(discogs_embeddings)

    return discogs_predictions

 
def get_danceability(discogs_embeddings):
    discogs_model = es.TensorflowPredict2D(graphFilename="models/danceability-discogs-effnet-1.pb", output="model/Softmax")
    discogs_predictions = discogs_model(discogs_embeddings)

    return discogs_predictions


def get_arousal_and_valence(embeddings):
    model = es.TensorflowPredict2D(graphFilename="emomusic-msd-musicnn-2.pb", output="model/Identity")
    predictions = model(embeddings)

    return predictions


def load_audio(filename):
    stereo_audio, _, num_channels, _, _, _ = es.AudioLoader(filename=filename)()
    mono_audio = es.MonoMixer()(stereo_audio, num_channels)

    return stereo_audio, mono_audio