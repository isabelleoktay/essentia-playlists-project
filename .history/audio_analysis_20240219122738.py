import essentia.standard as es
import IPython.display as ipd
import numpy as np


def get_tempo(mono_audio):
    bpm, _, _, _, _ = es.RhythmExtractor2013()(mono_audio)

    return bpm


def get_key(mono_audio):
    keys_and_scales = {'temperley': None,
                       'krumhansl': None,
                       'edma': None}

    for profile in keys_and_scales:
        key, scale, _ = es.KeyExtractor(profileType=profile)(mono_audio)
        keys_and_scales[profile] = [key, scale]

    return keys_and_scales


def get_loudness(stereo_audio):
    _, _, integrated_loudness, _ = es.LoudnessEBUR128()(stereo_audio)

    return integrated_loudness


def get_embeddings(mono_audio):
    discogs_model = es.TensorflowPredictEffnetDiscogs(graphFilename="models/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
    discogs_embeddings = discogs_model(mono_audio)

    musiCNN_model = es.TensorflowPredictMusiCNN(graphFilename="models/msd-musicnn-1.pb", output="model/dense/BiasAdd")
    musiCNN_embeddings = musiCNN_model(mono_audio)

    return discogs_embeddings, musiCNN_embeddings


def get_music_styles(discogs_embeddings):
    discogs_model = es.TensorflowPredict2D(graphFilename="models/genre_discogs400-discogs-effnet-1.pb", input="serving_default_model_Placeholder", output="PartitionedCall:0")
    discogs_predictions = discogs_model(discogs_embeddings)
    discogs_mean_predictions = np.mean(discogs_predictions, axis=0)

    return discogs_mean_predictions


def classify_voice_or_instrument(discogs_embeddings):
    discogs_model = es.TensorflowPredict2D(graphFilename="models/voice_instrumental-discogs-effnet-1.pb", output="model/Softmax")
    discogs_predictions = discogs_model(discogs_embeddings)
    discogs_mean_predictions = np.mean(discogs_predictions, axis=0)

    return discogs_mean_predictions

 
def get_danceability(discogs_embeddings):
    discogs_model = es.TensorflowPredict2D(graphFilename="models/danceability-discogs-effnet-1.pb", output="model/Softmax")
    discogs_predictions = discogs_model(discogs_embeddings)
    discogs_mean_predictions = np.mean(discogs_predictions, axis=0)
    print(discogs_mean_predictions)

    return discogs_mean_predictions


def get_arousal_and_valence(musiCNN_embeddings):
    musiCNN_model = es.TensorflowPredict2D(graphFilename="models/emomusic-msd-musicnn-2.pb", output="model/Identity")
    musiCNN_predictions = musiCNN_model(musiCNN_embeddings)
    musiCNN_mean_predictions = np.mean(musiCNN_predictions, axis=0)

    return musiCNN_mean_predictions


def load_audio(filename):
    stereo_audio, _, num_channels, _, _, _ = es.AudioLoader(filename=filename)()
    mono_audio = es.MonoMixer()(stereo_audio, num_channels)

    return stereo_audio, mono_audio