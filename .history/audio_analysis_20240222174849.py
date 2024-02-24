import essentia.standard as es
import numpy as np
import os


def get_tempo(mono_audio):
    try: 
        bpm, _, _, _, _ = es.RhythmExtractor2013()(mono_audio)
        return bpm
    
    except Exception as e:
        print(f"Error in get_tempo: {e}")
        return None


def get_key(mono_audio):
    keys_and_scales = {'temperley': None,
                       'krumhansl': None,
                       'edma': None}
    try:
        for profile in keys_and_scales:
            key, scale, _ = es.KeyExtractor(profileType=profile)(mono_audio)
            keys_and_scales[profile] = [key, scale]

        return keys_and_scales
    
    except Exception as e:
        print(f"Error in get_key: {e}")
        return None


def get_loudness(stereo_audio):
    try:
        _, _, integrated_loudness, _ = es.LoudnessEBUR128()(stereo_audio)

        return integrated_loudness
    
    except Exception as e:
        print(f"Error in get_loudness: {e}")
        return None


def get_embeddings(mono_audio):
    discogs_embeddings = None
    musiCNN_embeddings = None

    try:
        discogs_model = es.TensorflowPredictEffnetDiscogs(graphFilename="models/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
        discogs_embeddings = discogs_model(mono_audio)
    except Exception as e:
        print(f"Error in get_discogs_embeddings: {e}")

    try:
        musiCNN_model = es.TensorflowPredictMusiCNN(graphFilename="models/msd-musicnn-1.pb", output="model/dense/BiasAdd")
        musiCNN_embeddings = musiCNN_model(mono_audio)
    except Exception as e:
        print(f"Error in get_musiCNN_embeddings: {e}")

    return discogs_embeddings, musiCNN_embeddings


def get_music_styles(discogs_embeddings):
    try:
        discogs_model = es.TensorflowPredict2D(graphFilename="models/genre_discogs400-discogs-effnet-1.pb", input="serving_default_model_Placeholder", output="PartitionedCall:0")
        discogs_predictions = discogs_model(discogs_embeddings)
        discogs_mean_predictions = np.mean(discogs_predictions, axis=0)

        return discogs_mean_predictions
    
    except Exception as e:
        print(f"Error in get_music_styles: {e}")
        return None


def classify_voice_or_instrument(discogs_embeddings):
    try:
        discogs_model = es.TensorflowPredict2D(graphFilename="models/voice_instrumental-discogs-effnet-1.pb", output="model/Softmax")
        discogs_predictions = discogs_model(discogs_embeddings)
        discogs_mean_predictions = np.mean(discogs_predictions, axis=0)

        return discogs_mean_predictions
    
    except Exception as e:
        print(f"Error in classify_voice_or_instrument: {e}")
        return None

 
def get_danceability(discogs_embeddings):
    try:
        discogs_model = es.TensorflowPredict2D(graphFilename="models/danceability-discogs-effnet-1.pb", output="model/Softmax")
        discogs_predictions = discogs_model(discogs_embeddings)
        discogs_mean_predictions = np.mean(discogs_predictions, axis=0)

        return discogs_mean_predictions
    
    except Exception as e:
        print(f"Error in get_danceability: {e}")
        return None


def get_arousal_and_valence(musiCNN_embeddings):
    try:
        musiCNN_model = es.TensorflowPredict2D(graphFilename="models/emomusic-msd-musicnn-2.pb", output="model/Identity")
        musiCNN_predictions = musiCNN_model(musiCNN_embeddings)
        musiCNN_mean_predictions = np.mean(musiCNN_predictions, axis=0)

        return musiCNN_mean_predictions
    
    except Exception as e:
        print(f"Error in get_arousal_and_valence: {e}")
        return None


def load_audio(filename):
    stereo_audio = None
    mono_audio = None

    try:
        stereo_audio, _, num_channels, _, _, _ = es.AudioLoader(filename=filename)()
    except Exception as e:
        print(f"Error in loading stereo audio: {e}")

    try:
        if stereo_audio is not None:
            mono_audio = es.MonoMixer()(stereo_audio, num_channels)
    except Exception as e:
        print(f"Error in converting stereo to mono: {e}")

    return stereo_audio, mono_audio


def process_audio_files(directory):
    audio_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.ogg')):  
                audio_file = os.path.join(root, file)
                audio_files.append(audio_file)
    return audio_files